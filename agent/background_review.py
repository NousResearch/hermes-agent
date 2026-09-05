"""Background memory/skill review — fork the agent to evaluate the turn. After every turn
``AIAgent.run_conversation`` may spawn a daemon thread that replays the conversation snapshot in a
forked :class:`AIAgent` and asks "should any skill/memory be saved or updated?". Writes go
straight to the memory + skill stores; the main conversation and prompt cache are never touched.
The fork inherits the parent's live runtime (provider, model, credentials, cached system prompt)
so it hits the same prefix cache, and runs under a dispatch-side tool whitelist."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent.thread_scoped_output import thread_scoped_silence

logger = logging.getLogger(__name__)

_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS = 2.0


class _BackgroundReviewRun:
    """Per-review cancellation and request-completion handshake."""

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()
        self.request_done = threading.Event()
        self._lock = threading.Lock()
        self._review_agent = None
        self._request_finished = self._cancel_dispatched = False

    def begin_request(self, review_agent: Any) -> bool:
        """Atomically admit the first provider-capable review phase."""
        with self._lock:
            if self.cancel_requested.is_set() or self._request_finished:
                return False
            self._review_agent = review_agent
            return True

    def cancel(self) -> Any:
        """Fence startup and return the running fork, if one was admitted."""
        with self._lock:
            self.cancel_requested.set()
            if self._review_agent is None or self._cancel_dispatched:
                return None
            self._cancel_dispatched = True
            return self._review_agent

    def mark_request_finished(self) -> bool:
        """Latch request completion once; the caller publishes the event."""
        with self._lock:
            if self._request_finished:
                return False
            self._request_finished, self._review_agent = True, None
            return True


@contextmanager
def _optional_lock(agent: Any, attr: str) -> Iterator[None]:
    """``with`` over a lock attribute that may be absent (direct test stubs)."""
    lock = getattr(agent, attr, None)
    if lock is None:
        yield
        return
    with lock:
        yield


def prepare_background_review_run(agent: Any) -> Optional[_BackgroundReviewRun]:
    """Install a unique run token on the parent before ``Thread.start()``."""
    run = _BackgroundReviewRun()
    try:
        lock = getattr(agent, "_background_review_lock", None)
        if lock is None:
            lock = agent._background_review_lock = threading.Lock()
        with lock:
            current = getattr(agent, "_background_review_run", None)
            if current is not None and not current.request_done.is_set():
                return None
            agent._background_review_run = run
    except (AttributeError, TypeError):
        return None
    return run


def finish_background_review_run(agent: Any, run: Optional[_BackgroundReviewRun]) -> None:
    """Publish one run's request exit without clearing a successor (ABA-safe)."""
    if run is None or not run.mark_request_finished():
        return
    with _optional_lock(agent, "_background_review_lock"):
        if getattr(agent, "_background_review_run", None) is run:
            agent._background_review_run = None
    run.request_done.set()


def _interrupt_background_review(review_agent: Any) -> None:
    """Request abort off-thread so a wedged abort hook cannot stall the live turn (the bounded
    ``request_done`` wait in the canceller relies on this returning fast)."""
    def _interrupt() -> None:
        try:
            from agent.interrupt_compat import request_hard_interrupt

            request_hard_interrupt(
                review_agent, "superseded by a new live turn", tool_reason="background review superseded"
            )
        except Exception:
            logger.debug("Failed to cancel in-flight background review for a new turn", exc_info=True)

    try:
        threading.Thread(target=_interrupt, daemon=True, name="bg-review-cancel").start()
    except Exception:
        logger.debug("Failed to start background-review cancellation thread", exc_info=True)


def cancel_background_review_for_live_turn(agent: Any) -> None:
    """Cancel the current review and await its request-phase acknowledgement. Foreground priority:
    past the bounded deadline, warn and let the live turn proceed — self-improvement work must
    never block a user-facing turn.

    Foreground priority is preserved: if the review does not acknowledge within the bounded deadline, a
    warning is logged and the live turn proceeds anyway. See #84423.
    """
    with _optional_lock(agent, "_background_review_lock"):
        run = getattr(agent, "_background_review_run", None)
        legacy_agent = getattr(agent, "_background_review_agent", None)
    review_agent = legacy_agent if run is None else run.cancel()
    # Attribute the review fork's usage to the PARENT session. Snapshot BEFORE unregister/close so counters
    # survive teardown. Placed in this finally so a fork that consumed tokens and THEN raised is still
    # attributed (issue #87250). Best-effort: the recorder never raises into the review thread.
    if review_agent is not None:
        _interrupt_background_review(review_agent)
    if run is None:
        return
    if not run.request_done.wait(timeout=_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS):
        logger.warning(
            "Background review did not acknowledge cancellation within %.1fs; "
            "proceeding with foreground live turn",
            _BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS,
        )


# Aux-model routing: by default ("auto") the fork runs on the MAIN model and replays the full
# conversation as warm cache reads. When auxiliary.background_review.{provider,model} routes it
# to a DIFFERENT model the cache is cold anyway, so the fork replays a compact digest instead.
_REVIEW_MAX_ITERATIONS = 16
# Aggregate INPUT-token budget for one review fork (checked in conversation_loop's
# ``_review_input_budget_exhausted``). Request #1 replays the full snapshot as a warm cache read
# (both compression gates deferred until the first response); compaction then bounds each
# request, but nothing else caps the SUM across the tool loop. 2x the historical 300k foreground
# trigger. Override via ``auxiliary.background_review.max_input_tokens``; <= 0 disables.
_REVIEW_MAX_INPUT_TOKENS_DEFAULT = 600_000


def _task_block(cfg: Any) -> Dict[str, Any]:
    """``cfg["auxiliary"]["background_review"]`` as a dict (``{}`` on any shape mismatch)."""
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {})
    return task if isinstance(task, dict) else {}


def _background_review_task_config(task_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``auxiliary.background_review`` (or ``{}`` on any failure); pass a pre-loaded ``task_cfg``
    so the spawn / resolve / prompt paths do not re-read config on every turn."""
    if task_cfg is not None:
        return task_cfg if isinstance(task_cfg, dict) else {}
    try:
        from hermes_cli.config import load_config_readonly
        return _task_block(load_config_readonly())
    except Exception:
        return {}


def _review_input_token_budget(task_cfg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Aggregate input-token budget for one review fork (None = unlimited; <= 0 disables)."""
    raw = _background_review_task_config(task_cfg).get("max_input_tokens", _REVIEW_MAX_INPUT_TOKENS_DEFAULT)
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        budget = _REVIEW_MAX_INPUT_TOKENS_DEFAULT
    return budget if budget > 0 else None


def load_background_review_settings() -> tuple[bool, Dict[str, Any]]:
    """Single config read -> ``(enabled, task_cfg)``. Fail-open (``enabled=True``) so a broken
    config never silently disables reviews — but WARN so the cost is visible."""
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value
        task = _task_block(load_config_readonly())
        return is_truthy_value(task.get("enabled"), default=True), task
    except Exception:
        logger.warning(
            "Failed to read background_review.enabled; leaving automatic "
            "review enabled (fail-open)",
            exc_info=True,
        )
        return True, {}


def _resolve_review_runtime(agent: Any, task_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve provider/model/credentials for the review fork. Default (auto / unset / same as
    parent): the parent's live runtime with ``routed=False`` (codex_app_server -> codex_responses
    downgrade applied). When ``auxiliary.background_review.{provider,model}`` names a different
    concrete model, resolve that runtime and set ``routed=True``."""
    parent_runtime = agent._current_main_runtime()
    parent_api_mode = parent_runtime.get("api_mode") or None
    parent = {
        "provider": agent.provider, "model": agent.model,
        "api_key": parent_runtime.get("api_key") or None, "base_url": parent_runtime.get("base_url") or None,
        "api_mode": "codex_responses" if parent_api_mode == "codex_app_server" else parent_api_mode,
        "credential_pool": getattr(agent, "_credential_pool", None),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "max_tokens": getattr(agent, "max_tokens", None), "command": getattr(agent, "acp_command", None),
        "args": list(getattr(agent, "acp_args", []) or []), "routed": False,
    }
    task = _background_review_task_config(task_cfg)
    task_provider, task_model, task_base_url, task_api_key = (
        str(task.get(key, "")).strip() or None for key in ("provider", "model", "base_url", "api_key")
    )
    if not (task_provider and task_provider != "auto" and task_model) or (
        task_provider == (agent.provider or "") and task_model == (agent.model or "")  # same as parent
    ):
        return parent
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rp = resolve_runtime_provider(
            requested=task_provider, target_model=task_model,
            explicit_api_key=task_api_key, explicit_base_url=task_base_url,
        )
        return {
            "provider": rp.get("provider") or task_provider, "model": rp.get("model") or task_model,
            **{key: rp.get(key) for key in ("api_key", "base_url", "api_mode", "credential_pool", "command")},
            "request_overrides": dict(rp.get("request_overrides") or {}),
            "max_tokens": rp.get("max_output_tokens"), "args": list(rp.get("args") or []), "routed": True,
        }
    except Exception as e:
        logger.debug("background-review aux routing failed (%s); using main model", e)
        return parent


def _parent_can_emit_tool_calls(agent: Any) -> bool:
    """Whether a fork inheriting ``agent``'s runtime could act at all: an agent-as-provider client
    shim declaring ``SUPPORTS_HERMES_TOOL_CALLS = False`` (instance or class) is skipped — the fork
    would be a guaranteed no-op that still pays a full spawn. Silence means capable."""
    client = getattr(agent, "client", None)
    for candidate in (client, type(client) if client is not None else None):
        supported = getattr(candidate, "SUPPORTS_HERMES_TOOL_CALLS", None)
        if candidate is not None and supported is not None:
            return bool(supported)
    return True


def _msg_text(m: Dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c.strip() if isinstance(c, str) else ""


def _digest_history(messages_snapshot: List[Dict], tail: int = 24) -> List[Dict]:
    """Compact replay for the routed (different-model) path only: keep the recent ``tail``
    messages verbatim (extended so the kept run never starts on a tool result) and collapse older
    turns into one synthetic user-role digest, preserving role alternation."""
    msgs = list(messages_snapshot or [])
    while len(msgs) > tail:
        keep = msgs[-tail:]
        if not (isinstance(keep[0], dict) and keep[0].get("role") == "tool"):
            break
        tail += 1
    else:
        return msgs
    lines: List[str] = []
    for m in msgs[:-len(keep)]:
        if not isinstance(m, dict):
            continue
        role, text = m.get("role"), _msg_text(m).replace("\n", " ")
        if role == "user" and text:
            lines.append(f"USER: {text[:300]}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = [(tc.get("function") or {}).get("name", "?") for tc in m["tool_calls"] if isinstance(tc, dict)]
                lines.append(f"ASSISTANT[tools: {', '.join(names)}]")
            if text:
                lines.append(f"ASSISTANT: {text[:200]}")
    digest = (
        "[Earlier conversation digest — older turns summarised to bound the "
        "review's cold-write cost on the routed aux model. Recent turns "
        "follow verbatim below.]\n" + "\n".join(lines)
    )
    return [{"role": "user", "content": digest}] + keep


# Review prompts. AIAgent exposes them as class attributes (``_MEMORY_REVIEW_PROMPT`` etc.) so
# per-agent overrides work; the text lives here.
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, preferences, or "
    "personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work style, or ways "
    "they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. If nothing is worth saving, just say "
    "'Nothing to save.' and stop."
)

# Shared shape contract for anything written into a skill. The failure mode this prevents is the
# hoarding library: one references/ file per session, incident narration instead of rules, PR numbers
# and quotes as content, and duplicating what the repo's AGENTS.md / the tool schemas already teach.
_LESSON_LAYER_BLOCK = (
    "What a skill IS: the instructions for doing a class of task the most efficient and correct "
    "way, to THIS user's specifications — the procedure, the tools and commands that work, the "
    "order, the user's preferences for how the result should look, and the pitfalls that cost time. "
    "A future session should be able to follow it and produce what the user wants on the first "
    "try. Everything below is about writing that well:\n"
    "  • Procedure first: the steps in the order they are done, with the concrete commands, tool "
    "calls, and decision points. Lessons and pitfalls attach to the step they affect.\n"
    "  • A pitfall is a generalizable rule + one clause of WHY (the mechanism), imperative. 'Grep the "
    "test tree for the SYMBOL before widening a helper signature — hand-rolled mocks reimplement the "
    "old shape and fail on a shard you did not run.' Not a narrative of what happened this session.\n"
    "  • No PR/issue numbers, dates, ticket IDs, or quoted user text as content — the rule must stand "
    "without the incident behind it. Keep a short quote ONLY when the quote itself is the clearest "
    "statement of the rule.\n"
    "  • The same lesson learned twice is ONE rule. Before adding, search the skill (and its "
    "references/) for the rule already stated; strengthen or clarify it rather than appending a "
    "second copy.\n"
    "  • Not a duplicate of what the environment already teaches: repo AGENTS.md files, tool schema "
    "descriptions, and other always-loaded context. A skill carries the WORKFLOW and the pitfalls; "
    "it does not restate the codebase map or a tool's parameter list.\n"
    "  • Always-on rules (standing user preferences, gates that apply to every instance of the "
    "task) live in SKILL.md itself, whole. references/ is for depth that is only needed sometimes: "
    "a decision table, a recipe, a domain note — each file topical and reusable, never "
    "'<date>-<incident>.md'. Prefer extending an existing references/ file over creating one; "
    "a skill with dozens of one-off references is the failure shape, not the goal.\n"
    "  • Fix the skill in place when it is wrong: edit the sentence that misled, do not append "
    "'UPDATE: actually...' underneath it.\n\n"
)

# Shared tail of the skill and combined prompts: what NOT to persist as a skill.
_DO_NOT_CAPTURE_BLOCK = (
    " (these become persistent self-imposed constraints that bite you later when the environment "
    "changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install errors, post-migration "
    "path mismatches, 'command not found', unconfigured credentials, uninstalled packages. The "
    "user can fix these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not work', 'X tool is broken', "
    "'cannot use Y from execute_code'). These harden into refusals the agent cites against itself "
    "for months after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the conversation ended. If "
    "retrying worked, the lesson is the retry pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's market' or 'analyze this PR' is "
    "not a class of work that warrants a skill.\n\n"
    "  • Unresolved failures: if the session ended WITHOUT actually finding a working method — you "
    "tried several things, none worked, and told the user to check manually — do NOT write those "
    "attempts up as a 'reliable workflow' or 'recommended approach'. That presents an untested "
    "sequence of failures as validated guidance a future session will trust and repeat. Either say "
    "'Nothing to save', or, only if you are independently confident of a real working alternative "
    "(not something you are merely guessing might work), capture ONLY that alternative — never the "
    "dead ends, and never dressed up as best practice.\n\n"
    "If a tool failed because of setup state, capture the FIX (install command, config step, env "
    "var to set) under an existing setup or troubleshooting skill — never 'this tool does not "
    "work' as a standalone constraint.\n\n"
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be ACTIVE — most sessions produce "
    "at least one skill update, even if small. A pass that does nothing is a missed learning "
    "opportunity, not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a SKILL.md of always-on rules and a "
    "small `references/` set of topical depth. Not a flat list of narrow one-session skills, and "
    "not an umbrella hoarding a references/ file per session. This shapes HOW you update, not "
    "WHETHER you update.\n\n" + _LESSON_LAYER_BLOCK +
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or verbosity. Frustration signals "
    "like 'stop doing X', 'this is too verbose', 'don't format like this', 'why are you "
    "explaining', 'just give me the answer', 'you always do Y and I hate it', or an explicit "
    "'remember this' are FIRST-CLASS skill signals, not just memory signals. Update the relevant "
    "skill(s) to embed the preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. Encode the correction as a "
    "pitfall or explicit step in the skill that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or tool-usage pattern emerged "
    "that a future session would benefit from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out to be wrong, missing a step, "
    "or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do pick one when a signal above "
    "fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the conversation for skills the user "
    "loaded via /skill-name or you read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first (re-load it with skill_view during this review — see "
    "Read-before-write below). It is the skill that was in play, so it's the right one to extend — "
    "but only if it is curator-managed. Bundled, hub, pinned, and user-owned skills are off-limits "
    "to you no matter how relevant (see Protected skills below); for those, fall through to the "
    "next option.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). If no loaded skill fits but "
    "an existing class-level skill does, patch it. Add a subsection, a pitfall, or broaden a "
    "trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be packaged with three kinds "
    "of support files — use the right directory per kind:\n"
    "     • `references/<topic>.md` — topical depth needed only sometimes: a decision table, a "
    "reproduction recipe, provider quirks, condensed domain notes or API excerpts. Name it by "
    "TOPIC and extend an existing file when one covers the topic; do not create a per-session or "
    "per-incident file, and do not paste error transcripts — distill them to the rule.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be copied and modified (boilerplate "
    "configs, scaffolding, a known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions the skill can invoke directly "
    "(verification scripts, fixture generators, deterministic probes, anything the agent should "
    "run rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with file_path starting "
    "'references/', 'templates/', or 'scripts/'. The umbrella's SKILL.md should gain a one-line "
    "pointer to any new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing skill covers the class. The "
    "name MUST be at the class level. The name MUST NOT be a specific PR number, error string, "
    "feature codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' session artifact. "
    "If the proposed name only makes sense for today's task, it's wrong — fall back to (1), (2), "
    "or (3).\n\n"
    "Read-before-write (ENFORCED — skill_manage refuses otherwise): before you patch or edit an "
    "existing skill's SKILL.md, call skill_view(name) for that skill during this review. Before "
    "you overwrite or remove an EXISTING supporting file, call skill_view(name, file_path=...) for "
    "that exact file. Content quoted earlier in the conversation transcript does NOT count — the "
    "guard requires a fresh load within this review, and your write must be based on what "
    "skill_view just returned. Creating a brand-new skill or adding a NEW supporting file needs no "
    "prior read. If a write is refused with a read-before-write error, call skill_view for the "
    "named target once and retry the write once; do not loop.\n\n"
    "User-preference embedding (important): when the user expressed a style/format/workflow "
    "preference, the update belongs in the SKILL.md body, not just in memory. Memory captures 'who "
    "the user is and what the current situation and state of your operations are'; skills capture "
    "'how to do this class of task for this user'. When they complain about how you handled a "
    "task, the skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your reply — the background "
    "curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "  • Skills in skills.external_dirs (externally owned).\n"
    "  • PINNED skills (marked via 'hermes curator pin'). You are an autonomous no-user-present "
    "actor, so pin blocks your writes too — content updates included. Only the user, in a "
    "foreground session, can change a pinned skill.\n"
    "  • USER-OWNED skills — anything not curator-managed. A skill the user hand-wrote, installed "
    "by URL, or asked a foreground agent to create is theirs, not yours; your writes to it WILL be "
    "refused. This includes skills that were loaded or consulted this session: being in play does "
    "not make one yours to edit. If such a skill is wrong or outdated, say so in your reply and "
    "recommend 'hermes curator adopt <name>' — do not try to patch it.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture" + _DO_NOT_CAPTURE_BLOCK +
    "'Nothing to save.' is a real option but should NOT be the default. If the session ran "
    "smoothly with no corrections and produced no new technique, just say 'Nothing to save.' and "
    "stop. Otherwise, act."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, desires, preferences, personal "
    "details, or expectations about how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most sessions produce at least one "
    "skill update. A pass that does nothing is a missed learning opportunity, not a neutral "
    "outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a SKILL.md of always-on rules and a "
    "small `references/` set of topical depth — not narrow one-session skills, and not an umbrella "
    "hoarding a references/ file per session.\n\n" + _LESSON_LAYER_BLOCK +
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, verbosity, or approach. Frustration "
    "is a FIRST-CLASS skill signal, not just a memory signal. 'stop doing X', 'don't format like "
    "this', 'I hate when you Y' — embed the lesson in the skill that governs that task so the next "
    "session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, missing, or outdated — patch it "
    "now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were loaded via /skill-name or "
    "skill_view in the conversation. If one of them covers the learning, PATCH it first (re-load "
    "it with skill_view during this review — see Read-before-write below). It was in play; it's "
    "the right place — provided it is curator-managed. Protected and user-owned skills are "
    "off-limits however relevant; fall through when one of those is the best fit.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via skill_manage action=write_file. Three "
    "kinds: `references/<topic>.md` for topical depth (decision tables, recipes, quirks, condensed "
    "domain notes) — extend an existing topical file before creating one, never a per-session file; "
    "`templates/<name>.<ext>` for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions (verification, fixture generators, "
    "probes). Add a one-line pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. Name at the class level — NOT a "
    "PR number, error string, codename, library-alone name, or 'fix-X / debug-Y' session artifact. "
    "If the name only fits today's task, fall back to (1), (2), or (3).\n\n"
    "Read-before-write (ENFORCED — skill_manage refuses otherwise): before patching or editing an "
    "existing skill's SKILL.md, call skill_view(name) during this review; before overwriting or "
    "removing an EXISTING supporting file, call skill_view(name, file_path=...) for that exact "
    "file. Content quoted earlier in the transcript does NOT count — base the write on what "
    "skill_view just returned. New skills and NEW supporting files need no prior read. On a "
    "read-before-write refusal: view the named target once, retry the write once, do not loop.\n\n"
    "User-preference embedding: when the user complains about how you handled a task, update the "
    "skill that governs that task — memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; skills say 'how to do this "
    "class of task for this user'. Both should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the background curator handles "
    "consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "  • Skills in skills.external_dirs (externally owned).\n"
    "  • PINNED skills (marked via 'hermes curator pin'). Pin blocks autonomous writes entirely — "
    "content updates included — because no user is present to consent. Only a foreground session "
    "can change one.\n"
    "  • USER-OWNED skills — anything not curator-managed (hand-written, URL-installed, or created "
    "by a foreground agent at the user's request). Your writes to these WILL be refused, including "
    "to skills loaded or consulted this session. If one is wrong, say so in your reply and "
    "recommend 'hermes curator adopt <name>' instead.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills" + _DO_NOT_CAPTURE_BLOCK +
    "Act on whichever of the two dimensions has real signal. If genuinely nothing stands out on "
    "either, say 'Nothing to save.' and stop — but don't reach for that conclusion as a default."
)


def _preview(text: str, limit: int) -> str:
    return text[:limit] + ("…" if len(text) > limit else "")


# Memory op -> (glyph, which field carries the preview, preview length).
_MEMORY_OP_FORMATS: Dict[str, Tuple[str, str, int]] = {
    "add": ("➕", "content", 120), "replace": ("✏️", "content", 120), "remove": ("➖", "old_text", 60)
}


def _memory_op_line(label: str, action: str, fields: Dict[str, str]) -> Optional[str]:
    """Verbose line for one memory add/replace/remove, or None when no preview text."""
    glyph, field_name, limit = _MEMORY_OP_FORMATS.get(action) or (None, "", 0)
    text = fields.get(field_name) or "" if glyph else ""
    return f"{label} {glyph} {_preview(text, limit)}" if text else None


def _verbose_skill_line(data: Dict, detail: Dict, message: str) -> str:
    action = detail.get("action", "")
    skill_name = detail.get("name", "")
    # ``_change`` is free-form (wrapper MCP backends return lists/scalars).
    change_raw = data.get("_change")
    change: dict = change_raw if isinstance(change_raw, dict) else {}
    old_string = change.get("old", "") or detail.get("old_string", "")
    new_string = change.get("new", "") or detail.get("new_string", "")
    if action == "patch" and (old_string or new_string):
        old_preview, new_preview = (_preview(t, 80).replace("\n", " ") for t in (old_string, new_string))
        return f"📝 Skill '{skill_name}' patched: \"{old_preview}\" → \"{new_preview}\""
    verb = {"create": "created", "edit": "rewritten"}.get(action)
    if verb and change.get("description"):
        return f"📝 Skill '{skill_name}' {verb}: {change['description']}"
    return f"📝 {message}" if message else f"Skill {action}"


def _verbose_memory_lines(label: str, detail: Dict) -> List[str]:
    # ``operations`` may be any JSON value; only a list of dicts is usable.
    ops_raw = detail.get("operations")
    if isinstance(ops_raw, list) and ops_raw:
        lines = [_memory_op_line(label, op.get("action", ""), op) for op in ops_raw if isinstance(op, dict)]
        return [line for line in lines if line]
    return [_memory_op_line(label, detail.get("action", ""), detail) or f"{label} updated"]


# Tool-call argument fields surfaced in action summaries, with their defaults.
_CALL_DETAIL_DEFAULTS = (
    ("action", "?"), ("target", "memory"), ("content", ""), ("old_text", ""), ("name", ""),
    ("old_string", ""), ("new_string", ""),
)


def _collect_review_call_details(review_messages: List[Dict]) -> Tuple[set, dict]:
    """Map review-agent tool_call ids -> parsed call arguments for notify tools. Result JSON only
    says "Entry added"; the call arguments carry action, target and content previews. Restricting
    to notify tools keeps helper tools from surfacing as memory work just because they succeeded."""
    notify_tools = {"memory", "skill_manage"}
    all_tool_call_ids: set = set()
    call_details: dict = {}
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            fn_name = fn.get("name", "")
            tcid = tc.get("id")
            if tcid:
                all_tool_call_ids.add(tcid)
            if fn_name not in notify_tools or not tcid:
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            call_details[tcid] = {
                "tool": fn_name, "operations": args.get("operations") or [],
                **{k: args.get(k, default) for k, default in _CALL_DETAIL_DEFAULTS},
            }
    return all_tool_call_ids, call_details


def _tool_messages(messages: List[Dict]) -> Iterator[Dict]:
    return (m for m in messages or [] if isinstance(m, dict) and m.get("role") == "tool")


def _prior_tool_keys(prior_snapshot: List[Dict]) -> Tuple[set, set]:
    """``(tool_call_ids, contents)`` of tool messages already in the parent snapshot."""
    priors = list(_tool_messages(prior_snapshot))
    ids = {m["tool_call_id"] for m in priors if m.get("tool_call_id")}
    contents = {m["content"] for m in priors if not m.get("tool_call_id") and isinstance(m.get("content"), str)}
    return ids, contents


def _action_lines(data: Dict, detail: Dict, verbose: bool) -> List[str]:
    """Summary line(s) for one successful notify-tool result (``[]`` when nothing to report)."""
    message = data.get("message", "")
    target = data.get("target", "") or detail.get("target", "")
    is_skill = detail.get("tool") == "skill_manage"
    lower = message.lower()
    if not verbose and ("created" in lower or "updated" in lower or (is_skill and "patched" in lower)):
        return [message]
    if not is_skill and not target:
        return []
    label = "Skill" if is_skill else {"memory": "Memory", "user": "User profile"}.get(target, target)
    if verbose:
        return [_verbose_skill_line(data, detail, message)] if is_skill else _verbose_memory_lines(label, detail)
    hit = any(k in lower for k in ("added", "replaced", "removed", "applied")) or (target and "add" in lower)
    return [f"{label} updated"] if hit else []


def summarize_background_review_actions(
    review_messages: List[Dict], prior_snapshot: List[Dict], notification_mode: str = "on"
) -> List[str]:
    """Human-facing action summary for a background review pass: successful memory /
    skill-management tool results from the review agent's messages, skipping tool messages already
    present in ``prior_snapshot`` so inherited results are not re-surfaced as fresh work.
    ``notification_mode``: ``off`` -> no actions; ``on`` -> generic "Memory updated"/tool messages;
    ``verbose`` -> content previews from the tool-call arguments.

    See #14944.
    """
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    verbose = mode == "verbose"
    existing_tool_call_ids, existing_tool_contents = _prior_tool_keys(prior_snapshot)
    all_tool_call_ids, call_details = _collect_review_call_details(review_messages)
    actions: List[str] = []
    for msg in _tool_messages(review_messages):
        tcid = msg.get("tool_call_id")
        if tcid:
            if tcid in existing_tool_call_ids or (all_tool_call_ids and tcid not in call_details):
                continue
        elif isinstance(msg.get("content"), str) and msg["content"] in existing_tool_contents:
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        # Wrapper MCP servers may return a top-level list/scalar; only dict payloads carry
        # ``success``/``_change``.
        if not isinstance(data, dict) or not data.get("success"):
            continue
        actions.extend(_action_lines(data, call_details.get(tcid) or {}, verbose))
    return actions


def build_memory_write_metadata(
    agent: Any, *, write_origin: Optional[str] = None, execution_context: Optional[str] = None,
    task_id: Optional[str] = None, tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for external memory-provider mirrors."""
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": execution_context or getattr(agent, "_memory_write_context", "foreground"),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
        "task_id": task_id or None,
        "tool_call_id": tool_call_id or None,
    }
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


_USAGE_COUNTERS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_calls",
)


def _snapshot_review_usage(review_agent: Any) -> Dict[str, Any]:
    """Snapshot in-memory usage counters from a review fork (pre-close)."""
    return {
        **{key: getattr(review_agent, key, None) for key in ("model", "provider", "base_url")},
        **{key: int(getattr(review_agent, f"session_{key}", 0) or 0) for key in _USAGE_COUNTERS},
        "estimated_cost_usd": getattr(review_agent, "session_estimated_cost_usd", None),
    }


def _record_review_usage_to_parent(parent_agent: Any, usage: Dict[str, Any]) -> None:
    """Record a fork's usage against the parent session (best-effort, never raises). The fork has
    ``_session_db = None`` so conversation_loop's DB-gated accounting never sees its calls; route
    them through the aux-accounting chokepoint, which writes only ``session_model_usage`` — never
    the transcript or ``sessions`` row."""
    try:
        session_db = getattr(parent_agent, "_session_db", None)
        session_id = getattr(parent_agent, "session_id", None)
        counts = {key: int(usage.get(key) or 0) for key in _USAGE_COUNTERS}
        if session_db is None or not session_id or not any(counts.values()):
            return  # no DB, or the fork made no successful API calls (e.g. failed at spawn)
        session_db.record_auxiliary_usage(
            session_id, task="background_review", model=usage.get("model"),
            billing_provider=usage.get("provider"), billing_base_url=usage.get("base_url"),
            estimated_cost_usd=usage.get("estimated_cost_usd"),
            api_call_count=counts.pop("api_calls"), **counts,
        )
    except Exception as e:
        logger.debug("Background review usage recording failed (non-fatal): %s", e)


def _classify_review_result(actions: List[str]) -> str:
    """Map a review action summary to ``none`` / ``skill`` / ``memory`` / ``skill+memory``.
    Prefix-based on the formats :func:`summarize_background_review_actions` emits (``Skill …``,
    ``📝 Skill …``, ``Memory …``, ``User profile …``), so a free-text line like ``Skipped: no
    skill worth saving`` stays ``none``."""
    lowers = [str(action).lstrip().removeprefix("📝").lstrip().lower() for action in actions or []]
    has_skill = any(t.startswith("skill") for t in lowers)
    has_memory = any(t.startswith(("memory", "user profile")) for t in lowers)
    return "+".join(kind for kind, hit in (("skill", has_skill), ("memory", has_memory)) if hit) or "none"


def _log_review_completion(usage: Dict[str, Any], result: str) -> None:
    """Emit a per-fork completion line so cost is visible where it is incurred."""
    logger.info(
        "Background review complete: thread=bg-review calls=%d in=%d out=%d "
        "cache_read=%d result=%s",
        *(int(usage.get(k) or 0) for k in ("api_calls", "input_tokens", "output_tokens", "cache_read_tokens")),
        result,
    )


# OpenRouter provider-routing pins: prompt caches live per UPSTREAM provider, so a fork without
# the parent's pins can land on a different upstream and miss the warm cache even with
# byte-identical prompt/tools bytes.
_PROVIDER_PIN_ATTRS = (
    "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
    "provider_require_parameters", "provider_data_collection",
)


def _same_model_parity_kwargs(agent: Any) -> Dict[str, Any]:
    """AIAgent kwargs that keep a SAME-model fork's request bytes identical to the parent's. Only
    for the un-routed path: on a different model the cache is cold anyway, and the parent's
    reasoning-effort vocabulary may be invalid for the routed provider (OpenRouter forwards
    ``reasoning.effort`` unclamped; codex_responses passes ``max``/``ultra`` through unmapped)."""
    kwargs: Dict[str, Any] = {
        # Anthropic's cache key is namespaced by ``thinking`` presence; the gateway session context
        # is appended to the cached system prompt at API-call time (without it the prompt diverges).
        "reasoning_config": getattr(agent, "reasoning_config", None),
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None),
        **{attr: val for attr in _PROVIDER_PIN_ATTRS if (val := getattr(agent, attr, None))},
    }
    # Prefill sits right after the system message, so a parent with prefill would diverge at
    # index 1. Deep copy: unicode-error recovery sanitizes prefill entries IN PLACE and must not
    # rewrite the parent's bytes.
    if parent_prefill := copy.deepcopy(getattr(agent, "prefill_messages", None) or []):
        kwargs["prefill_messages"] = parent_prefill
    return kwargs


def _detach_fork_compression(review_agent: Any) -> None:
    """Detached in-memory compaction for a fork sharing the parent's session_id. Disabling
    compression (the old guard against compacting the parent's live session) removed the only
    bound on the review's snapshot. Persistence is already off, so compaction can only rewrite the
    fork's transcript — but the compressor's own SessionDB/session_id binding must be severed too,
    or cooldown/streak counters land on the parent's row. Force in-place mode and re-enable
    compression ONLY after the rebind succeeded (fail-closed); gates stay deferred until the first
    response so request #1 is a warm cache read."""
    bind = getattr(getattr(review_agent, "context_compressor", None), "bind_session_state", None)
    detached = False
    if callable(bind):
        try:
            # Plugin/third-party context engines may reject these kwargs; they own their
            # persistence policy, so a failed rebind never aborts the review.
            bind(session_db=None, session_id="")
            detached = True
        except Exception:
            # FAIL-CLOSED: the compressor may still point at the parent's SessionDB; enabling
            # compression would re-open the sibling race.
            logger.warning(
                "background-review compressor detachment failed; "
                "keeping compression DISABLED on this review fork "
                "(fail-closed, issue #93057 / #38727)",
                exc_info=True,
            )
    review_agent.compression_in_place = True
    review_agent.compression_enabled = detached
    if detached:
        review_agent._review_defer_compaction_before_first_response = True


def _fork_init_kwargs(agent: Any, rt: Dict[str, Any], routed: bool, max_iterations: int) -> Dict[str, Any]:
    """AIAgent constructor kwargs for the review fork. skip_memory=True: an external memory plugin
    scoped to the parent's session_id would leak the harness prompt into the user's real memory
    namespace; built-in MEMORY.md/USER.md state is re-bound by the caller. Toolsets match the
    parent so ``tools[]`` is byte-identical (Anthropic's cache key includes it); the runtime
    whitelist restricts dispatch."""
    kwargs: Dict[str, Any] = {
        "model": rt.get("model") or agent.model, "max_iterations": max_iterations, "quiet_mode": True,
        "platform": agent.platform, "provider": rt.get("provider") or agent.provider,
        "api_mode": rt.get("api_mode"), "base_url": rt.get("base_url") or None,
        "api_key": rt.get("api_key") or None, "credential_pool": rt.get("credential_pool"),
        "request_overrides": rt.get("request_overrides") or {}, "parent_session_id": agent.session_id,
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None),
        "disabled_toolsets": getattr(agent, "disabled_toolsets", None), "skip_memory": True,
    }
    if isinstance(rt.get("max_tokens"), int):
        kwargs["max_tokens"] = rt["max_tokens"]
    if isinstance(rt.get("command"), str) and rt["command"]:
        kwargs.update(acp_command=rt["command"], acp_args=rt.get("args") or [])
    if not routed:
        kwargs.update(_same_model_parity_kwargs(agent))
    return kwargs


# Sentinel generation above any live registry generation. Mid-review compaction
# boundaries re-run refresh_agent_mcp_tools(content_aware=True); freezing the
# snapshot generation prevents compaction from rebuilding agent.tools and dropping
# inherited provider tools (#103579).
_FROZEN_TOOL_SNAPSHOT_GENERATION: int = 2_147_483_647


def _inherit_parent_tool_surface(review_agent: Any, agent: Any) -> None:
    """Copy the parent's advertised tools for a same-model cache-parity fork.

    The parent's array is the last outbound payload: ``agent.tools`` is read
    directly at request time and is not mutated between review turns because
    ``_skip_mcp_refresh`` blocks the between-turn MCP refresh. Dispatch remains
    restricted by the thread whitelist. Copying the whole surface covers every
    dynamically injected tool (memory-provider, late MCP, and plugin-registered),
    not just memory tools.

    Freezes ``_tool_snapshot_generation`` so mid-review compaction boundaries
    (which invoke ``refresh_agent_mcp_tools(content_aware=True)``) refuse the
    rebuild and do not clobber the inherited tools (#103579).
    """
    parent_tools = getattr(agent, "tools", None)
    if not isinstance(parent_tools, (list, tuple)) or not parent_tools:
        return
    review_agent.tools = copy.deepcopy(list(parent_tools))
    review_agent.valid_tool_names = {
        entry["function"]["name"]
        for entry in review_agent.tools
        if isinstance(entry, dict)
        and isinstance(entry.get("function"), dict)
        and isinstance(entry["function"].get("name"), str)
    }
    review_agent._tool_snapshot_generation = _FROZEN_TOOL_SNAPSHOT_GENERATION


def build_cache_parity_fork(
    agent: Any, task_cfg: Optional[Dict[str, Any]] = None, *, max_iterations: int,
    write_origin: str = "background_review",
) -> Tuple[Any, Dict[str, Any], bool]:
    """Construct a detached AIAgent fork with warm prompt-cache parity (shared with ``/btw``): same
    runtime/credentials as the parent, byte-identical system prompt / tools[] / reasoning config on
    the same-model path, shared session_id for prefix warmth, full persistence detachment (no
    state.db writes, rotation, or external memory providers; in-place-only compaction). Returns
    ``(fork_agent, runtime_dict, routed)``; ``routed`` means a different model (cache cold —
    replay a digest). The caller owns registration, whitelisting, running, usage attribution and
    teardown."""
    from run_agent import AIAgent  # local: avoids a circular import at load
    # Inherit the parent's live runtime: AIAgent.__init__'s env auto-resolution fails for
    # OAuth-only providers, session-scoped creds and credential pools.
    _rt = _resolve_review_runtime(agent, task_cfg)
    _routed = bool(_rt.get("routed"))
    review_agent = AIAgent(**_fork_init_kwargs(agent, _rt, _routed, max_iterations))
    review_agent._memory_write_origin = review_agent._memory_write_context = write_origin
    review_agent._memory_store = agent._memory_store
    review_agent._memory_enabled = agent._memory_enabled
    review_agent._user_profile_enabled = agent._user_profile_enabled
    review_agent._memory_nudge_interval = review_agent._skill_nudge_interval = 0
    # _skip_mcp_refresh: the between-turns MCP refresh would add late-connecting MCP tools and
    # break tools[] parity. PERSISTENCE ISOLATION (curator-takeover root cause): sharing the
    # parent's session_id, the fork would otherwise write its harness turn into the REAL session,
    # which the next live turn re-reads as a standing instruction; close() must likewise not
    # finalize the parent's still-active session row. suppress_status_output: fork status/warning
    # emits go via _print_fn/status_callback, which bypass the stdout redirect.
    review_agent._skip_mcp_refresh = review_agent._persist_disabled = review_agent.suppress_status_output = True
    review_agent._session_json_enabled = review_agent._end_session_on_close = False
    review_agent._session_db = None
    review_agent.session_id = agent.session_id
    # Same model only: share the warm cached system prompt (~26% cost cut; a rebuilt prompt misses
    # the byte-exact prefix key) and pin session_start so any re-render (compression, plugin
    # hooks) stays byte-identical.
    # Inherit the parent's cached system prompt verbatim so the review fork's outbound HTTP request hits the
    # same Anthropic/OpenRouter prefix cache the parent warmed. Without this, the fork rebuilds the system
    # prompt from scratch (fresh _hermes_now() timestamp, fresh session_id, narrower toolset → different
    # skills_prompt) and the byte-exact prefix-cache key misses. See issue #25322 and PR #17276 for the full
    # analysis + measured impact (~26% end-to-end cost reduction on Sonnet 4.5). When routed to a different
    # model the parent's cached prompt is for the wrong model/cache key and would miss anyway, so let the
    # routed fork build its own.
    if not _routed:
        review_agent._cached_system_prompt = agent._cached_system_prompt
        review_agent.session_start = agent.session_start
        _inherit_parent_tool_surface(review_agent, agent)
    _detach_fork_compression(review_agent)
    # Compaction bounds a single request; this bounds the WHOLE review (checked in
    # conversation_loop via _review_input_budget_exhausted).
    review_agent._review_input_token_budget = _review_input_token_budget(task_cfg)
    return review_agent, _rt, _routed


# Install a non-interactive approval callback on this worker thread so any dangerous-command guard the
# review agent trips resolves to "deny" instead of falling back to input() -- which deadlocks against the
# parent's prompt_toolkit TUI (#15216). Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
def _bg_review_auto_deny(command, description, **kwargs):
    """Non-interactive approval: dangerous-command guards resolve to "deny" instead of input(),
    which would deadlock against the parent's TUI."""
    logger.warning("Background review auto-denied dangerous command: %s (%s)", command, description)
    return "deny"


def _set_thread_approval_callback(callback: Any) -> None:
    from tools.terminal_tool import set_approval_callback

    with suppress(Exception):
        set_approval_callback(callback)


def _track_review_fork(agent: Any, review_agent: Any, *, register: bool) -> None:
    """Add (``register=True``) or remove the fork on the PARENT's tracking slots:
    ``_background_review_agent`` (direct pointer the next live turn interrupts) and
    ``_active_children`` (interrupt() fan-out). Removal is identity-scoped and idempotent; both
    are best-effort for direct test stubs — the prepared run token is the live-turn cancellation
    authority."""
    if review_agent is None:
        return
    if hasattr(agent, "_background_review_agent"):
        with _optional_lock(agent, "_background_review_lock"):
            if register:
                agent._background_review_agent = review_agent
            elif agent._background_review_agent is review_agent:
                agent._background_review_agent = None
    if hasattr(agent, "_active_children"):
        with _optional_lock(agent, "_active_children_lock"):
            if register:
                agent._active_children.append(review_agent)
            else:
                with suppress(ValueError, AttributeError):
                    agent._active_children.remove(review_agent)


def _review_tool_whitelist(review_agent: Any, task_cfg: Optional[Dict[str, Any]]) -> Tuple[set, set]:
    """``(whitelist, configured_extra_tools)`` for the review fork — DISPATCH-side only, so the
    advertised ``tools[]`` stays byte-identical to the parent's (prompt-cache parity)."""
    from model_tools import get_tool_definitions
    # Gate the built-in memory tool on the profile's memory flags so a memory-disabled profile
    # is never contaminated by the review LLM.
    memory_on = review_agent._memory_enabled or review_agent._user_profile_enabled
    review_toolsets = ["memory", "skills"] if memory_on else ["skills"]
    whitelist = {t["function"]["name"] for t in get_tool_definitions(enabled_toolsets=review_toolsets, quiet_mode=True)}
    # Read-only file tools: denying read_file/search_files caused a per-review denial storm that
    # starved the loop (read_file also registers the read with the read-before-write guard).
    # Write tools stay denied — autonomous maintenance goes through skill_manage's validation.
    whitelist |= {"read_file", "search_files"}
    # ``extra_tools`` admits named parent tools (e.g. a human-gated proposal tool). The whitelist
    # can only admit, never advertise: a listed tool must already exist in the inherited schema.
    # Read-only file tools are whitelisted too (#61521, #39996): the model naturally reaches for
    # read_file/search_files to inspect a skill before patching it. Denying them caused a per-review denial
    # storm (~142 denials + ~204 read-before-write refusals over 2 days on one deployment) that starved the
    # self-improvement loop — the model never loaded SKILL.md the way the read-before-write guard requires,
    # so almost no patch landed. This is a DISPATCH-side change only: the advertised ``tools[]`` stays
    # byte-identical to the parent's, so prompt-cache parity is untouched. read_file registers the read with
    # the read-before-write guard (tools/file_tools.py), so a read_file → skill_manage(patch) sequence now
    # succeeds. Write tools (write_file/patch/terminal) stay denied — autonomous maintenance must go through
    # skill_manage's validation, and the deny message below names that substitute so one denial redirects
    # the model instead of a storm.
    # Profile-configured opt-in tools (#44672, salvage #82146 by @BrinShadewater):
    # ``auxiliary.background_review.extra_tools`` admits named parent tools to the review whitelist — e.g. a
    # human-gated proposal tool or a memory-provider write surface. Read from task_cfg (the
    # auxiliary.background_review block already loaded for this spawn) so no extra config I/O happens per
    # review.
    configured_extra_tools: set = set()
    try:
        extra_raw = _background_review_task_config(task_cfg).get("extra_tools", [])
        if isinstance(extra_raw, list):
            configured_extra_tools = {name.strip() for name in extra_raw if isinstance(name, str) and name.strip()}
    except Exception:
        logger.debug("background_review extra_tools parse failed", exc_info=True)
    return whitelist | configured_extra_tools, configured_extra_tools


@dataclass
class _ReviewForkState:
    """Mutable hand-off between the fork phase and the outer worker's error/cleanup paths."""

    review_agent: Any = None
    review_messages: List[Dict] = field(default_factory=list)
    review_usage: Dict[str, Any] = field(default_factory=dict)


def _release_fork_clients(review_agent: Any) -> None:
    """The fork shares the foreground session ID: close() / shutdown_memory_provider() are
    session-bound (close() kills that session's terminal processes), so release only clients."""
    with suppress(Exception):
        review_agent.release_clients()


def _run_review_fork(
    agent: Any, messages_snapshot: List[Dict], prompt: str, task_cfg: Optional[Dict[str, Any]],
    review_run: Optional[_BackgroundReviewRun], st: _ReviewForkState,
) -> None:
    """Fork phase (inside thread-scoped silence): build the fork, run the prompt under the tool
    whitelist, snapshot its messages/usage, release its clients. Partial progress lands on ``st``
    so the caller's error path still sees usage and the fork to clean up."""
    st.review_agent, _rt, _routed = build_cache_parity_fork(agent, task_cfg, max_iterations=_REVIEW_MAX_ITERATIONS)
    _track_review_fork(agent, st.review_agent, register=True)
    from hermes_cli.plugins import set_thread_tool_whitelist, clear_thread_tool_whitelist
    review_whitelist, configured_extra_tools = _review_tool_whitelist(st.review_agent, task_cfg)
    extra_list = ", ".join(sorted(configured_extra_tools))
    deny_extra = f" Configured extra tools also allowed: {extra_list}." if configured_extra_tools else ""
    prompt_extra = f" Exception — these configured tools are also allowed: {extra_list}." if configured_extra_tools else ""
    set_thread_tool_whitelist(
        review_whitelist,
        deny_msg_fmt=(
            "Background review denied non-whitelisted tool: "
            "{tool_name}. Allowed here: skill_view/skills_list/read_file/search_files to read, "
            "skill_manage(action='patch'|...) to change skills, and "
            "memory for notes." + deny_extra + " Do not retry {tool_name}."
        ),
    )
    with suppress(Exception):
        from tools.skill_manager_guards import _reset_background_review_read_marks

        _reset_background_review_read_marks()
    try:
        if review_run is None or review_run.begin_request(st.review_agent):
            # Routed -> digest (cache cold anyway); same model -> full snapshot (warm cache reads).
            st.review_agent.run_conversation(
                user_message=(
                    prompt + "\n\nYou can only call memory and skill "
                    "management tools. Other tools will be denied "
                    "at runtime — do not attempt them." + prompt_extra
                ),
                conversation_history=_digest_history(messages_snapshot) if _routed else messages_snapshot,
            )
    finally:
        clear_thread_tool_whitelist()
        # Attribute usage to the PARENT session. Snapshot BEFORE unregister/close so counters
        # survive teardown, and in this finally so a fork that consumed tokens then raised is
        # still attributed. The recorder never raises.
        if st.review_agent is not None:
            st.review_usage.update(_snapshot_review_usage(st.review_agent))
            _record_review_usage_to_parent(agent, st.review_usage)
        # Publish completion as soon as the provider-capable phase has returned or startup
        # cancellation has fenced it out (unregister + finish are identity-scoped and idempotent).
        _track_review_fork(agent, st.review_agent, register=False)
        finish_background_review_run(agent, review_run)
    st.review_messages = list(getattr(st.review_agent, "_session_messages", []))
    _release_fork_clients(st.review_agent)
    st.review_agent = None


def _publish_review_summary(agent: Any, actions: List[str]) -> None:
    summary = " · ".join(dict.fromkeys(actions))
    agent._safe_print(f"  💾 Self-improvement review: {summary}")
    if agent.background_review_callback:
        with suppress(Exception):
            agent.background_review_callback(f"💾 Self-improvement review: {summary}")


def _run_review_in_thread(
    agent: Any, messages_snapshot: List[Dict], prompt: str,
    task_cfg: Optional[Dict[str, Any]] = None, review_run: Optional[_BackgroundReviewRun] = None,
) -> None:
    """Daemon-thread worker: build the fork, run the prompt, surface the action summary via
    ``agent._safe_print`` / ``background_review_callback``. ``review_run`` (from
    :func:`prepare_background_review_run`) cancelled before the first provider call aborts
    without entering ``run_conversation()``.

    See #84423.
    """
    if review_run is not None and review_run.cancel_requested.is_set():
        finish_background_review_run(agent, review_run)
        return
    _set_thread_approval_callback(_bg_review_auto_deny)
    # A client that can't carry Hermes tool calls back would spawn a fork that cannot write
    # anything. Checked BEFORE the thread-scoped silence so the warning is not swallowed; cheap
    # check first so the normal path never resolves the runtime twice.
    if not _parent_can_emit_tool_calls(agent) and not _resolve_review_runtime(agent, task_cfg).get("routed"):
        logger.warning(
            "Background review skipped: provider %r cannot emit Hermes tool calls, "
            "so the review fork could not write memories or skills. Set "
            "auxiliary.background_review.{provider,model} to route the review to a normal model.",
            getattr(agent, "provider", "?"),
        )
        _set_thread_approval_callback(None)
        return
    st = _ReviewForkState()
    try:
        # Silence stdout/stderr for THIS thread only: a process-global redirect would blank every
        # other thread's console for the whole review.
        # A process-global ``contextlib.redirect_stdout(devnull)`` here would also blank
        # ``sys.stdout``/``sys.stderr`` for every other thread — including a gateway event-loop thread
        # driving a Telegram long-poll — for the full duration of the review (tens of seconds), swallowing
        # their console output (#55769 / #55925). ``thread_scoped_silence`` routes only this thread's writes
        # to devnull and leaves all other threads on the real streams.
        with thread_scoped_silence():
            _run_review_fork(agent, messages_snapshot, prompt, task_cfg, review_run, st)
        # A buggy/legacy tool response shape must NOT take down the whole review (the outer
        # except would discard every action the fork DID complete), so coerce to an empty list.
        try:
            # Scan the review agent's messages for successful tool actions and surface a compact summary to
            # the user. Tool messages already present in messages_snapshot must be skipped, since the review
            # agent inherits that history and would otherwise re-surface stale "created"/"updated" messages
            # from the prior conversation as if they just happened (issue #14944). ``_change`` returned as a
            # list instead of a dict, #59437) must NOT take down the whole review with an AttributeError,
            # since the caller's outer except logs only "Background memory/skill review failed" and discards
            # every successful action the fork DID complete before the crash. Coerce an exception into an
            # empty actions list so the partial valid actions from earlier in the messages are returned
            # instead.
            actions = summarize_background_review_actions(
                st.review_messages, messages_snapshot,
                notification_mode=getattr(agent, "memory_notifications", "on"),
            )
        except Exception as e:
            logger.warning(
                "summarize_background_review_actions returned partial results "
                "after exception (treating as empty); suppressing AttributeError "
                "that previously aborted the entire review (#59437): %s",
                e,
            )
            actions = []
        _log_review_completion(st.review_usage, _classify_review_result(actions))
        if actions:
            _publish_review_summary(agent, actions)
    except Exception as e:
        logger.warning("Background memory/skill review failed: %s", e)
        if st.review_usage:
            _log_review_completion(st.review_usage, "error")
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # Safety net for the exception path (setup failures before the request-phase finally).
        # Both cleanups are identity-scoped and idempotent; re-enter thread-scoped silence so
        # cleanup output stays quiet without blanking other threads.
        _track_review_fork(agent, st.review_agent, register=False)
        finish_background_review_run(agent, review_run)
        if st.review_agent is not None:
            with suppress(Exception), thread_scoped_silence():
                _release_fork_clients(st.review_agent)
        # Clear the approval callback so a recycled thread-id doesn't inherit it.
        _set_thread_approval_callback(None)


# (review_memory, review_skills) -> prompt attribute name; skills-only is also the default.
_PROMPT_NAME_BY_SCOPE = {
    (True, True): "_COMBINED_REVIEW_PROMPT", (True, False): "_MEMORY_REVIEW_PROMPT",
    (False, True): "_SKILL_REVIEW_PROMPT", (False, False): "_SKILL_REVIEW_PROMPT",
}


def spawn_background_review_thread(
    agent: Any, messages_snapshot: List[Dict], review_memory: bool = False,
    review_skills: bool = False, focus: Optional[str] = None,
    task_cfg: Optional[Dict[str, Any]] = None, review_run: Optional[_BackgroundReviewRun] = None,
):
    """Return ``(target, prompt)``; the caller builds the ``threading.Thread`` so test patches of
    ``run_agent.threading.Thread`` keep working. ``focus`` (``/refine [instructions]``) is appended
    to the chosen prompt; automatic reviews pass ``None``. ``task_cfg`` is the pre-loaded
    ``auxiliary.background_review`` block; when omitted it is read once here."""
    if task_cfg is None:
        task_cfg = _background_review_task_config()
    # Per-agent overrides (agent._MEMORY_REVIEW_PROMPT etc.) keep working.
    name = _PROMPT_NAME_BY_SCOPE[(review_memory, review_skills)]
    prompt = getattr(agent, name, globals()[name])
    if focus := (focus or "").strip():
        prompt = (
            f"{prompt}\n\nThe user explicitly requested this review with the following "
            f"focus — prioritize it over the general instructions above:\n{focus}"
        )

    def _target() -> None:  # resolves _run_review_in_thread at call time (tests patch it)
        _run_review_in_thread(agent, messages_snapshot, prompt, task_cfg=task_cfg, review_run=review_run)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT", "_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT", "load_background_review_settings",
    "spawn_background_review_thread", "summarize_background_review_actions", "build_memory_write_metadata",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402

def is_background_review_enabled(
    task_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return whether automatic post-turn background review may spawn.

    Controlled by ``auxiliary.background_review.enabled`` (default ``true``).
    Explicit ``/refine`` (``focus`` set) bypasses this gate — same contract as
    zeroing the nudge intervals, which stops automatic forks but leaves manual
    refine working (issue #87250).

    Prefer :func:`load_background_review_settings` at the spawn call site so
    the task block is not re-read on the same turn.
    """
    if task_cfg is not None:
        try:
            from utils import is_truthy_value

            return is_truthy_value(task_cfg.get("enabled"), default=True)
        except Exception:
            logger.warning(
                "Failed to interpret background_review.enabled; leaving "
                "automatic review enabled (fail-open)",
                exc_info=True,
            )
            return True
    enabled, _ = load_background_review_settings()
    return enabled
# ---- END PLUGIN-COMPAT ----
