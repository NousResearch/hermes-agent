"""Background memory/skill review — fork the agent to evaluate the turn.

After every turn, ``AIAgent.run_conversation`` may call
:func:`spawn_background_review` to fire off a daemon thread that replays
the conversation snapshot in a forked :class:`AIAgent` and asks itself
"should any skill/memory be saved or updated?".  Writes go straight to
the memory + skill stores.  Main conversation and prompt cache are never
touched.

The fork inherits the parent's live runtime (provider, model, base_url,
credentials, cached system prompt) so it hits the same prefix cache and
uses the same auth.  It runs with a tool whitelist limited to memory and
skill management tools; everything else is denied at runtime.

See the ``hermes-agent-dev`` skill (``references/self-improvement-loop.md``)
for invariants and PR review criteria.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.thread_scoped_output import thread_scoped_silence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background-review aux-model selector + routed digest.
#
# The review fork runs on the MAIN model by default ("auto"), replaying the
# full conversation — already warm in the prompt cache, so cheap cache reads.
# Optimal and unchanged. A user can route the review to a different, cheaper
# model via auxiliary.background_review.{provider,model}. A different model
# cannot reuse the parent's cache (different key), so the fork is cold
# regardless — replaying the full transcript would just cold-write it. So when
# (and only when) routed to a different model, we replay a compact DIGEST to
# minimise cold-written tokens. Same model -> full replay; different model ->
# digest. That's the whole policy.
# ---------------------------------------------------------------------------

# Historical hardcoded value — kept as the default so existing behavior is
# unchanged when the config key is unset.
_DEFAULT_REVIEW_MAX_ITERATIONS = 16


def _background_review_task_config(
    task_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return ``auxiliary.background_review`` (or ``{}`` on any failure).

    Pass ``task_cfg`` when the caller already loaded the block once so spawn /
    resolve / prompt paths do not re-read config on every turn.
    """
    if task_cfg is not None:
        return task_cfg if isinstance(task_cfg, dict) else {}
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {})
    return task if isinstance(task, dict) else {}


def load_background_review_settings() -> tuple[bool, Dict[str, Any]]:
    """Single config read for the automatic-review gate + task block.

    Returns ``(enabled, task_cfg)``. Fail-open on config errors (``enabled=True``)
    so a broken config file does not silently disable reviews — but log at
    WARNING so the cost-incurring path is visible.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        cfg = load_config_readonly()
        aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
        task = aux.get("background_review", {})
        task = task if isinstance(task, dict) else {}
        return is_truthy_value(task.get("enabled"), default=True), task
    except Exception:
        logger.warning(
            "Failed to read background_review.enabled; leaving automatic "
            "review enabled (fail-open)",
            exc_info=True,
        )
        return True, {}


def is_background_review_enabled(
    task_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return whether automatic post-turn background review may spawn.

    Controlled by ``auxiliary.background_review.enabled`` (default ``true``).
    Explicit ``/refine`` (``focus`` set) bypasses this gate — same contract as
    zeroing the nudge intervals, which stops automatic forks but leaves manual
    refine working (issue #87250).

    Prefer :func:`load_background_review_settings` at the spawn call site so
    the task block is not re-read for prompt / max_iterations on the same turn.
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


def _resolve_review_max_iterations(
    task_cfg: Optional[Dict[str, Any]] = None,
) -> int:
    """Resolve the review fork's tool-calling iteration budget.

    Default 16 (unchanged historical behavior). Configurable via
    ``auxiliary.background_review.max_iterations`` so operators can bound the
    worst-case cost of a review that never concludes "nothing to save".
    Clamped to [1, 64]; any config-read or parse failure falls back to the
    default.
    """
    try:
        val = _background_review_task_config(task_cfg).get("max_iterations")
        if val is None:
            return _DEFAULT_REVIEW_MAX_ITERATIONS
        return max(1, min(int(val), 64))
    except Exception:
        return _DEFAULT_REVIEW_MAX_ITERATIONS


def _load_review_prompt_file(
    task_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Load ``auxiliary.background_review.prompt_file`` if configured.

    Returns the file contents (stripped) or ``None`` when unset / unreadable.
    Relative paths resolve against ``HERMES_HOME`` (same pattern as TTS
    ``persona_prompt_file``).
    """
    raw = _background_review_task_config(task_cfg).get("prompt_file")
    if not isinstance(raw, str) or not raw.strip():
        return None
    expanded = os.path.expandvars(raw.strip())
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        try:
            from hermes_constants import get_hermes_home

            path = get_hermes_home() / path
        except Exception:
            path = Path.cwd() / path
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "background_review.prompt_file unavailable at %s: %s", path, exc
        )
        return None
    return text or None


def _resolve_review_runtime(
    agent: Any,
    task_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve provider/model/credentials for the review fork.

    Default (auto / unset / same as parent): inherit the parent's live runtime
    (with codex_app_server -> codex_responses downgrade). ``routed`` is False —
    the fork uses the main model and the warm cache, exactly as before. When
    ``auxiliary.background_review.{provider,model}`` names a concrete model
    different from the parent's, resolve that runtime and set ``routed=True``.
    """
    parent_runtime = agent._current_main_runtime()
    parent_api_mode = parent_runtime.get("api_mode") or None
    if parent_api_mode == "codex_app_server":
        parent_api_mode = "codex_responses"
    parent = {
        "provider": agent.provider,
        "model": agent.model,
        "api_key": parent_runtime.get("api_key") or None,
        "base_url": parent_runtime.get("base_url") or None,
        "api_mode": parent_api_mode,
        "credential_pool": getattr(agent, "_credential_pool", None),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "max_tokens": getattr(agent, "max_tokens", None),
        "command": getattr(agent, "acp_command", None),
        "args": list(getattr(agent, "acp_args", []) or []),
        "routed": False,
    }
    task = _background_review_task_config(task_cfg)
    task_provider = (str(task.get("provider", "")).strip() or None)
    task_model = (str(task.get("model", "")).strip() or None)
    task_base_url = (str(task.get("base_url", "")).strip() or None)
    task_api_key = (str(task.get("api_key", "")).strip() or None)
    if not (task_provider and task_provider != "auto" and task_model):
        return parent
    if task_provider == (agent.provider or "") and task_model == (agent.model or ""):
        return parent  # same model/provider as parent -> not routed
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rp = resolve_runtime_provider(
            requested=task_provider,
            target_model=task_model,
            explicit_api_key=task_api_key,
            explicit_base_url=task_base_url,
        )
        return {
            "provider": rp.get("provider") or task_provider,
            "model": rp.get("model") or task_model,
            "api_key": rp.get("api_key"),
            "base_url": rp.get("base_url"),
            "api_mode": rp.get("api_mode"),
            "credential_pool": rp.get("credential_pool"),
            "request_overrides": dict(rp.get("request_overrides") or {}),
            "max_tokens": rp.get("max_output_tokens"),
            "command": rp.get("command"),
            "args": list(rp.get("args") or []),
            "routed": True,
        }
    except Exception as e:
        logger.debug("background-review aux routing failed (%s); using main model", e)
        return parent


def _msg_text(m: Dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()
    return ""


def _digest_history(messages_snapshot: List[Dict], tail: int = 24) -> List[Dict]:
    """Compact replay for the routed (different-model) path only.

    Keeps the recent ``tail`` messages verbatim, collapses older turns into one
    synthetic user-role digest, preserving role alternation. Used ONLY when
    routed to a different model (cache cold regardless, so fewer cold-written
    tokens is a pure win). Never on the main-model path (full replay stays warm).
    """
    msgs = list(messages_snapshot or [])
    if len(msgs) <= tail:
        return msgs
    keep = msgs[-tail:]
    while keep and isinstance(keep[0], dict) and keep[0].get("role") == "tool":
        tail += 1
        if len(msgs) <= tail:
            return msgs
        keep = msgs[-tail:]
    old = msgs[:-len(keep)]
    lines: List[str] = []
    for m in old:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _msg_text(m).replace("\n", " ")
        if role == "user" and text:
            lines.append(f"USER: {text[:300]}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                names = [(tc.get("function") or {}).get("name", "?") for tc in tcs if isinstance(tc, dict)]
                lines.append(f"ASSISTANT[tools: {', '.join(names)}]")
            if text:
                lines.append(f"ASSISTANT: {text[:200]}")
    digest = {
        "role": "user",
        "content": (
            "[Earlier conversation digest — older turns summarised to bound the "
            "review's cold-write cost on the routed aux model. Recent turns "
            "follow verbatim below.]\n" + "\n".join(lines)
        ),
    }
    return [digest] + keep


# Review-prompt strings — used by ``spawn_background_review_thread`` to build
# the user-message that the forked review agent receives.  AIAgent exposes
# them as class attributes (``_MEMORY_REVIEW_PROMPT`` etc.) for back-compat;
# the actual text lives here so future edits are one-place.
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend — but only if it is "
    "curator-managed. Bundled, hub, pinned, and user-owned skills are "
    "off-limits to you no matter how relevant (see Protected skills "
    "below); for those, fall through to the next option.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with "
    "file_path starting 'references/', 'templates/', or 'scripts/'. "
    "The umbrella's SKILL.md should gain a one-line pointer to any "
    "new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
    "skill covers the class. The name MUST be at the class level. "
    "The name MUST NOT be a specific PR number, error string, feature "
    "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
    "session artifact. If the proposed name only makes sense for "
    "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
    "User-preference embedding (important): when the user expressed a "
    "style/format/workflow preference, the update belongs in the "
    "SKILL.md body, not just in memory. Memory captures 'who the user "
    "is and what the current situation and state of your operations "
    "are'; skills capture 'how to do this class of task for this "
    "user'. When they complain about how you handled a task, the "
    "skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your "
    "reply — the background curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "  • Skills in skills.external_dirs (externally owned).\n"
    "  • PINNED skills (marked via 'hermes curator pin'). You are an "
    "autonomous no-user-present actor, so pin blocks your writes too — "
    "content updates included. Only the user, in a foreground session, "
    "can change a pinned skill.\n"
    "  • USER-OWNED skills — anything not curator-managed. A skill the "
    "user hand-wrote, installed by URL, or asked a foreground agent to "
    "create is theirs, not yours; your writes to it WILL be refused. "
    "This includes skills that were loaded or consulted this session: "
    "being in play does not make one yours to edit. If such a skill is "
    "wrong or outdated, say so in your reply and recommend "
    "'hermes curator adopt <name>' — do not try to patch it.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture (these become persistent self-imposed constraints "
    "that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "  • Unresolved failures: if the session ended WITHOUT actually "
    "finding a working method — you tried several things, none worked, "
    "and told the user to check manually — do NOT write those attempts "
    "up as a 'reliable workflow' or 'recommended approach'. That presents "
    "an untested sequence of failures as validated guidance a future "
    "session will trust and repeat. Either say 'Nothing to save', or, "
    "only if you are independently confident of a real working alternative "
    "(not something you are merely guessing might work), capture ONLY that "
    "alternative — never the dead ends, and never dressed up as best practice.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
    "loaded via /skill-name or skill_view in the conversation. If one "
    "of them covers the learning, PATCH it first. It was in play; "
    "it's the right place — provided it is curator-managed. Protected "
    "and user-owned skills are off-limits however relevant; fall "
    "through when one of those is the best fit.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file. Three kinds: "
    "`references/<topic>.md` for session-specific detail OR condensed "
    "knowledge banks (quoted research, API docs excerpts, domain "
    "notes) written concise and task-focused; `templates/<name>.<ext>` "
    "for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification, fixture generators, probes). Add a one-line "
    "pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact. If the name only fits today's task, fall back to (1), "
    "(2), or (3).\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the "
    "background curator handles consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "  • Skills in skills.external_dirs (externally owned).\n"
    "  • PINNED skills (marked via 'hermes curator pin'). Pin blocks "
    "autonomous writes entirely — content updates included — because no "
    "user is present to consent. Only a foreground session can change one.\n"
    "  • USER-OWNED skills — anything not curator-managed (hand-written, "
    "URL-installed, or created by a foreground agent at the user's "
    "request). Your writes to these WILL be refused, including to skills "
    "loaded or consulted this session. If one is wrong, say so in your "
    "reply and recommend 'hermes curator adopt <name>' instead.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills (these become persistent self-imposed "
    "constraints that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "  • Unresolved failures: if the session ended WITHOUT actually "
    "finding a working method — you tried several things, none worked, "
    "and told the user to check manually — do NOT write those attempts "
    "up as a 'reliable workflow' or 'recommended approach'. That presents "
    "an untested sequence of failures as validated guidance a future "
    "session will trust and repeat. Either say 'Nothing to save', or, "
    "only if you are independently confident of a real working alternative "
    "(not something you are merely guessing might work), capture ONLY that "
    "alternative — never the dead ends, and never dressed up as best practice.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "Act on whichever of the two dimensions has real signal. If "
    "genuinely nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)



def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
    notification_mode: str = "on",
) -> List[str]:
    """Build the human-facing action summary for a background review pass.

    Walks the review agent's session messages and collects successful memory
    and skill-management actions to surface to the user. Tool messages already
    present in ``prior_snapshot`` are skipped so stale inherited results are
    not re-surfaced as fresh background work (issue #14944).

    ``notification_mode`` controls display detail:
    - ``off``: return no actions.
    - ``on``: generic "Memory updated"/tool messages.
    - ``verbose``: include compact content previews from tool-call arguments.
    """
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    verbose = mode == "verbose"

    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    # Map review-agent tool results back to the calls that produced them.  The
    # result JSON only says "Entry added"; the call arguments contain action,
    # target, and content previews.  Restricting to notify_tools also prevents
    # helper tools from surfacing as memory work just because they succeeded.
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
            if fn_name not in notify_tools:
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if tcid:
                call_details[tcid] = {
                    "tool": fn_name,
                    "action": args.get("action", "?"),
                    "target": args.get("target", "memory"),
                    "content": args.get("content", ""),
                    "old_text": args.get("old_text", ""),
                    "operations": args.get("operations") or [],
                    "name": args.get("name", ""),
                    "old_string": args.get("old_string", ""),
                    "new_string": args.get("new_string", ""),
                }

    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        if tcid and all_tool_call_ids and tcid not in call_details:
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        # ``data`` may not be a dict — some memory/skill tool responses in
        # older codepaths or wrapper MCP servers return a top-level JSON
        # list (e.g. ``[{"success": true, ...}]``) or a scalar.  The original
        # isinstance check below silently skips non-dict payloads, which
        # is correct, but ``data.get("_change")`` further down can still
        # hand back a list and break ``change.get("description", "")``.
        # Defensively normalize everything through a dict-typed alias so
        # the rest of the function can stay terse without per-call
        # ``isinstance`` guards (#59437).
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        detail = call_details.get(tcid) or {}
        if not isinstance(detail, dict):
            detail = {}
        target = data.get("target", "") or detail.get("target", "")
        is_skill = detail.get("tool") == "skill_manage"

        message_lower = message.lower()
        if not verbose:
            if "created" in message_lower:
                actions.append(message)
                continue
            if "updated" in message_lower:
                actions.append(message)
                continue
            if is_skill and "patched" in message_lower:
                actions.append(message)
                continue

        if is_skill:
            label = "Skill"
        elif target:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
        else:
            continue

        if verbose:
            action = detail.get("action", "")
            content = detail.get("content", "")
            old_text = detail.get("old_text", "")
            skill_name = detail.get("name", "")
            # ``operations`` may be anything callable put into the JSON
            # arguments.  Anything non-iterable that isn't a list[str]
            # of dicts becomes unusable here, so coerce defensively.
            ops_raw = detail.get("operations")
            operations: list = (
                ops_raw if isinstance(ops_raw, list) else []
            )
            max_preview = 120
            if is_skill:
                # ``_change`` is a free-form dict the skill tool leaves in
                # the response.  Older / wrapper MCP backends return it
                # as a list, an int, or a JSON-shaped scalar — normalize
                # to a dict so the .get() calls downstream don't
                # AttributeError (#59437).
                change_raw = data.get("_change")
                change: dict = (
                    change_raw if isinstance(change_raw, dict) else {}
                )
                old_string = (
                    change.get("old", "") or detail.get("old_string", "")
                )
                new_string = (
                    change.get("new", "") or detail.get("new_string", "")
                )
                description = change.get("description", "")
                if action == "patch" and (old_string or new_string):
                    old_preview = old_string[:80].replace("\n", " ") + (
                        "…" if len(old_string) > 80 else ""
                    )
                    new_preview = new_string[:80].replace("\n", " ") + (
                        "…" if len(new_string) > 80 else ""
                    )
                    actions.append(
                        f"📝 Skill '{skill_name}' patched: "
                        f"\"{old_preview}\" → \"{new_preview}\""
                    )
                elif action == "create" and description:
                    actions.append(f"📝 Skill '{skill_name}' created: {description}")
                elif action == "edit" and description:
                    actions.append(f"📝 Skill '{skill_name}' rewritten: {description}")
                else:
                    actions.append(f"📝 {message}" if message else f"Skill {action}")
            elif operations:
                for op in operations:
                    # Each element must be a dict-of-fields; some
                    # legacy codepaths serialize the entry as a bare
                    # string and the message dict doesn't exist.  Skip
                    # non-dict items defensively — they have no
                    # actionable fields anyway (#59437).
                    if not isinstance(op, dict):
                        continue
                    op_act = op.get("action", "")
                    op_content = (op.get("content") or "")
                    op_old = (op.get("old_text") or "")
                    if op_act == "add" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ➕ {preview}")
                    elif op_act == "replace" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ✏️ {preview}")
                    elif op_act == "remove" and op_old:
                        preview = op_old[:60] + ("…" if len(op_old) > 60 else "")
                        actions.append(f"{label} ➖ {preview}")
            elif action == "add" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ➕ {preview}")
            elif action == "replace" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ✏️ {preview}")
            elif action == "remove" and old_text:
                preview = old_text[:60] + ("…" if len(old_text) > 60 else "")
                actions.append(f"{label} ➖ {preview}")
            else:
                actions.append(f"{label} updated")
        elif (
            "added" in message_lower
            or "replaced" in message_lower
            or "removed" in message_lower
            or "applied" in message_lower
            or (target and "add" in message.lower())
            or "Entry added" in message
        ):
            actions.append(f"{label} updated")
    return actions


def build_memory_write_metadata(
    agent: Any,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for external memory-provider mirrors."""
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": (
            execution_context
            or getattr(agent, "_memory_write_context", "foreground")
        ),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
    }
    if task_id:
        metadata["task_id"] = task_id
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


def _snapshot_review_usage(review_agent: Any) -> Dict[str, Any]:
    """Snapshot in-memory usage counters from a review fork (pre-close)."""
    return {
        "model": getattr(review_agent, "model", None),
        "provider": getattr(review_agent, "provider", None),
        "base_url": getattr(review_agent, "base_url", None),
        "input_tokens": int(getattr(review_agent, "session_input_tokens", 0) or 0),
        "output_tokens": int(getattr(review_agent, "session_output_tokens", 0) or 0),
        "cache_read_tokens": int(
            getattr(review_agent, "session_cache_read_tokens", 0) or 0
        ),
        "cache_write_tokens": int(
            getattr(review_agent, "session_cache_write_tokens", 0) or 0
        ),
        "reasoning_tokens": int(
            getattr(review_agent, "session_reasoning_tokens", 0) or 0
        ),
        "api_calls": int(getattr(review_agent, "session_api_calls", 0) or 0),
        "estimated_cost_usd": getattr(review_agent, "session_estimated_cost_usd", None),
    }


def _record_review_usage_to_parent(
    parent_agent: Any,
    usage: Dict[str, Any],
) -> None:
    """Record a background-review fork's usage against the parent session.

    Background-review forks run with ``_session_db = None`` for persistence
    isolation (see the PERSISTENCE ISOLATION comment in
    :func:`_run_review_in_thread`): the fork must never write its harness turn
    into the user's real session. A side effect of that isolation is that the
    fork's API calls — which the provider bills — were never recorded in
    ``session_model_usage``, because the accounting path in
    ``conversation_loop`` is gated on the DB handle. This hides the
    background-review volume from billing analytics (issue #87250).

    The fork still accumulates the same in-memory counters the main loop does
    (``session_input_tokens`` etc.) and shares the parent's ``session_id``, so
    its usage can be attributed to the parent session through the
    aux-accounting chokepoint, which writes only ``session_model_usage`` —
    never the transcript or the ``sessions`` summary row.

    Best-effort by contract: accounting must never fail the review.
    """
    try:
        session_db = getattr(parent_agent, "_session_db", None)
        session_id = getattr(parent_agent, "session_id", None)
        if session_db is None or not session_id:
            return
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_tokens") or 0)
        cache_write = int(usage.get("cache_write_tokens") or 0)
        reasoning = int(usage.get("reasoning_tokens") or 0)
        api_calls = int(usage.get("api_calls") or 0)
        if not (
            input_tokens
            or output_tokens
            or cache_read
            or cache_write
            or reasoning
            or api_calls
        ):
            return  # fork made no successful API calls (e.g. failed at spawn)
        session_db.record_auxiliary_usage(
            session_id,
            task="background_review",
            model=usage.get("model"),
            billing_provider=usage.get("provider"),
            billing_base_url=usage.get("base_url"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
            estimated_cost_usd=usage.get("estimated_cost_usd"),
            api_call_count=api_calls,
        )
    except Exception as e:
        logger.debug(
            "Background review usage recording failed (non-fatal): %s", e
        )


def _classify_review_result(actions: List[str]) -> str:
    """Map a review action summary to ``none`` / ``skill`` / ``memory`` / both.

    Matching is prefix-based on the formats
    :func:`summarize_background_review_actions` emits
    (``Skill …``, ``📝 Skill …``, ``Memory …``, ``User profile …``), not
    free-text substring search — so a line like
    ``Skipped: no skill worth saving`` stays ``none``.
    """
    if not actions:
        return "none"
    has_skill = False
    has_memory = False
    for action in actions:
        text = str(action).lstrip()
        if text.startswith("📝"):
            text = text[1:].lstrip()
        lower = text.lower()
        if lower.startswith("skill"):
            has_skill = True
        elif lower.startswith("memory") or lower.startswith("user profile"):
            has_memory = True
    if has_skill and has_memory:
        return "skill+memory"
    if has_skill:
        return "skill"
    if has_memory:
        return "memory"
    return "none"


def _log_review_completion(usage: Dict[str, Any], result: str) -> None:
    """Emit a per-fork completion line so cost is visible where it is incurred."""
    logger.info(
        "Background review complete: thread=bg-review calls=%d in=%d out=%d "
        "cache_read=%d result=%s",
        int(usage.get("api_calls") or 0),
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        int(usage.get("cache_read_tokens") or 0),
        result,
    )


def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str,
    task_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Worker function executed in the background-review daemon thread.

    Spawns a forked ``AIAgent`` inheriting the parent's runtime, runs the
    review prompt, and surfaces a compact action summary back to the user
    via ``agent._safe_print`` and ``agent.background_review_callback``.
    """
    # Local import to avoid a hard circular dep at module load.
    from run_agent import AIAgent
    from tools.terminal_tool import set_approval_callback as _set_approval_callback

    # Install a non-interactive approval callback on this worker
    # thread so any dangerous-command guard the review agent trips
    # resolves to "deny" instead of falling back to input() -- which
    # deadlocks against the parent's prompt_toolkit TUI (#15216).
    # Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
    def _bg_review_auto_deny(command, description, **kwargs):
        logger.warning(
            "Background review auto-denied dangerous command: %s (%s)",
            command, description,
        )
        return "deny"
    try:
        _set_approval_callback(_bg_review_auto_deny)
    except Exception:
        pass

    review_agent = None
    review_messages: List[Dict] = []
    review_usage: Dict[str, Any] = {}

    def _unregister_review_agent(agent_ref) -> None:
        """Idempotent: clears the review fork from both tracking slots.
        Called from the run_conversation finally and the outer safety-net finally.
        """
        if agent_ref is None:
            return
        if hasattr(agent, "_background_review_agent"):
            _br_lock = getattr(agent, "_background_review_lock", None)
            if _br_lock is not None:
                with _br_lock:
                    if agent._background_review_agent is agent_ref:
                        agent._background_review_agent = None
            elif agent._background_review_agent is agent_ref:
                agent._background_review_agent = None
        if hasattr(agent, "_active_children"):
            try:
                _ac_lock = getattr(agent, "_active_children_lock", None)
                if _ac_lock is not None:
                    with _ac_lock:
                        agent._active_children.remove(agent_ref)
                else:
                    agent._active_children.remove(agent_ref)
            except (ValueError, AttributeError):
                pass

    try:
        # Silence stdout/stderr for THIS worker thread only.  A process-global
        # ``contextlib.redirect_stdout(devnull)`` here would also blank
        # ``sys.stdout``/``sys.stderr`` for every other thread — including a
        # gateway event-loop thread driving a Telegram long-poll — for the full
        # duration of the review (tens of seconds), swallowing their console
        # output (#55769 / #55925).  ``thread_scoped_silence`` routes only this
        # thread's writes to devnull and leaves all other threads on the real
        # streams.
        with thread_scoped_silence():
            # Inherit the parent agent's live runtime (provider, model,
            # base_url, api_key, api_mode) so the fork uses the exact
            # same credentials the main turn is using.  Without this,
            # AIAgent.__init__ re-runs auto-resolution from env vars,
            # which fails for OAuth-only providers, session-scoped
            # creds, or credential-pool setups where the resolver can't
            # reconstruct auth from scratch -- producing the spurious
            # "No LLM provider configured" warning at end of turn.
            # _resolve_review_runtime() returns the parent's live runtime by
            # default (routed=False; main model, warm cache), or — when the user
            # set auxiliary.background_review.{provider,model} to a different
            # model — that model's runtime (routed=True). The codex_app_server
            # -> codex_responses downgrade is applied inside the resolver.
            _rt = _resolve_review_runtime(agent, task_cfg)
            _routed = bool(_rt.get("routed"))
            # skip_memory=True keeps the review fork from
            # touching external memory plugins (honcho, mem0,
            # supermemory, etc.).  Without it, the fork's
            # __init__ rebuilds its own _memory_manager from
            # config, scoped to the parent's session_id, and
            # run_conversation() then leaks the harness prompt
            # into the user's real memory namespace via three
            # ingestion sites: on_turn_start (cadence + turn
            # message), prefetch_all (recall query), and
            # sync_all (harness prompt + review output recorded
            # as a (user, assistant) turn pair).  Built-in
            # MEMORY.md / USER.md state is re-bound from the
            # parent below so memory(action="add") writes from
            # the review still land on disk; the review just
            # has zero side effects on external providers.
            # Match parent's toolset config so ``tools[]`` is byte-identical
            # in the request body — Anthropic's cache key includes it.
            # (The runtime whitelist below still restricts dispatch.)
            _fork_kwargs: Dict[str, Any] = {}
            if isinstance(_rt.get("max_tokens"), int):
                _fork_kwargs["max_tokens"] = _rt["max_tokens"]
            if isinstance(_rt.get("command"), str) and _rt["command"]:
                _fork_kwargs["acp_command"] = _rt["command"]
                _fork_kwargs["acp_args"] = _rt.get("args") or []
            # Match parent's reasoning config so the fork's ``thinking`` /
            # ``output_config`` are byte-identical in the request body —
            # Anthropic's cache key is namespaced by ``thinking`` presence.
            # Same-model path only: when routed to a different aux model the
            # cache is cold regardless (parity buys nothing) and the parent's
            # effort vocabulary may not be valid for the routed model/provider
            # (e.g. OpenRouter ``extra_body.reasoning.effort`` is forwarded
            # unclamped; codex_responses passes ``max``/``ultra`` through
            # unmapped except on gpt-5.6/xAI). Let the routed fork use
            # provider defaults — matching the ``not _routed`` gate on
            # _cached_system_prompt below.
            if not _routed:
                _fork_kwargs["reasoning_config"] = getattr(agent, "reasoning_config", None)
                # Gateway session context is appended to the parent's cached
                # system prompt at API-call time through this field.  Preserve
                # it on same-model forks so the complete effective system
                # prompt remains byte-identical and can reuse the warm prefix.
                _fork_kwargs["ephemeral_system_prompt"] = getattr(
                    agent, "ephemeral_system_prompt", None
                )
                # Prefill messages are inserted immediately after the system
                # message at API-call time (chat_completion_helpers.py /
                # conversation_loop.py), so a parent with prefill configured
                # (gateway prefill_messages_file) would otherwise diverge
                # from the warm prefix at message index 1 — same bug class
                # as the ephemeral prompt above, one position later.
                # Deep copy: the unicode-error recovery path mutates
                # prefill entries IN PLACE (_sanitize_messages_surrogates
                # via conversation_loop), so sharing dicts would let a
                # fork-side sanitize rewrite the parent's prefill bytes.
                _parent_prefill = copy.deepcopy(
                    getattr(agent, "prefill_messages", None) or []
                )
                if _parent_prefill:
                    _fork_kwargs["prefill_messages"] = _parent_prefill
                # OpenRouter provider-routing pins: prompt caches live per
                # UPSTREAM provider, so a fork without the parent's pins can
                # be routed to a different upstream and miss the warm cache
                # even with byte-identical prompt/tools bytes.
                for _pref_attr in (
                    "providers_allowed",
                    "providers_ignored",
                    "providers_order",
                    "provider_sort",
                    "provider_require_parameters",
                    "provider_data_collection",
                ):
                    _pref_val = getattr(agent, _pref_attr, None)
                    if _pref_val:
                        _fork_kwargs[_pref_attr] = _pref_val
            review_agent = AIAgent(
                model=_rt.get("model") or agent.model,
                max_iterations=_resolve_review_max_iterations(task_cfg),
                quiet_mode=True,
                platform=agent.platform,
                provider=_rt.get("provider") or agent.provider,
                api_mode=_rt.get("api_mode"),
                base_url=_rt.get("base_url") or None,
                api_key=_rt.get("api_key") or None,
                credential_pool=_rt.get("credential_pool"),
                request_overrides=_rt.get("request_overrides") or {},
                parent_session_id=agent.session_id,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                skip_memory=True,
                **_fork_kwargs,
            )
            review_agent._memory_write_origin = "background_review"
            review_agent._memory_write_context = "background_review"
            # The review fork pins the parent's cached system prompt and keeps
            # ``tools[]`` byte-identical to the parent so its outbound request
            # hits the same provider cache prefix (see the toolset-parity note
            # above). The between-turns MCP refresh in build_turn_context would
            # add late-connecting MCP tools to this fork and break that parity,
            # so opt the review fork out of it.
            review_agent._skip_mcp_refresh = True
            review_agent._memory_store = agent._memory_store
            review_agent._memory_enabled = agent._memory_enabled
            review_agent._user_profile_enabled = agent._user_profile_enabled
            review_agent._memory_nudge_interval = 0
            review_agent._skill_nudge_interval = 0
            # PERSISTENCE ISOLATION (the curator-takeover root cause): the fork
            # shares the parent's session_id (set below, for prompt-cache
            # warmth), so without this it would write its harness turn ("Review
            # the conversation above and update the skill library…") + its own
            # response straight into the user's REAL session in state.db. On the
            # user's next live turn the agent re-reads that injected user message
            # as a standing instruction and "becomes" the curator, refusing the
            # actual task. _persist_disabled hard-stops every DB write/lazy-open
            # path (_flush_messages_to_session_db, _ensure_db_session,
            # _get_session_db_for_recall); the review writes only to the skill
            # and memory stores via its tools, which is all it needs.
            review_agent._persist_disabled = True
            review_agent._session_db = None
            review_agent._session_json_enabled = False
            # Suppress all status/warning emits from the fork so the
            # user only sees the final successful-action summary.
            # Without this, mid-review "Iteration budget exhausted",
            # rate-limit retries, compression warnings, and other
            # lifecycle messages bubble up through _emit_status ->
            # _vprint and leak past the stdout redirect (they go via
            # _print_fn/status_callback, which bypass sys.stdout).
            review_agent.suppress_status_output = True
            # Inherit the parent's cached system prompt verbatim so
            # the review fork's outbound HTTP request hits the same
            # Anthropic/OpenRouter prefix cache the parent warmed.
            # Without this, the fork rebuilds the system prompt from
            # scratch (fresh _hermes_now() timestamp, fresh
            # session_id, narrower toolset → different skills_prompt)
            # and the byte-exact prefix-cache key misses. See
            # issue #25322 and PR #17276 for the full analysis +
            # measured impact (~26% end-to-end cost reduction on
            # Sonnet 4.5).
            # Share the parent's warm cached system prompt ONLY when the review
            # runs on the SAME model (not routed). When routed to a different
            # model the parent's cached prompt is for the wrong model/cache key
            # and would miss anyway, so let the routed fork build its own.
            if not _routed:
                review_agent._cached_system_prompt = agent._cached_system_prompt
                # Defensive: pin session_start + session_id to the
                # parent's so any code path that re-renders parts of
                # the system prompt (compression, plugin hooks) still
                # produces byte-identical output. The cached-prompt
                # assignment above already short-circuits the normal
                # rebuild path, but these pins guarantee parity even
                # if a future code path bypasses the cache.
                review_agent.session_start = agent.session_start
            review_agent.session_id = agent.session_id
            # The fork shares the parent's live session_id (pinned above for
            # prefix-cache parity). It is single-lifecycle and calls close()
            # right after this run_conversation(); without opting out, close()
            # would finalize the parent's still-active session row mid
            # conversation (the review fires every ~10 turns). Leave session
            # finalization to the real owner (CLI close / gateway reset / cron).
            review_agent._end_session_on_close = False
            # Never let the review fork compress. It shares the parent's
            # session_id, so if it won a compression race it would rotate the
            # parent into a NEW child that the gateway never adopts (the fork
            # is single-lifecycle and dies right after this run_conversation).
            # The foreground turn would then start from the stale parent and
            # compress it again, leaving the same parent with two sibling
            # children (issue #38727). Review also needs full context to
            # produce a good memory/skill summary — compressing would strip
            # detail. Both compression triggers in conversation_loop.py gate on
            # agent.compression_enabled, so this short-circuits both paths.
            review_agent.compression_enabled = False

            # Register this fork on the PARENT's _active_children (the same
            # list interrupt() fans out to for subagent delegation) and
            # _background_review_agent (a direct pointer the next live turn
            # uses to proactively cancel a still-running review). Without
            # this, a review still streaming when the next turn starts races
            # the live turn against the same session_id/credentials — producing
            # doubled prompt-token accounting and a Ctrl+C-proof lockup.
            # Best-effort: agents built without agent_init.py (test stubs)
            # degrade to "no cross-cancellation" rather than aborting the review.
            if hasattr(agent, "_background_review_agent"):
                _br_lock = getattr(agent, "_background_review_lock", None)
                if _br_lock is not None:
                    with _br_lock:
                        agent._background_review_agent = review_agent
                else:
                    agent._background_review_agent = review_agent
            if hasattr(agent, "_active_children"):
                _ac_lock = getattr(agent, "_active_children_lock", None)
                if _ac_lock is not None:
                    with _ac_lock:
                        agent._active_children.append(review_agent)
                else:
                    agent._active_children.append(review_agent)

            from model_tools import get_tool_definitions
            from hermes_cli.plugins import (
                set_thread_tool_whitelist,
                clear_thread_tool_whitelist,
            )

            # Gate the built-in memory tool on the profile's memory_enabled flag.
            # Hardcoding ["memory", "skills"] granted the review LLM the MEMORY.md
            # read/write tool even when a profile set memory_enabled: false,
            # contaminating a memory-disabled profile (#54937 layer 2).
            review_toolsets = ["skills"]
            if review_agent._memory_enabled or review_agent._user_profile_enabled:
                review_toolsets.insert(0, "memory")
            review_whitelist = {
                t["function"]["name"]
                for t in get_tool_definitions(
                    enabled_toolsets=review_toolsets,
                    quiet_mode=True,
                )
            }
            set_thread_tool_whitelist(
                review_whitelist,
                deny_msg_fmt=(
                    "Background review denied non-whitelisted tool: "
                    "{tool_name}. Only memory/skill tools are allowed."
                ),
            )
            try:
                from tools.skill_manager_tool import _reset_background_review_read_marks

                _reset_background_review_read_marks()
            except Exception:
                pass

            try:
                # Routed to a different model -> replay a digest (cache is cold
                # on that model anyway, so minimise cold-written tokens). Same
                # model -> replay the full snapshot (warm cache reads).
                _review_history = (
                    _digest_history(messages_snapshot) if _routed
                    else messages_snapshot
                )
                review_agent.run_conversation(
                    user_message=(
                        prompt
                        + "\n\nYou can only call memory and skill "
                        "management tools. Other tools will be denied "
                        "at runtime — do not attempt them."
                    ),
                    conversation_history=_review_history,
                )
            finally:
                clear_thread_tool_whitelist()
                # Attribute the review fork's usage to the PARENT session.
                # Snapshot BEFORE unregister/close so counters survive teardown.
                # Placed in this finally so a fork that consumed tokens and THEN
                # raised is still attributed (issue #87250). Best-effort: the
                # recorder never raises into the review thread.
                if review_agent is not None:
                    review_usage.update(_snapshot_review_usage(review_agent))
                    _record_review_usage_to_parent(agent, review_usage)
                # Unregister as soon as run_conversation() itself has
                # returned — that's the only phase making outbound API
                # calls, i.e. the only phase that can race the parent's
                # next live turn. Runs on both the success and exception
                # path (this whole block is inside the try/finally above).
                _unregister_review_agent(review_agent)

            # Snapshot review actions before teardown. close() is allowed to
            # clean per-session state, but the user-visible self-improvement
            # summary still needs the completed review agent's tool results.
            review_messages = list(getattr(review_agent, "_session_messages", []))

            # Tear down memory providers while stdout is still
            # redirected so background thread teardown (Honcho flush,
            # Hindsight sync, etc.) stays silent.  The finally block
            # below is a safety net for the exception path.
            try:
                review_agent.shutdown_memory_provider()
            except Exception:
                pass
            try:
                review_agent.close()
            except Exception:
                pass
            review_agent = None

        # Scan the review agent's messages for successful tool actions
        # and surface a compact summary to the user. Tool messages
        # already present in messages_snapshot must be skipped, since
        # the review agent inherits that history and would otherwise
        # re-surface stale "created"/"updated" messages from the prior
        # conversation as if they just happened (issue #14944).
        #
        # Wrapped in try/except: a buggy/legacy tool response shape
        # (e.g. ``_change`` returned as a list instead of a dict, #59437)
        # must NOT take down the whole review with an AttributeError,
        # since the caller's outer except logs only "Background
        # memory/skill review failed" and discards every successful
        # action the fork DID complete before the crash. Coerce an
        # exception into an empty actions list so the partial valid
        # actions from earlier in the messages are returned instead.
        try:
            actions = summarize_background_review_actions(
                review_messages,
                messages_snapshot,
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

        _log_review_completion(
            review_usage, _classify_review_result(actions)
        )

        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            agent._safe_print(
                f"  💾 Self-improvement review: {summary}"
            )
            _bg_cb = agent.background_review_callback
            if _bg_cb:
                try:
                    _bg_cb(
                        f"💾 Self-improvement review: {summary}"
                    )
                except Exception:
                    pass

    except Exception as e:
        logger.warning("Background memory/skill review failed: %s", e)
        if review_usage:
            _log_review_completion(review_usage, "error")
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # Safety-net cleanup for the exception path.  Normal completion already
        # shut down inside the thread-scoped silence above.  Re-enter the
        # thread-scoped silence here so teardown output (Honcho flush, Hindsight
        # sync, background thread joins) stays quiet even on the exception path,
        # without blanking other threads' streams.
        # Also a safety-net unregister: covers exceptions raised during setup
        # (between registration and the run_conversation try/finally above)
        # that the primary _unregister_review_agent call site never reaches.
        # _unregister_review_agent is idempotent (checks `is`/`in` membership),
        # so calling it again here after the primary call site already ran is
        # a harmless no-op.
        _unregister_review_agent(review_agent)
        if review_agent is not None:
            try:
                with thread_scoped_silence():
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
            except Exception:
                pass
        # Clear the approval callback on this bg-review thread so a
        # recycled thread-id doesn't inherit a stale reference.
        try:
            _set_approval_callback(None)
        except Exception:
            pass


def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
    focus: Optional[str] = None,
    task_cfg: Optional[Dict[str, Any]] = None,
):
    """Build the review thread target and prompt for a background review.

    Returns a ``(target, prompt)`` tuple.  The caller (``AIAgent._spawn_background_review``)
    owns the actual ``threading.Thread`` construction so test-level patches
    of ``run_agent.threading.Thread`` keep working.

    ``focus`` is optional user steering (the ``/refine [instructions]``
    path): appended to the chosen review prompt so the fork prioritizes what
    the user asked for while keeping the same guardrails. Automatic
    post-turn reviews pass ``None`` — their prompts are byte-identical to
    before this parameter existed.

    ``task_cfg`` is the already-loaded ``auxiliary.background_review`` block
    from :func:`load_background_review_settings`. When omitted, config is
    read once here and shared with the worker (prompt file, max_iterations,
    aux routing) so a single turn does not re-parse the config file.
    """
    if task_cfg is None:
        task_cfg = _background_review_task_config()
    # Pick the right prompt based on which triggers fired.  Allow per-agent
    # override (the prompts moved to module-level constants but old code paths
    # that set agent._MEMORY_REVIEW_PROMPT etc. directly keep working).
    # ``auxiliary.background_review.prompt_file`` replaces the built-in text
    # when set — operators can make the reviewer conservative or redefine
    # "worth saving" without patching core (issue #87250).
    prompt_override = _load_review_prompt_file(task_cfg)
    if prompt_override:
        prompt = prompt_override
    elif review_memory and review_skills:
        prompt = getattr(agent, "_COMBINED_REVIEW_PROMPT", _COMBINED_REVIEW_PROMPT)
    elif review_memory:
        prompt = getattr(agent, "_MEMORY_REVIEW_PROMPT", _MEMORY_REVIEW_PROMPT)
    else:
        prompt = getattr(agent, "_SKILL_REVIEW_PROMPT", _SKILL_REVIEW_PROMPT)

    focus = (focus or "").strip()
    if focus:
        prompt = (
            f"{prompt}\n\n"
            f"The user explicitly requested this review with the following "
            f"focus — prioritize it over the general instructions above:\n"
            f"{focus}"
        )

    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot, prompt, task_cfg)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "is_background_review_enabled",
    "load_background_review_settings",
    "spawn_background_review_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
