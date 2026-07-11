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

import json
import logging
import os
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


def _resolve_review_runtime(agent: Any) -> Dict[str, Any]:
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
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return parent
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {}) if isinstance(aux.get("background_review"), dict) else {}
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
    "READ-BEFORE-WRITE HANDSHAKE (load the target before patching — #62397):\n"
    "  Every background-review write goes through a guard that refuses the\n"
    "  mutation when the exact target file wasn't loaded via skill_view in\n"
    "  this review turn. The guard's failure mode is a hard refusal — the\n"
    "  write is silently dropped and the learning is lost. To avoid that:\n"
    "    • Before any skill_manage(action='patch' | 'edit' | 'write_file' |\n"
    "      'delete' | 'remove_file'), call skill_view FIRST in this turn.\n"
    "    • For SKILL.md: skill_view(name) returns the current body.\n"
    "    • For a support file (under references/, templates/, scripts/):\n"
    "      skill_view(name, file_path='references/<topic>.md') — the file_path\n"
    "      form is required because the guard tracks reads per-path, not\n"
    "      per-skill.\n"
    "    • If the guard returns {\"_read_before_write_required\": true}, call\n"
    "      skill_view for the target, then REISSUE the write using the\n"
    "      content just returned by skill_view (do not re-derive it from\n"
    "      memory — the guard compares against the in-memory version).\n"
    "    • This applies even when you're CREATING a new support file: the\n"
    "      guard refuses the write whenever the file already exists, so a\n"
    "      single re-edit must be preceded by a skill_view of the existing\n"
    "      content. First-time creations (no existing file) are fine\n"
    "      without a read.\n"
    "  Skipping this handshake is the #1 cause of \"the background review\n"
    "  ran but nothing got patched\" — don't skip it.\n\n"
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
    "play, so it's the right one to extend.\n"
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
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
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
    "it's the right place.\n"
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
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
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


def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str,
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
            _rt = _resolve_review_runtime(agent)
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
            review_agent = AIAgent(
                model=_rt.get("model") or agent.model,
                max_iterations=16,
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
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # Safety-net cleanup for the exception path.  Normal completion already
        # shut down inside the thread-scoped silence above.  Re-enter the
        # thread-scoped silence here so teardown output (Honcho flush, Hindsight
        # sync, background thread joins) stays quiet even on the exception path,
        # without blanking other threads' streams.
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
):
    """Build the review thread target and prompt for a background review.

    Returns a ``(target, prompt)`` tuple.  The caller (``AIAgent._spawn_background_review``)
    owns the actual ``threading.Thread`` construction so test-level patches
    of ``run_agent.threading.Thread`` keep working.
    """
    # Pick the right prompt based on which triggers fired.  Allow per-agent
    # override (the prompts moved to module-level constants but old code paths
    # that set agent._MEMORY_REVIEW_PROMPT etc. directly keep working).
    if review_memory and review_skills:
        prompt = getattr(agent, "_COMBINED_REVIEW_PROMPT", _COMBINED_REVIEW_PROMPT)
    elif review_memory:
        prompt = getattr(agent, "_MEMORY_REVIEW_PROMPT", _MEMORY_REVIEW_PROMPT)
    else:
        prompt = getattr(agent, "_SKILL_REVIEW_PROMPT", _SKILL_REVIEW_PROMPT)

    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot, prompt)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "spawn_background_review_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
