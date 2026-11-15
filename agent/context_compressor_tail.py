"""Lean-mode stale-tail demotion for ContextCompressor (old tail tool results become one-line recovery stubs), and
the Skill Safety Rule the compaction handoff carries whenever its transcript holds a skill prune marker."""

from __future__ import annotations

import re
from typing import Any

# Demote tool results older than the newest N rounds so the tail budget binds
# (the tool-group alignment floor otherwise keeps ~32K of tool output alive).
_LEAN_TAIL_KEEP_TOOL_ROUNDS = 6
_LEAN_TAIL_DEMOTE_MIN_CHARS = 1_500


# Matches what ``_lean_recovery_stub`` emits, so later passes treat a demoted row as already summarised: with a session
# id and a reload marker the stub can pass the Phase-1 prune floor, and re-summarising it would drop both pointers.
_DEMOTED_STUB_RE = re.compile(r"\[[^\s\]]* output demoted at compaction — [\d,]+ chars preserved in session history\.")

# :func:`lean_tail_budget`: the lean tail never claims more than this share of the trigger. A configured
# ``threshold_tokens`` (or aux summariser ceiling) far below the window left the 10K floor at 83% of a 12K trigger, and
# the 2.5%-of-window term at 216% of it on a 1M window, so head + summary + tail stayed over threshold and every pre-API
# pass refused as "would grow" (the window-side twin is ``TAIL_MAX_CONTEXT_FRACTION``).
TAIL_MAX_THRESHOLD_FRACTION = 0.50


def lean_tail_budget(context_length: int, threshold_tokens: int) -> int:
    """Lean-mode tail selection budget: 2.5% of the window clamped to the 10K floor / 25K cap, then held to
    :data:`TAIL_MAX_THRESHOLD_FRACTION` of the trigger. When ``threshold_tokens`` is lowered below the default
    ``0.50 * context_length`` (e.g. ``0.15`` cliff-avoidance floor #CF-83), scale the recency window against
    ``min(context_length, threshold_tokens * 2)`` so a 1M window does not pin the 25K cap against a 157K trigger.
    At or below ``_MIN_SUMMARY_TOKENS`` the summary alone fills the trigger, so no tail leaves room under it and
    only the floor is held: trimming the recency window there would lose context without ending the loop."""
    from agent.context_compressor import _MIN_SUMMARY_TOKENS, LEAN_TAIL_CAP_TOKENS, LEAN_TAIL_FLOOR_TOKENS
    effective_window = (
        min(context_length, threshold_tokens * 2)
        if threshold_tokens > _MIN_SUMMARY_TOKENS
        else context_length
    )
    floor, window_term = LEAN_TAIL_FLOOR_TOKENS, min(LEAN_TAIL_CAP_TOKENS, int(effective_window * 0.025))
    if threshold_tokens > 0:
        share = int(threshold_tokens * TAIL_MAX_THRESHOLD_FRACTION)
        floor = min(floor, share)
        if threshold_tokens > _MIN_SUMMARY_TOKENS:
            window_term = min(window_term, share)
    return max(floor, window_term)


def is_demoted_stub(content: str) -> bool:
    """True for a tail tool result already replaced by ``_lean_recovery_stub`` (plus an optional reload marker)."""
    return len(content) < 1_000 and _DEMOTED_STUB_RE.match(content) is not None


def _lean_recovery_stub(tool_name: str, content_len: int, session_id: str) -> str:
    """One-line replacement for a demoted tail tool result."""
    hint = f" Recover with session_search(query=..., session_id='{session_id}')" if session_id else ""
    return (
        f"[{tool_name or 'tool'} output demoted at compaction — {content_len:,} "
        f"chars preserved in session history.{hint}]"
    )


def skill_view_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map each ``skill_view`` tool-call id to the skill name it loaded."""
    from agent.context_compressor import _json_dict, _tc_get

    names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = _tc_get(tc, "function", {})
            skill = _json_dict(_tc_get(fn, "arguments")).get("name") if _tc_get(fn, "name") == "skill_view" else None
            if isinstance(skill, str) and skill:
                names[str(_tc_get(tc, "id") or "")] = skill
    return names


def demote_stale_tail_tools(
    messages: list[dict[str, Any]], tail_start: int, session_id: str,
) -> tuple[list[dict[str, Any]], int]:
    """Demote tail tool results older than the newest ``_LEAN_TAIL_KEEP_TOOL_ROUNDS`` rounds to recovery stubs;
    skill-marker rows untouched. A demoted ``skill_view`` body carries the canonical reload marker, as its
    Phase-1 stub does, so the model does not act on instructions it no longer has (#32106). Returns a new list
    (untouched rows shared, demoted copied) and the number of rows demoted."""
    from agent.context_compressor import (
        SKILL_PRUNED_MARKER_PREFIX, _is_summary_stub, _result_failure_suffix, _rewritten, _skill_pruned_marker,
    )

    rounds_seen = 0
    protected: set[int] = set()
    prev_idx = None
    for i in (i for i in range(len(messages) - 1, tail_start - 1, -1) if messages[i].get("role") == "tool"):
        rounds_seen += prev_idx is None or prev_idx - i > 1
        prev_idx = i
        if rounds_seen > _LEAN_TAIL_KEEP_TOOL_ROUNDS:
            break
        protected.add(i)
    from agent.context_compressor import _collect_protected_skill_names
    protected_skills = _collect_protected_skill_names(messages, min(protected, default=len(messages)))
    result = list(messages)
    demoted = 0
    skills: dict[str, str] | None = None
    for i in range(tail_start, len(messages)):
        msg = messages[i]
        content = msg.get("content")
        if msg.get("role") != "tool" or i in protected or not isinstance(content, str):
            continue
        if len(content) < _LEAN_TAIL_DEMOTE_MIN_CHARS or SKILL_PRUNED_MARKER_PREFIX in content or _is_summary_stub(content):
            continue
        skills = skill_view_names_by_call_id(messages) if skills is None else skills
        skill = skills.get(str(msg.get("tool_call_id") or ""))
        if skill and (skill.strip().lower() in protected_skills or skill.strip().lower().rsplit("/", 1)[-1] in protected_skills) and not _result_failure_suffix(content):
            continue
        stub = _lean_recovery_stub(msg.get("tool_name") or "", len(content), session_id)
        if skill and not _result_failure_suffix(content):
            stub += " " + _skill_pruned_marker(skill)
        result[i] = _rewritten(msg, stub)
        demoted += 1
    return result, demoted


# Sent only where ``[SKILL_PRUNED:`` markers exist — after a compaction, riding its handoff — rather than in every
# fresh session's system prompt; adding it to a compacted transcript is the one sanctioned cache break.
SKILL_SAFETY_HEADING = "## Skill Safety Rule"
SKILL_SAFETY_RULE = (
    f"{SKILL_SAFETY_HEADING}\n"
    "A skill placeholder containing `[SKILL_PRUNED]` lost its full body in context compression — "
    "do NOT batch-reload pruned skills at the compaction boundary; reload a skill with `skill_view(name='...')` "
    "ONLY at the exact step that actively needs its detailed procedure (at most 1-2 skills per turn, never an eager reload sweep). "
    "After reloading, ignore any remaining `[SKILL_PRUNED]` markers for that same skill; they are historical artifacts of earlier compactions.\n"
    "Pinned Invariant Floor (always active even while a skill reload is pending):\n"
    "- Outbound (outbound-comms / google-chat-comms / humanizer): stage drafts only (never auto-send); Gmail create/update_draft = FULL REPLACE (pass BOTH body and html_body + thread_id/reply_to_message_id on replies); Celina/Ola/Hanna max 1 sentence DM / 1-2 room sentences, zero bullet lists; devs=English, business/store=Swedish.\n"
    "- Plane (plane-tickets): engineering projects (NEWWEBSHOP/ONE/ECOM/ATLAS/2IZII) = English only, raw HTML, explicit priority + label, 2IZII requires 'internal' label, never assign/flip developer colleague tickets without a 24h Google Chat heads-up.\n"
    "- Verification (always-verify / dab / rca / deep-research / research-and-plan): re-run receipts LIVE at claim time, quote verbatim output lines, run polarity + disconfirming checks before claiming done, and queue non-urgent GO asks via decision-queue.py."
)


def native_anthropic_tail_prices(compressor: Any):
    """Tail prices ``messages -> [tokens per row]`` on a preserved-thinking native Anthropic route, else
    None. That route replays exactly the projected carriers minus this session's rejected signatures —
    what the preflight estimate prices — never the canonical ``reasoning`` keys. Priced over the whole
    list: whether a row's signed thinking survives conversion depends on its neighbours."""
    from agent.anthropic_thinking_policy import native_anthropic_preserves_prior_thinking

    if (getattr(compressor, "api_mode", "") or "") != "anthropic_messages" or not native_anthropic_preserves_prior_thinking(
        getattr(compressor, "base_url", ""), getattr(compressor, "model", "")
    ):
        return None
    from agent.anthropic_thinking_replay import session_rejected_thinking, strip_rejected_thinking
    from agent.context_compressor import _estimate_msg_budget_tokens
    from agent.message_sanitization import native_anthropic_row_costs

    rejected = session_rejected_thinking(
        compressor, getattr(compressor, "_session_db", None), getattr(compressor, "_session_id", "")
    )

    def prices(messages: list[dict[str, Any]]) -> list[int]:
        request_copies = [dict(m) if isinstance(m, dict) else m for m in messages]
        if rejected:
            for copy in request_copies:
                strip_rejected_thinking(copy, rejected)
        return [
            _estimate_msg_budget_tokens(shadow) + thinking if isinstance(shadow, dict) else 0
            for shadow, thinking in native_anthropic_row_costs(request_copies)
        ]

    return prices


def tail_cut_pricer(compressor: Any, messages: list[dict[str, Any]], head_end: int):
    """Tail-cut walk price ``index -> tokens`` in which rows at or before the newest handoff cost nothing.

    The handoff scan folds every row up to the newest in-transcript handoff into the summary window (the tail starts
    past it), so none of them can stay in the protected tail. Charging them let the handoff row alone break the soft
    ceiling: the walk stopped on it, the window held only that handoff, and compress() returned the
    ``empty_post_handoff_window`` no-op at every trigger while the session grew past its window."""
    price = compressor._tail_row_pricer(messages)
    newest = next(
        (i for i in range(len(messages) - 1, head_end - 1, -1) if compressor._is_context_summary_message(messages[i])), -1,
    )
    return price if newest < 0 else (lambda i: 0 if i <= newest else price(i))


def tail_cut_progress_floor(compressor: Any, messages: list[dict[str, Any]], head_end: int, cut_idx: int) -> int:
    """Lowest tail cut that leaves the summary window a row the handoff scan keeps.

    The scan drops standalone handoff rows from the window, so when every row before the cut (and up to the newest
    handoff) is a handoff, the planned window is empty and compress() no-ops with ``empty_post_handoff_window``. That
    happens when the cut lands inside the tool group right after the handoff and is aligned back onto the group's
    assistant. The floor then moves past that group, unless doing so would take the newest row or the active user
    request out of the tail; those cases keep the old ``head_end + 1`` floor. ``_compress_window``'s
    assistant/tool bridge before the active request obeys the same floor, so the two never disagree: when the floor
    sits on the active request, the summary merges into that row instead of the bridge emptying the window."""
    floor = head_end + 1
    newest = next(
        (i for i in range(len(messages) - 1, head_end - 1, -1) if compressor._is_context_summary_message(messages[i])), -1,
    )
    if newest < 0 or not all(map(compressor._is_context_summary_message, messages[head_end:max(cut_idx, newest + 1)])):
        return floor
    raised = compressor._align_boundary_forward(messages, newest + 2)
    last_user = compressor._find_last_user_message_idx(messages, newest + 1)
    return floor if raised >= len(messages) or 0 <= last_user < raised else raised


def _unpruned_loaded_skill_names(rows: list[dict[str, Any]]) -> set[str]:
    """Return skill names (both full and leaf slug) whose unpruned body is present in *rows*."""
    from agent.context_compressor import (
        SKILL_PRUNED_MARKER_PREFIX,
        _content_text_for_contains,
        _is_summary_stub,
        _json_dict,
        _skill_view_failed,
    )
    from agent.skill_commands import skills_loaded_by_scaffold

    call_map = skill_view_names_by_call_id(rows)
    loaded: set[str] = set()
    for msg in rows:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        text = _content_text_for_contains(content)
        if not text:
            continue
        role = msg.get("role")
        if role == "user":
            for name in skills_loaded_by_scaffold(text):
                if name:
                    loaded.add(name)
                    loaded.add(name.rsplit("/", 1)[-1])
        elif (
            role == "tool"
            and SKILL_PRUNED_MARKER_PREFIX not in text
            and not _is_summary_stub(text)
            and not is_demoted_stub(text)
            and not _skill_view_failed(content)
        ):
            name = call_map.get(str(msg.get("tool_call_id") or ""), "")
            if not name and text.lstrip().startswith("{") and '"success": true' in text[:120]:
                payload = _json_dict(text)
                if (
                    isinstance(payload, dict)
                    and payload.get("success") is True
                    and not payload.get("dedup")
                    and not payload.get("file")
                    and isinstance(payload.get("content"), str)
                ):
                    name = str(payload.get("name") or "")
            if name:
                loaded.add(name)
                loaded.add(name.rsplit("/", 1)[-1])
    return loaded


def strip_loaded_skill_prune_markers(text: str, loaded_skills: set[str]) -> str:
    """Remove ``[SKILL_PRUNED: ...]`` lines from *text* for skills in *loaded_skills*, dropping an
    empty ``## Pruned Skills`` section when all of its markers were satisfied by the active tail."""
    if not text or not loaded_skills:
        return text
    from agent.context_compressor import (
        SKILL_PRUNED_MARKER_PREFIX,
        _PRUNED_SKILLS_SECTION_HEADING,
        _extract_pruned_skill_names,
    )

    if SKILL_PRUNED_MARKER_PREFIX not in text:
        return text
    kept_lines: list[str] = []
    changed = False
    for line in text.splitlines():
        names = _extract_pruned_skill_names(line) if SKILL_PRUNED_MARKER_PREFIX in line else []
        if names and all(n in loaded_skills or n.rsplit("/", 1)[-1] in loaded_skills for n in names):
            changed = True
            continue
        kept_lines.append(line)
    if not changed:
        return text
    rebuilt = "\n".join(kept_lines)
    # Drop an empty `## Pruned Skills` heading left behind when every marker under it was stripped.
    rebuilt = re.sub(
        rf"(?:^|\n\n){re.escape(_PRUNED_SKILLS_SECTION_HEADING)}\s*(?=\n## |\Z)",
        "",
        rebuilt,
    )
    return rebuilt.strip("\n")


def with_skill_safety_rule(
    summary: str, head: list[dict[str, Any]], tail: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """``(summary, head, tail)`` carrying :data:`SKILL_SAFETY_RULE` exactly once when the summary or a kept row holds a
    prune marker (summary-side ``## Pruned Skills`` markers, Phase-1 and lean-demoted ``skill_view`` stubs). A kept
    prune-only stub's copy (:func:`with_pruned_row_skill_safety_rule`) moves onto the handoff: compaction is the
    sanctioned rewrite, and the token-budget tail often keeps a stub the count-protected prune-only tail ruled."""
    from agent.context_compressor import SKILL_PRUNED_MARKER_PREFIX, _content_text_for_contains

    def unruled(rows):
        return [{**m, "content": without_skill_safety_rule(m["content"])}
                if isinstance(m, dict) and m.get("role") != "system" and isinstance(m.get("content"), str)
                and SKILL_SAFETY_RULE in m["content"] else m for m in rows]

    head, tail = unruled(head), unruled(tail)
    loaded_skills = _unpruned_loaded_skill_names([*head, *tail])
    if loaded_skills:
        summary = strip_loaded_skill_prune_markers(summary, loaded_skills)
    texts = (summary, *(_content_text_for_contains(m.get("content")) for m in (*head, *tail) if isinstance(m, dict)))
    if any(SKILL_SAFETY_RULE in text for text in texts) or not any(SKILL_PRUNED_MARKER_PREFIX in t for t in texts):
        return summary, head, tail
    return f"{summary}{_SKILL_SAFETY_SUFFIX}", head, tail


_SKILL_SAFETY_SUFFIX = f"\n\n{SKILL_SAFETY_RULE}"


def without_skill_safety_rule(content: str) -> str:
    """*content* minus an attached :data:`SKILL_SAFETY_RULE` (a ruled prune stub is still a stub)."""
    return content.replace(_SKILL_SAFETY_SUFFIX, "")


def elide_keeping_skill_safety_rule(content: str, limit: int) -> str:
    """``elide(content, limit)`` that keeps an attached rule: the head-only cut would drop it from a long handoff
    while marker rows beside it survive (anti-growth salvage)."""
    from agent.compression_marker import elide
    if SKILL_SAFETY_RULE not in content:
        return elide(content, limit)
    return elide(without_skill_safety_rule(content), limit - len(_SKILL_SAFETY_SUFFIX)) + _SKILL_SAFETY_SUFFIX


def with_pruned_row_skill_safety_rule(original: list[dict[str, Any]], pruned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*pruned* (a prune-only rewrite of *original*) with the rule on the first row it turned into a prune marker,
    when no row carries the rule yet: prune-only has no handoff for :func:`with_skill_safety_rule` to ride."""
    from agent.context_compressor import SKILL_PRUNED_MARKER_PREFIX, _content_text_for_contains
    if any(SKILL_SAFETY_RULE in _content_text_for_contains(m.get("content")) for m in pruned if isinstance(m, dict)):
        return pruned
    for i, (before, after) in enumerate(zip(original, pruned)):
        content = after.get("content") if isinstance(after, dict) else None
        if isinstance(content, str) and SKILL_PRUNED_MARKER_PREFIX in content and content != before.get("content"):
            return [*pruned[:i], {**after, "content": content + _SKILL_SAFETY_SUFFIX}, *pruned[i + 1:]]
    return pruned
