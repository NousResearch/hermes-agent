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
    :data:`TAIL_MAX_THRESHOLD_FRACTION` of the trigger. At or below ``_MIN_SUMMARY_TOKENS`` the summary alone fills
    the trigger, so no tail leaves room under it and only the floor is held: trimming the recency window there
    would lose context without ending the loop."""
    from agent.context_compressor import _MIN_SUMMARY_TOKENS, LEAN_TAIL_CAP_TOKENS, LEAN_TAIL_FLOOR_TOKENS
    floor, window_term = LEAN_TAIL_FLOOR_TOKENS, min(LEAN_TAIL_CAP_TOKENS, int(context_length * 0.025))
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
        stub = _lean_recovery_stub(msg.get("tool_name") or "", len(content), session_id)
        skills = skill_view_names_by_call_id(messages) if skills is None else skills
        if (skill := skills.get(str(msg.get("tool_call_id") or ""))) and not _result_failure_suffix(content):
            stub += " " + _skill_pruned_marker(skill)
        result[i] = _rewritten(msg, stub)
        demoted += 1
    return result, demoted


# Sent only where ``[SKILL_PRUNED:`` markers exist — after a compaction, riding its handoff — rather than in every
# fresh session's system prompt; adding it to a compacted transcript is the one sanctioned cache break.
SKILL_SAFETY_HEADING = "## Skill Safety Rule"
SKILL_SAFETY_RULE = (
    f"{SKILL_SAFETY_HEADING}\n"
    "A skill placeholder containing `[SKILL_PRUNED]` lost its content in context compression and is inaccessible — "
    "reload it with skill_view(name='...') before acting on anything that depends on it. After reloading, ignore any "
    "remaining `[SKILL_PRUNED]` markers for that same skill; they are historical artifacts of earlier compactions."
)


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
