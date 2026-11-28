"""Lean-mode stale-tail demotion for ContextCompressor (old tail tool results become one-line recovery stubs), and
the Skill Safety Rule the compaction handoff carries whenever its transcript holds a skill prune marker."""

from __future__ import annotations

import json
import re
from typing import Any

# Demote tool results older than the newest N rounds so the tail budget binds
# (the tool-group alignment floor otherwise keeps ~32K of tool output alive).
_LEAN_TAIL_KEEP_TOOL_ROUNDS = 6
_LEAN_TAIL_DEMOTE_MIN_CHARS = 1_500

# …and the argument half of the same stale rounds. A tool round is two rows: what the model sent
# (``tool_calls[].function.arguments`` — the whole body of a ``write_file``, both sides of a ``patch``, an inline
# shell script) and what it got back. Only the result half demoted, so the arguments rode the tail verbatim — the
# larger half, and pure duplication once the write is on disk. Same threshold and same round protection as the
# results, so the two halves of a round demote together and the call/result pairing survives by id.
_ARGS_STUB_KEY = "_demoted"
_ARGS_STUB_MARK = "arguments demoted at compaction"
_ARGS_POINTER_SEP = " — full arguments: "


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
    """True for a tail tool result already replaced by ``_lean_recovery_stub`` (plus an optional reload marker),
    or for a prune stub naming its archived original (re-summarizing it would drop the only pointer back)."""
    from tools.tool_result_storage import has_archive_pointer
    return len(content) < 1_000 and _DEMOTED_STUB_RE.match(content) is not None or has_archive_pointer(content)


def _lean_recovery_stub(tool_name: str, content_len: int, session_id: str) -> str:
    """One-line replacement for a demoted tail tool result."""
    # Search defaults to user/assistant rows, so demoted tool output needs an explicit tool filter.
    hint = (f" Recover with session_search(query=..., session_id='{session_id}', role_filter='tool', detail='full')"
            if session_id else "")
    return (
        f"[{tool_name or 'tool'} output demoted at compaction — {content_len:,} "
        f"chars preserved in session history.{hint}]"
    )


def _arguments_stub(tool_name: str, args_len: int, session_id: str) -> str:
    """One-line replacement for the arguments of a demoted tail tool call, valid JSON with the call's shape kept.

    The stub rides where the arguments rode, so anything reading the row back (``_json_dict`` callers, the
    OpenAI-shaped replay) must still parse it: only the values inside the object are replaced.
    """
    # Search defaults to user/assistant rows, so the call's own row needs the assistant filter to be found.
    hint = (f" Recover with session_search(query=..., session_id='{session_id}', role_filter='assistant', detail='full')"
            if session_id else "")
    return json.dumps(
        {_ARGS_STUB_KEY: f"{tool_name or 'tool'} {_ARGS_STUB_MARK} — {args_len:,} chars.{hint}"},
        ensure_ascii=False,
    )


def is_arguments_stub(arguments: Any) -> bool:
    """True for a demotion stub standing in for a stale call's arguments (short JSON, our one key inside)."""
    if not isinstance(arguments, str) or len(arguments) > 400 or _ARGS_STUB_MARK not in arguments:
        return False
    from agent.context_compressor import _json_dict
    return _ARGS_STUB_KEY in _json_dict(arguments)


def with_arguments_pointer(arguments: str, display_path: str) -> str:
    """Point a demoted argument stub at the archive of its original (the result side's ``_with_pointer``).

    Returns *arguments* unchanged when it is not one of our stubs or already names a file, so a second pass
    (or a resumed transcript) cannot stack two pointers onto the same stub.
    """
    from agent.context_compressor import _json_dict
    parsed = _json_dict(arguments)
    if _ARGS_STUB_KEY not in parsed or _ARGS_POINTER_SEP in str(parsed[_ARGS_STUB_KEY]):
        return arguments
    parsed[_ARGS_STUB_KEY] = f"{parsed[_ARGS_STUB_KEY]} {_ARGS_POINTER_SEP}{display_path}"
    return json.dumps(parsed, ensure_ascii=False)


def _demote_stale_tail_arguments(
    rows: list[dict[str, Any]], stale_calls: set[str], session_id: str,
) -> int:
    """Replace the arguments of stale call rows in *rows* with a stub; returns how many rows it changed.

    Only the calls of a stale round (a call whose own result row sits outside the protected newest rounds) are
    touched, so a kept round's arguments - and a call whose result row is not in the tail at all - survive whole.
    Long enough to pay for itself (``_LEAN_TAIL_DEMOTE_MIN_CHARS``), and never a ``skill_view`` call: its handle is
    the reload pointer the marker protocol depends on (#32106). The ``id`` and ``name`` are kept verbatim so the call
    still pairs with its result row (the API rejects a result whose call is gone).

    A stub is committed only where the original was archived (the caller's ``archive_compaction_drops`` names the file
    and puts rows back when the write fails) — the same contract as the result half, so a demoted call never loses the
    text it replaced. Where the archive folder is unavailable (plugin or hosted backend) the plain stub stands:
    reclaiming still beats keeping the arguments, exactly as the result half decides.
    """
    from agent.context_compressor import _tc_get

    demoted = 0
    for i, row in enumerate(rows):
        calls = row.get("tool_calls") if isinstance(row, dict) else None
        if not calls:
            continue
        kept_calls, changed = [], False
        for tc in calls:
            fn = _tc_get(tc, "function", {})
            raw = _tc_get(fn, "arguments")
            if (str(_tc_get(tc, "id") or "") not in stale_calls or not isinstance(raw, str)
                    or len(raw) < _LEAN_TAIL_DEMOTE_MIN_CHARS
                    or is_arguments_stub(raw) or _tc_get(fn, "name") == "skill_view"):
                kept_calls.append(tc)
                continue
            stub_fn = dict(fn) if isinstance(fn, dict) else {}
            stub_fn["arguments"] = _arguments_stub(_tc_get(fn, "name"), len(raw), session_id)
            kept_calls.append({**(dict(tc) if isinstance(tc, dict) else {}), "function": stub_fn})
            changed = True
        if changed:
            rows[i] = {**row, "tool_calls": kept_calls}
            demoted += 1
    return demoted


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
    Phase-1 stub does, so the model does not act on instructions it no longer has (#32106). On the local backend
    each demoted original is archived and its stub names the file (a failed write keeps that row whole). Returns
    a new list (untouched rows shared, demoted copied) and the number of rows demoted."""
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
    # The stale rounds' arguments go with their results: same rounds, same protection, so the tail keeps the two
    # halves of a round together (and a kept round keeps both). Counted as rows, like the results above.
    stale_calls = {
        str(messages[i].get("tool_call_id") or "") for i in range(tail_start, len(messages))
        if messages[i].get("role") == "tool" and i not in protected
    }
    demoted += _demote_stale_tail_arguments(result, stale_calls, session_id)
    if demoted:  # the demoted originals (results and call arguments) are saved and named like the window's stubs
        from tools.tool_result_storage import archive_compaction_drops

        result, _, kept = archive_compaction_drops(messages, result, session_id)
        demoted -= kept
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
        compressor, getattr(compressor, "_session_db", None), getattr(compressor, "_session_id", ""),
        persistence=getattr(compressor, "_persistence", None),
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


def with_skill_safety_rule(
    summary: str, head: list[dict[str, Any]], tail: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """``(summary, head, tail)`` carrying :data:`SKILL_SAFETY_RULE` exactly once when the summary or a kept row holds a
    prune marker (summary-side ``## Pruned Skills`` markers, Phase-1 and lean-demoted ``skill_view`` stubs). A kept
    prune-only stub's copy (:func:`with_pruned_row_skill_safety_rule`) moves onto the handoff: compaction is the
    sanctioned rewrite, and the token-budget tail often keeps a stub the count-protected prune-only tail ruled."""
    from agent.context_compressor import SKILL_PRUNED_MARKER_PREFIX
    from session_rules.turns import _content_text_for_contains

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
    from agent.context_compressor import SKILL_PRUNED_MARKER_PREFIX
    from session_rules.turns import _content_text_for_contains
    if any(SKILL_SAFETY_RULE in _content_text_for_contains(m.get("content")) for m in pruned if isinstance(m, dict)):
        return pruned
    for i, (before, after) in enumerate(zip(original, pruned)):
        content = after.get("content") if isinstance(after, dict) else None
        if isinstance(content, str) and SKILL_PRUNED_MARKER_PREFIX in content and content != before.get("content"):
            return [*pruned[:i], {**after, "content": content + _SKILL_SAFETY_SUFFIX}, *pruned[i + 1:]]
    return pruned
