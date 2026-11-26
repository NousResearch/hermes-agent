"""Lean-mode stale-tail demotion, deterministic anchor/file-evidence augmentation for ContextCompressor, and
the Skill Safety Rule the compaction handoff carries whenever its transcript holds a skill prune marker."""

from __future__ import annotations

import re
from typing import Any

from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide
from session_rules.turns import _content_text_for_contains

# 2.5% of the context window, clamped; floor keeps small models workable.
LEAN_TAIL_FLOOR_TOKENS = 10_000
LEAN_TAIL_CAP_TOKENS = 25_000
# Hard share of the window the verbatim tail may occupy, applied after either formula.
TAIL_MAX_CONTEXT_FRACTION = 0.20

_LEAN_USER_MESSAGES_BUDGET_CHARS = 24_000  # ~6K tokens
_LEAN_USER_MESSAGE_MAX_CHARS = 4_000
_LEAN_USER_MESSAGES_HEADING = "## User Messages (verbatim, newest first)"
_LEAN_RECOVERY_HEADING = "## Context Recovery"
_SYNTHETIC_USER_ROW_PREFIXES = (
    "[System:", "[CONTEXT", "[PRIOR CONTEXT", "[IMPORTANT: Background", "[Your active task list",
    "[Planning state preserved", "[ASYNC DELEGATION", "[OUT-OF-BAND", "Cronjob Response:",
)

_LEAN_ANCHOR_HEADING = "## Anchor Index (mechanically extracted, exact)"
_LEAN_ANCHOR_BUDGET_CHARS = 7_000
_FILE_EXT_PATTERN = (
    r"(?:py|ts|tsx|js|jsx|rs|go|java|c|cpp|h|sql|md|txt|rst|yaml|yml|json|jsonl|"
    r"toml|ini|cfg|sh|ps1|csv|tsv|xlsx|xls|pdf|docx|doc|pptx|eml|msg|ics|vcf|html|htm|xml|log)"
)
_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("PRs/issues", re.compile(r"#\d{3,6}\b"), 120),
    ("commits", re.compile(r"\b[0-9a-f]{9,40}\b"), 40),
    ("branches", re.compile(r"\b(?:fix|feat|docs|refactor|chore|salvage|ent)/[A-Za-z0-9._/-]{3,60}"), 40),
    ("files", re.compile(rf"\b(?:[\w./-]+/)?[\w.-]+\.{_FILE_EXT_PATTERN}(?:\.\w{{2,4}})?\b"), 90),
    ("record_ids", re.compile(r"\b[A-Z]{2,10}-\d{2,10}(?:-[A-Z0-9]{1,10})?\b"), 80),
    ("amounts_dates", re.compile(r"(?:\$|€|£)\d[\d,]*(?:\.\d{2})?\b|\b20\d{2}-[01]\d-[0-3]\d(?:[T\s][0-2]\d:[0-5]\d)?\b"), 60),
    ("errors", re.compile(r"\b(?:[A-Z][a-zA-Z]*Error|Exception|ENOSPC|EACCES|SIGKILL|Traceback)\b[^\n]{0,90}"), 40),
    ("handles", re.compile(r"@[A-Za-z0-9-]{3,30}\b"), 40),
    ("urls", re.compile(r"https?://[^\s)\"']{10,110}"), 30),
]
_ANCHOR_NOISE = frozenset({"@teknium", "@teknium1"})

_LEAN_FILE_EVIDENCE_HEADING = "## File Evidence Extracts (verbatim, redacted)"
_LEAN_FILE_EVIDENCE_BUDGET_CHARS = 6_000
_LEAN_FILE_PER_DOC_MAX_CHARS = 520
_EMAIL_TRANSPORT_HDR_RE = re.compile(
    r"^(?:Return-Path|Received|Message-ID|MIME-Version|Content-Type|Content-Transfer-Encoding|X-[A-Za-z0-9-]+)\s*:",
    re.IGNORECASE,
)
_VCAL_NOISE_RE = re.compile(r"^(?:BEGIN:VCALENDAR|END:VCALENDAR|BEGIN:VEVENT|END:VEVENT|VERSION:|PRODID:)\s*$", re.IGNORECASE)


def _synthetic_user_row(content: str) -> bool:
    """True for scaffolding user rows that carry no real user words."""
    if not isinstance(content, str) or not content.strip():
        return True
    return content.lstrip().startswith(_SYNTHETIC_USER_ROW_PREFIXES)


def _typed_user_text(content: str) -> str:
    """A ``/skill``, bundle or auto-load turn as the user typed it (``/skill do X``); other text unchanged."""
    from agent.skill_commands import skills_loaded_by_scaffold
    from session_rules.turns import describe_skill_invocation
    return (describe_skill_invocation(content, separator=" ") or "") if skills_loaded_by_scaffold(content) else content


def _build_verbatim_user_section(turns: list[dict[str, Any]]) -> str:
    """Compacted region's REAL user messages verbatim, newest-first under a char budget (straddler truncated); "" if none."""
    collected: list[str] = []
    used = 0
    for msg in reversed(turns):
        if msg.get("role") != "user":
            continue
        content = _typed_user_text(_content_text_for_contains(msg.get("content")))
        if _synthetic_user_row(content):
            continue
        remaining = _LEAN_USER_MESSAGES_BUDGET_CHARS - used
        if remaining <= 0:
            break
        text = content.strip()
        if len(text) > remaining and remaining <= ELISION_MARKER_MAX_LEN:
            break
        text = elide(text, min(_LEAN_USER_MESSAGE_MAX_CHARS, remaining))
        collected.append("> " + text.replace("\n", "\n> "))
        used += len(text)
    if not collected:
        return ""
    return (
        "\n\n" + _LEAN_USER_MESSAGES_HEADING + "\n"
        + "\n\n".join(collected)
        + "\n(Every real user message from the compacted region, quoted "
        "verbatim. These are the user's actual words and override any "
        "paraphrase of them above.)"
    )


def _build_recovery_footer(session_id: str, region_len: int) -> str:
    """Deterministic pointer to the compacted region in session history."""
    if not session_id:
        return ""
    return (
        "\n\n" + _LEAN_RECOVERY_HEADING + "\n"
        f"The {region_len} compacted message(s) remain fully preserved in "
        "session history. If you need any detail this summary does not carry "
        "(exact command output, file contents, error text, earlier "
        "reasoning), recover it with: "
        f"session_search(query='<keywords>', session_id='{session_id}') — "
        "do not guess at lost specifics when you can look them up."
    )


def _iter_turn_anchor_texts(turns: list[dict[str, Any]]):
    """Yield message text AND serialized tool_call arguments so file paths/IDs in tool calls are indexed."""
    from agent.context_compressor import _extract_tool_call_name_and_args

    for m in turns:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str) and content:
            yield _typed_user_text(content) if role == "user" else content
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                _name, raw_args = _extract_tool_call_name_and_args(tc)
                if raw_args:
                    yield raw_args


def _build_anchor_index(turns: list[dict[str, Any]]) -> str:
    """Regex-harvest exact identifiers from the compacted region (LLM-free); per-category caps, most-frequent first."""
    text = "\n".join(_iter_turn_anchor_texts(turns))
    if not text:
        return ""
    sections: list[str] = []
    used = 0
    for label, pattern, cap in _ANCHOR_PATTERNS:
        counts: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        for n, m in enumerate(pattern.finditer(text)):
            val = m.group(0).strip().rstrip(".,;:")
            if val.lower() in _ANCHOR_NOISE:
                continue
            counts[val] = counts.get(val, 0) + 1
            last_seen[val] = n
        if not counts:
            continue
        ranked = sorted(counts, key=lambda v: (-counts[v], -last_seen[v]))[:cap]
        line = f"{label}: " + ", ".join(f"{v}(x{counts[v]})" if counts[v] > 1 else v for v in ranked)
        if used + len(line) > _LEAN_ANCHOR_BUDGET_CHARS:
            break
        sections.append(line)
        used += len(line)
    if not sections:
        return ""
    return (
        "\n\n" + _LEAN_ANCHOR_HEADING + "\n"
        + "\n".join(sections)
        + "\n(Exact identifiers from the compacted region — use these verbatim, "
        "and as session_search query anchors to recover their full context.)"
    )


def _strip_repeated_boilerplate_chunks(text: str) -> str:
    """Collapse bracketed blocks or sentences repeated >=3 times inside a single read file."""
    for pat in (r"\[[^\]\n]{20,240}\]", r"[^.\n\[]{35,200}\."):
        found = re.findall(pat, text)
        counts: dict[str, int] = {}
        for ch in found:
            key = " ".join(ch.split()).lower()
            counts[key] = counts.get(key, 0) + 1
        for ch in set(found):
            if counts.get(" ".join(ch.split()).lower(), 0) >= 3:
                text = text.replace(ch, " ")
    return text


def _distill_file_body(raw_content: str, max_chars: int = _LEAN_FILE_PER_DOC_MAX_CHARS) -> str:
    """LLM-free structural distillation of a read file body: strips transport headers, HTML tags,
    and repetitive boilerplate blocks/sentences while keeping unique factual lines verbatim."""
    from agent.context_compressor import _redact_compaction_text

    text = _redact_compaction_text(raw_content or "")
    if "<html" in text.lower() or "<body" in text.lower() or "<div" in text.lower():
        text = re.sub(r"<[^>]+>", " ", text)
    text = _strip_repeated_boilerplate_chunks(text)
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_counts: dict[str, int] = {}
    for ln in raw_lines:
        norm = re.sub(r"\s+", " ", ln).lower()
        line_counts[norm] = line_counts.get(norm, 0) + 1

    kept: list[str] = []
    seen_once: set[str] = set()
    used = 0
    for ln in raw_lines:
        if ln == "--" or _EMAIL_TRANSPORT_HDR_RE.match(ln) or _VCAL_NOISE_RE.match(ln):
            continue
        compact = re.sub(r"\s+", " ", ln).strip()
        norm = compact.lower()
        if not compact or line_counts.get(norm, 0) >= 3 or norm in seen_once:
            continue
        seen_once.add(norm)
        if used + len(compact) + 3 > max_chars:
            remain = max_chars - used - 4
            if remain > 30:
                kept.append(compact[:remain].rstrip() + "...")
            break
        kept.append(compact)
        used += len(compact) + 3
    return " | ".join(kept)


def _build_file_evidence_ledger(turns: list[dict[str, Any]]) -> str:
    """Mechanically preserve file provenance and non-boilerplate verbatim extracts for `read_file`,
    `write_file`, and `patch` calls in the compacted window (0 LLM calls, strictly bounded)."""
    from agent.context_compressor import (
        _is_summary_stub,
        _json_dict,
        _result_failure_suffix,
        _str_arg,
        _summarize_refused_tool_result,
        _tool_calls_by_id,
    )

    call_map = _tool_calls_by_id(turns)
    file_entries: dict[str, str] = {}
    for msg in turns:
        content = msg.get("content")
        if msg.get("role") != "tool" or not isinstance(content, str) or not content.strip() or _is_summary_stub(content):
            continue
        if _result_failure_suffix(content) or _summarize_refused_tool_result("tool", {}, content):
            continue
        tool_name, raw_args = call_map.get(str(msg.get("tool_call_id") or ""), (str(msg.get("tool_name") or ""), ""))
        if tool_name not in ("read_file", "write_file", "patch"):
            continue
        args = _json_dict(raw_args)
        path = _str_arg(args, "path") or _str_arg(args, "file_path")
        if not path:
            continue
        if tool_name == "read_file":
            if digest := _distill_file_body(content):
                file_entries[path] = f"- `{path}`: {digest}"
        else:
            file_entries[path] = f"- `{path}` [{tool_name}]"
    if not file_entries:
        return ""

    selected_rev: list[str] = []
    used = 0
    for entry in reversed(list(file_entries.values())):
        if used + len(entry) + 1 > _LEAN_FILE_EVIDENCE_BUDGET_CHARS:
            break
        selected_rev.append(entry)
        used += len(entry) + 1
    if not selected_rev:
        return ""
    selected_rev.reverse()
    return "\n\n" + _LEAN_FILE_EVIDENCE_HEADING + "\n" + "\n".join(selected_rev)


def collect_head_pruned_exchanges(
    plan: list[dict[str, Any]], canonical_messages: list[dict[str, Any]], compress_start: int,
) -> list[dict[str, Any]]:
    """Return ONLY the Phase-1-pruned head ``tool`` rows (``0..compress_start``) and synthetic
    ``assistant`` stubs carrying their matching ``tool_calls`` (``content=""``).

    Protected ``system`` / ``user`` rows and unpruned sibling head tool calls stay intact in the
    head and are excluded so their identifiers never leak into the summary's Anchor Index or
    File Evidence Extracts.
    """
    from agent.context_compressor import _tc_get

    bound = min(max(compress_start, 0), len(plan), len(canonical_messages))
    pruned_indices = {
        i for i in range(bound)
        if plan[i].get("role") == "tool" and plan[i].get("content") != canonical_messages[i].get("content")
    }
    if not pruned_indices:
        return []
    pruned_call_ids = {
        str(canonical_messages[i].get("tool_call_id") or "") for i in pruned_indices
    } - {""}
    rescued: list[dict[str, Any]] = []
    for i in range(bound):
        msg = canonical_messages[i]
        if i in pruned_indices:
            rescued.append(msg)
        elif msg.get("role") == "assistant" and pruned_call_ids:
            matched_tcs = [
                tc for tc in (msg.get("tool_calls") or [])
                if str(_tc_get(tc, "id") or "") in pruned_call_ids
            ]
            if matched_tcs:
                rescued.append({"role": "assistant", "content": "", "tool_calls": matched_tcs})
    return rescued


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
    from agent.context_compressor import _MIN_SUMMARY_TOKENS
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
    if demoted:  # the demoted originals are saved and named like the window's prune stubs
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
