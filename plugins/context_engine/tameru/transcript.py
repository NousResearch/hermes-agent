"""Transcript-level extractive prune — harness-agnostic, no host imports.

Operates on OpenAI-style message lists (``[{"role": ..., "content": ...}]``),
the convention shared by Hermes, OpenCode, Codex, and most agent harnesses.
Turns bulky old tool-result payloads into extractive keep/drop using the
same compress_context contract. Safe to unit-test standalone.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from itertools import chain
from typing import Any

from .contract_gates import GENERIC_WORDS
from .compress_context import (
    _SUMMARY_STRUCTURED_TOKEN_RE,
    _detect_json_payloads,
    _extract_entities,
    _extract_terms,
    _json_query_needles,
    _selector_patterns,
    _summary_preserves_required_facts,
    compress_context,
    preprocess_json,
)

MIN_TOOL_CHARS = 800
PROTECT_LAST_TOOL = 2

_PRESERVATION_VERBS = {"carry", "keep", "preserve", "remember", "retain"}
_PRESERVATION_QUANTIFIERS = {"all", "entire", "every", "everything", "full"}
_PRESERVATION_OBJECTS = {
    "content",
    "context",
    "detail",
    "details",
    "fact",
    "facts",
    "history",
    "info",
    "information",
}


_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})

# task_query limits. Tool-call argument leaves are usually paths, commands or
# patterns, so they get a tighter per-part cap than free text; that keeps one
# huge ``old_string``/``content`` argument from using up the whole budget.
_QUERY_PART_CHARS = 600
_QUERY_ARG_CHARS = 240
_QUERY_MAX_DEPTH = 8

# ``data:<mime>[;param]*,<payload>`` -- the payload charset is base64 plus
# percent-encoding, so text that follows the URI is not swallowed. Every part is
# printable ASCII and length-bounded: unbounded, one ``data:a/b;`` with no comma
# scans to the end of a whitespace-free run and a text with many of them costs
# O(n^2); and ``_clean_query_text`` relies on a URI being a single run of
# printable ASCII.
_DATA_URI_RE = re.compile(
    r"(?<![\w])data:[A-Za-z0-9_.+-]{1,127}/[A-Za-z0-9_.+-]{1,127}"
    r"(?:;[\x21-\x2b\x2d-\x3a\x3c-\x7e]{0,255}){0,16},"
    r"[A-Za-z0-9+/=%_.~-]*"
)
# A run of >=200 printable ASCII chars without whitespace is a blob (base64,
# hex, minified data), never a task description. ASCII-only on purpose:
# CJK prose has no spaces, so a long Japanese paragraph must survive.
_LONG_TOKEN_RE = re.compile(r"[\x21-\x7e]{200,}")


def text_of(content: Any) -> str:
    """Plain text of a message ``content`` field, whatever its shape.

    - ``str`` -> returned as is.
    - ``list`` (multimodal parts) -> the text of parts whose ``type`` is
      ``text``/``input_text``/``output_text``, plus bare ``str`` parts, joined
      with ``"\\n"``. Image, base64 and every other part type is skipped.
    - ``dict`` with a ``str`` ``"text"`` -> that text.
    - anything else (``None``, numbers, ...) -> ``""``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return text_of(msg.get("content"))
    return ""


# Smallest prefix ``_clean_query_text`` looks at when it is given a ``limit``.
_CLEAN_WINDOW_MIN = 4096


def _clean_query_text(text: str, limit: int | None = None) -> str:
    """Drop ``data:`` URIs and base64-like blobs (>=200 printable non-space chars).

    ``limit`` says how many leading characters of the result the caller will
    use. ``task_query`` runs once per tool result against the same (possibly
    huge) user messages, so cleaning them in full each time would cost O(size)
    per call. With a ``limit`` only a prefix is cleaned, and it grows (x4) until
    it yields ``limit`` characters or covers the whole text. The cut can split a
    blob or a word, so the trailing run of printable characters is dropped from
    a truncated result; what is returned is always a prefix of the full clean.
    """
    window = len(text) if limit is None else max(4 * limit, _CLEAN_WINDOW_MIN)
    while True:
        cleaned = _LONG_TOKEN_RE.sub(" ", _DATA_URI_RE.sub(" ", text[:window]))
        if window >= len(text):
            return cleaned.strip()
        end = len(cleaned)
        while end and "!" <= cleaned[end - 1] <= "~":
            end -= 1
        cleaned = cleaned[:end].strip()
        if len(cleaned) >= (limit or 0):
            return cleaned
        window *= 4


def _string_leaves(value: Any, depth: int = 0) -> list[str]:
    """String leaf values of a JSON-like structure, in document order."""
    if isinstance(value, str):
        return [value]
    if depth >= _QUERY_MAX_DEPTH:
        return []
    out: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            out.extend(_string_leaves(child, depth + 1))
    elif isinstance(value, (list, tuple)):
        for child in value:
            out.extend(_string_leaves(child, depth + 1))
    return out


def _call_name_and_args(call: Any) -> tuple[str, Any]:
    """(function name, arguments) of an OpenAI-style or flat tool call."""
    if not isinstance(call, dict):
        return "", None
    fn = call.get("function")
    if not isinstance(fn, dict):
        fn = {}
    name = fn.get("name") or call.get("name") or ""
    args = fn.get("arguments", call.get("arguments"))
    return (name if isinstance(name, str) else ""), args


def _arg_leaves(args: Any) -> list[str]:
    """String leaves of tool-call arguments given as a JSON string or a dict."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (ValueError, RecursionError):
            return [args]
        return _string_leaves(parsed)
    return _string_leaves(args)


def _id_variants(*values: Any) -> frozenset[str]:
    """Every spelling of one tool-call id.

    Responses-style transports (Hermes' Codex path) give a call an ``id``, a
    ``call_id`` and a ``response_item_id`` that differ, and the ``role: tool``
    message answers with ``call_id`` or a composite ``call_id|response_item_id``.
    All of those name ONE call, so ids are compared as sets of variants.
    """
    variants: set[str] = set()
    for raw in values:
        if raw is None or isinstance(raw, (dict, list, bool)):
            continue
        text = str(raw).strip()
        if text:
            variants.add(text)
            variants.update(part for part in (p.strip() for p in text.split("|")) if part)
    return frozenset(variants)


def _call_id_variants(call: Any) -> frozenset[str]:
    if not isinstance(call, dict):
        return frozenset()
    return _id_variants(call.get("id"), call.get("call_id"), call.get("response_item_id"))


def _find_tool_call(messages: list[dict[str, Any]], idx: int, call_id: str) -> Any:
    """The assistant tool_call that ``call_id`` answers, nearest before ``idx`` first."""
    wanted = _id_variants(call_id)
    order = chain(
        range(min(idx, len(messages)) - 1, -1, -1), range(max(idx, 0), len(messages))
    )
    for i in order:
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if wanted & _call_id_variants(call):
                return call
    return None


def task_query(
    messages: list[dict[str, Any]],
    idx: int,
    *,
    focus: str | None = None,
    max_chars: int = 2000,
    skip_user: Callable[[dict[str, Any]], bool] | None = None,
) -> list[str]:
    """Query parts describing what the agent is doing around ``messages[idx]``.

    ``messages[idx]`` is normally a tool result about to be compressed. A last
    user message such as "continue" says nothing about it, so the query is built
    from the surrounding task. Parts, in priority order:

    1. ``focus`` (an explicit topic, e.g. a ``/compress <topic>`` argument);
    2. the function name and string argument values of the assistant tool call
       that produced this result (``messages[idx]["tool_call_id"]``, matched
       against the call's ``id``/``call_id``/``response_item_id`` and either half
       of a ``call_id|response_item_id`` composite);
    3. the first assistant message after ``idx``: its text and the string
       arguments of its tool calls (what the agent did with the result);
    4. the last user message with non-empty text, scanning backwards
       (``skip_user(msg)`` true skips a message, e.g. synthetic injections);
    5. the first such user message (the original task).

    Returns an ordered list: parts are de-duplicated, empty ones dropped, each
    capped, and ``data:`` URIs and base64-like tokens (runs of >=200 printable
    ASCII characters without whitespace; CJK prose is kept) removed. The list never exceeds ``max_chars`` characters in
    total (counting one ``"\\n"`` between parts); when the budget runs out,
    lower-priority parts are cut or dropped first. Join with ``"\\n"`` or
    ``" "`` to get a query string.
    """
    if max_chars <= 0 or not messages:
        return []

    parts: list[str] = []
    seen: set[str] = set()
    used = 0
    full = False

    def add(raw: Any, cap: int = _QUERY_PART_CHARS) -> None:
        nonlocal used, full
        if full or not isinstance(raw, str):
            return
        text = _clean_query_text(raw, cap)[:cap]
        if not text:
            return
        room = max_chars - used - (1 if parts else 0)
        if room <= 0:
            full = True
            return
        if len(text) > room:
            text = text[:room]
            full = True
        text = text.rstrip()
        # De-duplicate what is actually emitted (after the cap and the budget
        # cut), so two long parts that differ only past the cap are not both kept.
        key = " ".join(text.split())
        if not key or key in seen:
            return
        seen.add(key)
        parts.append(text)
        used += len(text) + (1 if len(parts) > 1 else 0)

    if isinstance(focus, str):
        add(focus)
    elif focus:
        add(text_of(focus))

    msg = messages[idx] if isinstance(idx, int) and 0 <= idx < len(messages) else None
    call_id = str((msg or {}).get("tool_call_id") or "") if isinstance(msg, dict) else ""
    if call_id:
        name, args = _call_name_and_args(_find_tool_call(messages, idx, call_id))
        add(name, _QUERY_ARG_CHARS)
        for leaf in _arg_leaves(args):
            add(leaf, _QUERY_ARG_CHARS)

    if isinstance(idx, int):
        for i in range(max(idx + 1, 0), len(messages)):
            follow = messages[i]
            if isinstance(follow, dict) and follow.get("role") == "assistant":
                add(text_of(follow.get("content")))
                for call in follow.get("tool_calls") or []:
                    for leaf in _arg_leaves(_call_name_and_args(call)[1]):
                        add(leaf, _QUERY_ARG_CHARS)
                break

    def real_user_text(m: Any) -> str:
        """Text of a user message that counts (non-blank, not skipped), else ``""``."""
        if not isinstance(m, dict) or m.get("role") != "user":
            return ""
        text = text_of(m.get("content"))
        if not text or text.isspace() or (skip_user is not None and skip_user(m)):
            return ""
        return text

    # Scan lazily from each end: ``skip_user`` runs only until a message is
    # accepted, not once per user message on every call.
    last_user = next((t for t in map(real_user_text, reversed(messages)) if t), "")
    add(last_user)
    add(next((t for t in map(real_user_text, messages) if t), ""))
    return parts


def _requests_full_preservation(query: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", (query or "").casefold()))
    if not words.intersection(_PRESERVATION_VERBS):
        return False
    if "everything" in words:
        return True
    return bool(
        words.intersection(_PRESERVATION_QUANTIFIERS)
        and words.intersection(_PRESERVATION_OBJECTS)
    )


def _distinctive_evidence_tokens(text: str) -> set[str]:
    """Return exact anchors strong enough to prove bulky evidence survived."""
    candidates = (
        _SUMMARY_STRUCTURED_TOKEN_RE.findall(text)
        + _extract_entities(text)
        + _extract_terms(text)
    )
    anchors: set[str] = set()
    for candidate in candidates:
        token = str(candidate).strip().casefold()
        if not token or token in GENERIC_WORDS:
            continue
        if (
            len(token) >= 8
            or any(character.isdigit() for character in token)
            or any(separator in token for separator in "._:/-")
        ):
            anchors.add(token)
    return anchors


def _json_query_answer_requirements(
    content: str, query: str
) -> tuple[set[str], set[tuple[str, bool]]]:
    """Return text answers and typed boolean requirements from matching JSON."""
    parsed_values: list[Any] = []

    def parse_payload(payload: str) -> None:
        normalised = preprocess_json(payload, query)
        try:
            parsed_values.append(json.loads(normalised))
        except (ValueError, RecursionError):  # JSONDecodeError + the 4300-digit int limit
            return

    parse_payload(content)
    if not parsed_values:
        for _start, _end, payload in _detect_json_payloads(content):
            parse_payload(payload)
    if not parsed_values:
        return set(), set()

    needles = [needle.casefold() for needle in _json_query_needles(query)]
    selectors = _selector_patterns(needles)
    query_words = set(re.findall(r"[a-z0-9]+", (query or "").casefold()))
    answers: set[str] = set()
    booleans: set[tuple[str, bool]] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(value, dict):
            try:
                record = json.dumps(value, ensure_ascii=False).casefold()
            except (TypeError, ValueError, RecursionError):
                record = ""
            if selectors and any(selector.search(record) for selector in selectors):
                for key, scalar in value.items():
                    key_words = set(re.findall(r"[a-z0-9]+", str(key).casefold()))
                    if not key_words.intersection(query_words):
                        continue
                    if isinstance(scalar, bool):
                        field = str(key).strip().casefold()
                        if field:
                            booleans.add((field, scalar))
                        continue
                    if not isinstance(scalar, (str, int, float)):
                        continue
                    answer = str(scalar).strip()
                    if answer and answer.casefold() not in (query or "").casefold():
                        answers.add(answer)
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    for parsed in parsed_values:
        visit(parsed)
    return answers, booleans


def _json_query_answers(content: str, query: str) -> set[str]:
    """Return scalar values under queried keys in query-matching JSON records."""
    answers, booleans = _json_query_answer_requirements(content, query)
    return answers | {str(value).casefold() for _field, value in booleans}


def _boolean_requirement_satisfied(field: str, expected: bool, post: str) -> bool:
    """Recognise explicit and natural-language boolean answers for one field."""
    words = re.findall(r"[a-z0-9]+", field.casefold())
    if not words:
        return False
    field_pattern = r"[\s_-]+".join(re.escape(word) for word in words)
    field_ref = rf"(?<![a-z0-9_]){field_pattern}(?![a-z0-9_])"
    explicit = re.search(
        rf"{field_ref}\s*(?:(?:is|equals?)\s+|[=:]\s*)?"
        r"(true|yes|on|false|no|off)\b",
        post,
        re.I,
    )
    if explicit:
        return (explicit.group(1).casefold() in {"true", "yes", "on"}) is expected

    negated = bool(
        re.search(
            rf"\b(?:is\s+)?(?:not|never)\s+(?:currently\s+)?{field_ref}",
            post,
            re.I,
        )
    )
    positive = bool(re.search(field_ref, post, re.I)) and not negated

    field_key = "_".join(words)
    if field_key == "enabled":
        not_disabled = bool(re.search(r"\bnot\s+disabled\b", post, re.I))
        disabled = bool(re.search(r"\bdisabled\b", post, re.I)) and not not_disabled
        positive = positive or not_disabled
        negated = negated or disabled
    elif field_key == "active":
        not_inactive = bool(re.search(r"\bnot\s+inactive\b", post, re.I))
        inactive = bool(re.search(r"\binactive\b", post, re.I)) and not not_inactive
        positive = positive or not_inactive
        negated = negated or inactive

    return positive if expected else negated


def _pending_tool_calls(messages: list[dict[str, Any]]) -> set[str]:
    """Tool-call ids of the trailing assistant turn that no tool message answers.

    Only the trailing turn counts: the LAST assistant message, if it carries
    ``tool_calls``, plus the ``tool`` messages after it. An orphan call earlier
    in the history (a result lost to a crash, an interrupted turn, a history
    rewrite) is not "in flight" once the agent has produced a later assistant
    message, so it must not block ``trajectory_gate`` for the rest of the
    session. A trailing assistant message without ``tool_calls`` leaves nothing
    pending.

    A call is answered when any spelling of its id (``id``, ``call_id``,
    ``response_item_id``, or either half of a ``call_id|response_item_id``
    composite) matches the ``tool_call_id`` of a later tool message. The
    returned set holds each pending call's ``id`` (``call_id`` when it has none).
    """
    last = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last = i
            break
    if last < 0:
        return set()
    calls = [
        call
        for call in messages[last].get("tool_calls") or []
        if _call_id_variants(call)
    ]
    if not calls:
        return set()
    answered: set[str] = set()
    for m in messages[last + 1 :]:
        if m.get("role") == "tool":
            answered |= _id_variants(m.get("tool_call_id"))
    return {
        str(call.get("id") or call.get("call_id") or call.get("response_item_id"))
        for call in calls
        if not _call_id_variants(call) & answered
    }


def _call_signature(msg: dict[str, Any]) -> list[tuple[str, str]]:
    """Deterministic (name, canonical-args) pairs for an assistant message."""
    sigs: list[tuple[str, str]] = []
    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = str(fn.get("name") or call.get("name") or "")
        args = fn.get("arguments", call.get("arguments"))
        try:
            canonical = (
                args
                if isinstance(args, str)
                else json.dumps(args, sort_keys=True, ensure_ascii=False)
            )
        except (TypeError, ValueError, RecursionError):
            canonical = repr(args)
        sigs.append((name, str(canonical)))
    return sigs


def trajectory_gate(
    messages: list[dict[str, Any]], *, stuck_run: int = 3
) -> tuple[bool, str]:
    """SelfCompact-style timing rubric, deterministic version.

    Compaction is suppressed when:
    - ``pending-tool-calls``: the trailing assistant turn requested tools that
      have no result yet — the trajectory is mid-derivation and pruning could
      evict context the in-flight call needs (SelfCompact C1 closed-unit).
      Orphan calls earlier in the history do not count.
    - ``stuck-loop``: the last ``stuck_run`` assistant turns issued identical
      tool calls — the agent is looping; the correct move is to diagnose,
      not to erase evidence of the loop (SelfCompact N1 not-stuck).

    Returns ``(allowed, reason)``. The gate only ever suppresses — the
    fail-safe direction — so it is safe to leave enabled by default.
    """
    if not messages:
        return True, "ok"
    if _pending_tool_calls(messages):
        return False, "pending-tool-calls"
    assistant = [m for m in messages if m.get("role") == "assistant"]
    if len(assistant) >= stuck_run:
        tail = assistant[-stuck_run:]
        sigs = [_call_signature(m) for m in tail]
        if all(s and s == sigs[0] for s in sigs):
            return False, "stuck-loop"
    return True, "ok"


def apply_extractive_tool_prune(
    messages: list[dict[str, Any]],
    query: str | None = None,
    *,
    min_chars: int = MIN_TOOL_CHARS,
    protect_last_tool: int = PROTECT_LAST_TOOL,
    timing_gate: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Compress old bulky tool payloads. Returns (messages, n_changed).

    If nothing changes — or ``timing_gate`` suppresses the pass — returns
    the same list object.
    """
    if not messages:
        return messages, 0
    if timing_gate:
        allowed, _reason = trajectory_gate(messages)
        if not allowed:
            return messages, 0
    q = query if query is not None else last_user_text(messages)
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    skip = set(tool_idxs[-protect_last_tool:]) if protect_last_tool else set()
    out: list[dict[str, Any]] | None = None
    changed = 0
    for i, msg in enumerate(messages):
        if i in skip or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < min_chars:
            continue
        # Live tool payloads can contain credentials or other secrets. This
        # adapter has no retrieval path, so persisting originals in CCR only
        # adds exposure, unbounded disk retention, and a semantic marker.
        result = compress_context(content, q, ccr=False, citations=False)
        new = result.compressed_text
        if result.fail_open or new == content or len(new) >= len(content):
            continue
        if out is None:
            out = [dict(m) for m in messages]
        out[i] = {**msg, "content": new}
        changed += 1
    return (out if out is not None else messages), changed


def query_facts_lost(before: list[dict[str, Any]], after: list[dict[str, Any]], query: str) -> bool:
    """True when distinctive query facts survived prune but not the summarizer.

    .. deprecated:: 1.4.0
        Kept for compatibility, with no runtime warning and no behaviour
        change. It re-scans every tool payload and the whole transcript (O(n^2)
        on long sessions) and the Hermes plugin no longer calls it: loss is
        bounded per result at demotion time instead of being re-checked after
        the summary. Do not use it on long transcripts.
    """
    raw_tool_evidence = [
        str(message.get("content") or "")
        for message in before
        if message.get("role") == "tool"
    ]
    normalised_tool_evidence = [
        preprocess_json(evidence, query or "") for evidence in raw_tool_evidence
    ]
    requirements_by_evidence = [
        _json_query_answer_requirements(evidence, query or "")
        for evidence in raw_tool_evidence
    ]
    tool_evidence = [
        (
            f"{query}\n" + "\n".join(sorted(answers))
            if answers
            else "" if booleans else normalised
        )
        for normalised, (answers, booleans) in zip(
            normalised_tool_evidence, requirements_by_evidence
        )
    ]
    pre = "\n".join(tool_evidence)
    post = "\n".join(
        str(message.get("content") or "")
        for message in after
        if message.get("role") != "user"
    )
    json_answers = {
        answer
        for answers, _booleans in requirements_by_evidence
        for answer in answers
    }
    boolean_requirements = {
        requirement
        for _answers, booleans in requirements_by_evidence
        for requirement in booleans
    }
    post_fold = post.casefold()
    if any(answer.casefold() not in post_fold for answer in json_answers):
        return True
    if any(
        not _boolean_requirement_satisfied(field, expected, post)
        for field, expected in boolean_requirements
    ):
        return True
    if _requests_full_preservation(query):
        return any(
            evidence.strip() and evidence.strip() not in post
            for evidence in normalised_tool_evidence
        )
    if not pre.strip():
        return False
    valid, _recall = _summary_preserves_required_facts(pre, post, query or "")
    return not valid


def bulky_tools_dropped(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    """True when no distinctive anchor from any bulky tool survives.

    .. deprecated:: 1.4.0
        Kept for compatibility, with no runtime warning and no behaviour
        change. It recomputes anchor sets for every bulky payload (O(n^2) on
        long sessions) and the Hermes plugin no longer calls it.
    """
    pre = [
        str(m.get("content") or "")
        for m in before
        if m.get("role") == "tool"
        and isinstance(m.get("content"), str)
        and len(m["content"]) >= MIN_TOOL_CHARS
    ]
    if not pre:
        return False
    post = "\n".join(
        str(message.get("content") or "")
        for message in after
        if message.get("role") != "user"
    )
    post_anchors = _distinctive_evidence_tokens(post)
    for evidence in pre:
        evidence_anchors = _distinctive_evidence_tokens(evidence)
        if not evidence_anchors or evidence_anchors.isdisjoint(post_anchors):
            return True
    return False
