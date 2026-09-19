"""Hermes-facing extractive prune — no Hermes imports.

Turns bulky old tool-result payloads into extractive keep/drop using the
same compress_context contract. Safe to unit-test from the skill.
"""
from __future__ import annotations

import json
import re
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


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


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
        except (json.JSONDecodeError, RecursionError):
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


def apply_extractive_tool_prune(
    messages: list[dict[str, Any]],
    query: str | None = None,
    *,
    min_chars: int = MIN_TOOL_CHARS,
    protect_last_tool: int = PROTECT_LAST_TOOL,
) -> tuple[list[dict[str, Any]], int]:
    """Compress old bulky tool payloads. Returns (messages, n_changed).

    If nothing changes, returns the same list object.
    """
    if not messages:
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
    """True when distinctive query facts survived prune but not the summarizer."""
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
    """True when no distinctive anchor from any bulky tool survives."""
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
