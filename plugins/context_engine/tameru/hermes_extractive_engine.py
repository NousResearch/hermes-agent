"""Hermes-facing extractive prune — no Hermes imports.

Turns bulky old tool-result payloads into extractive keep/drop using the
same compress_context contract. Safe to unit-test from the skill.
"""
from __future__ import annotations

from typing import Any

from .compress_context import compress_context

MIN_TOOL_CHARS = 800
PROTECT_LAST_TOOL = 2


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


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
    from .contract_gates import distinctive_query_terms

    def blob(msgs: list[dict[str, Any]]) -> str:
        return " ".join(
            str(m.get("content") or "")
            for m in msgs
            if m.get("role") == "tool"
        )

    pre = blob(before)
    post = blob(after)
    needles = list(distinctive_query_terms(query or ""))
    for tok in str(query or "").replace("/", " ").split():
        t = tok.strip("?.!,")
        if len(t) >= 4 and t.casefold() not in {n.casefold() for n in needles}:
            needles.append(t)
    hits = [n for n in needles if n and n.casefold() in pre.casefold()]
    if not hits:
        return False
    return any(n.casefold() not in post.casefold() for n in hits)


def bulky_tools_dropped(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    """True when a bulky tool payload vanished from the compressed transcript."""
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
        str(m.get("content") or "") for m in after if m.get("role") == "tool"
    )
    if not post.strip():
        return True
    return any(blob[:120] not in post for blob in pre)
