"""Deterministic extractive residuals for Catalog / Hybrid compaction.

Catalog replaces the auxiliary LLM summary with a redacted, bounded index of
handles already present in the compacted middle. Hybrid appends a smaller
unique-handle index after the existing Standard summary. Neither path invents
facts, writes MemoryStore, or adds a new persistence surface — archived
source stays on SessionDB compacted=1 rows and session_search.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from agent.redact import redact_sensitive_text
from agent.user_turn import is_catalog_user_turn

COMPRESSION_MODES = ("standard", "catalog", "hybrid")
DEFAULT_COMPRESSION_MODE = "standard"

CATALOG_HEADING = "## Catalog Residual"
HYBRID_INDEX_HEADING = "## Unique handles"
LEAN_ANCHOR_HEADING = "## Anchor Index (mechanically extracted, exact)"
CATALOG_FILES_HEADING = "## Files"
CATALOG_TOOLS_HEADING = "## Tools"
CATALOG_IDENTIFIERS_HEADING = "## Identifiers"
CATALOG_TOPICS_HEADING = "## Topics"
CATALOG_STORY_HEADING = "## Story"
CATALOG_RECEIPT_HEADING = "## Receipt"

# Body budget excludes SUMMARY_PREFIX (wrapped by the compressor).
CATALOG_BUDGET_CHARS = 4_000
HYBRID_INDEX_BUDGET_CHARS = 1_200
_STORY_LIMIT = 3
_STEM_LIMIT = 8
_ITEM_MAX_CHARS = 160
_STORY_MAX_CHARS = 200

_PATH_RE = re.compile(r"(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}<>]+")
_FILE_TOKEN_RE = re.compile(
    r"\b[\w./-]+/[\w.-]+\.(?:py|ts|tsx|js|jsx|rs|go|md|yaml|yml|json|toml|sh|c|h|cpp)\b"
)
_URL_RE = re.compile(r"https?://[^\s)\"'<>]{6,160}")
_ISSUE_RE = re.compile(r"(?:^|[\s(])(#[1-9]\d{1,6})\b")
# Require at least one digit so hex-only English words ("acceded", "defaced")
# are not admitted as commit identifiers.
_SHA_RE = re.compile(r"\b(?=[0-9a-f]{7,40}\b)[0-9a-f]*\d[0-9a-f]*\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
_TICKET_RE = re.compile(r"\b[A-Z]{2,8}-\d{1,7}\b")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_][A-Za-z0-9_-]{1,30}\b")
_SENTENCE_RE = re.compile(r"(?s).+?(?:[.!?](?:\s|$)|$)")

_PATH_ARG_KEYS = frozenset({
    "path", "paths", "file", "file_path", "filepath", "filename",
    "workdir", "cwd", "output_path", "target", "dest", "destination",
})
_NOISE_IDENTIFIERS = frozenset({
    "true", "false", "null", "none", "success", "error",
})


def normalize_compression_mode(value: Any) -> str:
    """Return a canonical compaction mode; invalid values fall back to standard."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in COMPRESSION_MODES:
            return mode
    return DEFAULT_COMPRESSION_MODE


def _redact(text: Any) -> str:
    return redact_sensitive_text(
        text or "",
        force=True,
        redact_url_credentials=True,
    )


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _compact_text(text: str, limit: int) -> str:
    text = _redact(re.sub(r"\s+", " ", text).strip())
    if len(text) > limit:
        text = text[: limit - 15].rstrip() + " ...[truncated]"
    return text


def _tool_name_and_args(tool_call: Any) -> tuple[str, str]:
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        return str(fn.get("name") or "unknown"), str(fn.get("arguments") or "")
    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "unknown", ""
    return (
        str(getattr(fn, "name", None) or "unknown"),
        str(getattr(fn, "arguments", None) or ""),
    )


def _remember(store: dict[str, str], key: str, value: str) -> None:
    """Last-wins by recency: a later duplicate key moves to the newest slot."""
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return
    # dict assignment keeps the original insertion index; pop+reinsert so a
    # tight last-N window or budget prefers the newest conflicting value.
    store.pop(key, None)
    store[key] = value


def _collect_paths_from_jsonish(obj: Any, files: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _PATH_ARG_KEYS and isinstance(val, str) and val.strip():
                cleaned = val.strip().rstrip(".,:;")
                _remember(files, cleaned.lower(), cleaned)
            elif key in _PATH_ARG_KEYS and isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        cleaned = item.strip().rstrip(".,:;")
                        _remember(files, cleaned.lower(), cleaned)
            _collect_paths_from_jsonish(val, files)
    elif isinstance(obj, list):
        for val in obj:
            _collect_paths_from_jsonish(val, files)
    elif isinstance(obj, str):
        _collect_path_mentions(obj, files)


def _collect_path_mentions(text: str, files: dict[str, str]) -> None:
    for match in _PATH_RE.findall(text):
        cleaned = match.rstrip(".,:;")
        if cleaned:
            _remember(files, cleaned.lower(), cleaned)
    for match in _FILE_TOKEN_RE.findall(text):
        cleaned = match.rstrip(".,:;")
        if cleaned:
            _remember(files, cleaned.lower(), cleaned)


def _collect_identifiers(text: str, identifiers: dict[str, str]) -> None:
    for pattern in (_URL_RE, _UUID_RE, _TICKET_RE, _HANDLE_RE):
        for match in pattern.findall(text):
            value = match.strip().rstrip(".,:;")
            if value.lower() in _NOISE_IDENTIFIERS:
                continue
            _remember(identifiers, value.lower(), value)
    for match in _ISSUE_RE.findall(text):
        _remember(identifiers, match.lower(), match)
    for match in _SHA_RE.findall(text):
        if len(match) >= 7:
            _remember(identifiers, match.lower(), match)


def _first_stem(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    sentence = _SENTENCE_RE.match(text)
    stem = (sentence.group(0) if sentence else text).strip()
    return _compact_text(stem, _ITEM_MAX_CHARS)


def _is_usable_user_turn(message: dict[str, Any]) -> bool:
    """True for a human ask — compression real-user plus catalog prefixes."""
    return is_catalog_user_turn(message)


def extract_catalog_items(
    messages: Iterable[dict[str, Any]],
    previous_residual: str = "",
) -> dict[str, dict[str, str]]:
    """Extract last-wins catalog maps from prior residual + message order."""
    files: dict[str, str] = {}
    tools: dict[str, str] = {}
    identifiers: dict[str, str] = {}
    topics: dict[str, str] = {}
    story: dict[str, str] = {}

    if previous_residual:
        _reingest_prior_residual(
            previous_residual, files, tools, identifiers, topics, story,
        )

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        text = _redact(_content_text(msg.get("content")))
        _collect_path_mentions(text, files)
        _collect_identifiers(text, identifiers)

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls") or []:
                name, raw_args = _tool_name_and_args(tc)
                name = name.strip()
                if name and name != "unknown":
                    _remember(tools, name.lower(), name)
                args = _redact(raw_args)
                if args:
                    try:
                        parsed = json.loads(args)
                    except Exception:
                        parsed = args
                    _collect_paths_from_jsonish(parsed, files)
                    if isinstance(parsed, str):
                        _collect_identifiers(parsed, identifiers)
                    else:
                        _collect_identifiers(json.dumps(parsed, ensure_ascii=False), identifiers)

        if role == "tool":
            name = str(msg.get("name") or msg.get("tool_name") or "").strip()
            if name:
                _remember(tools, name.lower(), name)

        if role == "user" and _is_usable_user_turn(msg) and text:
            stem = _first_stem(text)
            if stem:
                _remember(topics, stem.lower(), stem)
                _remember(story, stem.lower(), _compact_text(text, _STORY_MAX_CHARS))
        elif role == "assistant" and text and not msg.get("tool_calls"):
            stem = _first_stem(text)
            if stem:
                _remember(topics, stem.lower(), stem)

    # Story keeps only the newest few unique asks (dict is last-wins ordered).
    if len(story) > _STORY_LIMIT:
        keep_keys = list(story.keys())[-_STORY_LIMIT:]
        story = {key: story[key] for key in keep_keys}
    if len(topics) > _STEM_LIMIT:
        keep_keys = list(topics.keys())[-_STEM_LIMIT:]
        topics = {key: topics[key] for key in keep_keys}

    return {
        "files": files,
        "tools": tools,
        "identifiers": identifiers,
        "topics": topics,
        "story": story,
    }


_SECTION_HEADINGS = {
    "files": CATALOG_FILES_HEADING,
    "tools": CATALOG_TOOLS_HEADING,
    "identifiers": CATALOG_IDENTIFIERS_HEADING,
    "topics": CATALOG_TOPICS_HEADING,
    "story": CATALOG_STORY_HEADING,
}


def _reingest_prior_residual(
    residual: str,
    files: dict[str, str],
    tools: dict[str, str],
    identifiers: dict[str, str],
    topics: dict[str, str],
    story: dict[str, str],
) -> None:
    """Parse a prior catalog (or hybrid index) and seed last-wins maps.

    Prior items are older than the current middle, so subsequent extracts
    overwrite them. Reingest is extractive: only listed bullets / handles.
    """
    text = _redact(residual)
    current: str | None = None
    stores = {
        "files": files,
        "tools": tools,
        "identifiers": identifiers,
        "topics": topics,
        "story": story,
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current = None
            for key, heading in _SECTION_HEADINGS.items():
                if line.startswith(heading):
                    current = key
                    break
            if line.startswith(HYBRID_INDEX_HEADING) or line.startswith(
                LEAN_ANCHOR_HEADING
            ):
                # Unique handles and the lean Anchor Index share csv handle
                # lines (files:/tools:/ids:). Parse both before the public
                # Unique section is stripped on Hybrid+lean rebuilds.
                current = "hybrid"
            continue
        if current == "hybrid":
            _reingest_hybrid_line(line, files, tools, identifiers)
            continue
        if current is None or not line.startswith("- "):
            continue
        value = line[2:].strip()
        if not value:
            continue
        _remember(stores[current], value.lower(), value)


def _reingest_hybrid_line(
    line: str,
    files: dict[str, str],
    tools: dict[str, str],
    identifiers: dict[str, str],
) -> None:
    lowered = line.lower()
    payload = line.split(":", 1)[1] if ":" in line else ""
    items = [_strip_anchor_freq(item) for item in payload.split(",") if item.strip()]
    items = [item for item in items if item]
    if lowered.startswith("files:"):
        for item in items:
            _remember(files, item.lower(), item)
    elif lowered.startswith("tools:"):
        for item in items:
            _remember(tools, item.lower(), item)
    elif (
        lowered.startswith("ids:")
        or lowered.startswith("identifiers:")
        or lowered.startswith("urls:")
        or lowered.startswith("handles:")
        or lowered.startswith("commits:")
        or lowered.startswith("prs/")
        or lowered.startswith("branches:")
    ):
        for item in items:
            _remember(identifiers, item.lower(), item)


def _strip_anchor_freq(item: str) -> str:
    """Drop lean Anchor Index frequency suffixes like ``(x2)``."""
    return re.sub(r"\(x\d+\)$", "", item.strip()).strip()


def merge_handles_into_anchor_index(
    summary: str,
    items: dict[str, dict[str, str]],
) -> str:
    """Fold last-wins file handles into the lean Anchor Index.

    Hybrid+lean strips the public Unique handles section so only one index
    representation remains. First-window paths must still survive in the
    Anchor Index across repeated compactions.
    """
    if not summary or LEAN_ANCHOR_HEADING not in summary:
        return summary or ""
    files = [value for value in (items.get("files") or {}).values() if value]
    if not files:
        return summary

    lines = summary.splitlines()
    in_anchor = False
    files_line_idx: int | None = None
    insert_after: int | None = None
    existing: set[str] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            in_anchor = stripped.startswith(LEAN_ANCHOR_HEADING)
            if in_anchor:
                insert_after = idx
            continue
        if not in_anchor:
            continue
        if stripped.lower().startswith("files:"):
            files_line_idx = idx
            payload = stripped.split(":", 1)[1]
            for item in payload.split(","):
                cleaned = _strip_anchor_freq(item)
                if cleaned:
                    existing.add(cleaned.lower())

    missing = [value for value in files if value.lower() not in existing]
    if not missing:
        return summary
    addition = ", ".join(missing)
    if files_line_idx is not None:
        lines[files_line_idx] = lines[files_line_idx].rstrip() + ", " + addition
    elif insert_after is not None:
        lines.insert(insert_after + 1, f"files: {addition}")
    return "\n".join(lines)


def _pack_section(
    heading: str,
    values: Iterable[str],
    *,
    budget: int,
    used: int,
    kept: list[str],
    dropped: list[str],
) -> tuple[str, int]:
    ordered = [value for value in values if value]
    selected: list[str] = []
    skipped: list[str] = []
    section_used = len(heading) + 1
    # Newest first so a tight cap keeps the latest last-wins handles.
    for value in reversed(ordered):
        line = f"- {value}"
        cost = len(line) + 1
        if used + section_used + cost > budget:
            skipped.append(value)
            continue
        selected.append(value)
        section_used += cost
    if not selected:
        dropped.extend(reversed(skipped))
        return "", used
    selected.reverse()
    skipped.reverse()
    kept.extend(selected)
    dropped.extend(skipped)
    return (
        "\n".join([heading] + [f"- {value}" for value in selected]),
        used + section_used,
    )


def build_catalog_residual(
    messages: list[dict[str, Any]],
    *,
    previous_residual: str = "",
    budget: int = CATALOG_BUDGET_CHARS,
    task_snapshot: str = "",
) -> str:
    """Build a prefixed-ready catalog body under a hard character budget."""
    items = extract_catalog_items(messages, previous_residual=previous_residual)
    reserved_receipt = 96
    pack_budget = max(240, budget - reserved_receipt)
    used = 0
    kept: list[str] = []
    dropped: list[str] = []
    sections: list[str] = [CATALOG_HEADING]
    used += len(CATALOG_HEADING) + 1

    if task_snapshot:
        # Keep the Standard historical snapshot / no-user sentinel intact.
        # Do not clip provenance semantics to the 200-char story-item cap.
        snapshot = _redact(str(task_snapshot).strip())
        block = f"## Historical Task Snapshot\n{snapshot}"
        sections.append(block)
        used += len(block) + 1

    packed, used = _pack_section(
        CATALOG_FILES_HEADING, items["files"].values(),
        budget=pack_budget, used=used, kept=kept, dropped=dropped,
    )
    if packed:
        sections.append(packed)
    packed, used = _pack_section(
        CATALOG_TOOLS_HEADING, items["tools"].values(),
        budget=pack_budget, used=used, kept=kept, dropped=dropped,
    )
    if packed:
        sections.append(packed)
    packed, used = _pack_section(
        CATALOG_IDENTIFIERS_HEADING, items["identifiers"].values(),
        budget=pack_budget, used=used, kept=kept, dropped=dropped,
    )
    if packed:
        sections.append(packed)
    packed, used = _pack_section(
        CATALOG_TOPICS_HEADING, items["topics"].values(),
        budget=pack_budget, used=used, kept=kept, dropped=dropped,
    )
    if packed:
        sections.append(packed)
    packed, used = _pack_section(
        CATALOG_STORY_HEADING, items["story"].values(),
        budget=pack_budget, used=used, kept=kept, dropped=dropped,
    )
    if packed:
        sections.append(packed)

    receipt = (
        f"{CATALOG_RECEIPT_HEADING}\n"
        f"kept: {len(kept)} items ({used} chars)\n"
        f"dropped: {len(dropped)} items"
        + (" (over budget)" if dropped else "")
    )
    sections.append(receipt)
    body = "\n\n".join(sections)
    body = _redact(body)
    if len(body) > budget:
        # Receipt is packed last and must survive a mid-body clip.
        marker = "\n...[catalog truncated]...\n"
        prefix_budget = max(0, budget - len(receipt) - len(marker))
        head = body[:prefix_budget].rstrip()
        body = head + marker + receipt
        if len(body) > budget:
            prefix = head + marker
            body = prefix[: max(0, budget - len(receipt))] + receipt
    return body


def build_hybrid_handle_index(
    messages: list[dict[str, Any]],
    *,
    previous_residual: str = "",
    budget: int = HYBRID_INDEX_BUDGET_CHARS,
) -> str:
    """Compact unique-handle index appended after a Standard summary."""
    items = extract_catalog_items(messages, previous_residual=previous_residual)
    lines = [HYBRID_INDEX_HEADING]
    dropped = 0

    def _csv(label: str, values: Iterable[str]) -> None:
        nonlocal dropped
        values = [value for value in values if value]
        if not values:
            return
        newest_first = list(reversed(values))
        while newest_first:
            line = f"{label}: " + ", ".join(reversed(newest_first))
            if sum(len(existing) + 1 for existing in lines) + len(line) <= budget:
                lines.append(line)
                return
            newest_first.pop()
            dropped += 1

    _csv("files", items["files"].values())
    _csv("tools", items["tools"].values())
    _csv("ids", items["identifiers"].values())
    if len(lines) == 1:
        return ""
    if dropped:
        lines.append(f"dropped: {dropped} items (over budget)")
    body = "\n".join(lines)
    return _redact(body)[:budget]


_HYBRID_INDEX_SECTION_RE = re.compile(
    rf"(?ms)^(?:{re.escape(HYBRID_INDEX_HEADING)})[^\n]*(?:\n(?!## ).*)*"
)


def strip_hybrid_handle_index(summary: str) -> str:
    """Remove every Unique handles section so a rebuild can replace them."""
    if not summary or HYBRID_INDEX_HEADING not in summary:
        return summary or ""
    stripped = _HYBRID_INDEX_SECTION_RE.sub("", summary)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def append_hybrid_handle_index(summary: str, index: str) -> str:
    """Rebuild exactly one Unique handles section, even if the LLM echoed it."""
    body = strip_hybrid_handle_index(summary or "")
    if not index:
        return body
    return body.rstrip() + "\n\n" + index
