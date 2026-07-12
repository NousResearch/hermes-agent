"""Feishu Card JSON 2.0 renderer for final assistant replies.

Converts a platform-neutral ``MessageDocument`` into Feishu Card JSON 2.0 dicts
suitable for ``msg_type=interactive``. This module is purely functional — it
produces card dictionaries and has no transport dependency on ``FeishuAdapter``
or the lark SDK.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gateway.rendering.document import (
    CodeBlock,
    DividerBlock,
    HeadingBlock,
    ImageBlock,
    ListBlock,
    MessageDocument,
    ParagraphBlock,
    TableBlock,
)

_PLAINTEXT_SUMMARY_MAX_LENGTH = 80
_FIXED_CARD_TITLE = "Hermes"
_UNDESIRED_SUMMARY_CHARS_RE = re.compile(r"[`*_~\[\]!#>"">]")
_DEFAULT_MAX_MARKDOWN_CHARS = 3000
_DEFAULT_MAX_ELEMENTS_PER_CARD = 120
_MAX_ELEMENTS_PER_CARD = 200
_DEFAULT_MAX_CARD_CHARS = 6000
_DEFAULT_MAX_CARD_BYTES = 28 * 1024
_DEFAULT_MAX_ROWS_PER_TABLE = 10
_DEFAULT_MAX_TABLES_PER_CARD = 5
_MAX_TABLE_COLUMNS = 50
_DEFAULT_MAX_OUTER_REQUEST_BYTES = 30 * 1024
_MAX_INLINE_PLAIN_TEXT_BYTES = 1024
_NUMBERING_TITLE_RESERVE = " 99999999999999999999/99999999999999999999"


class FeishuCardRenderingError(ValueError):
    """Raised when a card cannot be represented within Feishu's hard limits."""


def _require_single_card(
    cards: list[dict[str, Any]], *, entrypoint: str
) -> dict[str, Any]:
    if len(cards) != 1:
        raise FeishuCardRenderingError(
            f"{entrypoint} requires multiple cards ({len(cards)}); "
            "use the plural Feishu card builder"
        )
    return cards[0]


def render_document_to_feishu_card_v2(
    doc: MessageDocument,
    *,
    title: str = _FIXED_CARD_TITLE,
    table_policy: str = "table",
    table_cell_type: str = "markdown",
    max_tables: int = _DEFAULT_MAX_TABLES_PER_CARD,
    max_columns: int = 8,
    max_rows: int = 20,
    image_key_by_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a document into exactly one hard-limit-compliant Card v2 dict.

    Callers that can deliver multiple cards must use
    :func:`render_document_to_feishu_card_v2_parts`; this singular entry point
    fails explicitly rather than returning an oversized payload.
    """
    cards = render_document_to_feishu_card_v2_parts(
        doc,
        title=title,
        table_policy=table_policy,
        table_cell_type=table_cell_type,
        max_tables=max_tables,
        max_columns=max_columns,
        max_rows=max_rows,
        image_key_by_source=image_key_by_source,
    )
    return _require_single_card(cards, entrypoint="singular Feishu card renderer")


def render_document_to_feishu_card_v2_parts(
    doc: MessageDocument,
    *,
    title: str = _FIXED_CARD_TITLE,
    table_policy: str = "table",
    table_cell_type: str = "markdown",
    max_tables: int = _DEFAULT_MAX_TABLES_PER_CARD,
    max_columns: int = 8,
    max_rows: int = 20,
    image_key_by_source: dict[str, str] | None = None,
    max_markdown_chars: int = _DEFAULT_MAX_MARKDOWN_CHARS,
    max_elements_per_card: int = _DEFAULT_MAX_ELEMENTS_PER_CARD,
    max_card_chars: int = _DEFAULT_MAX_CARD_CHARS,
    max_card_bytes: int = _DEFAULT_MAX_CARD_BYTES,
) -> list[dict[str, Any]]:
    """Convert a document into one or more Feishu Card v2 payload dicts.

    The returned cards satisfy both the 28 KiB inner-card limit and the
    30 KiB nested SDK request-body limit. Content is split or safely degraded
    at semantic boundaries; it is never silently truncated.
    """
    markdown_budget = min(max_markdown_chars, _DEFAULT_MAX_MARKDOWN_CHARS)
    card_budget = min(max_card_bytes, _DEFAULT_MAX_CARD_BYTES)
    if markdown_budget < 8 or card_budget <= 0:
        raise FeishuCardRenderingError(
            "Feishu card byte budgets must be positive and markdown budget >= 8"
        )
    for name, value in (
        ("max_tables", max_tables),
        ("max_columns", max_columns),
        ("max_rows", max_rows),
        ("max_elements_per_card", max_elements_per_card),
        ("max_card_chars", max_card_chars),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FeishuCardRenderingError(f"{name} must be a positive integer")
    effective_max_tables = min(max_tables, _DEFAULT_MAX_TABLES_PER_CARD)
    effective_max_columns = min(max_columns, _MAX_TABLE_COLUMNS)
    effective_max_rows = min(max_rows, _DEFAULT_MAX_ROWS_PER_TABLE)
    effective_max_elements = min(max_elements_per_card, _MAX_ELEMENTS_PER_CARD)
    effective_title = title
    title_elements: list[dict[str, Any]] = []
    if len((title + _NUMBERING_TITLE_RESERVE).encode("utf-8")) > (
        _MAX_INLINE_PLAIN_TEXT_BYTES
    ):
        effective_title = _FIXED_CARD_TITLE
        title_elements.append(
            {"tag": "markdown", "content": title, "text_size": "heading"}
        )

    elements = title_elements + _document_to_elements(
        doc,
        table_policy=table_policy,
        table_cell_type=table_cell_type,
        max_columns=effective_max_columns,
        max_rows=effective_max_rows,
        image_key_by_source=image_key_by_source,
    )

    normalized: list[dict[str, Any]] = []
    for element in elements:
        normalized.extend(_normalize_non_markdown_element(element))

    expanded: list[dict[str, Any]] = []
    for element in normalized:
        expanded.extend(_split_oversized_markdown_element(element, markdown_budget))

    paginated: list[dict[str, Any]] = []
    for element in expanded:
        if element.get("tag") == "table":
            paginated.extend(_paginate_table_element(element, effective_max_rows))
        else:
            paginated.append(element)

    groups = _partition_elements(
        paginated,
        max_elements_per_card=effective_max_elements,
        max_card_chars=max_card_chars,
        max_tables_per_card=effective_max_tables,
    ) or [[]]

    cards: list[dict[str, Any]] = []
    for group in groups:
        cards.extend(
            _enforce_card_byte_limit(
                group,
                doc_blocks=doc.blocks,
                title=effective_title,
                max_bytes=card_budget,
                max_markdown_chars=markdown_budget,
            )
        )

    if len(cards) > 1:
        total = len(cards)
        for idx, card in enumerate(cards, start=1):
            card["header"]["title"]["content"] = f"{effective_title} {idx}/{total}"

    for index, card in enumerate(cards, start=1):
        final_title = str(card["header"]["title"].get("content") or "")
        final_title_bytes = len(final_title.encode("utf-8"))
        if final_title_bytes > _MAX_INLINE_PLAIN_TEXT_BYTES:
            raise FeishuCardRenderingError(
                f"final plain_text title is {final_title_bytes} UTF-8 bytes; "
                f"limit is {_MAX_INLINE_PLAIN_TEXT_BYTES}"
            )
        if not _serialized_card_fits_limits(card, card_budget):
            raise FeishuCardRenderingError(
                f"final numbered Feishu card {index}/{len(cards)} exceeds hard limits: "
                f"inner={_check_serialized_card_size(card)} bytes, "
                f"outer={_check_serialized_outer_request_size(card)} bytes"
            )
        for element in card["body"]["elements"]:
            if element.get("tag") == "markdown" and len(
                str(element.get("content") or "").encode("utf-8")
            ) > _DEFAULT_MAX_MARKDOWN_CHARS:
                raise FeishuCardRenderingError(
                    "final markdown element exceeds 3000 UTF-8 bytes"
                )
    return cards


def build_feishu_card_v2_payload(text: str, *, table_policy: str = "table") -> str:
    from gateway.rendering.markdown_parser import parse_markdown_document

    return json.dumps(
        render_document_to_feishu_card_v2(
            parse_markdown_document(text), table_policy=table_policy
        ),
        ensure_ascii=False,
    )


def build_feishu_card_v2_payloads(
    text: str,
    *,
    table_policy: str = "table",
    max_markdown_chars: int = _DEFAULT_MAX_MARKDOWN_CHARS,
    max_elements_per_card: int = _DEFAULT_MAX_ELEMENTS_PER_CARD,
    max_card_chars: int = _DEFAULT_MAX_CARD_CHARS,
) -> list[str]:
    from gateway.rendering.markdown_parser import parse_markdown_document

    return [
        json.dumps(card, ensure_ascii=False)
        for card in render_document_to_feishu_card_v2_parts(
            parse_markdown_document(text),
            table_policy=table_policy,
            max_markdown_chars=max_markdown_chars,
            max_elements_per_card=max_elements_per_card,
            max_card_chars=max_card_chars,
        )
    ]


def build_feishu_card_v2_payload_from_document(
    doc: MessageDocument,
    *,
    table_policy: str = "table",
    image_key_by_source: dict[str, str] | None = None,
) -> str:
    return json.dumps(
        render_document_to_feishu_card_v2(
            doc,
            table_policy=table_policy,
            image_key_by_source=image_key_by_source,
        ),
        ensure_ascii=False,
    )


def build_feishu_card_v2_payloads_from_document(
    doc: MessageDocument,
    *,
    table_policy: str = "table",
    image_key_by_source: dict[str, str] | None = None,
    max_markdown_chars: int = _DEFAULT_MAX_MARKDOWN_CHARS,
    max_elements_per_card: int = _DEFAULT_MAX_ELEMENTS_PER_CARD,
    max_card_chars: int = _DEFAULT_MAX_CARD_CHARS,
) -> list[str]:
    return [
        json.dumps(card, ensure_ascii=False)
        for card in render_document_to_feishu_card_v2_parts(
            doc,
            table_policy=table_policy,
            image_key_by_source=image_key_by_source,
            max_markdown_chars=max_markdown_chars,
            max_elements_per_card=max_elements_per_card,
            max_card_chars=max_card_chars,
        )
    ]


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _build_card_payload(
    blocks: list[Any],
    elements: list[Any],
    *,
    title: str,
) -> dict[str, Any]:
    summary = _build_summary(blocks)
    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "width_mode": "fill",
            "summary": {"content": summary},
        },
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "blue",
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 8px 12px 8px",
            "elements": elements,
        },
    }


def _document_to_elements(
    doc: MessageDocument,
    *,
    table_policy: str,
    table_cell_type: str,
    max_columns: int,
    max_rows: int,
    image_key_by_source: dict[str, str] | None,
) -> list[Any]:
    elements: list[Any] = []

    for block in doc.blocks:
        if isinstance(block, ParagraphBlock):
            elements.append(
                {"tag": "markdown", "content": block.text, "text_size": "normal"}
            )
        elif isinstance(block, HeadingBlock):
            elements.append(
                {"tag": "markdown", "content": block.text, "text_size": "heading"}
            )
        elif isinstance(block, CodeBlock):
            content = _render_code_block_content(block)
            elements.append(
                {"tag": "markdown", "content": content, "text_size": "normal"}
            )
        elif isinstance(block, ListBlock):
            content = "\n".join(
                f"{index}. {item}" if block.ordered else f"- {item}"
                for index, item in enumerate(block.items, start=1)
            )
            elements.append(
                {"tag": "markdown", "content": content, "text_size": "normal"}
            )
        elif isinstance(block, DividerBlock):
            elements.append({"tag": "hr"})
        elif isinstance(block, ImageBlock):
            image_key = (image_key_by_source or {}).get(block.source, "")
            if image_key:
                elements.append(
                    {
                        "tag": "img",
                        "img_key": image_key,
                        "alt": {
                            "tag": "plain_text",
                            "content": block.alt or "image",
                        },
                    }
                )
            else:
                elements.append(
                    {
                        "tag": "markdown",
                        "content": f"[Image: {block.alt or block.source}]",
                        "text_size": "normal",
                    }
                )
        elif isinstance(block, TableBlock):
            use_table = (
                table_policy == "table"
                and len(block.headers) <= max_columns
                and bool(block.headers)
                and bool(block.rows)
                and all(len(row) == len(block.headers) for row in block.rows)
            )
            if use_table:
                elements.append(
                    _build_table_element(
                        block,
                        table_cell_type,
                        min(len(block.rows), max_rows),
                    )
                )
            else:
                content = _render_code_block_content_from_raw(
                    language="markdown", code=block.raw_markdown or _table_to_markdown(block)
                )
                elements.append(
                    {"tag": "markdown", "content": content, "text_size": "normal"}
                )
        else:
            raise FeishuCardRenderingError(
                f"unsupported message document block: {type(block).__name__}"
            )
    return elements


def _normalize_non_markdown_element(element: dict[str, Any]) -> list[dict[str, Any]]:
    """Move oversized plain-text fields into splittable markdown elements.

    Feishu image ``alt`` is a single unsplittable plain-text field. Keeping a
    short placeholder on the image while emitting the complete description as
    adjacent markdown preserves the information and gives the normal UTF-8
    splitter a semantic boundary to work with.
    """
    if element.get("tag") != "img":
        return [element]
    alt = element.get("alt")
    if not isinstance(alt, dict):
        return [element]
    content = str(alt.get("content") or "")
    if len(content.encode("utf-8")) <= _MAX_INLINE_PLAIN_TEXT_BYTES:
        return [element]
    image = {**element, "alt": {**alt, "content": "image"}}
    return [
        image,
        {"tag": "markdown", "content": content, "text_size": "normal"},
    ]


def _split_oversized_markdown_element(
    element: dict[str, Any],
    max_markdown_chars: int,
) -> list[dict[str, Any]]:
    """Split an oversized markdown element by UTF-8 byte budget.

    - Uses ``len(content.encode("utf-8"))`` for the size check.
    - Prefers splitting at line boundaries (``\\n``).
    - When no line boundary is available within the budget, splits at a
      UTF-8 character-safe position (never mid-multibyte).
    - Avoids splitting inside ``[...](...)`` links, `` `...` `` inline code,
      or backslash escape sequences.
    - When the content contains an open code fence (````​```​````), each
      split piece gets a closing fence and the next piece re-opens it.
    - Never produces empty chunks.
    """
    if element.get("tag") != "markdown":
        return [element]
    content = str(element.get("content") or "")
    if len(content.encode("utf-8")) <= max_markdown_chars:
        return [element]
    parts = _split_markdown_by_utf8_bytes(content, max_markdown_chars)
    return [{**element, "content": part} for part in parts if part]


def _is_markdown_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _markdown_link_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        if text[index] != "[" or _is_markdown_escaped(text, index):
            index += 1
            continue
        label_depth = 1
        cursor = index + 1
        while cursor < len(text) and label_depth:
            if not _is_markdown_escaped(text, cursor):
                if text[cursor] == "[":
                    label_depth += 1
                elif text[cursor] == "]":
                    label_depth -= 1
            cursor += 1
        if label_depth or cursor >= len(text) or text[cursor] != "(":
            index += 1
            continue
        paren_depth = 1
        cursor += 1
        while cursor < len(text) and paren_depth:
            if not _is_markdown_escaped(text, cursor):
                if text[cursor] == "(":
                    paren_depth += 1
                elif text[cursor] == ")":
                    paren_depth -= 1
            cursor += 1
        if paren_depth == 0:
            spans.append((index, cursor, "link"))
            index = cursor
        else:
            index += 1
    return spans


def _markdown_atom_spans(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping link/inline-code spans.

    Inline-code delimiters are matched by exact backtick-run length. Fenced
    runs (three or more backticks) are intentionally left to the fence-aware
    splitter.
    """
    spans: list[tuple[int, int, str]] = _markdown_link_spans(text)

    index = 0
    while index < len(text):
        if text[index] != "`" or _is_markdown_escaped(text, index):
            index += 1
            continue
        end_run = index + 1
        while end_run < len(text) and text[end_run] == "`":
            end_run += 1
        run_length = end_run - index
        if run_length >= 3:
            index = end_run
            continue
        cursor = end_run
        closing = -1
        while cursor < len(text):
            if text[cursor] != "`" or _is_markdown_escaped(text, cursor):
                cursor += 1
                continue
            candidate_end = cursor + 1
            while candidate_end < len(text) and text[candidate_end] == "`":
                candidate_end += 1
            if candidate_end - cursor == run_length:
                closing = candidate_end
                break
            cursor = candidate_end
        if closing > 0:
            spans.append((index, closing, "inline_code"))
            index = closing
        else:
            index = end_run

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[tuple[int, int, str]] = []
    for span in spans:
        if result and span[0] < result[-1][1]:
            continue
        result.append(span)
    return result


def _escape_oversized_markdown_atom(raw: str, kind: str) -> str:
    """Make atom delimiters literal without flipping existing escape parity."""
    special = "`" if kind == "inline_code" else "`[]()"
    escaped: list[str] = []
    for index, char in enumerate(raw):
        if char in special and not _is_markdown_escaped(raw, index):
            escaped.append("\\")
        escaped.append(char)
    return "".join(escaped)


def _degrade_oversized_markdown_atoms(text: str, max_bytes: int) -> str:
    spans = _markdown_atom_spans(text)
    if not spans:
        return text
    chunks: list[str] = []
    cursor = 0
    for start, end, kind in spans:
        chunks.append(text[cursor:start])
        raw = text[start:end]
        chunks.append(
            _escape_oversized_markdown_atom(raw, kind)
            if len(raw.encode("utf-8")) > max_bytes
            else raw
        )
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _degrade_oversized_fence_openers(text: str, max_bytes: int) -> str:
    """Escape a fenced block whose complete opener cannot fit atomically.

    Fence opener lines are indivisible: cutting one creates a partial language
    string that can be reinserted indefinitely when the splitter reopens the
    fence. Escaping backticks across that block turns it into pageable visible
    plain source while preserving every source character after unescaping.
    """
    lines = text.splitlines(keepends=True)
    degraded: list[str] = []
    escaping_block = False

    for line in lines:
        stripped = line.rstrip("\r\n").strip()
        if not escaping_block:
            is_opener = re.fullmatch(r"```[^`]*", stripped) is not None
            if is_opener and len(line.encode("utf-8")) > max_bytes:
                escaping_block = True
        if escaping_block:
            degraded.append(line.replace("`", "\\`"))
            if stripped == "```":
                escaping_block = False
        else:
            degraded.append(line)

    return "".join(degraded)


def _find_safe_split_point(text: str, max_bytes: int) -> int:
    """Return a UTF-8-safe character boundary outside Markdown atoms."""
    if max_bytes <= 0:
        return 0
    if len(text.encode("utf-8")) <= max_bytes:
        return len(text)

    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(text[:mid].encode("utf-8")) <= max_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        return 0

    window = text[:best]
    candidates = [window.rfind("\n\n"), window.rfind("\n"), window.rfind("。"), window.rfind(" ")]
    boundary = max(candidates)
    split_at = boundary + 1 if boundary >= max(40, best // 4) else best

    for start, end, _kind in _markdown_atom_spans(text):
        if start < split_at < end:
            split_at = start if start > 0 else end
            break

    while split_at > 0:
        slash_count = 0
        cursor = split_at - 1
        while cursor >= 0 and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            break
        split_at -= 1
    return split_at


def _open_fence_language(text: str) -> str | None:
    """Return the language of an unmatched triple-backtick fence in *text*."""
    language: str | None = None
    in_fence = False
    for line in text.splitlines():
        match = re.match(r"^```([^`]*)$", line.strip())
        if not match:
            continue
        if in_fence:
            in_fence = False
            language = None
        else:
            in_fence = True
            language = match.group(1).strip()
    return language if in_fence else None


def _split_markdown_by_utf8_bytes(content: str, max_bytes: int) -> list[str]:
    """Split markdown under a final UTF-8 budget without breaking atoms.

    Oversized links and inline-code spans cannot fit atomically, so they are
    first escaped into plain source text. When a split lands inside a fenced
    block, closing/reopening fences are included in the budget before the cut
    is selected.
    """
    if max_bytes < 8:
        raise FeishuCardRenderingError(
            f"markdown byte budget {max_bytes} is too small for safe splitting"
        )
    remaining = _degrade_oversized_fence_openers(content, max_bytes)
    parts: list[str] = []

    while remaining:
        # Splitting can expose a previously protected atom at the start of the
        # rewritten remainder. Re-scan every round instead of assuming the
        # initial degradation remains sufficient after synthetic markup and
        # escape boundaries are introduced.
        remaining = _degrade_oversized_markdown_atoms(remaining, max_bytes)
        remaining_bytes = len(remaining.encode("utf-8"))
        if remaining_bytes <= max_bytes:
            parts.append(remaining)
            break

        previous_remaining_bytes = remaining_bytes
        split_at = _find_safe_split_point(remaining, max_bytes)
        if split_at <= 0:
            raise FeishuCardRenderingError(
                "no UTF-8/Markdown-safe split point fits the markdown budget"
            )

        part = remaining[:split_at]
        rest = remaining[split_at:]
        language = _open_fence_language(part)
        if language is not None:
            close_fence = "```" if part.endswith("\n") else "\n```"
            content_budget = max_bytes - len(close_fence.encode("utf-8"))
            split_at = _find_safe_split_point(remaining, content_budget)
            if split_at <= 0:
                raise FeishuCardRenderingError(
                    "code-fence overhead leaves no safe markdown split point"
                )
            part = remaining[:split_at]
            rest = remaining[split_at:]
            language = _open_fence_language(part)
            if language is not None:
                close_fence = "```" if part.endswith("\n") else "\n```"
                part += close_fence
                rest = f"```{language}\n" + rest

        if not part:
            raise FeishuCardRenderingError("markdown splitter produced an empty chunk")
        if len(part.encode("utf-8")) > max_bytes:
            raise FeishuCardRenderingError(
                f"markdown splitter produced {len(part.encode('utf-8'))} bytes "
                f"for a {max_bytes}-byte budget"
            )
        next_remaining_bytes = len(rest.encode("utf-8"))
        if next_remaining_bytes >= previous_remaining_bytes:
            raise FeishuCardRenderingError(
                "markdown splitter made no UTF-8 byte progress: "
                f"remaining={previous_remaining_bytes}, next={next_remaining_bytes}"
            )
        parts.append(part)
        remaining = rest

    return parts


def _detect_open_code_fence(text: str) -> str | None:
    """Backward-compatible wrapper for fence-state tests/callers."""
    return _open_fence_language(text)


def _enforce_card_byte_limit(
    elements: list[dict[str, Any]],
    *,
    doc_blocks: list[Any],
    title: str,
    max_bytes: int,
    max_markdown_chars: int,
) -> list[dict[str, Any]]:
    """Recursively split/degrade until inner and outer hard limits fit.

    A conservative numbering suffix is included while sizing so adding the
    final ``N/M`` title cannot push an otherwise valid card over the wire
    boundary. No content-truncating fallback exists.
    """
    card = _build_card_payload(doc_blocks, elements, title=title)
    reserved_card = _build_card_payload(
        doc_blocks, elements, title=title + _NUMBERING_TITLE_RESERVE
    )
    if _serialized_card_fits_limits(card, max_bytes) and _serialized_card_fits_limits(
        reserved_card, max_bytes
    ):
        return [card]

    if len(elements) > 1:
        mid = len(elements) // 2
        result: list[dict[str, Any]] = []
        for half in (elements[:mid], elements[mid:]):
            result.extend(
                _enforce_card_byte_limit(
                    half,
                    doc_blocks=doc_blocks,
                    title=title,
                    max_bytes=max_bytes,
                    max_markdown_chars=max_markdown_chars,
                )
            )
        return result

    if len(elements) == 1:
        element = elements[0]
        tag = element.get("tag")
        if tag == "table":
            row_count = len(element.get("rows", []))
            if row_count > 1:
                parts = _paginate_table_element(element, max(1, row_count // 2))
                result: list[dict[str, Any]] = []
                for part in parts:
                    result.extend(
                        _enforce_card_byte_limit(
                            [part],
                            doc_blocks=doc_blocks,
                            title=title,
                            max_bytes=max_bytes,
                            max_markdown_chars=max_markdown_chars,
                        )
                    )
                return result

            markdown = {
                "tag": "markdown",
                "content": _table_element_to_markdown(element),
                "text_size": "normal",
            }
            result: list[dict[str, Any]] = []
            for part in _split_oversized_markdown_element(
                markdown, max_markdown_chars
            ):
                result.extend(
                    _enforce_card_byte_limit(
                        [part],
                        doc_blocks=doc_blocks,
                        title=title,
                        max_bytes=max_bytes,
                        max_markdown_chars=max_markdown_chars,
                    )
                )
            return result

        if tag == "markdown":
            content = str(element.get("content") or "")
            smaller_budget = min(max_markdown_chars, max(8, len(content.encode("utf-8")) // 2))
            parts = _split_markdown_by_utf8_bytes(content, smaller_budget)
            if len(parts) > 1:
                result: list[dict[str, Any]] = []
                for part in parts:
                    result.extend(
                        _enforce_card_byte_limit(
                            [{**element, "content": part}],
                            doc_blocks=doc_blocks,
                            title=title,
                            max_bytes=max_bytes,
                            max_markdown_chars=max_markdown_chars,
                        )
                    )
                return result

        raise FeishuCardRenderingError(
            f"unsplittable Feishu card element tag={tag!r} exceeds hard limits: "
            f"inner={_check_serialized_card_size(card)} bytes, "
            f"outer={_check_serialized_outer_request_size(card)} bytes"
        )

    raise FeishuCardRenderingError(
        f"empty Feishu card exceeds hard limits because of its envelope/title: "
        f"inner={_check_serialized_card_size(card)} bytes, "
        f"outer={_check_serialized_outer_request_size(card)} bytes"
    )


def _table_element_to_markdown(table_el: dict) -> str:
    """Convert a table element dict back to a markdown table string."""
    columns = table_el.get("columns", [])
    headers = [c.get("display_name", c.get("name", "")) for c in columns]
    col_names = [c.get("name", f"col_{i}") for i, c in enumerate(columns)]
    rows = table_el.get("rows", [])

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(row.get(name, "")) for name in col_names]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _check_serialized_card_size(card_dict: dict) -> int:
    """Serialize *card_dict* to JSON and return its UTF-8 byte size."""
    return len(json.dumps(card_dict, ensure_ascii=False).encode("utf-8"))


def _check_serialized_outer_request_size(card_dict: dict) -> int:
    """Return a conservative SDK create-message request size estimate.

    The lark SDK stores the card JSON as a string in ``content``. Serializing
    that outer body escapes the card JSON a second time, so checking the card
    payload alone is insufficient for quote/backslash-heavy content. The
    placeholder receive ID is intentionally longer than normal IDs to reserve
    transport-envelope headroom without importing lark-oapi in the renderer.
    """
    content = json.dumps(card_dict, ensure_ascii=False)
    outer_body = {
        "receive_id": "r" * 256,
        "msg_type": "interactive",
        "content": content,
        "uuid": "u" * 50,
    }
    return len(json.dumps(outer_body, ensure_ascii=False).encode("utf-8"))


def _serialized_card_fits_limits(card_dict: dict, max_bytes: int) -> bool:
    return (
        _check_serialized_card_size(card_dict) <= max_bytes
        and _check_serialized_outer_request_size(card_dict)
        < _DEFAULT_MAX_OUTER_REQUEST_BYTES
    )


def _paginate_table_element(
    table_element: dict,
    max_rows: int = _DEFAULT_MAX_ROWS_PER_TABLE,
) -> list[dict]:
    """Split a table element into multiple sub-tables, each with at most
    *max_rows* rows.  Each sub-table retains the original header/columns.

    Returns a list of table element dicts.  If the original table has
    at most *max_rows* rows, returns a single-element list.
    """
    rows = table_element.get("rows", [])
    if len(rows) <= max_rows:
        return [dict(table_element)]

    parts: list[dict] = []
    for start in range(0, len(rows), max_rows):
        chunk = rows[start : start + max_rows]
        part = dict(table_element)
        part["rows"] = chunk
        part["page_size"] = min(len(chunk), max_rows)
        parts.append(part)
    return parts


def _partition_elements(
    elements: list[dict[str, Any]],
    *,
    max_elements_per_card: int,
    max_card_chars: int,
    max_tables_per_card: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    current_tables = 0

    for element in elements:
        element_chars = _element_char_size(element)
        is_table = element.get("tag") == "table"
        would_overflow_count = len(current) >= max_elements_per_card
        would_overflow_chars = bool(current) and current_chars + element_chars > max_card_chars
        would_overflow_tables = is_table and current_tables >= max_tables_per_card
        if would_overflow_count or would_overflow_chars or would_overflow_tables:
            groups.append(current)
            current = []
            current_chars = 0
            current_tables = 0
        current.append(element)
        current_chars += element_chars
        current_tables += int(is_table)
    if current:
        groups.append(current)
    return groups


def _element_char_size(element: dict[str, Any]) -> int:
    if element.get("tag") == "markdown":
        return len(str(element.get("content") or ""))
    return len(json.dumps(element, ensure_ascii=False))


def _build_summary(blocks: list[Any]) -> str:
    for block in blocks:
        if isinstance(block, ParagraphBlock):
            return _strip_inline_markdown(block.text)[:_PLAINTEXT_SUMMARY_MAX_LENGTH]
    for block in blocks:
        if isinstance(block, HeadingBlock):
            return _strip_inline_markdown(block.text)[:_PLAINTEXT_SUMMARY_MAX_LENGTH]
    return ""


def _strip_inline_markdown(text: str) -> str:
    """Best-effort inline Markdown removal for notification summaries."""
    return _UNDESIRED_SUMMARY_CHARS_RE.sub("", text).strip()


def _render_code_block_content(block: CodeBlock) -> str:
    code = block.code.replace("\r\n", "\n")
    closing_separator = "" if code.endswith("\n") else "\n"
    return f"```{block.language}\n{code}{closing_separator}```"


def _render_code_block_content_from_raw(*, language: str, code: str) -> str:
    normalized = code.replace("\r\n", "\n")
    closing_separator = "" if normalized.endswith("\n") else "\n"
    return f"```{language}\n{normalized}{closing_separator}```"


def _build_table_element(data: TableBlock, cell_type: str, row_count: int) -> dict[str, Any]:
    return {
        "tag": "table",
        "page_size": min(row_count, 10),
        "row_height": "auto",
        "row_max_height": "124px",
        "freeze_first_column": len(data.headers) > 2,
        "header_style": {
            "text_align": "left",
            "text_size": "normal",
            "background_style": "none",
            "text_color": "grey",
            "bold": True,
            "lines": 1,
        },
        "columns": [
            {"name": f"col_{i}", "display_name": header, "data_type": cell_type}
            for i, header in enumerate(data.headers)
        ],
        "rows": [
            {f"col_{i}": cell for i, cell in enumerate(row)}
            for row in data.rows
        ],
    }


def _table_to_markdown(data: TableBlock) -> str:
    lines = ["| " + " | ".join(data.headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in data.headers) + " |")
    for row in data.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
