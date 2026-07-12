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
    MessageDocument,
    ParagraphBlock,
    TableBlock,
)

_PLAINTEXT_SUMMARY_MAX_LENGTH = 80
_FIXED_CARD_TITLE = "Hermes"
_UNDESIRED_SUMMARY_CHARS_RE = re.compile(r"[`*_~\[\]!#>"">]")
_DEFAULT_MAX_MARKDOWN_CHARS = 3000
_DEFAULT_MAX_ELEMENTS_PER_CARD = 120
_DEFAULT_MAX_CARD_CHARS = 6000


def render_document_to_feishu_card_v2(
    doc: MessageDocument,
    *,
    title: str = _FIXED_CARD_TITLE,
    table_policy: str = "table",
    table_cell_type: str = "markdown",
    max_tables: int = 5,
    max_columns: int = 8,
    max_rows: int = 20,
    image_key_by_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a ``MessageDocument`` into a single Feishu Card JSON 2.0 dict."""
    elements = _document_to_elements(
        doc,
        table_policy=table_policy,
        table_cell_type=table_cell_type,
        max_tables=max_tables,
        max_columns=max_columns,
        max_rows=max_rows,
        image_key_by_source=image_key_by_source,
    )
    return _build_card_payload(doc.blocks, elements, title=title)


def render_document_to_feishu_card_v2_parts(
    doc: MessageDocument,
    *,
    title: str = _FIXED_CARD_TITLE,
    table_policy: str = "table",
    table_cell_type: str = "markdown",
    max_tables: int = 5,
    max_columns: int = 8,
    max_rows: int = 20,
    image_key_by_source: dict[str, str] | None = None,
    max_markdown_chars: int = _DEFAULT_MAX_MARKDOWN_CHARS,
    max_elements_per_card: int = _DEFAULT_MAX_ELEMENTS_PER_CARD,
    max_card_chars: int = _DEFAULT_MAX_CARD_CHARS,
) -> list[dict[str, Any]]:
    """Convert a document into one or more Feishu Card v2 payload dicts.

    Large single cards can be clipped by Feishu clients or rejected by OpenAPI.
    This renderer keeps the response complete by splitting oversized markdown
    elements, then partitioning elements into multiple cards.
    """
    elements = _document_to_elements(
        doc,
        table_policy=table_policy,
        table_cell_type=table_cell_type,
        max_tables=max_tables,
        max_columns=max_columns,
        max_rows=max_rows,
        image_key_by_source=image_key_by_source,
    )
    expanded: list[dict[str, Any]] = []
    for element in elements:
        expanded.extend(_split_oversized_markdown_element(element, max_markdown_chars))

    groups = _partition_elements(
        expanded,
        max_elements_per_card=max_elements_per_card,
        max_card_chars=max_card_chars,
    )
    if not groups:
        groups = [[]]
    if len(groups) == 1:
        return [_build_card_payload(doc.blocks, groups[0], title=title)]

    total = len(groups)
    return [
        _build_card_payload(doc.blocks, group, title=f"{title} {idx}/{total}")
        for idx, group in enumerate(groups, start=1)
    ]


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
    max_tables: int,
    max_columns: int,
    max_rows: int,
    image_key_by_source: dict[str, str] | None,
) -> list[Any]:
    elements: list[Any] = []
    table_count = 0

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
                and table_count < max_tables
                and len(block.headers) <= max_columns
                and len(block.rows) <= max_rows
                and bool(block.headers)
                and bool(block.rows)
            )
            if use_table:
                elements.append(_build_table_element(block, table_cell_type, len(block.rows)))
                table_count += 1
            else:
                content = _render_code_block_content_from_raw(
                    language="markdown", code=block.raw_markdown or _table_to_markdown(block)
                )
                elements.append(
                    {"tag": "markdown", "content": content, "text_size": "normal"}
                )
        # Unknown block types are silently ignored.
    return elements


def _split_oversized_markdown_element(
    element: dict[str, Any],
    max_markdown_chars: int,
) -> list[dict[str, Any]]:
    if element.get("tag") != "markdown":
        return [element]
    content = str(element.get("content") or "")
    if len(content) <= max_markdown_chars:
        return [element]
    parts = _split_text_preserving_words(content, max_markdown_chars)
    return [{**element, "content": part} for part in parts]


def _split_text_preserving_words(text: str, limit: int) -> list[str]:
    if limit <= 0 or len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind("。"), window.rfind(" "))
        if split_at < max(80, limit // 3):
            split_at = limit
        else:
            split_at += 1
        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _partition_elements(
    elements: list[dict[str, Any]],
    *,
    max_elements_per_card: int,
    max_card_chars: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for element in elements:
        element_chars = _element_char_size(element)
        would_overflow_count = len(current) >= max_elements_per_card
        would_overflow_chars = bool(current) and current_chars + element_chars > max_card_chars
        if would_overflow_count or would_overflow_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(element)
        current_chars += element_chars
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
    code = block.code.replace("\r\n", "\n").rstrip("\n")
    return f"```{block.language}\n{code}\n```"


def _render_code_block_content_from_raw(*, language: str, code: str) -> str:
    normalized = code.replace("\r\n", "\n").rstrip("\n")
    return f"```{language}\n{normalized}\n```"


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
            {f"col_{i}": cell for i, cell in enumerate(padded_row)}
            for padded_row in (
                _fit_row(row, len(data.headers)) for row in data.rows
            )
        ],
    }


def _table_to_markdown(data: TableBlock) -> str:
    lines = ["| " + " | ".join(data.headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in data.headers) + " |")
    for row in data.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fit_row(row: list[str], width: int) -> list[str]:
    fitted = list(row)[:width]
    if len(fitted) < width:
        fitted.extend("" for _ in range(width - len(fitted)))
    return fitted
