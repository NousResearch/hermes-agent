"""Markdown processing — heading demotion, table demotion, image-key stripping, long-text chunking."""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger("hermes_lark_streaming")

_MAX_CARD_TABLES = 8
_MAX_CHUNK_CHARS = 2400

__all__ = [
    "_downgrade_tables",
    "_find_tables_outside_code_blocks",
    "_split_long_text",
    "_strip_invalid_image_keys",
    "optimize_markdown_style",
]


def _find_tables_outside_code_blocks(text: str) -> list[tuple[int, int, str]]:
    """Find markdown tables outside code blocks; returns [(start, end, raw), ...]."""
    code_ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"```[\s\S]*?```", text):
        code_ranges.append((m.start(), m.end()))

    def _in_code(idx: int) -> bool:
        return any(s <= idx < e for s, e in code_ranges)

    results: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\|.+\|\n\|[-:| ]+\|[\s\S]*?(?=\n\n|\n(?!\|)|$)", text):
        if not _in_code(m.start()):
            results.append((m.start(), m.end(), m.group(0)))
    return results


def _downgrade_tables(text: str, limit: int = _MAX_CARD_TABLES) -> str:
    """Demote an oversized table to a code block (content stays visible; Feishu just renders no table element)."""
    matches = _find_tables_outside_code_blocks(text)
    if len(matches) <= limit:
        return text
    result = text
    for start, end, raw in reversed(matches[limit:]):
        replacement = f"```\n{raw}\n```"
        result = result[:start] + replacement + result[end:]
    return result


def _strip_invalid_image_keys(text: str) -> str:
    """Remove image references that lack the img_ prefix."""
    if "![" not in text:
        return text

    def _replace(m: re.Match) -> str:
        return m.group(0) if m.group(2).startswith("img_") else ""

    return re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", _replace, text)


def optimize_markdown_style(text: str) -> str:
    """Optimize streaming markdown for Feishu CardKit rendering.

    1. lift code blocks out behind placeholders
    2. demote headings: H1 -> H4, H2-H6 -> H5
    3. restore code blocks
    4. collapse excess blank lines
    5. strip invalid image keys (non img_xxx forms)
    """
    try:
        # 1. lift code blocks
        mark = "___CB_"
        code_blocks: list[str] = []

        def _extract(m: re.Match) -> str:
            prefix = m.group(1) or ""
            block = m.group(0)[len(prefix) :]
            idx = len(code_blocks)
            code_blocks.append(block)
            return f"{prefix}{mark}{idx}___"

        r = re.sub(r"(^|\n)(`{3,})([^\n]*)\n[\s\S]*?\n\2(?=\n|$)", _extract, text)

        # 2. demote headings (only when H1-H3 present)
        if re.search(r"^#{1,3} ", text, re.MULTILINE):
            r = re.sub(r"^#{2,6} (.+)$", r"##### \1", r, flags=re.MULTILINE)
            r = re.sub(r"^# (.+)$", r"#### \1", r, flags=re.MULTILINE)

        # 3. restore code blocks
        for i, block in enumerate(code_blocks):
            r = r.replace(f"{mark}{i}___", block)

        # 4. collapse excess blank lines
        r = re.sub(r"\n{3,}", "\n\n", r)

        # 5. strip invalid image keys
        r = _strip_invalid_image_keys(r)

        return r
    except Exception:
        _logger.debug("optimize_markdown_style failed", exc_info=True)
        return text


def _split_long_text(text: str, limit: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Chunk overlong text on paragraph/line boundaries into blocks of at most limit characters."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks
