"""Telegram rich-message conversion helpers."""

from __future__ import annotations

from typing import Any, Iterable, List


def rich_message_to_plaintext(rich: dict) -> str:
    """Best-effort plaintext fallback for Telegram rich messages."""
    try:
        parts: List[str] = []
        text_keys = {"text", "summary", "caption", "credit", "expression", "label"}

        def walk(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                parts.append(value)
                return
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if isinstance(value, dict):
                for key in text_keys:
                    if key in value:
                        walk(value.get(key))
                for key, child in value.items():
                    if key in text_keys:
                        continue
                    if isinstance(child, (dict, list)):
                        walk(child)

        walk(rich)
        return "\n".join(parts)
    except Exception:
        return ""


def rich_message_to_markdown(rich: dict) -> str:
    """Convert Telegram rich-message blocks into markdown."""
    try:
        blocks = rich.get("blocks") if isinstance(rich, dict) else rich
        if not isinstance(blocks, list):
            return ""
        lines = _render_blocks(blocks)
        return "\n".join(line.rstrip() for line in lines if line is not None).strip()
    except Exception:
        return rich_message_to_plaintext(rich)


def _render_blocks(blocks: Iterable[Any]) -> List[str]:
    lines: List[str] = []
    for block in blocks:
        rendered = _render_block(block)
        if not rendered:
            continue
        if isinstance(rendered, str):
            lines.extend(rendered.splitlines() or [rendered])
        else:
            lines.extend(rendered)
    return lines


def _render_block(block: Any) -> List[str] | str:
    if not isinstance(block, dict):
        text = _render_rich_text(block)
        return text

    block_type = str(block.get("type") or "").lower()

    if block_type in {"thinking", "anchor"}:
        return []
    if block_type == "paragraph":
        return _render_rich_text(block.get("text"))
    if block_type == "heading":
        text = _render_rich_text(block.get("text"))
        size = _safe_int(block.get("size"), default=1)
        size = min(max(size, 1), 6)
        return f"{'#' * size} {text}".strip() if text else ""
    if block_type == "pre":
        text = _render_rich_text(block.get("text"))
        language = str(block.get("language") or "").strip()
        return f"```{language}\n{text}\n```" if language else f"```\n{text}\n```"
    if block_type == "footer":
        text = _render_rich_text(block.get("text"))
        return f"*{text}*" if text else ""
    if block_type == "divider":
        return "---"
    if block_type == "mathematical_expression":
        expr = _render_rich_text(block.get("expression"))
        return f"${expr}$" if expr else ""
    if block_type == "blockquote":
        inner = _render_blocks(block.get("blocks") or [])
        if not inner:
            return ""
        lines = [f"> {line}".rstrip() for line in "\n".join(inner).splitlines()]
        credit = _render_caption_text(block.get("credit"))
        if credit:
            lines.append(f"> — {credit}".rstrip())
        return lines
    if block_type == "pullquote":
        text = _render_rich_text(block.get("text"))
        if not text:
            return ""
        lines = [f"> {line}".rstrip() for line in text.splitlines() or [text]]
        credit = _render_caption_text(block.get("credit"))
        if credit:
            lines.append(f"> — {credit}".rstrip())
        return lines
    if block_type == "list":
        return _render_list(block)
    if block_type == "table":
        return _render_table(block)
    if block_type == "details":
        summary = _render_rich_text(block.get("summary"))
        nested = _render_blocks(block.get("blocks") or [])
        parts: List[str] = []
        if summary:
            parts.append(f"**{summary}**")
        parts.extend(nested)
        return parts
    if block_type in {"collage", "slideshow"}:
        parts = _render_blocks(block.get("blocks") or [])
        caption = _render_caption_text(block.get("caption"))
        if caption:
            parts.append(caption)
        return parts
    if block_type == "map":
        loc = block.get("location")
        lat = lon = None
        if isinstance(loc, dict):
            lat = loc.get("latitude", loc.get("lat"))
            lon = loc.get("longitude", loc.get("lon"))
        elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
            lat, lon = loc[0], loc[1]
        elif isinstance(loc, str):
            return [f"[map: {loc}]"]
        label = "[map]"
        if lat is not None and lon is not None:
            label = f"[map: {lat},{lon}]"
        parts = [label]
        caption = _render_caption_text(block.get("caption"))
        if caption:
            parts.append(caption)
        return parts
    if block_type in {"animation", "audio", "photo", "video", "voice_note"}:
        parts = [f"[media: {block_type}]"]
        caption = _render_caption_text(block.get("caption"))
        if caption:
            parts.append(caption)
        return parts

    text = _render_rich_text(block.get("text"))
    if text:
        return text
    return f"[rich block: {block_type or 'unknown'}]"


def _render_list(block: dict) -> List[str]:
    lines: List[str] = []
    for item in block.get("items") or []:
        if not isinstance(item, dict):
            continue
        content_lines = _render_blocks(item.get("blocks") or [])
        if not content_lines:
            continue
        content = "\n".join(content_lines)
        label = _render_rich_text(item.get("label"))
        checkbox = ""
        if item.get("has_checkbox"):
            checkbox = "[x] " if item.get("is_checked") else "[ ] "
        value = item.get("value")
        prefix = f"{value}. " if value is not None else "- "
        first_line = content.splitlines()[0]
        if label:
            first_line = f"{label} {first_line}".strip()
        lines.append(f"{prefix}{checkbox}{first_line}".rstrip())
        lines.extend(content.splitlines()[1:])
    return lines


def _render_table(block: dict) -> List[str]:
    rows = block.get("cells") or []
    if not isinstance(rows, list) or not rows:
        return []

    header_row_idx = None
    for idx, row in enumerate(rows):
        if any(isinstance(cell, dict) and cell.get("is_header") for cell in (row or [])):
            header_row_idx = idx
            break
    if header_row_idx is None:
        header_row_idx = 0

    caption = _render_caption_text(block.get("caption"))
    max_cols = max((len(row) for row in rows if isinstance(row, list)), default=0)
    if max_cols == 0:
        return [caption] if caption else []

    def cell_text(cell: Any) -> str:
        if not isinstance(cell, dict):
            return _render_rich_text(cell)
        return _render_rich_text(cell.get("text"))

    rendered_rows: List[List[str]] = []
    for row in rows:
        if not isinstance(row, list):
            row = []
        rendered_rows.append([cell_text(cell).replace("\n", " ").replace("|", "\\|").strip() for cell in row])

    header = rendered_rows[header_row_idx] if header_row_idx < len(rendered_rows) else []
    header = header + [""] * (max_cols - len(header))
    lines: List[str] = []
    if caption:
        lines.append(caption)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for idx, row in enumerate(rendered_rows):
        if idx == header_row_idx:
            continue
        padded = row + [""] * (max_cols - len(row))
        lines.append("| " + " | ".join(padded) + " |")
    return lines


def _render_caption_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict) and "text" in value:
        text = _render_rich_text(value.get("text"))
        credit = _render_rich_text(value.get("credit"))
        if text and credit:
            return f"{text} — {credit}"
        return text or credit
    return _render_rich_text(value)


def _render_rich_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_render_rich_text(item) for item in value)
    if isinstance(value, dict):
        rich_type = str(value.get("type") or "").lower()
        text = value.get("text")
        inner = _render_rich_text(text)

        if rich_type == "bold":
            return f"**{inner}**"
        if rich_type == "italic":
            return f"*{inner}*"
        if rich_type == "underline":
            return f"__{inner}__"
        if rich_type == "strikethrough":
            return f"~~{inner}~~"
        if rich_type == "spoiler":
            return f"||{inner}||"
        if rich_type == "marked":
            return f"=={inner}=="
        if rich_type == "subscript":
            return f"~{inner}~"
        if rich_type == "superscript":
            return f"^{inner}^"
        if rich_type == "code":
            return f"`{inner}`"
        if rich_type == "url":
            url = str(value.get("url") or "").strip()
            return f"[{inner}]({url})" if inner and url else (url or inner)
        if rich_type == "email_address":
            email = str(value.get("email_address") or "").strip()
            return f"[{inner}](mailto:{email})" if inner and email else (email or inner)
        if rich_type == "mention":
            mention_text = inner or str(value.get("mention") or "")
            return f"@{mention_text.lstrip('@')}"
        if rich_type == "text_mention":
            user = value.get("user") or {}
            user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            if user_id is not None and inner:
                return f"[{inner}](tg://user?id={user_id})"
            return inner
        if rich_type == "custom_emoji":
            return inner or str(value.get("custom_emoji_id") or "")
        if rich_type == "mathematical_expression":
            expr = str(value.get("expression") or inner)
            return f"${expr}$" if expr else ""
        if rich_type in {
            "hashtag",
            "cashtag",
            "bot_command",
            "phone_number",
            "bank_card_number",
            "datetime",
            "anchor",
            "anchor_link",
            "reference",
            "reference_link",
        }:
            return inner

        if inner:
            return inner
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return "".join(_render_rich_text(item) for item in text)
        return ""
    return str(value)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default
