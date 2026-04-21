"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence


def _detect_language() -> str:
    """Detect the configured language from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        approvals = config.get("approvals") or {}
        if isinstance(approvals, dict):
            lang = approvals.get("language", "")
            if lang:
                return lang.lower()
        display = config.get("display") or {}
        if isinstance(display, dict):
            lang = display.get("language", "")
            if lang:
                return lang.lower()
    except Exception:
        pass
    return ""


def _compress_ui_zh(text: str, **kwargs) -> str:
    """Translate compression UI strings to Chinese if language is 'zh'."""
    lang = _detect_language()
    if lang != "zh":
        return text
    
    # Format the template with kwargs first
    formatted = text.format(**kwargs) if kwargs else text
    
    # Chinese translations
    zh_templates = {
        "No changes from compression: {count} messages": "压缩未改变内容：{count} 条消息",
        "Rough transcript estimate: ~{from} tokens (unchanged)": "粗略估算：~{from} tokens（未改变）",
        "Rough transcript estimate: ~{from} → ~{to} tokens": "粗略估算：~{from} → ~{to} tokens",
        "Compressed: {from} → {to} messages": "已压缩：{from} → {to} 条消息",
        "Note: fewer messages can still raise this rough transcript estimate when compression rewrites the transcript into denser summaries.": "注意：消息数减少时，如果压缩将转录重写为更紧凑的摘要，粗略估算仍可能增加。",
    }
    
    for en_template, zh_template in zh_templates.items():
        if text == en_template:
            return zh_template.format(**kwargs)
    
    return formatted


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    if noop:
        headline = _compress_ui_zh(
            "No changes from compression: {count} messages",
            count=before_count,
        )
        if after_tokens == before_tokens:
            token_line = _compress_ui_zh(
                "Rough transcript estimate: ~{from} tokens (unchanged)",
                from_=before_tokens,
            )
        else:
            token_line = _compress_ui_zh(
                "Rough transcript estimate: ~{from} → ~{to} tokens",
                from_=before_tokens,
                to=after_tokens,
            )
    else:
        headline = _compress_ui_zh(
            "Compressed: {from} → {to} messages",
            from_=before_count,
            to=after_count,
        )
        token_line = _compress_ui_zh(
            "Rough transcript estimate: ~{from} → ~{to} tokens",
            from_=before_tokens,
            to=after_tokens,
        )

    note = None
    if not noop and after_count < before_count and after_tokens > before_tokens:
        note = _compress_ui_zh(
            "Note: fewer messages can still raise this rough transcript estimate "
            "when compression rewrites the transcript into denser summaries."
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
