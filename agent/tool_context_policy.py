"""Compact tool results before they enter model context.

This module keeps large tool outputs useful without letting raw HTML, logs, or
file dumps dominate the next prompt. It is deliberately deterministic: no LLM
summaries, no external storage, and only a small set of built-in tool classes.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 12_000
_TERMINAL_MAX_CHARS = 12_000
_FILE_MAX_CHARS = 16_000
_WEB_MAX_CHARS = 8_000
_JSON_MAX_CHARS = 12_000
_HEAD_CHARS = 3_000
_TAIL_CHARS = 2_000


@dataclass(frozen=True)
class ToolContextConfig:
    enabled: bool = True
    default_max_chars: int = _DEFAULT_MAX_CHARS
    terminal_max_chars: int = _TERMINAL_MAX_CHARS
    file_max_chars: int = _FILE_MAX_CHARS
    web_max_chars: int = _WEB_MAX_CHARS
    json_max_chars: int = _JSON_MAX_CHARS
    head_chars: int = _HEAD_CHARS
    tail_chars: int = _TAIL_CHARS


def _coerce_int(value: Any, default: int, *, minimum: int = 1000) -> int:
    try:
        coerced = int(value)
    except Exception:
        return default
    return max(minimum, coerced)


def load_tool_context_config(config: Mapping[str, Any] | None = None) -> ToolContextConfig:
    """Load tool-context config from a config dict or config.yaml."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
        except Exception:
            config = {}

    raw = config.get("tool_context") if isinstance(config, Mapping) else None
    if not isinstance(raw, Mapping):
        raw = {}

    return ToolContextConfig(
        enabled=bool(raw.get("enabled", True)),
        default_max_chars=_coerce_int(raw.get("default_max_chars"), _DEFAULT_MAX_CHARS),
        terminal_max_chars=_coerce_int(raw.get("terminal_max_chars"), _TERMINAL_MAX_CHARS),
        file_max_chars=_coerce_int(raw.get("file_max_chars"), _FILE_MAX_CHARS),
        web_max_chars=_coerce_int(raw.get("web_max_chars"), _WEB_MAX_CHARS),
        json_max_chars=_coerce_int(raw.get("json_max_chars"), _JSON_MAX_CHARS),
        head_chars=_coerce_int(raw.get("head_chars"), _HEAD_CHARS, minimum=200),
        tail_chars=_coerce_int(raw.get("tail_chars"), _TAIL_CHARS, minimum=200),
    )


def _json_loads(value: str) -> Any | None:
    try:
        return json.loads(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def classify_tool(tool_name: str, args: Mapping[str, Any] | None = None, content: str = "") -> str:
    name = (tool_name or "").lower()
    if name in {"terminal", "process", "execute_code"}:
        return "terminal"
    if name in {"read_file", "search_files", "skill_view"}:
        return "file"
    if name.startswith("web_") or name in {"web_extract", "web_search", "browser"}:
        return "web"
    if _looks_like_html(content):
        return "web"
    if _json_loads(content) is not None:
        return "json"
    return "generic"


def _max_chars_for(kind: str, cfg: ToolContextConfig) -> int:
    if kind == "terminal":
        return cfg.terminal_max_chars
    if kind == "file":
        return cfg.file_max_chars
    if kind == "web":
        return cfg.web_max_chars
    if kind == "json":
        return cfg.json_max_chars
    return cfg.default_max_chars


def _notice(tool_name: str, original_chars: int, kept_chars: int) -> str:
    return (
        f"[tool output compacted: tool={tool_name or 'unknown'} "
        f"original_chars={original_chars} kept_chars={kept_chars}]"
    )


def _head_tail(text: str, max_chars: int, cfg: ToolContextConfig) -> str:
    if len(text) <= max_chars:
        return text
    head = min(cfg.head_chars, max_chars // 2)
    tail = min(cfg.tail_chars, max_chars - head - 200)
    tail = max(200, tail)
    omitted = len(text) - head - tail
    return (
        text[:head].rstrip()
        + f"\n\n[... compacted: {omitted} chars omitted ...]\n\n"
        + text[-tail:].lstrip()
    )


def _looks_like_html(text: str) -> bool:
    sample = text[:4096].lower()
    return "<html" in sample or "<!doctype html" in sample or ("<head" in sample and "<body" in sample)


def _strip_html_noise(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|svg|noscript)\b.*?</\1>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    return text


def _tag_text(pattern: str, text: str, limit: int = 12) -> list[str]:
    values = []
    for match in re.finditer(pattern, text, flags=re.I | re.S):
        value = re.sub(r"<[^>]+>", " ", match.group(1))
        value = html.unescape(re.sub(r"\s+", " ", value)).strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _compact_html(tool_name: str, text: str, max_chars: int, cfg: ToolContextConfig) -> str:
    original = len(text)
    clean = _strip_html_noise(text)
    title = _tag_text(r"<title[^>]*>(.*?)</title>", clean, limit=1)
    meta = _tag_text(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', clean, limit=1)
    headings = _tag_text(r"<h[1-3][^>]*>(.*?)</h[1-3]>", clean, limit=16)
    body = re.sub(r"(?is)<[^>]+>", " ", clean)
    body = html.unescape(re.sub(r"\s+", " ", body)).strip()

    lines = [_notice(tool_name, original, max_chars)]
    if title:
        lines.append(f"Title: {title[0]}")
    if meta:
        lines.append(f"Description: {meta[0]}")
    if headings:
        lines.append("Headings:")
        lines.extend(f"- {h}" for h in headings)
    if body:
        lines.append("Text preview:")
        lines.append(_head_tail(body, max_chars=max_chars, cfg=cfg))
    compacted = "\n".join(lines)
    return compacted[:max_chars]


def _error_lines(text: str, limit: int = 30) -> list[str]:
    found = []
    for line in text.splitlines():
        low = line.lower()
        if any(marker in low for marker in ("error", "exception", "failed", "warning", "traceback")):
            found.append(line[:500])
        if len(found) >= limit:
            break
    return found


def _compact_terminal(tool_name: str, text: str, max_chars: int, cfg: ToolContextConfig) -> str:
    errors = _error_lines(text)
    body = _head_tail(text, max_chars=max_chars, cfg=cfg)
    if not errors:
        return f"{_notice(tool_name, len(text), max_chars)}\n{body}"[:max_chars]
    prefix = [_notice(tool_name, len(text), max_chars), "Important lines:"]
    prefix.extend(f"- {line}" for line in errors)
    return ("\n".join(prefix) + "\n\nOutput preview:\n" + body)[:max_chars]


def _compact_json_value(value: Any, tool_name: str, kind: str, max_chars: int, cfg: ToolContextConfig) -> str:
    if isinstance(value, dict):
        changed = False
        result = dict(value)
        for key, item in list(result.items()):
            if not isinstance(item, str):
                continue
            item_kind = "web" if _looks_like_html(item) else kind
            item_max = max(1000, max_chars - 1500)
            if len(item) > item_max:
                changed = True
                if item_kind == "web":
                    result[key] = _compact_html(tool_name, item, item_max, cfg)
                elif kind == "terminal":
                    result[key] = _compact_terminal(tool_name, item, item_max, cfg)
                else:
                    result[key] = _head_tail(item, item_max, cfg)
        dumped = _json_dumps(result)
        if len(dumped) <= max_chars:
            if changed:
                logger.info("tool_context compacted JSON fields for %s", tool_name)
            return dumped

        skeleton = {}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                skeleton[key] = item if not isinstance(item, str) else _head_tail(item, 1000, cfg)
            elif isinstance(item, list):
                skeleton[key] = f"[list len={len(item)}]"
            elif isinstance(item, dict):
                skeleton[key] = f"[object keys={list(item.keys())[:12]}]"
            else:
                skeleton[key] = f"[{type(item).__name__}]"
        dumped = _json_dumps({
            "_hermes_context_compacted": True,
            "_original_chars": len(_json_dumps(value)),
            "summary": skeleton,
        })
        if len(dumped) <= max_chars:
            return dumped

        fallback = {
            "_hermes_context_compacted": True,
            "_original_chars": len(_json_dumps(value)),
            "keys": list(value.keys())[:50],
            "preview": _head_tail(_json_dumps(value), max(1000, max_chars - 500), cfg),
        }
        dumped = _json_dumps(fallback)
        if len(dumped) <= max_chars:
            return dumped
        fallback["preview"] = fallback["preview"][: max(200, max_chars - 800)]
        return _json_dumps(fallback)

    dumped = _json_dumps(value)
    return _head_tail(dumped, max_chars, cfg)


def compact_tool_result(
    tool_name: str,
    args: Mapping[str, Any] | None,
    content: Any,
    *,
    config: Mapping[str, Any] | ToolContextConfig | None = None,
) -> Any:
    """Return the tool result content that should be sent back to the model."""
    if not isinstance(content, str):
        return content

    cfg = config if isinstance(config, ToolContextConfig) else load_tool_context_config(config)
    if not cfg.enabled:
        return content

    kind = classify_tool(tool_name, args, content)
    max_chars = _max_chars_for(kind, cfg)
    if len(content) <= max_chars:
        return content

    parsed = _json_loads(content)
    if parsed is not None:
        compacted = _compact_json_value(parsed, tool_name, kind, max_chars, cfg)
    elif kind == "web":
        compacted = _compact_html(tool_name, content, max_chars, cfg)
    elif kind == "terminal":
        compacted = _compact_terminal(tool_name, content, max_chars, cfg)
    else:
        compacted = f"{_notice(tool_name, len(content), max_chars)}\n{_head_tail(content, max_chars, cfg)}"
        compacted = compacted[:max_chars]

    if len(compacted) < len(content):
        logger.info(
            "tool_context compacted tool=%s kind=%s original_chars=%d context_chars=%d",
            tool_name, kind, len(content), len(compacted),
        )
    return compacted
