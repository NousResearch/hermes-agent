"""Configurable response style prompts and lightweight output guardrails."""

from __future__ import annotations

import re
from typing import Any

_DETAIL_REQUEST_RE = re.compile(
    r"(详细|展开|复盘|查日志|日志|证据|完整|具体|细节|路径|代码|脚本|命令|配置|测试|报错|错误|diff|patch|debug|details?|verbose)",
    re.IGNORECASE,
)

_SECTION_PATTERNS = {
    "result": re.compile(r"(?:^|\n)\s*(?:Result|结果|结论)\s*[:：]\s*(.*?)(?=\n\s*(?:Blocker|阻塞|卡点|Next step|下一步)\s*[:：]|\Z)", re.IGNORECASE | re.DOTALL),
    "blocker": re.compile(r"(?:^|\n)\s*(?:Blocker|阻塞|卡点)\s*[:：]\s*(.*?)(?=\n\s*(?:Result|结果|结论|Next step|下一步)\s*[:：]|\Z)", re.IGNORECASE | re.DOTALL),
    "next": re.compile(r"(?:^|\n)\s*(?:Next step|下一步)\s*[:：]\s*(.*?)(?=\n\s*(?:Result|结果|结论|Blocker|阻塞|卡点)\s*[:：]|\Z)", re.IGNORECASE | re.DOTALL),
}

_BLOCKER_RE = re.compile(r"(blocker|阻塞|卡点|失败|不能|无法|没有|待|pending|error|failed|stale)", re.IGNORECASE)
_NEXT_RE = re.compile(r"(next step|下一步|建议|后面|之后|我会|继续|接下来)", re.IGNORECASE)
_MEDIA_ONLY_RE = re.compile(r"^\s*(?:MEDIA:\S+\s*)+$")
_DELIVERY_CONTROL_MARKERS = {"[SILENT]"}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _style_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    raw = config.get("response_style") or config.get("communication_style") or {}
    return raw if isinstance(raw, dict) else {}


def _platform_enabled(style: dict[str, Any], platform: str | None) -> bool:
    platforms = style.get("platforms") or []
    if not platforms:
        return True
    platform_key = (platform or "").strip().lower()
    return any(str(p).strip().lower() == platform_key for p in platforms)


def is_response_style_enabled(config: dict[str, Any] | None, platform: str | None = None) -> bool:
    style = _style_config(config)
    if not _truthy(style.get("enabled"), False):
        return False
    return _platform_enabled(style, platform)


def build_response_style_prompt(config: dict[str, Any] | None, platform: str | None = None) -> str:
    """Return an extra system-prompt block for the configured response style."""
    if not is_response_style_enabled(config, platform):
        return ""
    style = _style_config(config)
    profile = str(style.get("profile") or "secretary").strip().lower()
    if profile not in {"secretary", "executive_secretary"}:
        return ""

    return (
        "# User-facing communication style — secretary mode\n"
        "Treat the user-facing reply like a capable human secretary/assistant would.\n"
        "Default front-channel replies must be short, plain-language, and decision-oriented.\n"
        "Use this visible structure unless the user explicitly asks for detail:\n"
        "Result:\n"
        "<one short conclusion; include only the practical impact>\n\n"
        "Blocker:\n"
        "<none, or the single blocker that matters>\n\n"
        "Next step:\n"
        "<the next useful action>\n\n"
        "Do not dump paths, logs, PIDs, stack traces, tool names, model/router details, or long evidence lists "
        "unless the user asks for them. If verification matters, say briefly that it was verified; keep the raw proof internal. "
        "专业词或英文缩写只有在必要时才出现，并立刻用中文解释。"
    )


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)].rstrip()
    # Prefer ending on a sentence or line boundary when possible.
    boundary = max(cut.rfind("。"), cut.rfind("."), cut.rfind("\n"), cut.rfind("；"), cut.rfind(";"))
    if boundary >= max(40, int(limit * 0.55)):
        cut = cut[: boundary + 1].rstrip()
    return cut + "…"


def _first_content_line(text: str) -> str:
    for line in _clean(text).splitlines():
        stripped = line.strip(" -•*\t")
        if not stripped:
            continue
        if stripped.lower().startswith(("fresh evidence", "evidence", "next step", "blocker")):
            continue
        return stripped
    return _clean(text)


def _find_line(text: str, pattern: re.Pattern[str]) -> str:
    for line in _clean(text).splitlines():
        stripped = line.strip(" -•*\t")
        if stripped and pattern.search(stripped):
            return stripped
    return ""


def _extract_section(text: str, key: str) -> str:
    match = _SECTION_PATTERNS[key].search(text)
    return _clean(match.group(1)) if match else ""


def apply_response_style_guard(
    response: str,
    config: dict[str, Any] | None,
    platform: str | None = None,
    user_message: str | None = None,
) -> str:
    """Lightweight deterministic guard for concise secretary-style gateway replies.

    This is intentionally local and conservative: it does not invent evidence, and
    it skips media-only replies or explicit user requests for details.
    """
    if not response or not is_response_style_enabled(config, platform):
        return response
    stripped_response = response.strip()
    if stripped_response in _DELIVERY_CONTROL_MARKERS:
        return stripped_response
    if "MEDIA:" in response or "![" in response or _MEDIA_ONLY_RE.match(response):
        return response
    if user_message and _DETAIL_REQUEST_RE.search(user_message):
        return response

    style = _style_config(config)
    if str(style.get("profile") or "secretary").strip().lower() not in {"secretary", "executive_secretary"}:
        return response

    max_chars = int(style.get("max_chars") or 700)
    require_labels = _truthy(style.get("require_labels"), True)
    result_limit = int(style.get("result_chars") or 220)
    blocker_limit = int(style.get("blocker_chars") or 180)
    next_limit = int(style.get("next_chars") or 180)

    cleaned = _clean(response)
    already_labeled = all(_extract_section(cleaned, key) for key in ("result", "blocker", "next"))
    if not require_labels and len(cleaned) <= max_chars:
        return cleaned
    if already_labeled and len(cleaned) <= max_chars:
        return cleaned

    result = _extract_section(cleaned, "result") or _first_content_line(cleaned)
    blocker = _extract_section(cleaned, "blocker") or _find_line(cleaned, _BLOCKER_RE) or "未单独说明。"
    next_step = _extract_section(cleaned, "next") or _find_line(cleaned, _NEXT_RE) or "按当前方向继续推进。"

    guarded = (
        f"Result:\n{_truncate(result, result_limit)}\n\n"
        f"Blocker:\n{_truncate(blocker, blocker_limit)}\n\n"
        f"Next step:\n{_truncate(next_step, next_limit)}"
    )
    return _truncate(guarded, max_chars)
