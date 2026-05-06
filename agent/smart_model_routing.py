"""Conservative smart-primary model routing decisions.

This module is intentionally pure: no SDK imports, network calls, or config file
reads. Runtime code passes a single turn and the smart_model_routing config in,
and receives a small decision object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RouteDecision:
    target: str  # "primary" or "cheap"
    reason: str


_CODE_OR_COMPLEX_PATTERNS = (
    r"```",
    r"\b(traceback|exception|stack trace|error|debug|bug|fix|failing|failed)\b",
    r"\b(git|github|pr|issue|commit|branch|diff|patch|test|pytest|npm|pnpm|uv|pip)\b",
    r"\b(docker|compose|deploy|server|service|systemd|nginx|caddy|kubernetes|k8s)\b",
    r"\b(api|endpoint|database|sql|redis|mongodb|postgres|schema|migration)\b",
    r"\b(refactor|implement|modify|edit|write code|run|execute|terminal|file)\b",
    r"(修改|修复|调试|报错|错误|代码|运行|执行|部署|接口|数据库|提交|分支|测试|文件|项目)",
)

_MULTI_STEP_PATTERNS = (
    r"\b(first|then|next|finally|step by step)\b",
    r"(先.+再|然后|接着|最后|分步骤|一步步)",
)


def _word_count(text: str) -> int:
    # Whitespace-delimited words for Latin text plus a rough CJK fallback.
    latin_words = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_words) + max(1, len(cjk_chars) // 2) if cjk_chars else len(latin_words)


def _enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(isinstance(cfg, Mapping) and cfg.get("enabled") is True)


def _cheap_model(cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cheap = cfg.get("cheap_model") if isinstance(cfg, Mapping) else None
    if isinstance(cheap, Mapping) and cheap.get("provider") and cheap.get("model"):
        return cheap
    return None


def decide_route(user_message: str, cfg: Mapping[str, Any] | None) -> RouteDecision:
    """Return whether a turn should use the primary or configured cheap model.

    The classifier is deliberately conservative: it only routes short,
    standalone, plain-text turns to cheap_model. Anything tool-like, code-like,
    long, multi-step, or context-heavy stays on primary.
    """
    if not isinstance(cfg, Mapping) or not _enabled(cfg):
        return RouteDecision("primary", "disabled")
    if _cheap_model(cfg) is None:
        return RouteDecision("primary", "missing_cheap_model")
    if not isinstance(user_message, str) or not user_message.strip():
        return RouteDecision("primary", "empty")

    text = user_message.strip()
    max_chars = int(cfg.get("max_simple_chars") or 160)
    max_words = int(cfg.get("max_simple_words") or 28)

    if len(text) > max_chars:
        return RouteDecision("primary", "too_long_chars")
    if _word_count(text) > max_words:
        return RouteDecision("primary", "too_long_words")

    lowered = text.lower()
    for pattern in _CODE_OR_COMPLEX_PATTERNS + _MULTI_STEP_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return RouteDecision("primary", "complex_marker")

    if text.count("?") + text.count("？") > 1:
        return RouteDecision("primary", "multi_question")

    return RouteDecision("cheap", "short_plain_text")
