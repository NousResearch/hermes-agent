"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Canonical model-emitted control token for intentional silence.
SILENT_REPLY_TOKEN = "NO_REPLY"

# Exact whole-response markers that mean "the agent intentionally chose not to
# reply".  Keep this list small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})

_HOME_PATH_RE = re.compile(
    r"(?P<path>(?:~|/home/[^/\s`\"'<>，。；、,;:!?！？)）\]}]+)"
    r"(?:/[^\s`\"'<>，。；、,;:!?！？)）\]}]*)?)"
)
_MEDIA_PREFIX_RE = re.compile(r"MEDIA:\s*[`\"']?$", re.IGNORECASE)
_LOCAL_PATH_PLACEHOLDER = "[local file]"
_MODEL_IDENTITY_PLACEHOLDER = "当前大模型"
_MODEL_SERVICE_PLACEHOLDER = "当前模型服务"
_PROVIDER_PLACEHOLDER = "模型服务商"
_MODEL_IDENTITY_REPLACEMENTS = (
    (
        re.compile(
            r"\b(?:gpt[-_ ]?(?:5(?:\.5)?|4(?:\.1|o)?|4|3\.5)|o[134](?:[-_ ]?(?:mini|pro))?)\b",
            re.IGNORECASE,
        ),
        _MODEL_IDENTITY_PLACEHOLDER,
    ),
    (
        re.compile(r"\b(?:claude|gemini|grok|llama|qwen|deepseek)[-_ ]?[A-Za-z0-9._-]*\b", re.IGNORECASE),
        _MODEL_IDENTITY_PLACEHOLDER,
    ),
    (
        re.compile(r"\bopenai[-_ ]codex\b|\bcodex[-_ ]?(?:app[-_ ]?server|runtime|acp)\b|\bcodex_app_server\b", re.IGNORECASE),
        _MODEL_SERVICE_PLACEHOLDER,
    ),
    (re.compile(r"\bhermes(?:[-_ ]?agent)?\b", re.IGNORECASE), _MODEL_SERVICE_PLACEHOLDER),
    (re.compile(r"\bcodex\b", re.IGNORECASE), _MODEL_SERVICE_PLACEHOLDER),
    (re.compile(r"\bopenai\b|\banthropic\b|\bgoogle\b", re.IGNORECASE), _PROVIDER_PLACEHOLDER),
    (
        re.compile(r"https?://chatgpt\.com/backend-api/codex[^\s`\"'<>]*", re.IGNORECASE),
        _MODEL_SERVICE_PLACEHOLDER,
    ),
)
_MODEL_IDENTITY_QUERY_REPLY = (
    "我是当前系统提供的 AI 助手，具体模型、服务商、运行环境和程序细节不对外展示。"
)
_USER_VISIBLE_RUNTIME_ERROR_REPLY = (
    "抱歉，这次模型服务调用失败了，内部错误日志已隐藏。请稍后重试。"
)
_RAW_RUNTIME_ERROR_MARKERS = (
    "turn ended status=failed",
    "stderr (last",
    "invalid_request_error",
    "cloudflare",
    "cf_chl",
    "<!doctype html",
    "<html",
)
_MODEL_IDENTITY_QUERY_PATTERNS = (
    re.compile(
        r"(你|你们|当前|现在|这个|本助手|机器人|bot).{0,12}"
        r"(什么|哪个|哪种|哪一个|用的|使用|基于|基座|后端|底层|运行).{0,16}"
        r"(模型|大模型|程序|系统|服务|provider|backend|runtime|agent)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(模型|大模型|程序|系统|服务|provider|backend|runtime|agent).{0,16}"
        r"(什么|哪个|哪种|哪一个|名字|名称|版本|后端|底层).{0,12}"
        r"(你|你们|当前|现在|这个|本助手|机器人|bot)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what|which)\b.{0,24}\b(model|provider|backend|runtime|agent)\b.{0,24}\b(are you|you use|running)\b",
        re.IGNORECASE,
    ),
)


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def is_intentional_silence_response(response: Any) -> bool:
    """Return True only when ``response`` is exactly a silence marker.

    Substantive prose that merely mentions ``NO_REPLY`` or ``[SILENT]`` must be
    delivered normally.  A blank response is also not silence; blank output is
    handled by the empty-response failure path.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    if len(stripped) > 64:
        return False
    return _canonical_silence_candidate(stripped) in LIVE_GATEWAY_SILENT_MARKERS


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


def redact_user_visible_local_paths(text: Any) -> str:
    """Redact local Hermes/Codex paths from user-visible gateway text.

    ``MEDIA:/path`` directives are intentionally preserved so downstream
    platform adapters can still extract and deliver attachments. Callers that
    have already extracted media can run this on the cleaned message body.
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)
    if not text:
        return text

    home = os.path.expanduser("~")
    home_prefix = f"{home}/" if home and home != "~" else ""

    def repl(match: re.Match[str]) -> str:
        path = match.group("path")
        prefix = text[max(0, match.start() - 32):match.start()]
        if _MEDIA_PREFIX_RE.search(prefix):
            return path
        expanded = os.path.expanduser(path)
        if home_prefix and expanded.startswith(home_prefix):
            return _LOCAL_PATH_PLACEHOLDER
        return path

    return _HOME_PATH_RE.sub(repl, text)


def redact_user_visible_model_identity(text: Any) -> str:
    """Redact concrete model/provider/runtime names from user-visible text."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    if not text:
        return text
    protected_spans = [match.span() for match in _HOME_PATH_RE.finditer(text)]
    if protected_spans:
        chunks: list[str] = []
        last = 0
        for start, end in protected_spans:
            if start > last:
                chunks.append(redact_user_visible_model_identity(text[last:start]))
            chunks.append(text[start:end])
            last = end
        if last < len(text):
            chunks.append(redact_user_visible_model_identity(text[last:]))
        return "".join(chunks)

    redacted = text
    for pattern, replacement in _MODEL_IDENTITY_REPLACEMENTS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def collapse_user_visible_runtime_error(text: Any) -> str:
    """Replace raw runtime failure dumps with a short user-facing notice."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    if not text:
        return text

    normalized = text.strip().lower()
    if normalized.startswith("⚠️ turn ended status=failed"):
        return _USER_VISIBLE_RUNTIME_ERROR_REPLY
    if "turn ended status=failed" in normalized and any(
        marker in normalized for marker in _RAW_RUNTIME_ERROR_MARKERS[1:]
    ):
        return _USER_VISIBLE_RUNTIME_ERROR_REPLY
    if "cloudflare" in normalized and ("cf_chl" in normalized or "<html" in normalized):
        return _USER_VISIBLE_RUNTIME_ERROR_REPLY
    return text


def sanitize_user_visible_gateway_text(text: Any, *, redact_paths: bool = True) -> str:
    """Apply hard outbound redaction before text reaches chat platforms."""
    redacted = collapse_user_visible_runtime_error(text)
    redacted = redact_user_visible_local_paths(redacted) if redact_paths else redacted
    redacted = redact_user_visible_model_identity(redacted)
    return redacted


def is_model_identity_query(text: Any) -> bool:
    """Return True for user questions about this assistant's model/runtime."""
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    if len(normalized) > 160:
        return False
    return any(pattern.search(normalized) for pattern in _MODEL_IDENTITY_QUERY_PATTERNS)


def model_identity_private_reply() -> str:
    """Stable reply for model/runtime identity questions."""
    return _MODEL_IDENTITY_QUERY_REPLY
