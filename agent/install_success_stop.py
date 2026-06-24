"""Success-stop helpers for low-token install tasks."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


_INSTALL_INTENT_RE = re.compile(
    r"(?:安装|帮我装|装一下|\binstall\b|\bsetup\b)",
    re.IGNORECASE,
)
_INSTALL_SOURCE_RE = re.compile(
    r"(?:github\.com|gitlab\.com|bitbucket\.org|\.dmg\b|\.pkg\b|"
    r"\bbrew\b|\bnpm\b|\bpip\b|\bgo install\b)",
    re.IGNORECASE,
)

_ACQUIRE_RE = re.compile(
    r"(?:github\.com|git clone|curl\b|wget\b|download(?:ed|ing)?|"
    r"release|\.dmg\b|\.pkg\b|\bbrew install\b|\bnpm install\b|"
    r"\bpip install\b|\bgo install\b|获取|下载)",
    re.IGNORECASE,
)
_INSTALL_RE = re.compile(
    r"(?:successfully installed|already installed|\binstalled\b|"
    r"\bnpm install\b|\bpip install\b|\bbrew install\b|\bgo install\b|"
    r"copied|/applications|安装完成|已安装|已复制)",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(
    r"(?:\b--version\b|\bversion\b|\bwhich\b|\btest -[ef]\b|"
    r"\bls\b|verified|verification|校验|验证|存在|found)",
    re.IGNORECASE,
)
_FAIL_RE = re.compile(
    r"(?:exit_code[\"']?\s*:\s*[1-9]|failed|failure|"
    r"permission denied|not found|no such file|失败|错误|未找到)",
    re.IGNORECASE,
)


def is_install_task_message(platform: str, budget_key: str, message: str) -> bool:
    """Return True for ordinary Feishu install requests.

    Deep diagnostic turns are intentionally excluded so users keep the larger
    investigation path when they opt into it.
    """
    if (platform or "").strip().lower() != "feishu":
        return False
    if (budget_key or "").strip().lower() == "feishu_deep":
        return False
    text = message or ""
    return bool(_INSTALL_INTENT_RE.search(text) and _INSTALL_SOURCE_RE.search(text))


def install_success_stop_response(messages: list[dict[str, Any]], current_turn_user_idx: int) -> str | None:
    """Build a concise final response once an install task is verified.

    The detector is deliberately conservative: it needs acquisition, install,
    and verification evidence, and it refuses to fire if the recent evidence
    contains a clear failure marker.
    """
    evidence = _current_turn_tool_evidence(messages, current_turn_user_idx)
    if not evidence:
        return None

    combined = "\n".join(evidence)
    if _FAIL_RE.search(combined):
        return None

    has_acquire = bool(_ACQUIRE_RE.search(combined))
    has_install = bool(_INSTALL_RE.search(combined))
    has_verify = _has_verification_evidence(evidence)
    if not (has_acquire and has_install and has_verify):
        return None

    detail = _extract_install_detail(combined)
    if detail:
        return (
            f"已完成安装并通过验证：{detail}\n\n"
            "我已在验证通过后停止继续探测，避免继续消耗 token。"
        )
    return (
        "已完成安装并通过验证。\n\n"
        "我已在验证通过后停止继续探测，避免继续消耗 token。"
    )


def _current_turn_tool_evidence(messages: list[dict[str, Any]], current_turn_user_idx: int) -> list[str]:
    start = max(0, current_turn_user_idx + 1)
    chunks: list[str] = []
    for msg in messages[start:]:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    chunks.append(str(fn.get("name") or ""))
                    chunks.append(str(fn.get("arguments") or ""))
        elif msg.get("role") == "tool":
            chunks.append(_stringify_tool_content(msg.get("content")))
    return [chunk for chunk in chunks if chunk]


def _stringify_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _has_verification_evidence(chunks: Iterable[str]) -> bool:
    for chunk in chunks:
        text = chunk or ""
        lower = text.lower()
        if _VERIFY_RE.search(text) and not re.search(r"exit_code[\"']?\s*:\s*[1-9]", lower):
            return True
        if re.search(r"exit_code[\"']?\s*:\s*0", lower) and _VERIFY_RE.search(text):
            return True
    return False


def _extract_install_detail(text: str) -> str:
    candidates = []
    for pattern in (
        r"(/Applications/[^\s\"']+)",
        r"(v?\d+(?:\.\d+){1,3}[^\s,;)]*)",
        r"(https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    ):
        match = re.search(pattern, text)
        if match:
            candidates.append(match.group(1).strip())
    return " · ".join(list(dict.fromkeys(candidates))[:2])
