"""流式文本解析 — reasoning 标签提取与最终回答清理."""

from __future__ import annotations

import re

REASONING_PREFIX = "Reasoning:\n"

_REASONING_TAG = r"(?:think(?:ing)?|thought|antthinking)"
_REASONING_TAG_RE = re.compile(r"<\s*(/?)\s*" + _REASONING_TAG + r"\s*>", re.IGNORECASE)
_REASONING_OPEN_RE = re.compile(r"<\s*" + _REASONING_TAG + r"\s*>", re.IGNORECASE)
_REASONING_CLOSE_RE = re.compile(r"<\s*/\s*" + _REASONING_TAG + r"\s*>", re.IGNORECASE)


def split_reasoning_text(text: str | None) -> dict[str, str | None]:
    if not isinstance(text, str) or not text.strip():
        return {}
    trimmed = text.strip()
    if trimmed.startswith(REASONING_PREFIX) and len(trimmed) > len(REASONING_PREFIX):
        return {"reasoning_text": _clean_reasoning_prefix(trimmed)}
    tagged = extract_thinking_content(text)
    stripped = strip_reasoning_tags(text)
    if not tagged and stripped == text:
        return {"answer_text": text}
    return {
        "reasoning_text": tagged or None,
        "answer_text": stripped or None,
    }


def extract_thinking_content(text: str) -> str:
    if not text:
        return ""
    result = ""
    last_index = 0
    in_thinking = False
    for match in _REASONING_TAG_RE.finditer(text):
        idx = match.start()
        if in_thinking:
            result += text[last_index:idx]
        in_thinking = match.group(1) != "/"
        last_index = match.end()
    if in_thinking:
        result += text[last_index:]
    return result.strip()


def strip_reasoning_tags(text: str) -> str:
    result = _REASONING_OPEN_RE.sub(
        lambda _: "",
        _REASONING_CLOSE_RE.sub("", text),
    )
    result = re.sub(
        r"<\s*" + _REASONING_TAG + r"\s*>[\s\S]*?<\s*/\s*" + _REASONING_TAG + r"\s*>",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"<\s*" + _REASONING_TAG + r"\s*>[\s\S]*$",
        "",
        result,
        flags=re.IGNORECASE,
    )
    if result.strip().startswith(REASONING_PREFIX):
        result = ""
    return result


def _clean_reasoning_prefix(text: str) -> str:
    cleaned = re.sub(r"^Reasoning:\s*", "", text, flags=re.IGNORECASE)
    cleaned = "\n".join(
        line.replace("_", "") if line.startswith("_") and line.endswith("_") else line for line in cleaned.split("\n")
    )
    return cleaned.strip()
