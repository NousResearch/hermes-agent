"""Request-local system-prompt policy for direct Z.AI endpoints."""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import urlparse


_ZAI_SYSTEM_PREFIX = (
    "You are a precise local AI coding and operations assistant. "
    "Answer the user's task directly. Use the provided tools and conversation "
    "context when available. Do not mention internal platform branding."
)

_BRANDING_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"You are Hermes Agent, an intelligent AI assistant created by Nous Research\.\s*", ""),
    (r"You run on Hermes Agent \(by Nous Research\)\.\s*", ""),
    (r"Hermes Agent", "the local assistant"),
    (r"Nous Research", "the platform provider"),
)


def is_zai_request(*, provider: str = "", model: str = "", base_url: str = "") -> bool:
    """Return whether a request targets the direct Z.AI API endpoint."""
    provider_lower = str(provider or "").strip().lower()
    if provider_lower in {"zai", "z-ai", "z.ai", "glm", "zhipu"}:
        return True

    del model  # Model family alone is not endpoint identity (local GLM is valid).
    try:
        hostname = (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        hostname = ""
    return hostname in {"api.z.ai", "open.bigmodel.cn"}


def sanitize_zai_system_prompt(text: str) -> str:
    """Return a neutralized Z.AI system prompt while preserving instructions."""
    output = text
    for pattern, replacement in _BRANDING_REPLACEMENTS:
        output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
    output = re.sub(r"\n{3,}", "\n\n", output).strip()
    if not output:
        return _ZAI_SYSTEM_PREFIX
    if output.startswith(_ZAI_SYSTEM_PREFIX):
        return output
    return f"{_ZAI_SYSTEM_PREFIX}\n\n{output}"


def _sanitize_content(content: Any) -> Any:
    if isinstance(content, str):
        return sanitize_zai_system_prompt(content)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                part["text"] = sanitize_zai_system_prompt(part["text"])
            elif isinstance(part.get("content"), str):
                part["content"] = sanitize_zai_system_prompt(part["content"])
    return content


def apply_zai_prompt_policy(
    messages: list[dict[str, Any]],
    *,
    provider: str = "",
    model: str = "",
    base_url: str = "",
) -> list[dict[str, Any]]:
    """Rewrite only the outbound message copy for direct Z.AI requests."""
    if not is_zai_request(provider=provider, model=model, base_url=base_url):
        return messages

    sanitized = list(messages)
    saw_system = False
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue
        saw_system = True
        cloned = copy.deepcopy(message)
        cloned["content"] = _sanitize_content(cloned.get("content"))
        sanitized[index] = cloned
    if not saw_system:
        sanitized.insert(0, {"role": "system", "content": _ZAI_SYSTEM_PREFIX})
    return sanitized
