"""Z.AI / GLM prompt policy.

Z.AI GLM-5.x has repeatedly returned provider-side 429/code-1305 style
failures when the normal Hermes-branded system prompt is sent verbatim.  This
module is an API-boundary sanitizer: it rewrites only the per-request copy of
messages sent to Z.AI, never the cached prompt or conversation history.
"""
from __future__ import annotations

import copy
import re
from typing import Any


_ZAI_SYSTEM_PREFIX = (
    "You are a precise local AI coding and operations assistant. "
    "Answer the user's task directly. Use the provided tools and conversation "
    "context when available. Do not mention internal platform branding."
)

_BRANDING_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"You are Hermes Agent, an intelligent AI assistant created by Nous Research\.\s*", ""),
    (r"You run on Hermes Agent \(by Nous Research\)\.\s*", ""),
    (r"Hermes Agent", "the local assistant"),
    (r"hermes-agent", "local-agent"),
    (r"\bHermes\b", "the local assistant"),
    (r"Nous Research", "the platform provider"),
    (r"\bHERMES_[A-Z0-9_]+\b", "LOCAL_AGENT_ENV"),
)


def is_zai_request(agent: Any) -> bool:
    """Return True for any request routed to direct Z.AI/GLM endpoints."""
    provider = (getattr(agent, "provider", "") or "").lower()
    model = (getattr(agent, "model", "") or "").lower()
    base_url = (getattr(agent, "base_url", "") or getattr(agent, "_base_url_lower", "") or "").lower()
    return (
        provider in {"zai", "z-ai", "z.ai", "glm", "zhipu"}
        or model.startswith("glm-")
        or "api.z.ai" in base_url
        or "open.bigmodel.cn" in base_url
    )


def sanitize_zai_system_prompt(text: str) -> str:
    """Return the special Z.AI-safe system prompt variant.

    Keep operational/tool instructions, but strip/rewrite Hermes/Nous branding
    and prepend a stable neutral instruction. This preserves capability while
    avoiding the provider-triggering brand strings.
    """
    if not isinstance(text, str):
        return text
    out = text
    for pattern, repl in _BRANDING_REPLACEMENTS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if not out:
        return _ZAI_SYSTEM_PREFIX
    if out.startswith(_ZAI_SYSTEM_PREFIX):
        return out
    return f"{_ZAI_SYSTEM_PREFIX}\n\n{out}"


def _sanitize_content(content: Any) -> Any:
    if isinstance(content, str):
        return sanitize_zai_system_prompt(content)
    if isinstance(content, list):
        new_content = []
        for part in content:
            if isinstance(part, dict):
                item = part.copy()
                if isinstance(item.get("text"), str):
                    item["text"] = sanitize_zai_system_prompt(item["text"])
                elif isinstance(item.get("content"), str):
                    item["content"] = sanitize_zai_system_prompt(item["content"])
                new_content.append(item)
            else:
                new_content.append(part)
        return new_content
    return content


def apply_zai_special_prompt(agent: Any, api_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Z.AI special prompt policy to an API-message copy.

    The function returns a new list/dicts to avoid mutating cached prompt state
    or conversation history. It intentionally affects all direct Z.AI/GLM
    requests, not only council calls.

    Delegate subagents are special: their task contract lives in
    ``ephemeral_system_prompt`` while the normal cached system prompt can be a
    huge Hermes/persona/tool-policy prefix. GLM-5.x is fragile to that prefix,
    and the delegated task prompt must be the authority for the child. For
    Z.AI-routed subagents, replace the system prompt copy with a compact neutral
    prefix + the child task prompt instead of merely appending/sanitizing the
    full parent-derived prompt.
    """
    if not is_zai_request(agent):
        return api_messages

    platform = (getattr(agent, "platform", "") or "").lower()
    ephem = getattr(agent, "ephemeral_system_prompt", None)
    if platform == "subagent" and isinstance(ephem, str) and ephem.strip():
        child_system = sanitize_zai_system_prompt(ephem)
        new_messages = [copy.deepcopy(msg) for msg in api_messages if msg.get("role") != "system"]
        return [{"role": "system", "content": child_system}] + new_messages

    new_messages: list[dict[str, Any]] = []
    saw_system = False
    for msg in api_messages:
        cloned = copy.deepcopy(msg)
        if cloned.get("role") == "system":
            saw_system = True
            cloned["content"] = _sanitize_content(cloned.get("content"))
        new_messages.append(cloned)
    if not saw_system:
        return [{"role": "system", "content": _ZAI_SYSTEM_PREFIX}] + new_messages
    return new_messages
