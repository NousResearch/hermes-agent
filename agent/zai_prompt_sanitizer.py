"""Z.AI / Zhipu outbound prompt sanitization.

Z.ai/Zhipu Coding Plan returns a misleading HTTP 429 / code 1305 ("temporarily
overloaded") whenever the effective system prompt contains the exact phrase
``"Hermes Agent"``. This is a prompt-level block on Z.ai's side, not real
overload (see issues #47685, #53002).

This module sanitizes the **outbound API-message copy** at the request
boundary — it never mutates the cached or persisted system prompt. That way:

- resumed sessions that restore a branded prompt verbatim are still sanitized
  (the restore path returns early and bypasses assembly-time hooks);
- the byte-stability invariant enforced by
  ``tests/agent/test_system_prompt_restore.py`` (restored prompt must be
  byte-identical to the stored bytes, for prefix-cache warmth) is preserved;
- non-zai providers are byte-identical (the transformation is gated).

Only the single triggering phrase is rewritten. Unlike a full branding strip,
this preserves user customisation and assistant identity outside the exact
string Z.ai blocks on.
"""
from __future__ import annotations

from typing import Any

# The exact phrase Z.ai blocks on. Confirmed via controlled test:
# same key/endpoint/model/payload, only the system prompt differs —
# "Hermes Agent" → 0/5 pass (all 429/1305), "Hermes framework" → 5/5 pass.
_BLOCKED_PHRASE = "Hermes Agent"
_REPLACEMENT = "Hermes framework"


def is_zai_request(agent: Any) -> bool:
    """Return True if this request is routed to a Z.ai/Zhipu endpoint.

    Detection covers provider name, model slug, and base_url host so custom
    providers pointing at Z.ai are also caught.
    """
    provider = (getattr(agent, "provider", "") or "").lower()
    if provider in {"zai", "z-ai", "z.ai", "glm", "zhipu"}:
        return True
    model = (getattr(agent, "model", "") or "").lower()
    if model.startswith("glm-"):
        return True
    base_url = (
        getattr(agent, "base_url", "")
        or getattr(agent, "_base_url_lower", "")
        or ""
    )
    base_url = base_url.lower()
    return "api.z.ai" in base_url or "open.bigmodel.cn" in base_url


def sanitize_zai_system_content(content: Any) -> Any:
    """Rewrite the blocked phrase in a system-message content value.

    Handles both string content and the OpenAI content-block list form
    (``[{"type": "text", "text": "..."}]``). Non-string / non-list input is
    returned unchanged. Returns the same object type — callers assign the
    result back to a message copy.
    """
    if isinstance(content, str):
        if _BLOCKED_PHRASE in content:
            return content.replace(_BLOCKED_PHRASE, _REPLACEMENT)
        return content
    if isinstance(content, list):
        changed = False
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and _BLOCKED_PHRASE in text:
                    part = {**part, "text": text.replace(_BLOCKED_PHRASE, _REPLACEMENT)}
                    changed = True
            out.append(part)
        return out if changed else content
    return content


def sanitize_zai_api_messages(agent: Any, api_messages: list) -> list:
    """Return a sanitized copy of api_messages for Z.ai/Zhipu requests.

    Returns a new list (shallow copy of each message dict) so the caller's
    cached/history state is never mutated. Non-zai requests get the original
    list back unchanged (same reference) so there is zero overhead for other
    providers.

    Only ``role: system`` messages are touched — user/assistant/tool content
    is passed through verbatim.
    """
    if not is_zai_request(agent):
        return api_messages

    out = []
    for msg in api_messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            out.append(msg)
            continue
        content = msg.get("content")
        sanitized = sanitize_zai_system_content(content)
        if sanitized is content:
            out.append(msg)
        else:
            out.append({**msg, "content": sanitized})
    return out
