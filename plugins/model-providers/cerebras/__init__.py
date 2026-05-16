"""Cerebras provider profile.

Cerebras runs wafer-scale inference for open-weight models (e.g. Llama,
DeepSeek-R1-Distill, GPT-OSS) via an OpenAI-compatible endpoint.  Their
API accepts the standard Chat Completions schema but explicitly rejects
the ``reasoning_content`` field in replayed assistant messages, returning:

    HTTP 400: messages.N.assistant.reasoning_content:
    property '...' is unsupported

This profile strips ``reasoning_content`` from every assistant message
before the request goes out, which is the correct fix for sessions that
were initially started with a reasoning-capable provider (DeepSeek, Kimi,
MiMo, …) and later switched to Cerebras, or for sessions where the SDK
returned reasoning tokens that Cerebras does not accept on replay.

Configure in ``~/.hermes/config.yaml``::

    model: llama-4-scout-17b-16e-instruct   # or any Cerebras model
    provider: cerebras

Or point a custom endpoint at the Cerebras base URL and the URL-based
detection in ``AIAgent._strips_reasoning_content_from_api_messages``
handles the sanitization automatically for ``provider: custom`` users.
"""

from __future__ import annotations

import copy
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CerebrasProfile(ProviderProfile):
    """Cerebras wafer-scale inference — strips reasoning_content on replay."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove ``reasoning_content`` from assistant messages.

        Cerebras rejects the field with HTTP 400 when it appears in replayed
        history.  This belt-and-suspenders cleanup runs in the transport layer
        for sessions going through the profile path (``provider: cerebras``).
        The primary guard lives in
        ``AIAgent._strips_reasoning_content_from_api_messages`` / ``_copy_reasoning_content_for_api``
        which covers the ``provider: custom`` + Cerebras URL case as well.
        """
        sanitized = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "reasoning_content" in msg:
                msg = copy.copy(msg)
                del msg["reasoning_content"]
            sanitized.append(msg)
        return sanitized


cerebras = CerebrasProfile(
    name="cerebras",
    display_name="Cerebras",
    description="Cerebras wafer-scale chip inference (OpenAI-compatible)",
    signup_url="https://cerebras.ai",
    env_vars=("CEREBRAS_API_KEY",),
    base_url="https://api.cerebras.ai/v1",
    hostname="api.cerebras.ai",
    fallback_models=(
        "llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b",
        "llama3.1-8b",
        "llama3.1-70b",
        "deepseek-r1-distill-llama-70b",
        "qwen-3-32b",
    ),
)

register_provider(cerebras)
