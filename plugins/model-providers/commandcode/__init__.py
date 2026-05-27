"""CommandCode provider profile.

CommandCode provides a unified API that fronts 20+ models from DeepSeek, Qwen,
Kimi, GLM, MiniMax, StepFun, Xiaomi Mimo, Google Gemini, and OpenAI GPT — all
accessible through either OpenAI-compatible chat completions or Anthropic
Messages endpoints from a single base URL and API key.

Two provider profiles are registered:

``commandcode``
    ``api_mode=chat_completions`` — standard OpenAI-compatible endpoint.
    Model prefix: ``deepseek/deepseek-v4-pro``, ``Qwen/Qwen3.7-Max``, etc.

``commandcode-anthropic``
    ``api_mode=anthropic_messages`` — Anthropic Messages API-compatible.
    Model names: ``claude-sonnet-4-6``, ``claude-opus-4-7``,
    ``claude-haiku-4-5-20251001``.

Both use the same ``COMMANDCODE_API_KEY`` env var and
``https://api.commandcode.ai/provider/v1`` base URL.  The
``commandcode-anthropic`` profile relies on ``agent/anthropic_adapter.py``
recognizing the ``api.commandcode.ai`` hostname for Bearer auth (the
CommandCode /anthropic endpoint uses ``Authorization: Bearer``, not
Anthropic's native ``x-api-key`` header).
"""

from __future__ import annotations

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────
_COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"
_COMMANDCODE_MODELS_URL = f"{_COMMANDCODE_BASE}/models"
_COMMANDCODE_ENV = ("COMMANDCODE_API_KEY",)


def _fetch_commandcode_models(
    timeout: float = 10.0,
) -> list[str] | None:
    """Fetch the live model list from the CommandCode /models endpoint.

    Returns a flat list of model IDs or None on failure.
    No auth required — the public models endpoint is open.
    """
    try:
        req = urllib.request.Request(_COMMANDCODE_MODELS_URL)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        # Response shape: {"object": "list", "data": [{"id": "..."}, ...]}
        return [
            m["id"]
            for m in data.get("data", [])
            if isinstance(m, dict) and "id" in m
        ]
    except Exception as exc:
        logger.debug("fetch_models(commandcode): %s", exc)
        return None


# ── Chat Completions profile ──────────────────────────────────────────────────

class CommandCodeProfile(ProviderProfile):
    """CommandCode — OpenAI-compatible chat completions endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from the public CommandCode /models endpoint."""
        return _fetch_commandcode_models(timeout=timeout)


commandcode = CommandCodeProfile(
    name="commandcode",
    aliases=("commandcode-chat",),
    api_mode="chat_completions",
    env_vars=_COMMANDCODE_ENV,
    display_name="CommandCode",
    description="CommandCode — 20+ models via OpenAI-compatible API",
    signup_url="https://commandcode.ai/",
    base_url=_COMMANDCODE_BASE,
    models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=(
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "Qwen/Qwen3.7-Max",
        "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6",
        "zai-org/GLM-5.1",
        "MiniMaxAI/MiniMax-M2.7",
        "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro",
        "google/gemini-3.5-flash",
        "gpt-5.5",
    ),
    default_aux_model="deepseek/deepseek-v4-flash",
)


# ── Anthropic Messages profile ────────────────────────────────────────────────

class CommandCodeAnthropicProfile(ProviderProfile):
    """CommandCode — Anthropic Messages API-compatible endpoint.

    Uses Bearer auth (same API key), not Anthropic's native x-api-key header.
    ``agent/anthropic_adapter.py`` must recognize ``api.commandcode.ai``
    as a Bearer-auth domain for this to work.
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from the public CommandCode /models endpoint.

        Filter to Anthropic-family models only (claude-*).
        """
        all_models = _fetch_commandcode_models(timeout=timeout)
        if all_models is None:
            return None
        return [m for m in all_models if m.startswith("claude-")]


commandcode_anthropic = CommandCodeAnthropicProfile(
    name="commandcode-anthropic",
    aliases=("commandcode-claude",),
    api_mode="anthropic_messages",
    env_vars=_COMMANDCODE_ENV,
    display_name="CommandCode (Anthropic)",
    description="CommandCode — Claude models via Anthropic Messages API",
    signup_url="https://commandcode.ai/",
    base_url=_COMMANDCODE_BASE,
    models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=(
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
    ),
    default_aux_model="claude-haiku-4-5-20251001",
)


# ── Registration ──────────────────────────────────────────────────────────────
register_provider(commandcode)
register_provider(commandcode_anthropic)
