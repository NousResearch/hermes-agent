"""CommandCode provider profiles: ``commandcode`` (chat_completions) and
``commandcode-anthropic`` (anthropic_messages, Bearer auth — see
``agent/anthropic_adapter.py``). Same key and base URL for both."""

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

_COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"
_COMMANDCODE_MODELS_URL = f"{_COMMANDCODE_BASE}/models"


class CommandCodeProfile(ProviderProfile):
    """CommandCode — OpenAI-compatible chat completions endpoint."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Public (unauthenticated) /models endpoint. The picker passes base_url
        unconditionally, so only a value differing from the default is a custom endpoint."""
        caller_base = (base_url or "").strip().rstrip("/")
        custom = caller_base and caller_base != _COMMANDCODE_BASE
        models_url = caller_base + "/models" if custom else _COMMANDCODE_MODELS_URL
        try:
            req = urllib.request.Request(models_url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", _profile_user_agent())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(commandcode): %s", exc)
            return None


class CommandCodeAnthropicProfile(CommandCodeProfile):
    """CommandCode — Anthropic Messages API-compatible endpoint."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Public /models endpoint, filtered to Anthropic-family models."""
        all_models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        return None if all_models is None else [m for m in all_models if m.startswith("claude-")]


commandcode = CommandCodeProfile(
    name="commandcode", aliases=("commandcode-chat", "commandcode-api"), api_mode="chat_completions",
    # Same key as the anthropic profile; distinct base-URL override vars so each
    # profile renders its own card on the desktop Keys tab (rows keyed by env var).
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_BASE_URL"),
    display_name="Command Code (API Key)", description="Command Code — 20+ models via direct API key",
    signup_url="https://commandcode.ai/", base_url=_COMMANDCODE_BASE, models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=(
        "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "Qwen/Qwen3.7-Max", "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6", "zai-org/GLM-5.1", "MiniMaxAI/MiniMax-M2.7", "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro", "google/gemini-3.5-flash", "gpt-5.5",
    ),
    default_aux_model="deepseek/deepseek-v4-flash",
)

commandcode_anthropic = CommandCodeAnthropicProfile(
    name="commandcode-anthropic", aliases=("commandcode-claude",), api_mode="anthropic_messages",
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_ANTHROPIC_BASE_URL"),
    display_name="Command Code (Anthropic)",
    description="Command Code — Claude models via Anthropic Messages API",
    signup_url="https://commandcode.ai/", base_url=_COMMANDCODE_BASE, models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"),
    default_aux_model="claude-haiku-4-5-20251001",
)


class CommandCodeOAuthProfile(CommandCodeProfile):
    """Command Code OAuth profile."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if not models:
            return list(self.fallback_models)
        free = ["meituan/LongCat-2.0:free", "poolside/laguna-s-2.1-free"]
        for f in reversed(free):
            if f in models:
                models.remove(f)
            models.insert(0, f)
        return models


commandcode_oauth = CommandCodeOAuthProfile(
    name="commandcode-oauth", aliases=("command-code", "commandcode-auth"), api_mode="commandcode_alpha",
    env_vars=(),
    display_name="Command Code OAuth",
    description="Command Code — browser OAuth or CLI account (~/.commandcode/auth.json)",
    signup_url="https://commandcode.ai/", base_url="https://api.commandcode.ai", models_url=_COMMANDCODE_MODELS_URL,
    auth_type="oauth_external",
    fallback_models=(
        "meituan/LongCat-2.0:free", "poolside/laguna-s-2.1-free",
        "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro", "Qwen/Qwen3.7-Max", "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6", "zai-org/GLM-5.1", "MiniMaxAI/MiniMax-M2.7", "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro", "google/gemini-3.5-flash", "gpt-5.5",
    ),
    default_aux_model="meituan/LongCat-2.0:free",
)

register_provider(commandcode)
register_provider(commandcode_anthropic)
register_provider(commandcode_oauth)
