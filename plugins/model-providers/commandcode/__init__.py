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

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict, dict]:
        """Apply the native DeepSeek reasoning controls for DeepSeek-family ids.

        CommandCode fronts DeepSeek with vendor-prefixed ids
        (``deepseek/deepseek-v4-flash``). DeepSeek V4+ defaults to thinking
        mode when the ``thinking`` field is omitted, so without an explicit
        wire control a Hermes ``/reasoning none`` changes the session state
        but not the actual request — the turn sits in
        ``reflecting.../brainstorming...`` for minutes (#95232). Strip the
        vendor prefix and delegate to the native DeepSeek profile's logic
        (extra_body.thinking + reasoning_effort mapping); other CommandCode
        model families keep the base no-op behavior.
        """
        m = (model or "").strip()
        if not m.lower().startswith("deepseek/") or len(m) <= len("deepseek/"):
            return {}, {}
        try:
            from plugins.model_providers.deepseek import deepseek as _deepseek_profile
        except ImportError:
            # The bundled-plugin loader tolerates a plugin failing to load
            # (it pops the half-registered module and continues), so the
            # shim may be absent. Degrade to the pre-fix no-op instead of
            # crashing every DeepSeek-routed CommandCode turn.
            logger.warning(
                "DeepSeek provider plugin unavailable; CommandCode cannot "
                "forward reasoning controls (#95232)",
                exc_info=True,
            )
            return {}, {}
        return _deepseek_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config,
            model=m.split("/", 1)[1],
            **context,
        )


commandcode = CommandCodeProfile(
    name="commandcode", aliases=("commandcode-chat",), api_mode="chat_completions",
    # Same key as the anthropic profile; distinct base-URL override vars so each
    # profile renders its own card on the desktop Keys tab (rows keyed by env var).
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_BASE_URL"),
    display_name="CommandCode", description="CommandCode — 20+ models via OpenAI-compatible API",
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
    display_name="CommandCode (Anthropic)",
    description="CommandCode — Claude models via Anthropic Messages API",
    signup_url="https://commandcode.ai/", base_url=_COMMANDCODE_BASE, models_url=_COMMANDCODE_MODELS_URL,
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"),
    default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(commandcode)
register_provider(commandcode_anthropic)
