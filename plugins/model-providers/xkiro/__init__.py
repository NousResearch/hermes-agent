"""xKiro provider profile.

xKiro exposes an OpenAI-compatible gateway for models from OpenAI, Anthropic,
xAI, Moonshot, and other upstream providers. Model IDs use xKiro's full
``vendor/model`` form, for example ``openai/gpt-5.6-luna``.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

_XKIRO_BASE = "https://api.xkiro.com/v1"
_XKIRO_MODELS_URL = f"{_XKIRO_BASE}/models"
_XKIRO_ENV = ("XKIRO_API_KEY", "XKIRO_BASE_URL")


def _fetch_xkiro_models(
    *,
    api_key: str | None = None,
    timeout: float = 8.0,
    base_url: str | None = None,
) -> list[str] | None:
    """Fetch xKiro's live model catalog, returning model IDs on success."""
    caller_base = (base_url or "").strip()
    models_url = (
        caller_base.rstrip("/") + "/models"
        if caller_base and caller_base.rstrip("/") != _XKIRO_BASE.rstrip("/")
        else _XKIRO_MODELS_URL
    )

    try:
        request = urllib.request.Request(models_url)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", _profile_user_agent())
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        models = payload.get("data", [])
        return [
            model["id"]
            for model in models
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
    except Exception as exc:
        logger.debug("fetch_models(xkiro): %s", exc)
        return None


class XKiroProfile(ProviderProfile):
    """xKiro's OpenAI-compatible chat-completions endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return _fetch_xkiro_models(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )


xkiro = XKiroProfile(
    name="xkiro",
    aliases=("xkiro-ai",),
    api_mode="chat_completions",
    env_vars=_XKIRO_ENV,
    display_name="xKiro",
    description="xKiro — multi-model API gateway",
    signup_url="https://xkiro.com/",
    base_url=_XKIRO_BASE,
    models_url=_XKIRO_MODELS_URL,
    # xKiro's catalog is intentionally live-only. Do not duplicate the live
    # catalog here: a static fallback would become stale as models change.
    fallback_models=(),
    default_aux_model="qwen/qwen3.5-flash:free",
)

class XKiroAnthropicProfile(XKiroProfile):
    """xKiro's Anthropic Messages-compatible endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        models = _fetch_xkiro_models(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        if models is None:
            return None
        return [model for model in models if model.startswith("anthropic/claude-")]


xkiro_anthropic = XKiroAnthropicProfile(
    name="xkiro-anthropic",
    aliases=("xkiro-claude",),
    api_mode="anthropic_messages",
    env_vars=("XKIRO_API_KEY", "XKIRO_ANTHROPIC_BASE_URL"),
    display_name="xKiro (Anthropic)",
    description="xKiro — multi-model API via Anthropic Messages",
    signup_url="https://xkiro.com/",
    base_url=_XKIRO_BASE,
    models_url=_XKIRO_MODELS_URL,
    fallback_models=(),
    default_aux_model="",
)


register_provider(xkiro)
register_provider(xkiro_anthropic)
