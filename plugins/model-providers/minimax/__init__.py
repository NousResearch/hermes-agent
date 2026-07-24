"""MiniMax provider profiles (international + China).

The default API-key routes use anthropic_messages because their base URLs end
with /anthropic. Users can opt MiniMax-M3 into the OpenAI-compatible endpoint
with base_url=https://api.minimax.io/v1; that route needs MiniMax-specific
reasoning controls in extra_body.
"""

from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    if (parsed.hostname or "").lower() != "api.minimax.io":
        return False
    path = parsed.path.rstrip("/").lower()
    return path == "/v1"


def _is_minimax_m3(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized in {"minimax-m3", "minimax/minimax-m3"}


class MiniMaxProfile(ProviderProfile):
    """MiniMax — M3 OpenAI-compatible reasoning controls."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch MiniMax's live catalog with the provider's auth semantics.

        MiniMax's default Hermes route is the Anthropic-compatible endpoint,
        which expects ``x-api-key`` rather than ``Authorization: Bearer``.
        The base ``ProviderProfile`` fetcher is intentionally OpenAI-style, so
        use the shared probing helper with ``api_mode='anthropic_messages'``.
        """
        try:
            from hermes_cli.models import fetch_api_models

            return fetch_api_models(
                api_key,
                base_url or self.base_url,
                timeout=timeout,
                api_mode="anthropic_messages",
            )
        except Exception:
            return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit M3 reasoning controls for api.minimax.io/v1.

        MiniMax-M3's OpenAI-compatible endpoint keeps thinking inline unless
        ``reasoning_split`` is sent, so always request the split format on that
        route. ``thinking`` controls the M3 mode; Hermes' effort levels are not
        a MiniMax depth knob here, so they only select adaptive vs disabled.
        """
        if not _is_minimax_global_openai_base_url(base_url) or not _is_minimax_m3(model):
            return {}, {}

        extra_body: dict[str, Any] = {"reasoning_split": True}

        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, {}

        if reasoning_config is not None:
            extra_body["thinking"] = {"type": "adaptive"}

        return extra_body, {}


minimax = MiniMaxProfile(
    name="minimax",
    aliases=("mini-max",),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic",
    auth_type="api_key",
    default_aux_model="MiniMax-M3",
)

minimax_cn = MiniMaxProfile(
    name="minimax-cn",
    aliases=("minimax-china", "minimax_cn"),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",),
    base_url="https://api.minimaxi.com/anthropic",
    auth_type="api_key",
    default_aux_model="MiniMax-M3",
)

minimax_oauth = MiniMaxProfile(
    name="minimax-oauth",
    aliases=("minimax_oauth", "minimax-oauth-io"),
    api_mode="anthropic_messages",
    display_name="MiniMax (OAuth)",
    description="MiniMax via OAuth browser flow — no API key required",
    signup_url="https://api.minimax.io/",
    env_vars=(),  # OAuth — tokens in auth.json, not env
    base_url="https://api.minimax.io/anthropic",
    auth_type="oauth_external",
    default_aux_model="MiniMax-M2.7",
)

register_provider(minimax)
register_provider(minimax_cn)
register_provider(minimax_oauth)
