"""MiniMax provider profiles (international, China, OAuth).

Default routes use anthropic_messages (base URLs end in /anthropic). Users can
opt MiniMax-M3 into the OpenAI-compatible https://api.minimax.io/v1 route,
which needs MiniMax-specific reasoning controls in extra_body.
"""

from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import MatchingPrefixModelIdsMixin


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").lower() == "api.minimax.io" and parsed.path.rstrip("/").lower() == "/v1"


class MiniMaxProfile(MatchingPrefixModelIdsMixin, ProviderProfile):
    """MiniMax — M3 OpenAI-compatible reasoning controls."""

    model_prefix_exclusions = ("mini-max", "minimax-oauth-io")

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        base_url: str | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """M3 on api.minimax.io/v1 keeps thinking inline unless ``reasoning_split``
        is sent; effort levels only select adaptive vs disabled ``thinking``."""
        is_m3 = str(model or "").strip().lower() in {"minimax-m3", "minimax/minimax-m3"}
        if not _is_minimax_global_openai_base_url(base_url) or not is_m3:
            return {}, {}
        extra_body: dict[str, Any] = {"reasoning_split": True}
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_config is not None:
            extra_body["thinking"] = {"type": "adaptive"}
        return extra_body, {}


minimax = MiniMaxProfile(
    name="minimax", aliases=("mini-max",), display_name="MiniMax",
    description="MiniMax (Global direct API)", signup_url="https://www.minimax.io/",
    api_mode="anthropic_messages", env_vars=("MINIMAX_API_KEY",), base_url="https://api.minimax.io/anthropic",
    base_url_env_var="MINIMAX_BASE_URL", auth_type="api_key", default_aux_model="MiniMax-M3",
)

minimax_cn = MiniMaxProfile(
    name="minimax-cn", aliases=("minimax-china", "minimax_cn"), display_name="MiniMax (China)",
    description="MiniMax China (Domestic direct API)", signup_url="https://www.minimaxi.com/",
    api_mode="anthropic_messages", env_vars=("MINIMAX_CN_API_KEY",),
    base_url="https://api.minimaxi.com/anthropic", base_url_env_var="MINIMAX_CN_BASE_URL", auth_type="api_key",
    default_aux_model="MiniMax-M3",
)

minimax_oauth = MiniMaxProfile(
    name="minimax-oauth", aliases=("minimax_oauth", "minimax-portal", "minimax-global", "minimax-oauth-io"), api_mode="anthropic_messages",
    display_name="MiniMax (OAuth)", description="MiniMax via OAuth browser flow — no API key required",
    signup_url="https://api.minimax.io/",
    env_vars=(),  # OAuth — tokens in auth.json, not env
    base_url="https://api.minimax.io/anthropic", auth_type="oauth_external", default_aux_model="MiniMax-M2.7",
)

register_provider(minimax)
register_provider(minimax_cn)
register_provider(minimax_oauth)
