"""MiniMax provider profiles (international, China, OAuth).

Default routes use anthropic_messages (base URLs end in /anthropic). Users can
opt MiniMax-M3 or M3.1 into the OpenAI-compatible https://api.minimax.io/v1
route; M3 uses a thinking toggle, while M3.1 accepts reasoning_effort.
"""

from typing import Any
from urllib.parse import urlparse

from agent.anthropic_endpoints import _model_name_is_minimax_adaptive
from agent.reasoning_effort import MINIMAX_M31_EFFORTS, clamp_effort, requested_effort
from providers import register_provider
from providers.base import ProviderProfile


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").lower() == "api.minimax.io" and parsed.path.rstrip("/").lower() == "/v1"


class MiniMaxProfile(ProviderProfile):
    """MiniMax — model-specific OpenAI-compatible reasoning controls."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        base_url: str | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """M3 needs ``reasoning_split`` and a thinking toggle; M3.1 already splits
        reasoning and accepts graded effort, but rejects disabled thinking."""
        if not _is_minimax_global_openai_base_url(base_url):
            return {}, {}
        if _model_name_is_minimax_adaptive(model):
            # M3.1 rejects both disabled thinking and effort=none. Omit those
            # requests, as on the Anthropic route; the model keeps thinking.
            effort = requested_effort(reasoning_config)
            clamped = clamp_effort(None if effort == "none" else effort, MINIMAX_M31_EFFORTS)
            return ({}, {"reasoning_effort": clamped}) if clamped in MINIMAX_M31_EFFORTS else ({}, {})
        is_m3 = str(model or "").strip().lower() in {"minimax-m3", "minimax/minimax-m3"}
        if not is_m3:
            return {}, {}
        extra_body: dict[str, Any] = {"reasoning_split": True}
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_config is not None:
            extra_body["thinking"] = {"type": "adaptive"}
        return extra_body, {}


minimax = MiniMaxProfile(
    name="minimax", aliases=("mini-max",), api_mode="anthropic_messages", env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic", auth_type="api_key", default_aux_model="MiniMax-M3",
)

minimax_cn = MiniMaxProfile(
    name="minimax-cn", aliases=("minimax-china", "minimax_cn"), api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",), base_url="https://api.minimaxi.com/anthropic", auth_type="api_key",
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
