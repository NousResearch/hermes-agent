"""NVIDIA NIM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile
from utils import base_url_host_matches


class NvidiaProfile(ProviderProfile):
    """NVIDIA NIM — owns its reasoning wire-shape contract."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        base_url: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """NIM cloud (integrate.api.nvidia.com) rejects ``reasoning`` outright.

        Every model on the hosted endpoint returns HTTP 400
        ``Unsupported parameter(s): `reasoning``` for ANY value — including
        ``{"enabled": false}`` — so the generic fallback in
        ``agent/auxiliary_client._build_call_kwargs`` can never succeed there
        whenever a reasoning_config resolves (e.g. a global
        ``agent.reasoning_effort`` flowing into a MoA aggregator/advisor or
        compression call). Omit the field entirely on the cloud route.

        Local NIM endpoints (``NVIDIA_BASE_URL`` override) keep the previous
        generic-fallback behavior, replicated verbatim here — overriding this
        method marks the profile as reasoning-owning
        (``profile_handles_reasoning`` in the caller), which would otherwise
        suppress the fallback for local endpoints too.
        """
        if not (reasoning_config and isinstance(reasoning_config, dict)):
            return {}, {}
        if base_url_host_matches(str(base_url or ""), "integrate.api.nvidia.com"):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            return {"reasoning": {"enabled": False}}, {}
        effort = reasoning_config.get("effort") or "medium"
        return {"reasoning": {"enabled": True, "effort": effort}}, {}


nvidia = NvidiaProfile(
    name="nvidia",
    aliases=("nvidia-nim",),
    env_vars=("NVIDIA_API_KEY",),
    display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference",
    signup_url="https://build.nvidia.com/",
    fallback_models=(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.3-70b-instruct",
    ),
    base_url="https://integrate.api.nvidia.com/v1",
    default_max_tokens=16384,
)

register_provider(nvidia)
