"""Routing policy for auxiliary output budgets."""

from typing import Any, Optional

from agent.auxiliary_profile_gates import profile_forwards_max_tokens
from utils import base_url_host_matches


def forwards_max_tokens(provider: str, provider_norm: str, model: str, effective_base: str, task: Optional[str]) -> bool:
    """Whether an explicit max_tokens is forwarded on this route. Configured title/approval caps win.

    No default cap elsewhere (omitted = provider default; avoids max_completion_tokens / ZAI-vision
    quirks). Forward only where mandatory or honored: Anthropic Messages wire (400 without it);
    NVIDIA NIM (empty choices[] when omitted); MoA reference slots; Gemini native (fixed 65,535
    ceiling otherwise); OpenRouter (budgets the FULL window when omitted → 402 on low credit);
    managed local llama-server (uncapped decode with no EOS burns the GPU to the context window);
    a provider profile opting in via ``forwards_max_tokens``.
    """
    from agent import auxiliary_client as aux

    return (
        (task in {"title_generation", "approval"} and aux._get_auxiliary_task_config(task).get("max_tokens") is not None)
        or aux._is_anthropic_compat_endpoint(provider, effective_base)
        or aux._nous_on_messages_wire(provider_norm, model)
        or provider_norm in aux._NVIDIA_PROVIDER_NAMES
        or base_url_host_matches(effective_base, "integrate.api.nvidia.com")
        or str(task) == "moa_reference"
        or aux._is_gemini_native_route(provider_norm, effective_base)
        or provider_norm == "openrouter"
        or base_url_host_matches(effective_base, "openrouter.ai")
        or aux._is_managed_local_endpoint(effective_base)
        or profile_forwards_max_tokens(provider_norm)
    )


def responses_output_budget(kwargs: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Public OpenAI Responses accepts a cap; the subscription Codex backend rejects it."""
    cap = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
    if cap is not None and base_url_host_matches(base_url, "api.openai.com"):
        return {"max_output_tokens": cap}
    return {}
