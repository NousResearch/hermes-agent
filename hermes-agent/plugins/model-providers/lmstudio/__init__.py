"""LM Studio provider profile (local desktop app with an OpenAI-compatible server).

Request quirk: reasoning goes out as top-level ``reasoning_effort``, gated on and clamped to
the per-model ``allowed_options`` LM Studio publishes in ``/api/v1/models``. The agent probes
and caches those options and passes them as ``lmstudio_reasoning_options``.

Everything else stays where it lives today: endpoint, ``LM_API_KEY`` / ``LM_BASE_URL`` and the
no-auth placeholder (``hermes_cli/auth.py`` registry row), the chat-model picker probe
(``hermes_cli/models_local.py``). Hence ``base_url`` is left empty: a loopback ``base_url``
would map every ``127.0.0.1`` / ``localhost`` endpoint to LM Studio in URL->provider inference
and disable local-server detection for Ollama, llama.cpp and vLLM.
"""

from typing import Any

from agent.lmstudio_reasoning import resolve_lmstudio_effort
from agent.reasoning_effort import generic_nested_reasoning
from providers import register_provider
from providers.base import ProviderProfile


class LMStudioProfile(ProviderProfile):
    """LM Studio: top-level ``reasoning_effort`` from the model's published options."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False,
        lmstudio_reasoning_options: list[str] | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``supports_reasoning`` is true only when the model publishes a non-``off`` option;
        ``lmstudio_reasoning_options`` is the agent's cached probe of them. Auxiliary calls have no
        probe (``None``) and keep the generic nested ``reasoning`` they always sent."""
        if lmstudio_reasoning_options is None:
            return generic_nested_reasoning(reasoning_config), {}
        if not supports_reasoning:
            return {}, {}
        effort = resolve_lmstudio_effort(reasoning_config, lmstudio_reasoning_options)
        return {}, ({"reasoning_effort": effort} if effort is not None else {})


register_provider(LMStudioProfile(
    name="lmstudio", display_name="LM Studio", supports_model_listing=False, supports_health_check=False,
    env_vars=("LM_API_KEY", "LM_BASE_URL"),
))
