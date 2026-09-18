"""Phoenix Grove provider profile.

OpenAI-compatible endpoint over curated open-weight models (GLM, DeepSeek,
Kimi, Qwen, MiMo, Nemotron, Gemma, MiniMax), sold as a flat coding plan
(windows + bank usage) with per-token keys also available. Model ids are bare
(``glm-5.3-flash``, ``deepseek-v4-pro``, ...) and match the models.dev
``pgsgrove`` catalog; ``GET /v1/models`` (no auth) lists them with pricing and
context, so the live picker fetch works before a key is configured.

Reasoning: the endpoint relays ``reasoning_effort`` to the model unchanged and
returns thinking in ``reasoning_content``; ``none`` is honored as the off
switch on every model that can be switched off.
"""

from typing import Any

from agent.reasoning_effort import PGSGROVE_EFFORTS, clamp_effort, requested_effort
from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


class PhoenixGroveProfile(ProviderProfile):
    """Phoenix Grove - top-level ``reasoning_effort`` passthrough."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        supports_reasoning: bool = False, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict):
            return {}, {}
        # Explicit disable → the wire's own off switch (verified live: no reasoning_content).
        if reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = requested_effort(reasoning_config)
        if not effort:
            return {}, {}  # unset stays unset — the model's default is its recommended setting
        # Nearest weaker supported level, never escalate; ``none`` is never a clamp target.
        return {}, {"reasoning_effort": clamp_effort(effort, PGSGROVE_EFFORTS) or effort}


pgsgrove = PhoenixGroveProfile(
    name="pgsgrove", aliases=("phoenix-grove", "phoenixgrove", "pgs"),
    display_name="Phoenix Grove",
    description="Phoenix Grove — curated open-weights coding plan (flat monthly, zero retention)",
    signup_url="https://api.pgsgrove.com/",
    env_vars=("PGS_API_KEY", "PGS_BASE_URL"),
    base_url="https://api.pgsgrove.com/v1", auth_type="api_key",
    # Attribution so Phoenix Grove can identify Hermes Agent traffic.
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="glm-5.3-flash",
    fallback_models=(
        "glm-5.3", "glm-5.3-flash", "deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-flash-0731",
        "kimi-k2.7", "qwen-3.8-2.4t", "minimax-m3",
    ),
)

register_provider(pgsgrove)
