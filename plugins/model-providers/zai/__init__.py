"""ZAI / GLM provider profiles.

Z.AI (GLM) — api.z.ai (Global) and open.bigmodel.cn (China)
Both support Coding Plan endpoints at /api/coding/paas/v4.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — thinking parameter, tool_stream, Coding Plan headers."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Inject thinking parameter for Z.AI/GLM models."""
        reasoning_config = context.get("reasoning_config")
        body: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                body["thinking"] = {"type": "disabled"}
            else:
                body["thinking"] = {"type": "enabled"}
        else:
            # Default: thinking enabled (GLM-5+ models support it)
            body["thinking"] = {"type": "enabled"}

        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Z.AI returns no provider-specific top-level kwargs beyond defaults."""
        return {}, {}


# ── Global ──────────────────────────────────────────────────────────────
zai = ZaiProfile(
    name="zai",
    aliases=("z-ai", "z.ai"),
    env_vars=("ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI",
    description="Z.AI (GLM) — api.z.ai",
    signup_url="https://z.ai/",
    base_url="https://api.z.ai/api/paas/v4",
    hostname="api.z.ai",
    default_headers={
        "X-Title": "Hermes-Agent",
    },
    default_aux_model="glm-4.5-flash",
    fallback_models=(
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
    ),
)

# ── China ───────────────────────────────────────────────────────────────
zai_cn = ZaiProfile(
    name="zai-cn",
    aliases=("glm", "zhipu", "bigmodel"),
    env_vars=("GLM_API_KEY",),
    display_name="Zhipu AI",
    description="Zhipu AI (GLM) — open.bigmodel.cn",
    signup_url="https://open.bigmodel.cn/",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    hostname="open.bigmodel.cn",
    default_aux_model="glm-4.5-flash",
    fallback_models=(
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
    ),
)

# ── Global Coding Plan ─────────────────────────────────────────────────
zai_coding_global = ZaiProfile(
    name="zai-coding-global",
    aliases=("glm-coding-global", "z-ai-coding"),
    env_vars=("ZAI_CODING_API_KEY",),
    display_name="Z.AI Coding Plan",
    description="Z.AI Coding Plan — api.z.ai/api/coding",
    signup_url="https://z.ai/pricing",
    base_url="https://api.z.ai/api/coding/paas/v4",
    hostname="api.z.ai",
    default_headers={
        "X-Title": "Hermes-Agent",
    },
    default_aux_model="glm-4.7",
    fallback_models=(
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
    ),
)

# ── China Coding Plan ──────────────────────────────────────────────────
zai_coding_cn = ZaiProfile(
    name="zai-coding-cn",
    aliases=("glm-coding-cn",),
    env_vars=("GLM_CODING_API_KEY",),
    display_name="Zhipu AI Coding Plan",
    description="Zhipu AI Coding Plan — open.bigmodel.cn/api/coding",
    signup_url="https://open.bigmodel.cn/",
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    hostname="open.bigmodel.cn",
    default_aux_model="glm-4.7",
    fallback_models=(
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
    ),
)

register_provider(zai)
register_provider(zai_cn)
register_provider(zai_coding_global)
register_provider(zai_coding_cn)
