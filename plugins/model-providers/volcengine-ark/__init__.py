"""Volcengine Ark (火山引擎) — ByteDance unified model platform.

Ark is ByteDance's API gateway that gives access to models from multiple
vendors (doubao-seed, deepseek-v4, glm-5.2, kimi-k2, minimax-m3) through
a single API key.

The platform provides two subscription plans:
  - Agent Plan: General-purpose AI agent usage, AFP billing, multi-modal
  - Coding Plan: Coding-focused, token quotas, AI coding tools only

Both plans share the same Anthropic-compatible /api/coding endpoint.

Prompt caching: Ark's /api/coding endpoint supports Anthropic-style
cache_control markers (documented in the official plan overview).
Hermes enables caching for Ark providers and hosts automatically
via anthropic_prompt_cache_policy().

IMPORTANT — model list maintenance:
  Ark's /models endpoint returns 124+ stale model IDs from the full
  platform catalog, not the user's actual subscription. This provider
  hardcodes the current Agent Plan text-generation model list and must
  be updated when Ark adds/removes models.

  Last updated: 2026-06-26 (11 models)
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# ── Hardcoded model list (Agent Plan text generation, as of 2026-06-26) ──
# Ark's auto-discovery returns 124 stale models. We hardcode the actual
# working list to avoid confusion. Update when the provider's lineup changes.
_ARK_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.2",
    "doubao-seed-2.0-pro",
    "doubao-seed-2.0-lite",
    "doubao-seed-2.0-mini",
    "doubao-seed-2.0-code",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "minimax-m3",
    "minimax-m2.7",
]


class VolcengineArkProfile(ProviderProfile):
    """Volcengine Ark — hardcoded model list to avoid stale auto-discovery."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return hardcoded model list.

        We intentionally skip the API call because Ark's /models endpoint
        returns 124+ stale model IDs from the full platform catalog rather
        than the user's actual subscription.
        """
        return list(_ARK_MODELS)


volcengine_ark = VolcengineArkProfile(
    name="volcengine-ark",
    aliases=(
        "ark",
        "volcengine",
        "volcano",
        "bytedance",
    ),
    display_name="Volcengine Ark (火山引擎)",
    description="ByteDance model platform — doubao-seed, deepseek-v4, glm-5.2, kimi-k2, minimax",
    signup_url="https://console.volcengine.com/ark/region:ark+cn-beijing/",
    api_mode="anthropic_messages",
    env_vars=("ARK_API_KEY",),
    base_url="https://ark.cn-beijing.volces.com/api/coding",
    auth_type="api_key",
    default_aux_model="deepseek-v4-flash",
)

register_provider(volcengine_ark)
