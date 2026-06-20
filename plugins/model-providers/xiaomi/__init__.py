"""Xiaomi MiMo provider profile.

Xiaomi MiMo Token Plan keys are region/cluster-specific (Singapore, China,
Amsterdam).  The default ``base_url`` here targets the Singapore cluster;
users on other clusters should set ``XIAOMI_BASE_URL`` or use the cluster
selector in ``hermes setup`` (``_model_flow_xiaomi``).
"""

import os

from providers import register_provider
from providers.base import ProviderProfile

# Resolve base URL at registration time so the plugin reflects the user's
# configured cluster without requiring a separate config step.
_default_base = os.environ.get(
    "XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"
)

xiaomi = ProviderProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url=_default_base,
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
