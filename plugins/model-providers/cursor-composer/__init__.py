"""Cursor Composer provider profile.

Uses the Standard Agents API-for-Cursor OpenCode route so Chat Completions
requests are always primed into Cursor Composer Agent mode.  The generic
``/v1`` route is OpenAI-compatible too, but it only enters Agent mode when tools
are present; this provider intentionally targets ``/opencode/v1`` to make the
behavior unconditional. Do not use the OpenCode SDK harness route here: it emits
OpenCode-native tool names such as ``shell`` instead of Hermes tool names.
"""

from providers import register_provider
from providers.base import ProviderProfile

cursor_composer = ProviderProfile(
    name="cursor-composer",
    aliases=("cursor", "composer", "cursor-api", "api-for-cursor"),
    display_name="Cursor Composer",
    description="Cursor Composer via API for Cursor (forced Agent mode)",
    signup_url="https://cursor.com/dashboard",
    env_vars=("CURSOR_API_KEY", "CURSOR_COMPOSER_BASE_URL"),
    base_url="https://cursor-api.standardagents.ai/opencode/v1",
    fallback_models=(
        "composer-2.5",
        "composer-2.5-fast",
        "composer-2",
        "composer-latest",
    ),
    default_aux_model="composer-2.5-fast",
)

register_provider(cursor_composer)
