"""Cursor Agent (SDK) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

cursor = ProviderProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor_agent"),
    api_mode="cursor_sdk_runtime",
    display_name="Cursor Agent",
    description="Cursor SDK — local agent runtime with Hermes tools via MCP",
    signup_url="https://cursor.com/dashboard/integrations",
    env_vars=("CURSOR_API_KEY",),
    base_url="cursor://sdk",
    auth_type="api_key",
    default_aux_model="composer-2.5",
    fallback_models=(
        "composer-2.5",
        "composer-2.5-fast",
        "auto",
    ),
)

register_provider(cursor)
