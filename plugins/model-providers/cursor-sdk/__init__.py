"""Cursor SDK provider profile."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


_CURSOR_FALLBACK_MODELS = (
    "composer-latest",
    "composer-2.5",
    "composer-2",
    "gpt-5.5",
    "claude-opus-4.7",
    "claude-sonnet-4.6",
    "gemini-3.1-pro-preview",
)


class CursorSdkProfile(ProviderProfile):
    """Cursor SDK uses @cursor/sdk through Hermes' cursor_sdk transport."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return Cursor's live SDK model catalog when the Node SDK is available."""
        if not api_key:
            return None
        try:
            from agent.cursor_sdk_adapter import list_cursor_models

            models = list_cursor_models(api_key=api_key, timeout=timeout)
        except Exception:
            return None
        return models or None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}


cursor_sdk = CursorSdkProfile(
    name="cursor-sdk",
    aliases=("cursor", "cursor_sdk"),
    api_mode="cursor_sdk",
    display_name="Cursor SDK",
    description="Cursor agents through the official SDK",
    signup_url="https://cursor.com/settings/api-keys",
    env_vars=("CURSOR_API_KEY",),
    base_url="cursor-sdk://local",
    models_url="",
    auth_type="api_key",
    supports_health_check=False,
    fallback_models=_CURSOR_FALLBACK_MODELS,
)

register_provider(cursor_sdk)
