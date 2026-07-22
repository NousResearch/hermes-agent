"""Cursor Agent SDK provider profile.

Uses ``CURSOR_API_KEY`` and a custom OpenAI-compat shim
(``agent.cursor_agent_client.CursorAgentClient``) rather than a REST chat API.
Default model is ``auto`` (server-selected; survives usage-limit lockouts).
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile


class CursorProfile(ProviderProfile):
    """Cursor Agent — local SDK runtime with Auto model default."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return ["auto", "composer-2.5", "default"]


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor-sdk"),
    display_name="Cursor Agent",
    description="Cursor Agent SDK (local runtime; default model auto)",
    env_vars=("CURSOR_API_KEY",),
    base_url="cursor://agent",
    auth_type="api_key",
    default_aux_model="auto",
    api_mode="chat_completions",
)

register_provider(cursor)
