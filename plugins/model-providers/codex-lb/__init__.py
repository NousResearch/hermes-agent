"""Codex LB OpenAI-compatible provider profile."""

from typing import Any

from agent.portal_tags import get_conversation_context
from providers import register_provider
from providers.base import ProviderProfile


class CodexLBProfile(ProviderProfile):
    """Add conversation affinity without changing generic API behavior."""

    def build_api_kwargs_extras(
        self,
        *,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        sticky_id = get_conversation_context() or session_id
        if not sticky_id:
            return {}, {}
        return {}, {"extra_headers": {"session_id": sticky_id}}


codex_lb = CodexLBProfile(
    name="codex-lb",
    aliases=("custom:codex-lb",),
    env_vars=(),
    base_url="",
)

register_provider(codex_lb)
