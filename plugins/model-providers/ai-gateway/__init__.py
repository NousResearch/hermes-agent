"""Vercel AI Gateway provider profile: attribution + session-affinity headers + reasoning passthrough."""

from typing import Any

from agent.portal_tags import get_affinity_scope, get_conversation_context
from agent.prompt_cache_scope import is_fork_cache_scope
from agent.transports.codex import _cache_scope_from_session_id
from providers import register_provider
from providers.base import ProviderProfile


def _sticky_key(session_id: str | None) -> str | None:
    """Declared routing scope, then ambient conversation, then explicit session_id."""
    return _cache_scope_from_session_id(get_affinity_scope() or get_conversation_context() or session_id)


class VercelAIGatewayProfile(ProviderProfile):
    """Vercel AI Gateway — attribution headers, session affinity, and reasoning passthrough."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = True,
        session_id: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if supports_reasoning:
            extra_body["reasoning"] = (
                dict(reasoning_config) if reasoning_config is not None else {"enabled": True, "effort": "medium"}
            )
        # Keep multi-turn conversations pinned to the upstream provider holding the warm prompt cache
        # (e.g. Direct Anthropic vs Bedrock vs Vertex for Claude, or sticky DeepSeek/OpenAI/xAI workers).
        affinity = _sticky_key(session_id)
        if is_fork_cache_scope(ctx.get("cache_scope_id")):
            affinity = ctx["cache_scope_id"]
        if affinity:
            top_level["extra_headers"] = {"x-session-affinity": affinity}
        return extra_body, top_level


vercel = VercelAIGatewayProfile(
    name="ai-gateway", aliases=("vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"),
    env_vars=("AI_GATEWAY_API_KEY",), base_url="https://ai-gateway.vercel.sh/v1",
    default_headers={"HTTP-Referer": "https://hermes-agent.nousresearch.com", "X-Title": "Hermes Agent"},
    default_aux_model="google/gemini-3-flash",
)

register_provider(vercel)
