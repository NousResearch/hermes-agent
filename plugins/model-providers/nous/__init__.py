"""Nous Portal provider profile."""

from typing import Any

from agent.portal_tags import get_conversation_context, nous_portal_tags
from providers import register_provider
from providers.base import ProviderProfile


class NousProfile(ProviderProfile):
    """Nous Portal — product tags, reasoning with Nous-specific omission."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"tags": nous_portal_tags(session_id=session_id)}
        # Top-level session_id → provider sticky routing key. Pins every
        # turn of a session to the same upstream endpoint so explicit
        # Anthropic cache_control breakpoints stay warm instead of
        # cold-writing a fresh cache on each reroute (Anthropic/Vertex/
        # Bedrock caches are instance-local). Mirrors the OpenRouter
        # profile; without it the portal falls back to hashing the opening
        # messages, which breaks pinning whenever those shift.
        #
        # Resolve it exactly like ``nous_portal_tags`` resolves the
        # ``conversation=`` tag: ambient context first (the lineage ROOT id
        # published by the agent loop), explicit argument as fallback.
        #
        # The gap this closes is the auxiliary call sites — compression,
        # title generation, vision, web_extract, session_search, MoA slots.
        # They funnel through ``agent.auxiliary_client`` which has no session
        # handle, so they never pass ``session_id``: they carried the
        # ``conversation=`` tag but NO sticky key at all, and each one routed
        # independently of the conversation it belongs to. Reading the same
        # ambient contextvar the tag already uses fixes that with zero
        # per-call-site plumbing.
        #
        # For the main loop the two agree anyway under the default
        # ``compression.in_place: true`` (#38763), where compaction keeps the
        # session id; the ambient root additionally keeps the key stable for
        # installs that opt back into rotating compaction, and across
        # delegate-subagent trees.
        sticky_key = get_conversation_context() or session_id
        if sticky_key:
            body["session_id"] = sticky_key
        provider_preferences = context.get("provider_preferences")
        if provider_preferences:
            body["provider"] = provider_preferences
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Nous: passes full reasoning_config, but OMITS when disabled."""
        extra_body = {}
        if supports_reasoning:
            if reasoning_config is not None:
                rc = dict(reasoning_config)
                if rc.get("enabled") is False:
                    pass  # Nous omits reasoning when disabled
                else:
                    extra_body["reasoning"] = rc
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


nous = NousProfile(
    name="nous",
    aliases=("nous-portal", "nousresearch"),
    env_vars=("NOUS_API_KEY",),
    display_name="Nous Research",
    description="Nous Research — Hermes model family",
    signup_url="https://nousresearch.com/",
    fallback_models=(
        "hermes-3-405b",
        "hermes-3-70b",
    ),
    base_url="https://inference-api.nousresearch.com/v1",
    auth_type="oauth_device_code",
)

register_provider(nous)
