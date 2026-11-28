"""Regression for NousResearch/hermes-agent#154770: Vercel AI Gateway affinity.

The logical conversation key must stay stable across compression rotations of
the physical session ID, without mixing unrelated conversations.
"""

from agent.portal_tags import reset_affinity_scope, set_affinity_scope
from providers import get_provider_profile


def test_ai_gateway_affinity_uses_declared_scope_across_session_rotation():
    profile = get_provider_profile("ai-gateway")
    assert profile is not None

    token = set_affinity_scope("gwk_conversation_one")
    try:
        _body1, first = profile.build_api_kwargs_extras(session_id="physical-before-compression")
        _body2, after_rotation = profile.build_api_kwargs_extras(session_id="physical-after-compression")
    finally:
        reset_affinity_scope(token)

    other_token = set_affinity_scope("gwk_conversation_two")
    try:
        _body3, other_conversation = profile.build_api_kwargs_extras(session_id="physical-after-compression")
    finally:
        reset_affinity_scope(other_token)

    assert first.get("extra_headers", {}).get("x-session-affinity") == "gwk_conversation_one"
    assert after_rotation.get("extra_headers", {}).get("x-session-affinity") == "gwk_conversation_one"
    assert other_conversation.get("extra_headers", {}).get("x-session-affinity") == "gwk_conversation_two"
