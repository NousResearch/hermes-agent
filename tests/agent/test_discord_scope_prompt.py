"""Prompt-builder tests for Discord scope trust classification."""

from agent.prompt_builder import format_conversation_trust_context


def test_no_trust_override_is_empty_and_cache_safe():
    assert format_conversation_trust_context(None) == ""


def test_full_trusted_uses_operator_designation_not_transport_identity():
    text = format_conversation_trust_context("full_trusted")
    assert text.startswith("**Conversation trust:**")
    assert "operator-designated confidential workspace" in text
    assert "solely because transport is Discord" in text
    assert "multiple distinct authenticated participants may be present" in text
    assert "owner identity" in text
    assert "approvals" in text
    assert "tools" in text
    assert "credentials" in text
    assert "secrets" in text


def test_public_private_and_legacy_have_distinct_explicit_semantics():
    legacy = format_conversation_trust_context("legacy")
    public = format_conversation_trust_context("public")
    private = format_conversation_trust_context("private")
    assert "legacy trust semantics" in legacy
    assert "operator-designated public workspace" in public
    assert "operator-designated private workspace" in private
    for text in (legacy, public, private):
        assert "existing permissions apply" in text
        assert "does not change Discord membership or visibility" in text


def test_untyped_unknown_value_fails_closed():
    assert format_conversation_trust_context("owner") == ""
