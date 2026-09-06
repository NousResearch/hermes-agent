"""Regression tests for authenticated Discord conversation trust metadata."""

import pytest

from gateway.session import (
    Platform,
    SessionContext,
    SessionSource,
    build_session_context_prompt,
)


def _context(*, trust=None, platform=Platform.DISCORD, shared=False):
    return SessionContext(
        source=SessionSource(
            platform=platform,
            chat_id="channel-1",
            chat_type="channel",
            scope_id="guild-1",
            conversation_trust=trust,
        ),
        connected_platforms=[],
        home_channels={},
        shared_multi_user_session=shared,
    )


def test_missing_override_preserves_prompt_bytes():
    context = _context()
    assert build_session_context_prompt(context) == build_session_context_prompt(
        _context(trust=None)
    )
    assert "Conversation trust" not in build_session_context_prompt(context)


def test_trust_round_trips_without_affecting_legacy_sources():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        conversation_trust="full_trusted",
    )
    restored = SessionSource.from_dict(source.to_dict())
    assert restored.conversation_trust == "full_trusted"

    legacy = SessionSource.from_dict({"platform": "discord", "chat_id": "channel-1"})
    assert legacy.conversation_trust is None
    assert "conversation_trust" not in legacy.to_dict()


@pytest.mark.parametrize("trust", ["legacy", "public", "private", "full_trusted"])
def test_authenticated_trust_is_scoped_and_keeps_multi_user_accuracy(trust):
    prompt = build_session_context_prompt(_context(trust=trust, shared=True))
    assert f"Conversation trust:" in prompt
    assert "Multiple users may participate" in prompt
    assert "multiple distinct authenticated participants may be present" in prompt
    assert "does not grant owner identity, approvals, tools, credentials, or secrets" in prompt
    assert "does not change Discord membership or visibility" in prompt


def test_full_trusted_is_not_an_automatic_discord_private_claim():
    prompt = build_session_context_prompt(_context(trust="full_trusted"))
    assert "operator-designated confidential workspace" in prompt
    assert "do not presume public exposure solely because transport is Discord" in prompt
    assert "private Discord" not in prompt


def test_unknown_persisted_trust_is_rejected():
    with pytest.raises(ValueError, match="conversation_trust"):
        SessionSource.from_dict(
            {
                "platform": "discord",
                "chat_id": "channel-1",
                "conversation_trust": "owner",
            }
        )


def test_non_discord_source_does_not_receive_discord_trust_text():
    prompt = build_session_context_prompt(_context(trust="full_trusted", platform=Platform.SLACK))
    assert "Conversation trust" not in prompt
