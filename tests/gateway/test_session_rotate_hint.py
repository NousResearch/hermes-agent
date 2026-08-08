"""Tests for the session-size rotate hint.

A long-lived Telegram/Discord thread has no visible context meter: the user
cannot tell a 40k session from a 400k one, so bloat is invisible until the
bill arrives. The existing hygiene warning fires at 95% of the model's
context window, which for a 1M-token model means 950k — a threshold a real
conversation never reaches. This hint is an ABSOLUTE token threshold with a
cooldown, so it actually fires and does not spam.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import (
    SessionSource,
    build_session_rotate_hint,
    should_hint_session_rotate,
)


class TestShouldHintSessionRotate:
    def test_fires_when_over_threshold(self):
        assert should_hint_session_rotate(
            tokens=200_000, threshold=150_000, last_hint_at=None, now=1000.0, cooldown_s=3600
        )

    def test_silent_under_threshold(self):
        assert not should_hint_session_rotate(
            tokens=100_000, threshold=150_000, last_hint_at=None, now=1000.0, cooldown_s=3600
        )

    def test_disabled_when_threshold_zero_or_negative(self):
        for threshold in (0, -1):
            assert not should_hint_session_rotate(
                tokens=900_000, threshold=threshold, last_hint_at=None, now=1000.0, cooldown_s=3600
            )

    def test_cooldown_suppresses_repeat(self):
        """Every turn is over threshold once you are bloated — without a
        cooldown the user gets nagged on every single message."""
        assert not should_hint_session_rotate(
            tokens=200_000, threshold=150_000, last_hint_at=1000.0, now=1500.0, cooldown_s=3600
        )

    def test_fires_again_after_cooldown_elapses(self):
        assert should_hint_session_rotate(
            tokens=200_000, threshold=150_000, last_hint_at=1000.0, now=5000.0, cooldown_s=3600
        )

    def test_zero_cooldown_always_fires_when_over(self):
        assert should_hint_session_rotate(
            tokens=200_000, threshold=150_000, last_hint_at=1000.0, now=1000.1, cooldown_s=0
        )

    @pytest.mark.parametrize("tokens", [None, 0, -5])
    def test_unknown_token_count_is_silent(self, tokens):
        assert not should_hint_session_rotate(
            tokens=tokens, threshold=150_000, last_hint_at=None, now=1000.0, cooldown_s=3600
        )


class TestBuildSessionRotateHint:
    def test_message_states_size_and_remedy(self):
        msg = build_session_rotate_hint(tokens=232_000)
        assert "232,000" in msg
        assert "/reset" in msg

    def test_message_is_plain_text_for_chat_surfaces(self):
        """Telegram/Discord render markdown tables badly — keep it one line."""
        msg = build_session_rotate_hint(tokens=232_000)
        assert "|" not in msg
        assert "\n\n" not in msg


class TestSessionRotateHintDelivery:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("platform", [Platform.API_SERVER, Platform.WEBHOOK])
    async def test_machine_sources_never_receive_the_hint(self, platform):
        from gateway.run import GatewayRunner

        adapter = AsyncMock()
        runner = object.__new__(GatewayRunner)
        runner._adapter_for_source = lambda _source: adapter
        runner._reply_anchor_for_event = lambda _event: None
        runner._thread_metadata_for_source = lambda _source, _anchor: {}
        source = SessionSource(platform=platform, chat_id="machine-target")
        event = MessageEvent(text="payload", source=source)

        delivered = await runner._send_session_rotate_hint(
            source,
            event,
            tokens=232_000,
        )

        assert delivered is False
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_human_chat_receives_the_hint(self):
        from gateway.run import GatewayRunner

        adapter = AsyncMock()
        runner = object.__new__(GatewayRunner)
        runner._adapter_for_source = lambda _source: adapter
        runner._reply_anchor_for_event = lambda _event: None
        runner._thread_metadata_for_source = lambda _source, _anchor: {}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="human-chat")
        event = MessageEvent(text="hello", source=source)

        delivered = await runner._send_session_rotate_hint(
            source,
            event,
            tokens=232_000,
        )

        assert delivered is True
        adapter.send.assert_awaited_once()
