"""GatewayRunner._handle_message must drop genuinely empty inbound events.

Regression tests for the empty-inbound protection (salvaged from PR #56793):
a user-originated event with no text (or only the Discord no-text sentinel),
no media, no reply context, and no channel context must return before
authorization, pairing, or agent dispatch.  Media-only events, reply-context
events, channel-context ("catch me up") events, and internal synthetic events
must continue to flow.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


DISCORD_NO_TEXT_SENTINEL = "(The user sent a message with no text content)"


def _build_runner(monkeypatch, tmp_path) -> GatewayRunner:
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    return runner


def _instrument(monkeypatch, runner):
    """Track authorization and agent dispatch on ``runner``.

    Authorization is stubbed to False with chat_type='group' events so an
    event that gets PAST the empty-event guard is silently ignored (no
    pairing DM) — reaching the auth check at all is the signal we assert on.
    """
    calls = {"auth": 0, "agent": 0}

    def _tracking_auth(self, src):
        calls["auth"] += 1
        return False

    async def _tracking_agent(self, *a, **kw):
        calls["agent"] += 1
        raise RuntimeError("sentinel — stop before the agent runner")

    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", _tracking_auth)
    monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _tracking_agent)
    return calls


def _source(chat_type="group", user_id="user-1"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type=chat_type,
        user_id=user_id,
        user_name="alice",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   \n\t", DISCORD_NO_TEXT_SENTINEL])
async def test_blank_and_sentinel_events_return_before_authorization(
    monkeypatch, tmp_path, text
):
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(text=text, source=_source())
    result = await runner._handle_message(event)

    assert result is None
    assert calls["auth"] == 0, "empty event must be dropped before authorization"
    assert calls["agent"] == 0, "empty event must never reach agent dispatch"


@pytest.mark.asyncio
async def test_blank_dm_event_does_not_trigger_pairing(monkeypatch, tmp_path):
    """A blank DM from an unknown user is dropped before the pairing flow."""
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    generate_called = False
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return original_generate(*args, **kwargs)

    runner.pairing_store.generate_code = tracking_generate

    event = MessageEvent(
        text="",
        source=_source(chat_type="dm", user_id="unknown_user_999"),
    )
    result = await runner._handle_message(event)

    assert result is None
    assert not generate_called
    assert adapter.send.await_count == 0


@pytest.mark.asyncio
async def test_media_only_event_continues_past_the_empty_guard(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(
        text="",
        source=_source(),
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )
    await runner._handle_message(event)

    assert calls["auth"] == 1, "media-only event must reach authorization"


@pytest.mark.asyncio
async def test_sentinel_text_with_media_continues(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(
        text=DISCORD_NO_TEXT_SENTINEL,
        source=_source(),
        media_urls=["/tmp/photo.png"],
    )
    await runner._handle_message(event)

    assert calls["auth"] == 1


@pytest.mark.asyncio
async def test_reply_context_event_continues_past_the_empty_guard(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(
        text="",
        source=_source(),
        reply_to_message_id="m-1",
        reply_to_text="the assistant reply being pointed at",
    )
    await runner._handle_message(event)

    assert calls["auth"] == 1, "reply-context event must reach authorization"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply_metadata",
    [
        {"reply_to_message_id": "m-1"},
        {"reply_to_author_id": "user-2"},
        {"reply_to_author_name": "bob"},
        {"reply_to_is_own_message": True},
    ],
)
async def test_reply_metadata_without_quoted_text_continues_past_empty_guard(
    monkeypatch, tmp_path, reply_metadata
):
    """Platforms can supply reply identity even when quoted-text lookup misses."""
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(text="", source=_source(), **reply_metadata)
    await runner._handle_message(event)

    assert calls["auth"] == 1, "reply metadata must reach authorization"


@pytest.mark.asyncio
async def test_channel_context_catch_me_up_event_continues(monkeypatch, tmp_path):
    """A bare-mention trigger with backfill context is a 'catch me up' turn."""
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(
        text="",
        source=_source(),
        channel_context="[Recent channel messages]\nbob: anyone seen the deploy?",
    )
    await runner._handle_message(event)

    assert calls["auth"] == 1, "channel-context event must reach authorization"


@pytest.mark.asyncio
async def test_blank_internal_event_reaches_agent_dispatch(monkeypatch, tmp_path):
    """Internal synthetic events must continue even with blank visible text."""
    runner = _build_runner(monkeypatch, tmp_path)
    calls = _instrument(monkeypatch, runner)

    event = MessageEvent(text="", source=_source(), internal=True)

    with pytest.raises(RuntimeError, match="sentinel"):
        await runner._handle_message(event)

    assert calls["auth"] == 0, "internal events bypass authorization"
    assert calls["agent"] == 1, "internal event must reach agent dispatch"
