"""An MSC3245 voice event carries an EMPTY ``m.mentions`` block even when the user typed a mention
while recording — Element X sends the typed mention as a SEPARATE ``m.text`` event right after.
Under ``MATRIX_REQUIRE_MENTION=true`` the mention gate therefore dropped every voice message
silently (``plugins/platforms/matrix/adapter.py``: the ``return None`` in
``_resolve_message_context``'s require-mention branch), while the following bare-mention text
carried no content of its own to answer.

Invariants:
  1. An unmentioned voice is PARKED (not dropped) and the following bare-mention text from the
     same sender claims it — exactly one dispatch, carrying the VOICE event.
  2. A parked voice expires after its window, so a late bare mention dispatches the text only.
  3. The wake word path claims a voice whose transcript contains the bot's own name, without a
     typed mention.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_adapter(monkeypatch):
    # This box exports MATRIX_* (allowed rooms, home room, …) for the live gateway. Clear the
    # gate inputs so the test measures the mention gate, not whatever the host happens to export.
    for name in ("MATRIX_ALLOWED_ROOMS", "MATRIX_FREE_RESPONSE_ROOMS", "MATRIX_HOME_ROOM",
                 "MATRIX_SESSION_SCOPE", "MATRIX_READ_TOKEN", "MATRIX_READ_HOMESERVER",
                 "MATRIX_HOME_ROOM_THREAD_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}))
    assert adapter._allowed_rooms == set(), "test must not inherit a room whitelist"
    adapter._startup_ts = time.time() - 10
    adapter.handle_message = AsyncMock()
    adapter._client = None
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Group Room", room_topic=None, server_name="example.org", chat_type="group"))
    adapter._is_dm_room = AsyncMock(return_value=False)
    # Keep the wake-word transcode path out of these tests unless a test opts in.
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/cached.ogg")
    # Text events are debounced for 0.6s in production (_enqueue_text_event) and would never
    # reach handle_message inside a test. Disable the batching so a text dispatch is observable.
    adapter._text_batch_delay_seconds = 0.0
    return adapter


def _voice_event(body="voice message", sender="@alice:example.org", event_id="$voice1"):
    """An MSC3245 voice event: m.audio plus the voice flag, and an EMPTY m.mentions block."""
    return SimpleNamespace(
        sender=sender, event_id=event_id, room_id="!group:example.org",
        timestamp=int(time.time() * 1000),
        content={
            "body": body, "msgtype": "m.audio", "url": "mxc://example.org/voice",
            "info": {"mimetype": "audio/ogg", "size": 2048},
            "org.matrix.msc3245.voice": {},
            "m.mentions": {},
        })


def _text_event(body, sender="@alice:example.org", event_id="$text1"):
    content = {"body": body, "msgtype": "m.text"}
    if "@hermes:example.org" in body:
        content["m.mentions"] = {"user_ids": ["@hermes:example.org"]}
    return SimpleNamespace(
        sender=sender, event_id=event_id, room_id="!group:example.org",
        timestamp=int(time.time() * 1000), content=content)


# ---------------------------------------------------------------------------
# 1. Park + claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmentioned_voice_is_parked_not_dropped(monkeypatch):
    """The voice is parked; nothing is dispatched while it waits for the typed mention."""
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())

    assert adapter.handle_message.await_count == 0
    assert "$voice1" in {v[1] for v in (getattr(adapter, "_pending_voice", {}) or {}).values()}


@pytest.mark.asyncio
async def test_bare_mention_claims_parked_voice(monkeypatch):
    """A bare ``@bot`` text event dispatches the PARKED VOICE (not the empty text)."""
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())
    await adapter._on_room_message(_text_event("@hermes:example.org", event_id="$text1"))

    assert adapter.handle_message.await_count == 1
    msg = adapter.handle_message.await_args.args[0]
    assert msg.message_id == "$voice1"


@pytest.mark.asyncio
async def test_claimed_voice_is_not_left_parked(monkeypatch):
    """Claiming consumes the parked entry: the mention answers the voice, the next one is text."""
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())
    await adapter._on_room_message(_text_event("@hermes:example.org", event_id="$text1"))
    await adapter._on_room_message(_text_event("@hermes:example.org", event_id="$text2"))

    dispatched = [c.args[0].message_id for c in adapter.handle_message.await_args_list]
    assert dispatched == ["$voice1", "$text2"], (
        "the first bare mention must answer the VOICE and not be replayed by the second")


@pytest.mark.asyncio
async def test_other_user_bare_mention_does_not_claim(monkeypatch):
    """A parked voice is claimed only by a mention of THIS bot from the same sender."""
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())
    await adapter._on_room_message(_text_event("hey @alice:example.org", event_id="$text1"))

    assert adapter.handle_message.await_count == 0
    assert adapter._pending_voice, "the voice must stay parked for its real mention"


@pytest.mark.asyncio
async def test_mention_with_text_does_not_claim_parked_voice(monkeypatch):
    """Only a BARE mention claims: a mention carrying its own text is dispatched as text."""
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())
    await adapter._on_room_message(_text_event("@hermes:example.org what about this", event_id="$text1"))

    assert adapter.handle_message.await_count == 1
    msg = adapter.handle_message.await_args.args[0]
    assert msg.message_id == "$text1"
    assert adapter._pending_voice, "the voice stays parked — the mention was not bare"


# ---------------------------------------------------------------------------
# 2. Expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_parked_voice_is_not_dispatched(monkeypatch):
    """Past the claim window the parked voice is discarded: no dispatch may carry the voice event.

    (The bare mention itself still flows through as an ordinary — text-less — mention; that is
    upstream behaviour and not what this patch governs.)
    """
    adapter = _make_adapter(monkeypatch)

    await adapter._on_room_message(_voice_event())
    sender = "@alice:example.org"
    room_id, event_id, _ts, content, relates = adapter._pending_voice[sender]
    adapter._pending_voice[sender] = (room_id, event_id, time.time() - 121, content, relates)

    await adapter._on_room_message(_text_event("@hermes:example.org", event_id="$text1"))

    dispatched = [c.args[0].message_id for c in adapter.handle_message.await_args_list]
    assert "$voice1" not in dispatched, "an expired parked voice must never be dispatched"
    assert not adapter._pending_voice, "the expired entry is consumed, not left to rot"


# ---------------------------------------------------------------------------
# 3. Wake word
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("heard, claimed", [
    ("hermesbot wie ist das wetter", True),        # exact
    ("hermesport wie ist das wetter", True),       # Whisper mangles the name
    ("alexa wie ist das wetter", False),           # someone else
])
async def test_spoken_bot_name_claims_voice_without_typed_mention(monkeypatch, heard, claimed):
    """A voice whose transcript contains the bot's name is claimed with no typed mention."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda *a, **k: {"success": True, "transcript": heard}, raising=False)

    await adapter._on_room_message(_voice_event())
    await adapter._voice_wake_word_check(
        "!group:example.org", "@alice:example.org", "$voice1", _voice_event().content)

    assert adapter.handle_message.await_count == (1 if claimed else 0)
    if claimed:
        msg = adapter.handle_message.await_args.args[0]
        assert msg.message_id == "$voice1"
