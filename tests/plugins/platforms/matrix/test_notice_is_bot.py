"""m.notice senders arrive flagged as bots so the gateway's bot rungs see them (#143370).

Matrix has no per-event bot flag from the homeserver; ``m.notice`` is the msgtype convention
bots publish under. The adapter flags notice senders on ``SessionSource.is_bot`` for the loop
guard and no-pairing rule only: any sender picks the msgtype, so the flag never widens access.
"""

import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig

ROOM = "!room:example.org"


def _make_adapter(extra=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
        "process_notices": "true", **(extra or {})}))


def _stub_room_gates(adapter, captured):
    identity = types.SimpleNamespace(display_name="Room", room_topic=None, server_name="example.org")

    async def _resolve_room_identity(room_id, *, force_refresh=False):
        return identity

    async def _is_dm_room(room_id):
        return True

    async def _get_display_name(room_id, user_id):
        return "Human"

    async def mark_async(thread_id):
        pass

    async def handle_message(event):
        captured.append(event)

    adapter._resolve_room_identity = _resolve_room_identity
    adapter._is_dm_room = _is_dm_room
    adapter._get_display_name = _get_display_name
    adapter._threads = types.SimpleNamespace(mark_async=mark_async)
    adapter._background_read_receipt = lambda room_id, event_id: None
    adapter.handle_message = handle_message


async def _deliver(adapter, source_content):
    """Feed one text-path message through the real gate/normalise pipeline."""
    await adapter._handle_text_message(ROOM, "@peer:example.org", "$event", 1_000_000.0, source_content, {})


@pytest.mark.asyncio
async def test_notice_sender_arrives_flagged_as_bot(monkeypatch):
    monkeypatch.setenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", "0")
    adapter = _make_adapter()
    captured = []
    _stub_room_gates(adapter, captured)

    await _deliver(adapter, {"msgtype": "m.notice", "body": "bot says hi"})

    assert captured, "m.notice with process_notices=true must be admitted"
    assert captured[0].source.is_bot is True


@pytest.mark.asyncio
async def test_text_sender_still_arrives_as_person(monkeypatch):
    monkeypatch.setenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", "0")
    adapter = _make_adapter()
    captured = []
    _stub_room_gates(adapter, captured)

    await _deliver(adapter, {"msgtype": "m.text", "body": "human says hi"})

    assert captured
    assert captured[0].source.is_bot is False


@pytest.mark.asyncio
async def test_media_content_is_not_over_flagged():
    """Media msgtypes never equal m.notice; the source must stay a person's."""
    adapter = _make_adapter()
    _stub_room_gates(adapter, [])

    ctx = await adapter._resolve_message_context(
        ROOM, "@peer:example.org", "$media", "photo.jpg",
        {"msgtype": "m.image", "body": "photo.jpg", "url": "mxc://example.org/abc"}, {})

    assert ctx is not None
    assert ctx[5].is_bot is False


@pytest.mark.parametrize("allow_bots", ["mentions", "all"])
def test_unlisted_notice_sender_is_denied_even_with_allow_bots_env(monkeypatch, allow_bots):
    """m.notice is sender-chosen: a stranger sending notices must not get past
    MATRIX_ALLOWED_USERS, whatever MATRIX_ALLOW_BOTS says (Matrix does not honour it)."""
    from gateway.run import GatewayRunner
    from gateway.session import Platform, SessionSource

    for var in ("MATRIX_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    monkeypatch.setenv("MATRIX_ALLOW_BOTS", allow_bots)
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)

    def source(user_id):
        return SessionSource(platform=Platform.MATRIX, chat_id=ROOM, chat_type="dm",
                             user_id=user_id, user_name="Notice", is_bot=True)

    assert runner._is_user_authorized(source("@stranger:evil.org")) is False
    assert runner._is_user_authorized(source("@owner:example.org")) is True
