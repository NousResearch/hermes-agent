"""Matrix command gate: ``/cmd`` and ``!cmd`` in a thread reach only the thread starter or the
addressed bot, not every bot that ever touched the thread."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType

ROOM = "!room:example.org"
ROOT = "$root"
HUMAN = "@alice:example.org"
BOT_A = "@a:example.org"
BOT_B = "@b:example.org"


def _bot(user_id, *, root_author=HUMAN, owns=(), members=None, **extra):
    """An adapter for one bot in a group room. ``root_author`` is the sender the homeserver
    reports for the thread root event; ``owns`` pre-seeds threads this bot started."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test",
        extra={"homeserver": "https://matrix.example.org", "user_id": user_id, **extra}))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._resolve_room_identity = AsyncMock(
        return_value=MagicMock(display_name="Room", room_topic=None, server_name="example.org"))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    client = SimpleNamespace(get_event=AsyncMock(return_value=SimpleNamespace(sender=root_author)))
    if members is not None:
        client.state_store = SimpleNamespace(get_members=AsyncMock(return_value=members))
    adapter._client = client
    adapter._threads._threads.clear()
    adapter._owned_threads._threads.clear()
    adapter._threads._remember(ROOT)  # the bug: every bot that touched the thread bypasses the mention gate
    for thread_id in owns:
        adapter._owned_threads._remember(thread_id)
    return adapter


async def _send(adapter, body, *, thread: str | None = ROOT, mentions=None, in_reply_to=None, room=ROOM, event_id="$e1"):
    """Feed a text event through the real text path; return the MessageEvent handled, else None."""
    content = {"msgtype": "m.text", "body": body}
    if mentions is not None:
        content["m.mentions"] = {"user_ids": mentions}
    relates = {}
    if thread:
        relates.update({"rel_type": "m.thread", "event_id": thread})
    if in_reply_to:
        relates["m.in_reply_to"] = {"event_id": in_reply_to}
    if relates:
        content["m.relates_to"] = relates
    adapter.handle_message.reset_mock()
    await adapter._handle_text_message(room, HUMAN, event_id, time.time(), content, relates)
    return adapter.handle_message.await_args.args[0] if adapter.handle_message.await_count else None


def _thread_pair(**kwargs):
    """Bot A started the thread; bot B only participates."""
    return _bot(BOT_A, owns=(ROOT,), root_author=BOT_A, **kwargs), _bot(BOT_B, root_author=BOT_A)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_thread_command_acts_only_on_thread_starter(command):
    a, b = _thread_pair()
    event = await _send(a, command)
    assert event is not None and event.text == "/status" and event.message_type == MessageType.COMMAND
    assert await _send(b, command) is None


@pytest.mark.asyncio
async def test_thread_starter_known_from_root_author_without_local_state():
    """Restart/lost state: the root event was sent by A, so A is still the starter and B is not."""
    a = _bot(BOT_A, root_author=BOT_A)
    b = _bot(BOT_B, root_author=BOT_A)
    assert await _send(a, "!status") is not None
    assert await _send(b, "!status") is None
    assert ROOT in a._owned_threads  # learned once, persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_thread_command_mention_addresses_only_that_bot(command):
    a, b = _thread_pair()
    for body, mentions in ((f"{command} {BOT_B}", [BOT_B]), (f"{command} @b", None), (f"@b:example.org {command}", [BOT_B])):
        assert await _send(a, body, mentions=mentions) is None, body  # starter yields to the addressed bot
        event = await _send(b, body, mentions=mentions)
        assert event is not None, body
        assert event.text == "/status" and event.message_type == MessageType.COMMAND  # mention stripped


@pytest.mark.asyncio
async def test_thread_command_profile_name_as_first_argument():
    members = [BOT_A, BOT_B, HUMAN]
    a = _bot(BOT_A, owns=(ROOT,), root_author=BOT_A, members=members)
    b = _bot(BOT_B, root_author=BOT_A, members=members)
    event = await _send(b, "!status b")
    assert event is not None and event.text == "/status"  # the name was the address, not an argument
    assert await _send(a, "!status b") is None  # starter sees the name belongs to another member
    assert (await _send(a, "!status")).text == "/status"


@pytest.mark.asyncio
async def test_thread_plain_text_unchanged():
    a, b = _thread_pair()
    for adapter in (a, b):  # thread participation still bypasses require_mention for chat text
        event = await _send(adapter, "just chatting")
        assert event is not None and event.message_type == MessageType.TEXT
    c = _bot(BOT_B, root_author=BOT_A)
    c._threads._threads.clear()  # not a thread this bot holds -> still needs a mention
    assert await _send(c, "just chatting") is None
    assert await _send(c, f"hi {BOT_B}", mentions=[BOT_B]) is not None


@pytest.mark.asyncio
async def test_unknown_bang_word_stays_chat_text():
    a, b = _thread_pair()
    for adapter in (a, b):
        event = await _send(adapter, "!definitelynotacommand wow")
        assert event is not None
        assert event.text == "!definitelynotacommand wow" and event.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_thread_command_fails_closed_when_root_lookup_fails():
    b = _bot(BOT_B)
    b._client.get_event = AsyncMock(side_effect=RuntimeError("boom"))
    assert await _send(b, "!status") is None
    assert await _send(b, f"!status {BOT_B}", mentions=[BOT_B]) is not None  # addressed needs no lookup


@pytest.mark.asyncio
async def test_synthetic_root_creator_owns_thread():
    """Auto-thread: the bot that turned a room message into a thread root owns its commands."""
    a = _bot(BOT_A, auto_thread=True)
    a._threads._threads.clear()
    event = await _send(a, f"hey {BOT_A}", thread=None, mentions=[BOT_A], event_id="$synth")
    assert event is not None and event.source.thread_id == "$synth"
    assert "$synth" in a._owned_threads
    b = _bot(BOT_B, root_author=HUMAN)
    b._threads._remember("$synth")
    assert await _send(a, "!status", thread="$synth") is not None
    assert await _send(b, "!status", thread="$synth") is None


@pytest.mark.asyncio
async def test_reply_quote_pill_is_not_an_address():
    a, b = _thread_pair()
    body = f"> <{BOT_B}> earlier answer\n\n!status"
    assert await _send(b, body, in_reply_to="$prev") is None  # quoting B does not address B
    assert await _send(a, body, in_reply_to="$prev") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_room_command_needs_address_when_mention_required(command):
    a, b = _bot(BOT_A), _bot(BOT_B)
    assert await _send(a, command, thread=None) is None
    assert await _send(b, command, thread=None) is None  # no longer handled by every bot
    assert await _send(a, f"{command} {BOT_A}", thread=None, mentions=[BOT_A]) is not None
    assert await _send(b, f"{command} {BOT_A}", thread=None, mentions=[BOT_A]) is None
    assert (await _send(b, f"{command} b", thread=None)).text == "/status"


@pytest.mark.asyncio
async def test_room_command_free_room_and_open_rooms_keep_their_rule():
    free = _bot(BOT_A, free_response_rooms=ROOM)
    free._free_rooms = {ROOM}
    assert (await _send(free, "!status", thread=None)).text == "/status"
    open_room = _bot(BOT_A, require_mention=False)
    assert (await _send(open_room, "/status", thread=None)).text == "/status"


@pytest.mark.asyncio
async def test_dm_commands_unaffected():
    a = _bot(BOT_A)
    a._is_dm_room = AsyncMock(return_value=True)
    assert (await _send(a, "!status", thread=None)).text == "/status"
