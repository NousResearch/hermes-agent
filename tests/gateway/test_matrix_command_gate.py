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
BOB = "@bob:example.org"  # a second human, also allowed
BOT_A = "@a:example.org"
BOT_B = "@b:example.org"
BOT_C = "@c:example.org"


def _bot(user_id, *, root_author=HUMAN, owns=(), members=None, **extra):
    """An adapter for one bot in a group room. ``root_author`` is the sender the homeserver
    reports for the thread root event; ``owns`` pre-seeds threads this bot started."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test",
        extra={"homeserver": "https://matrix.example.org", "user_id": user_id,
               "allowed_users": f"{HUMAN},{BOB}", **extra}))
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


async def _send(adapter, body, *, thread: str | None = ROOT, mentions=None, in_reply_to=None, room=ROOM, event_id="$e1",
                sender=HUMAN, msgtype="m.text"):
    """Feed a text event through the real text path; return the MessageEvent handled, else None."""
    content = {"msgtype": msgtype, "body": body}
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
    await adapter._handle_text_message(room, sender, event_id, time.time(), content, relates)
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


# --- Sole-agent contexts need no mention; several agents still do ------------------------------

ONE_AGENT = [BOT_A, HUMAN, BOB]  # a user (plus another human) and a single agent
THREE_AGENTS = [BOT_A, BOT_B, BOT_C, HUMAN]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
@pytest.mark.parametrize("members", [ONE_AGENT, [BOT_A, HUMAN]])
async def test_one_on_one_room_command_needs_no_mention(command, members):
    a = _bot(BOT_A, members=members)
    event = await _send(a, command, thread=None)
    assert event is not None and event.text == "/status" and event.message_type == MessageType.COMMAND
    assert await _send(a, f"{command} @someoneelse", thread=None) is None  # addressed to someone else: still dropped


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_two_party_thread_command_needs_no_mention(command):
    # The user started the thread (root by HUMAN), only this agent takes part: no starter, no mention.
    a = _bot(BOT_A, root_author=HUMAN, members=ONE_AGENT)
    event = await _send(a, command)
    assert event is not None and event.text == "/status" and event.message_type == MessageType.COMMAND
    # Even a thread this agent has not posted in yet: one possible recipient, so no mention needed.
    a._threads._threads.clear()
    event = await _send(a, command)
    assert event is not None and event.text == "/status" and event.message_type == MessageType.COMMAND
    assert await _send(a, f"{command} @someoneelse") is None  # addressed to someone else: still dropped


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_three_agents_room_command_needs_mention_and_reaches_addressed_bot_only(command):
    bots = {u: _bot(u, members=THREE_AGENTS) for u in (BOT_A, BOT_B, BOT_C)}
    for bot in bots.values():
        assert await _send(bot, command, thread=None) is None  # no mention: nobody acts
    body, mentions = f"{command} {BOT_B}", [BOT_B]
    handled = {u: await _send(bot, body, thread=None, mentions=mentions) for u, bot in bots.items()}
    assert handled[BOT_A] is None and handled[BOT_C] is None
    assert handled[BOT_B] is not None and handled[BOT_B].text == "/status"


@pytest.mark.asyncio
async def test_three_agents_thread_still_needs_starter_or_mention():
    a = _bot(BOT_A, owns=(ROOT,), root_author=BOT_A, members=THREE_AGENTS)
    b = _bot(BOT_B, root_author=BOT_A, members=THREE_AGENTS)
    assert await _send(b, "!status") is None
    assert await _send(b, f"!status {BOT_B}", mentions=[BOT_B]) is not None
    assert await _send(a, "!status") is not None  # the starter keeps its existing rule


@pytest.mark.asyncio
async def test_unknown_membership_fails_closed_to_mention_required():
    a = _bot(BOT_A)  # no member list available
    assert await _send(a, "!status", thread=None) is None
    assert await _send(a, f"!status {BOT_A}", thread=None, mentions=[BOT_A]) is not None


# --- Only a human we can receive from issues commands -------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_commands_written_by_bots_are_ignored(command):
    a = _bot(BOT_A, members=ONE_AGENT + [BOT_B], process_notices="true")
    # Another agent's output: mentions us, echoes the command, or is an m.notice.
    assert await _send(a, f"{command} {BOT_A}", thread=None, mentions=[BOT_A], sender=BOT_B) is None
    assert await _send(a, command, thread=None, sender=BOT_B) is None
    assert await _send(a, f"{command} {BOT_A}", mentions=[BOT_A], sender=BOT_B) is None  # in a thread too
    assert await _send(a, f"{command} {BOT_A}", thread=None, mentions=[BOT_A], msgtype="m.notice") is None
    assert await _send(a, command, thread=None, sender="@_bridge_1:example.org") is None


@pytest.mark.asyncio
async def test_quoted_or_echoed_command_inside_a_human_reply_is_not_a_command():
    a = _bot(BOT_A, members=ONE_AGENT)
    quoted = f"> <{BOT_B}> /status\n\nthanks, that worked"
    event = await _send(a, quoted, in_reply_to="$prev")
    assert event is not None and event.message_type == MessageType.TEXT and event.text == "thanks, that worked"
    echo = await _send(a, "agent said: /status then !model", thread=None, mentions=[BOT_A])
    assert echo is not None and echo.message_type == MessageType.TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [None, ROOT])
async def test_command_only_from_a_user_the_agent_receives_from_and_who_is_in_the_room(thread):
    a = _bot(BOT_A, members=ONE_AGENT)
    stranger = "@mallory:example.org"  # in the room but not allowed
    a._joined_member_ids = AsyncMock(return_value={m.lower() for m in ONE_AGENT + [stranger]})
    assert await _send(a, f"!status {BOT_A}", thread=thread, mentions=[BOT_A], sender=stranger) is None
    assert await _send(a, "!status", thread=thread, sender=stranger) is None
    a._joined_member_ids = AsyncMock(return_value={m.lower() for m in ONE_AGENT})
    assert await _send(a, "!status", thread=thread, sender=BOB) is not None  # allowed human
    # Allowed on paper but no longer a member of this room/thread.
    left = _bot(BOT_A, members=[BOT_A, BOB])
    assert await _send(left, f"!status {BOT_A}", thread=thread, mentions=[BOT_A], sender=HUMAN) is None
    assert await _send(left, "!status", thread=thread, sender=BOB) is not None


# --- Agent identity comes from MATRIX_AGENT_USERS / local profiles, never from the allowlist ----

# The live shape: vera's allowlist names the human AND another agent (taxis's bot account).
ALLOW_WITH_AGENT = f"{HUMAN},{BOT_B}"
ROOM_WITH_TWO_AGENTS = [BOT_A, BOT_B, HUMAN]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_allowlisted_agent_still_counts_as_an_agent(command):
    a = _bot(BOT_A, members=ROOM_WITH_TWO_AGENTS, allowed_users=ALLOW_WITH_AGENT, agent_users=BOT_B)
    assert a._is_sole_agent({m.lower() for m in ROOM_WITH_TWO_AGENTS}) is False
    assert await _send(a, command, thread=None) is None  # two agents: a bare command needs a mention
    assert await _send(a, command) is None  # ... in a thread too
    assert await _send(a, f"{command} {BOT_A}", thread=None, mentions=[BOT_A]) is not None
    # Without the second agent in the room the same config is a sole-agent room again.
    alone = _bot(BOT_A, members=[BOT_A, HUMAN], allowed_users=ALLOW_WITH_AGENT, agent_users=BOT_B)
    assert await _send(alone, command, thread=None) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "!status"])
async def test_command_written_by_an_allowlisted_agent_is_ignored(command):
    """Hermes bots send m.text, so the msgtype cannot tell; the sender identity does."""
    a = _bot(BOT_A, members=ROOM_WITH_TWO_AGENTS, allowed_users=ALLOW_WITH_AGENT, agent_users=BOT_B)
    for thread in (None, ROOT):
        assert await _send(a, command, thread=thread, sender=BOT_B) is None
        assert await _send(a, f"{command} {BOT_A}", thread=thread, mentions=[BOT_A], sender=BOT_B) is None
    assert await _send(a, f"{command} {BOT_A}", thread=None, mentions=[BOT_A], sender=HUMAN) is not None


@pytest.mark.asyncio
async def test_agents_discovered_from_local_profile_env_files(tmp_path, monkeypatch):
    import hermes_constants

    (tmp_path / "profiles" / "taxis").mkdir(parents=True)
    (tmp_path / "profiles" / "taxis" / ".env").write_text(f'FOO=1\nMATRIX_USER_ID="{BOT_B}"\nMATRIX_ACCESS_TOKEN=secret\n')
    (tmp_path / "profiles" / "broken").mkdir()
    (tmp_path / "profiles" / "broken" / ".env").write_bytes(b"\xff\xfe MATRIX_USER_ID=\n")
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda *a, **k: tmp_path)
    a = _bot(BOT_A, members=ROOM_WITH_TWO_AGENTS, allowed_users=ALLOW_WITH_AGENT)  # no explicit agent_users
    assert a._local_agent_ids() == {BOT_B}
    assert a._is_known_agent(BOT_B.upper()) and not a._is_known_agent(HUMAN)
    assert await _send(a, "!status", thread=None) is None
    assert await _send(a, f"!status {BOT_A}", thread=None, mentions=[BOT_A], sender=BOT_B) is None
    assert await _send(a, f"!status {BOT_A}", thread=None, mentions=[BOT_A]) is not None


@pytest.mark.asyncio
async def test_agent_users_env_var(monkeypatch):
    monkeypatch.setenv("MATRIX_AGENT_USERS", f"{BOT_B}, {BOT_C}")
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="t", extra={"homeserver": "https://m.example.org", "user_id": BOT_A}))
    assert adapter._agent_user_ids == {BOT_B, BOT_C}


# --- Member lookup: joined only, API fallback, empty means unknown ----------------------------

@pytest.mark.asyncio
async def test_joined_members_ask_only_for_joined_state():
    from mautrix.types import Membership

    a = _bot(BOT_A, members=[BOT_A, HUMAN])
    assert await a._joined_member_ids(ROOM) == {BOT_A, HUMAN}
    a._client.state_store.get_members.assert_awaited_once_with(ROOM, memberships=(Membership.JOIN,))


@pytest.mark.asyncio
async def test_joined_members_fall_back_to_api_and_empty_is_unknown():
    a = _bot(BOT_A, members=[])  # MemoryStateStore answers [] for a room it never saw
    assert await a._joined_member_ids(ROOM) is None
    assert await _send(a, "!status", thread=None) is None  # unknown membership: mention still required
    a._client.get_joined_members = AsyncMock(return_value={BOT_A: object(), HUMAN: object()})  # {user_id: Member}
    assert await a._joined_member_ids(ROOM) == {BOT_A, HUMAN}
    assert (await _send(a, "!status", thread=None)).text == "/status"
    a._client.get_joined_members = AsyncMock(return_value={})
    assert await a._joined_member_ids(ROOM) is None
    a._client.get_joined_members = AsyncMock(side_effect=RuntimeError("boom"))
    assert await a._joined_member_ids(ROOM) is None
