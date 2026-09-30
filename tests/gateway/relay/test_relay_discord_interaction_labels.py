"""A relayed Discord interaction carries the same chat/user labels as a relayed message in that chat.

The text lane gets ``chat_name``, ``chat_topic`` and ``user_display_name`` from the connector; a
forwarded interaction (slash command, component) is the raw Discord body. Both land in one session,
and the pinned session-context prompt renders those labels, so a slash turn built without them
re-rendered the cached prefix and the next message rendered it back.
"""

import json
import threading
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.relay.ws_transport import _event_from_wire
from gateway.session import SessionStore, build_session_context, build_session_key
from tests.gateway.relay.test_relay_interactive import _adapter


def _forward(**payload):
    body = {"type": 2, "id": "i1", "channel_id": "ch1", "guild_id": "g1", "data": {"name": "status"}}
    body.update(payload)

    class Forward:
        platform, method, path = "discord", "POST", "/interactions/bot1"

    Forward.body = json.dumps(body).encode()
    return Forward()


def _pinned_prompt(source):
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    return runner._pinned_session_context_prompt(build_session_context(source, config), False, "k")


def _message(chat, user="u1", chat_name="Hermes Server / #ops", chat_topic="Incident triage"):
    return _event_from_wire({"text": "hi", "message_type": "text", "source": {
        "platform": "discord", **chat, "scope_id": "g1",
        "user_id": user, "user_name": "ben", "user_display_name": "Ben D",
        "chat_name": chat_name, "chat_topic": chat_topic, "message_id": f"{user}:{chat_name}"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "warm", "restart", "thread", "rename", "peer-rename", "late-session", "read-retry"])
async def test_slash_between_messages_keeps_one_pinned_prompt(tmp_path, case):
    """Every case but ``warm`` and ``thread`` restarts the gateway after the last message, so the
    slash command is the first event the new process sees in that chat; its chat labels are the
    ones the session store recorded when the text lane carried them.
    ``thread``: both events happen inside a thread, which the text lane keys on chat_type "thread" +
    thread_id; the slash command must land in that session, not in a per-user "group" one.
    ``rename`` / ``peer-rename``: the channel was renamed before the restart. In ``peer-rename``
    another user's message carried the new labels and this user's session was reset since: a reset
    inherits the origin (old labels) under a fresh ``created_at``.
    ``late-session``: another user's older message, still carrying the old labels, only gets its
    session after the rename was observed. What the chat is called follows the order the labels
    were observed in, not the order its sessions were created in.
    ``read-retry``: the first read after the restart fails. That interaction goes through without
    labels; the next one asks the store again instead of keeping "no labels"."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    chat = ({"chat_id": "th1", "chat_type": "thread", "thread_id": "th1", "parent_chat_id": "ch1"}
            if case == "thread" else {"chat_id": "ch1", "chat_type": "group"})
    renamed = {"chat_name": "Hermes Server / #triage", "chat_topic": "Renamed"}
    db, writers = store._routing_db, []
    set_meta = db.set_meta

    def recording(key, value, **kwargs):
        if key.startswith("gateway_chat_labels:"):
            writers.append(threading.get_ident())
        set_meta(key, value, **kwargs)

    db.set_meta = recording

    async def relay(event):
        await adapter._on_inbound(event)
        return store.get_or_create_session(event.source)

    if case == "late-session":
        late = _message(chat, "u2")
        await adapter._on_inbound(late)
        message = _message(chat, **renamed)
        await relay(message)
        store.get_or_create_session(late.source)
    else:
        if case == "peer-rename":
            await relay(_message(chat, "u2"))
        message = _message(chat)
        entry = await relay(message)
        if case in ("rename", "peer-rename"):
            await relay(_message(chat, "u1" if case == "rename" else "u2", **renamed))
            message = _message(chat, **renamed)
        if case == "peer-rename":
            store.reset_session(entry.session_key)
    if case not in ("warm", "thread"):
        store = SessionStore(tmp_path, config)
        adapter, _stub = _adapter(platform="discord")
        adapter.set_session_store(store)
    interaction = {"member": {"nick": "Ben D", "user": {"id": "u1", "username": "ben"}}}
    if case == "thread":
        interaction.update(channel_id="th1", channel={"id": "th1", "type": 11, "parent_id": "ch1"})
    if case == "read-retry":
        get_meta, faults = store._routing_db.get_meta, [OSError("database is locked")]

        def flaky(key):
            if faults:
                raise faults.pop()
            return get_meta(key)

        store._routing_db.get_meta = flaky
        assert adapter._discord_interaction_to_event(_forward(**interaction)).source.chat_name is None
    slash = adapter._discord_interaction_to_event(_forward(**interaction))

    assert build_session_key(slash.source) == build_session_key(message.source)
    prompts = {_pinned_prompt(event.source) for event in (message, slash, message)}
    assert len(prompts) == 1
    # Recording the labels is a disk write and _on_inbound runs on the gateway loop.
    assert writers and threading.get_ident() not in writers


@pytest.mark.asyncio
@pytest.mark.parametrize("member, expected", [
    ({"nick": "Benny", "user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Benny"),
    ({"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Ben D"),
    ({"user": {"id": "u1", "username": "ben"}}, "ben"),
])
async def test_interaction_names_the_user_as_the_text_lane_would_now(member, expected):
    """The text lane's user_display_name is the native author.display_name (guild nick, else global
    name, else username), and an interaction carries all three as of now. The name an earlier
    message carried ("Ben D" here) must not win: it predates a nickname change."""
    adapter, _stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(_message({"chat_id": "ch1", "chat_type": "group"}))
    assert adapter._discord_interaction_to_event(_forward(member=member)).source.user_name == expected
