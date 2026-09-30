"""A relayed Discord interaction carries the same chat/user labels as a relayed message in that chat.

The text lane gets ``chat_name``, ``chat_topic`` and ``user_display_name`` from the connector; a
forwarded interaction (slash command, component) is the raw Discord body. Both land in one session,
and the pinned session-context prompt renders those labels, so a slash turn built without them
re-rendered the cached prefix and the next message rendered it back.
"""

import json
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
@pytest.mark.parametrize("restart, nick, thread, renamed_by", [
    (False, "Benny", False, None), (True, "Ben D", False, None), (False, "Benny", True, None),
    (True, "Ben D", False, "u1"), (True, "Ben D", False, "u2"),
], ids=["warm", "restart", "thread", "rename", "peer-rename"])
async def test_slash_between_messages_keeps_one_pinned_prompt(tmp_path, restart, nick, thread, renamed_by):
    """``restart``: the gateway restarted after the message, so the slash command is the first
    event the new process sees in that chat; the chat labels come from the persisted session origin.
    ``thread``: both events happen inside a thread, which the text lane keys on chat_type "thread" +
    thread_id; the slash command must land in that session, not in a per-user "group" one.
    ``renamed_by``: the channel was renamed before the restart and that user's message carried the
    new labels. An origin is written when its session is created and inherited by a reset, so the
    rename must reach the origin of every session in the chat: in ``peer-rename`` another user's
    message carried it and this user's session was reset since."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    chat = ({"chat_id": "th1", "chat_type": "thread", "thread_id": "th1", "parent_chat_id": "ch1"}
            if thread else {"chat_id": "ch1", "chat_type": "group"})

    async def relay(event):
        await adapter._on_inbound(event)
        return store.get_or_create_session(event.source)

    if renamed_by == "u2":
        await relay(_message(chat, "u2"))
    message = _message(chat)
    entry = await relay(message)
    if renamed_by:
        renamed = {"chat_name": "Hermes Server / #triage", "chat_topic": "Renamed"}
        await relay(_message(chat, renamed_by, **renamed))
        message = _message(chat, **renamed)
        if renamed_by == "u2":
            store.reset_session(entry.session_key)
    if restart:
        adapter, _stub = _adapter(platform="discord")
        adapter.set_session_store(SessionStore(tmp_path, config))
    interaction = {"member": {"nick": nick, "user": {"id": "u1", "username": "ben"}}}
    if thread:
        interaction.update(channel_id="th1", channel={"id": "th1", "type": 11, "parent_id": "ch1"})
    slash = adapter._discord_interaction_to_event(_forward(**interaction))

    assert build_session_key(slash.source) == build_session_key(message.source)
    prompts = {_pinned_prompt(event.source) for event in (message, slash, message)}
    assert len(prompts) == 1


@pytest.mark.parametrize("member, expected", [
    ({"nick": "Benny", "user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Benny"),
    ({"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Ben D"),
    ({"user": {"id": "u1", "username": "ben"}}, "ben"),
])
def test_first_interaction_names_the_user_like_the_text_lane(member, expected):
    """With no relayed message seen yet, the name follows the text lane's contract:
    user_display_name is the native author.display_name (guild nick, else global name, else username)."""
    adapter, _stub = _adapter(platform="discord")
    assert adapter._discord_interaction_to_event(_forward(member=member)).source.user_name == expected
