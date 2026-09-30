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
from gateway.session import SessionStore, build_session_context
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


@pytest.mark.asyncio
@pytest.mark.parametrize("restart, nick", [(False, "Benny"), (True, "Ben D")], ids=["warm", "restart"])
async def test_slash_between_messages_keeps_one_pinned_prompt(tmp_path, restart, nick):
    """``restart``: the gateway restarted after the message, so the slash command is the first
    event the new process sees in that chat; the chat labels come from the persisted session origin."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    adapter, _stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    message = _event_from_wire({"text": "hi", "message_type": "text", "source": {
        "platform": "discord", "chat_id": "ch1", "chat_type": "group", "scope_id": "g1",
        "user_id": "u1", "user_name": "ben", "user_display_name": "Ben D",
        "chat_name": "Hermes Server / #ops", "chat_topic": "Incident triage", "message_id": "m1"}})
    await adapter._on_inbound(message)
    SessionStore(tmp_path, config).get_or_create_session(message.source)
    if restart:
        adapter, _stub = _adapter(platform="discord")
        adapter.set_session_store(SessionStore(tmp_path, config))
    slash = adapter._discord_interaction_to_event(
        _forward(member={"nick": nick, "user": {"id": "u1", "username": "ben"}}))

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
