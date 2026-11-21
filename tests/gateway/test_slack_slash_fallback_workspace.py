"""A slash reply's chat.postEphemeral fallback posts from the workspace the reply is for (#130313),
not whichever one the channel map or the primary client names."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_team, reply_team, expected", [
    (None, "T_SEC", "T_SEC"),     # shared channel: the map names no single owner
    ("T_PRI", "T_SEC", "T_SEC"),  # the map points at the other workspace
    ("T_SEC", "T_PRI", "T_PRI"),
], ids=["ambiguous-map", "map-says-primary", "map-says-secondary"])
async def test_send_slash_ephemeral_fallback_posts_from_the_reply_workspace(
    adapter, channel_team, reply_team, expected
):
    """When response_url fails, chat.postEphemeral goes out through the workspace the reply is
    for, not whichever one the channel map (or the primary client) names."""
    import time
    from plugins.platforms.slack.adapter import _slash_user_id

    clients = {"T_PRI": adapter._app.client, "T_SEC": AsyncMock()}
    for team, client in clients.items():
        client.chat_postEphemeral = AsyncMock(return_value=(
            {"ok": True} if team == expected else {"ok": False, "error": "channel_not_found"}))
    adapter._team_clients = clients
    adapter._channel_team = {"C1": channel_team} if channel_team else {}
    adapter._channel_teams = {} if channel_team else {"C1": {"T_PRI", "T_SEC"}}
    adapter._slash_command_contexts[(reply_team, "C1", "U1")] = {
        "response_url": "https://hooks.slack.com/commands/timeout",
        "user_id": "U1",
        "ts": time.monotonic(),
    }
    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=Exception("connection timeout"))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    token = _slash_user_id.set("U1")
    try:
        with patch(
            "plugins.platforms.slack.adapter.aiohttp.ClientSession", return_value=mock_session
        ):
            result = await adapter.send(
                "C1", "Some response", metadata={"slack_team_id": reply_team})
    finally:
        _slash_user_id.reset(token)

    assert result.success is True
    assert {team: c.chat_postEphemeral.await_count for team, c in clients.items()} == {
        team: int(team == expected) for team in clients}
