"""Slack native task cards name the channel's workspace when the turn does not (#148998).

``chat.startStream`` into a channel needs ``recipient_team_id``. The gateway copies it from the
turn's ``source.scope_id``, which a bot-authored channel turn can lack, so Slack answered
``missing_recipient_team_id`` and the turn fell back to plain-text progress. The adapter knows the
workspace that owns the channel (``scope_id_for_chat``, as the plain-text stream uses); a team the
turn names still wins.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("installed", "channel_owner", "turn_team", "expected"),
    [
        pytest.param(["T1"], None, None, "T1", id="single-workspace-unseen-channel"),
        pytest.param(["T1", "T2"], "T1", None, "T1", id="recorded-channel-owner"),
        pytest.param(["T1", "T2"], "T1", "T2", "T2", id="turn-team-wins"),
    ],
)
async def test_channel_card_start_names_a_workspace(installed, channel_owner, turn_team, expected):
    client = MagicMock(api_call=AsyncMock(return_value={"ts": "stream-1"}))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock(client=client)
    adapter._team_clients.update(dict.fromkeys(installed, client))
    if channel_owner:
        adapter._remember_channel_team("C1", channel_owner)
    metadata = {"thread_id": "1700.100", "recipient_user_id": "U_NOTIFIER"}
    if turn_team:
        metadata["slack_team_id"] = turn_team

    result = await adapter.send_native_task_card_progress(
        "C1", [{"id": "call-1", "title": "terminal", "status": "in_progress"}], metadata=metadata)

    assert result.success is True
    (start,) = [c.kwargs["json"] for c in client.api_call.await_args_list if c.args[0] == "chat.startStream"]
    assert start.get("recipient_team_id") == expected
