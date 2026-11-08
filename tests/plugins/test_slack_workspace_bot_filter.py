"""Bot mentions and self-message suppression use the receiving workspace's identity."""

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.delenv("SLACK_ALLOW_BOTS", raising=False)
    value = SlackAdapter(PlatformConfig(enabled=True, extra={"allow_bots": "mentions"}))
    value._bot_user_id = "U_PRIMARY"
    value._team_bot_user_ids = {"T_PRIMARY": "U_PRIMARY", "T_SECONDARY": "U_SECONDARY"}
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("event_team,payload", [
    ({"team": "T_SECONDARY"}, None),
    ({"team_id": "T_SECONDARY"}, None),
    ({}, {"team_id": "T_SECONDARY"}),
    ({}, {"authorizations": [{"team_id": "T_SECONDARY"}]}),
])
@pytest.mark.parametrize("mention,accepted", [("U_SECONDARY", True), ("U_PRIMARY", False)])
async def test_peer_bot_mention_is_workspace_scoped(adapter, event_team, payload, mention, accepted):
    event = {
        "user": "U_PEER", "bot_id": "B_PEER", "channel": "C_SECONDARY",
        "text": f"<@{mention}> what are you?", "ts": "100.001", **event_team,
    }
    result = await adapter._prefilter_inbound(event, payload)
    assert (result is not None) is accepted
    if accepted:
        assert result == (event, "T_SECONDARY", "C_SECONDARY")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["mentions", "all"])
@pytest.mark.parametrize("team,user", [("T_PRIMARY", "U_PRIMARY"), ("T_SECONDARY", "U_SECONDARY")])
async def test_own_bot_message_is_dropped_in_each_workspace(adapter, policy, team, user):
    adapter.config.extra["allow_bots"] = policy
    event = {
        "user": user, "bot_id": "B_SELF", "channel": "C_TEST",
        "text": f"<@{user}> my own reply", "ts": "100.002",
    }
    assert await adapter._prefilter_inbound(event, {"team_id": team}) is None


@pytest.mark.asyncio
async def test_primary_identity_remains_fallback_without_workspace(adapter):
    event = {"user": "U_PEER", "bot_id": "B_PEER", "text": "<@U_PRIMARY> hello"}
    assert await adapter._drop_bot_sender(event) is False
