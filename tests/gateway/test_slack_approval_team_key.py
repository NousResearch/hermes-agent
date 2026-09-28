"""Slack approval buttons: the click must resolve even when the send side and the click
side disagree about whether a team id is present.

Regression: a prompt stored team-scoped (``(team_id, ts)``) but clicked from an Enterprise
Grid payload carrying only ``enterprise`` (so ``_event_team_id`` is empty) missed the guard's
atomic pop, which returned its ``True`` default and returned early. The click was swallowed
with no log, the agent stayed blocked, and the card timed out as if nobody had clicked.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_approval_buttons import _make_adapter, _attach_auth_runner

TS = "1790030306.663649"
TEAM = "T1"


def _body(team):
    b = {"message": {"ts": TS, "blocks": []}, "channel": {"id": "D0AV0M0DRFE"},
         "user": {"name": "owner", "id": "U_OWNER"}}
    if team:
        b["team"] = {"id": team}
    else:  # Enterprise Grid org payload: enterprise only, no workspace team
        b["enterprise"] = {"id": "E015GUGD2V6"}
        b["authorizations"] = [{"enterprise_id": "E015GUGD2V6", "team_id": None}]
    return b


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_team,click_team", [
    (TEAM, ""),    # the bug: stored team-scoped, Grid click resolves no team
    ("", TEAM),    # the inverse the old one-way compensation already handled
    (TEAM, TEAM),
    ("", ""),
])
async def test_click_resolves_despite_team_key_mismatch(stored_team, click_team):
    adapter = _make_adapter()
    _attach_auth_runner(adapter)
    adapter._approval_resolved[
        adapter._workspace_message_marker(stored_team, TS)] = False
    adapter._team_clients[TEAM].chat_update = AsyncMock()

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as res:
        await adapter._handle_approval_action(
            AsyncMock(), _body(click_team),
            {"action_id": "hermes_approve_once", "value": "sess-key"})

    assert res.called, "click was swallowed — agent stays blocked and times out"
    assert res.call_args[0][1] == "once"


@pytest.mark.asyncio
async def test_second_click_is_still_ignored():
    """The guard must keep rejecting double-clicks."""
    adapter = _make_adapter()
    _attach_auth_runner(adapter)
    adapter._approval_resolved[adapter._workspace_message_marker(TEAM, TS)] = False
    adapter._team_clients[TEAM].chat_update = AsyncMock()
    action = {"action_id": "hermes_approve_once", "value": "sess-key"}

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as res:
        await adapter._handle_approval_action(AsyncMock(), _body(""), action)
        assert res.call_count == 1
        await adapter._handle_approval_action(AsyncMock(), _body(""), action)
        assert res.call_count == 1, "double-click guard regressed"
