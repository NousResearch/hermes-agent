"""Slack approval buttons: the click must resolve even when the send side and the click
side disagree about whether a team id is present.

Regression: a prompt stored team-scoped (``(team_id, ts)``) but clicked from an Enterprise
Grid payload carrying only ``enterprise`` (so ``_event_team_id`` is empty) missed both the
request-id lookup and the guard's atomic pop, which returned its ``True`` default and returned
early. The click was swallowed with no log, the agent stayed blocked, and the card timed out
as if nobody had clicked.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_approval_buttons import _make_adapter, _attach_auth_runner

TS = "1790030306.663649"
TEAM = "T1"
REQUEST_ID = "req-1"


def _body(team):
    b = {"message": {"ts": TS, "blocks": []}, "channel": {"id": "D0AV0M0DRFE"},
         "user": {"name": "owner", "id": "U_OWNER"}}
    if team:
        b["team"] = {"id": team}
    else:  # Enterprise Grid org payload: enterprise only, no workspace team
        b["enterprise"] = {"id": "E015GUGD2V6"}
        b["authorizations"] = [{"enterprise_id": "E015GUGD2V6", "team_id": None}]
    return b


def _stored_prompt(stored_team):
    """An adapter holding one live approval card, keyed the way the send side keys it."""
    adapter = _make_adapter()
    _attach_auth_runner(adapter)
    key = adapter._workspace_message_marker(stored_team, TS)
    adapter._approval_resolved[key] = False
    adapter._approval_request_ids[key] = REQUEST_ID
    adapter._team_clients[TEAM].chat_update = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_team,click_team", [
    (TEAM, ""),    # the bug: stored team-scoped, Grid click resolves no team
    ("", TEAM),    # the inverse the old one-way compensation already handled
    (TEAM, TEAM),
    ("", ""),
])
async def test_click_resolves_despite_team_key_mismatch(stored_team, click_team):
    adapter = _stored_prompt(stored_team)

    with patch("tools.approval.resolve_gateway_approval_card", return_value=1) as res:
        await adapter._handle_approval_action(
            AsyncMock(), _body(click_team),
            {"action_id": "hermes_approve_once", "value": "sess-key"})

    assert res.called, "click was swallowed — agent stays blocked and times out"
    assert res.call_args[0][1] == "once"
    # The card's own request id must travel with the click, or the resolver settles nothing.
    assert res.call_args[0][2] == REQUEST_ID
    assert not adapter._approval_resolved and not adapter._approval_request_ids


@pytest.mark.asyncio
async def test_second_click_is_still_ignored():
    """The guard must keep rejecting double-clicks."""
    adapter = _stored_prompt(TEAM)
    action = {"action_id": "hermes_approve_once", "value": "sess-key"}

    with patch("tools.approval.resolve_gateway_approval_card", return_value=1) as res:
        await adapter._handle_approval_action(AsyncMock(), _body(""), action)
        assert res.call_count == 1
        await adapter._handle_approval_action(AsyncMock(), _body(""), action)
        assert res.call_count == 1, "double-click guard regressed"


@pytest.mark.asyncio
async def test_click_for_another_message_does_not_steal_a_live_prompt():
    """The ts fallback matches only the clicked message, never a different live card."""
    adapter = _stored_prompt(TEAM)
    body = _body("")
    body["message"]["ts"] = "1790030399.000001"

    with patch("tools.approval.resolve_gateway_approval_card", return_value=1) as res:
        await adapter._handle_approval_action(
            AsyncMock(), body, {"action_id": "hermes_approve_once", "value": "sess-key"})

    res.assert_not_called()
    assert adapter._approval_resolved == {(TEAM, TS): False}
