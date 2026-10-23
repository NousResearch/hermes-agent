"""
Tests for Slack Connect external-org user handling (#143792).

In a channel shared to another workspace (``is_ext_shared_channel``), one external
@mention arrives as two events with the same ``ts`` — a ``message`` event whose inner
``event.team`` is the *sender's* org, and an ``app_mention`` event whose inner ``team``
is the host workspace. The envelope ``team_id`` is the installing workspace on both.
Resolving the workspace from the inner event first therefore split the dedup key and
the session scope across two teams (double reply, one session keyed to a workspace the
app is not installed in). External-org users can also DM the bot; those DMs must be
refused regardless of allowlists.

Event shapes follow the live payloads captured in the issue.
"""

import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.slack.adapter import SlackAdapter

BOT_USER_ID = "U_BOT_123"
HOST_TEAM = "T_HOST"
EXT_TEAM = "T_EXT"
SHARED_CHANNEL = "C_SHARED"
SHARED_TS = "1712345678.000001"


def _make_adapter():
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    adapter._dedup = MessageDeduplicator(ttl_seconds=1800)
    return adapter


def _connect_message_event():
    """The ``message`` half of the external @mention pair: inner ``team`` is the sender's org."""
    return {
        "type": "message",
        "team": EXT_TEAM,
        "user": "U_EXT_USER",
        "client_msg_id": "conn-msg-1",
        "channel": SHARED_CHANNEL,
        "channel_type": "channel",
        "text": f"<@{BOT_USER_ID}> hello",
        "ts": SHARED_TS,
        "is_ext_shared_channel": True,
    }


def _connect_message_body():
    return {
        "team_id": HOST_TEAM,
        "authorizations": [{"team_id": HOST_TEAM}],
        "context_team_id": HOST_TEAM,
        "is_ext_shared_channel": True,
    }


def _connect_mention_event():
    """The ``app_mention`` half: inner ``team`` is the host workspace, ``user_team`` the org."""
    return {
        "type": "app_mention",
        "team": HOST_TEAM,
        "user_team": EXT_TEAM,
        "source_team": EXT_TEAM,
        "user": "U_EXT_USER",
        "client_msg_id": "conn-msg-1",
        "channel": SHARED_CHANNEL,
        "text": f"<@{BOT_USER_ID}> hello",
        "ts": SHARED_TS,
        "is_ext_shared_channel": True,
    }


def _connect_mention_body():
    return {
        "team_id": HOST_TEAM,
        "authorizations": [{"team_id": HOST_TEAM}],
        "is_ext_shared_channel": True,
    }


def _dm_event(team: str, ts: str):
    return {
        "type": "message",
        "channel_type": "im",
        "team": team,
        "user": "U_EXT_USER" if team == EXT_TEAM else "U_LOCAL_USER",
        "client_msg_id": f"dm-{ts}",
        "channel": "D_INBOX_1",
        "text": "hi",
        "ts": ts,
    }


# ---------------------------------------------------------------------------
# Tests: workspace resolution prefers the envelope over the inner event
# ---------------------------------------------------------------------------


def test_connect_message_event_resolves_to_install_workspace():
    adapter = _make_adapter()
    team = adapter._event_team_id(_connect_message_event(), _connect_message_body())
    assert team == HOST_TEAM


def test_connect_mention_event_resolves_to_install_workspace():
    adapter = _make_adapter()
    team = adapter._event_team_id(_connect_mention_event(), _connect_mention_body())
    assert team == HOST_TEAM


def test_envelope_without_team_id_still_falls_back_to_inner_event():
    """Bolt paths that pass only the inner event keep resolving its team."""
    adapter = _make_adapter()
    team = adapter._event_team_id({"team": HOST_TEAM}, None)
    assert team == HOST_TEAM


# ---------------------------------------------------------------------------
# Tests: the Connect message+mention pair collapses to one turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_pair_starts_one_turn_only():
    adapter = _make_adapter()
    first = await adapter._prefilter_inbound(_connect_message_event(), _connect_message_body())
    assert first is not None
    _event, team_id, channel_id = first
    assert team_id == HOST_TEAM
    assert channel_id == SHARED_CHANNEL

    second = await adapter._prefilter_inbound(_connect_mention_event(), _connect_mention_body())
    assert second is None, "the same-ts app_mention must be deduped against the message event"


# ---------------------------------------------------------------------------
# Tests: external-org DMs are refused, local-org DMs pass the gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_org_dm_is_refused():
    adapter = _make_adapter()
    accepted = await adapter._prefilter_inbound(
        _dm_event(EXT_TEAM, "1712345680.000100"), {"team_id": HOST_TEAM})
    assert accepted is None


@pytest.mark.asyncio
async def test_local_org_dm_passes_external_gate():
    adapter = _make_adapter()
    accepted = await adapter._prefilter_inbound(
        _dm_event(HOST_TEAM, "1712345681.000100"), {"team_id": HOST_TEAM})
    assert accepted is not None
    assert accepted[1] == HOST_TEAM


@pytest.mark.asyncio
async def test_connect_channel_message_from_external_user_is_not_dm_gated():
    """Channel posts from external-org users are handled once via the shared workspace —
    the external-sender guard must only refuse DMs, not channel traffic."""
    adapter = _make_adapter()
    accepted = await adapter._prefilter_inbound(
        _connect_message_event(), _connect_message_body())
    assert accepted is not None
    assert accepted[1] == HOST_TEAM
