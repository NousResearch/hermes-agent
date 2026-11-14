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
from unittest.mock import AsyncMock, MagicMock

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
    adapter._team_clients = {HOST_TEAM: MagicMock()}
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


# ---------------------------------------------------------------------------
# Tests: the resolver filters candidates against the authenticated install set
# (no Slack field is guaranteed to name an installing workspace; a partner team
# must never scope a session or a dedupe id)
# ---------------------------------------------------------------------------


def test_resolver_ignores_foreign_envelope_with_single_install():
    """Production shape where the envelope names the partner org: with exactly one
    authenticated workspace that install wins, so the pair still collapses."""
    adapter = _make_adapter()
    event = _connect_message_event()
    body = {"team_id": EXT_TEAM, "authorizations": [{"team_id": EXT_TEAM}],
            "is_ext_shared_channel": True}
    assert adapter._event_team_id(event, body, install_set={HOST_TEAM}) == HOST_TEAM


def test_resolver_grid_example_prefers_authorization_install():
    """Enterprise Grid documented shape: envelope workspace differs from
    ``authorizations[0].team_id``; the authorization naming an install wins."""
    adapter = _make_adapter()
    event = {"type": "message", "team": "T111"}
    body = {"team_id": "T111", "authorizations": [{"team_id": "T222"}]}
    assert adapter._event_team_id(event, body, install_set={"T222"}) == "T222"


def test_resolver_multi_install_no_match_is_empty():
    """A foreign team is never adopted when several workspaces are authenticated."""
    adapter = _make_adapter()
    event = {"type": "message", "team": EXT_TEAM}
    body = {"team_id": EXT_TEAM}
    assert adapter._event_team_id(event, body, install_set={"T_A", "T_B"}) == ""


@pytest.mark.asyncio
async def test_multi_install_unattributed_external_dm_is_refused():
    """A resolver miss must reject the DM at intake, not dispatch it with an empty scope."""
    adapter = _make_adapter()
    adapter._team_clients = {"T_A": MagicMock(), "T_B": MagicMock()}
    event = _dm_event(EXT_TEAM, "1712345689.000100")
    body = {"team_id": EXT_TEAM, "authorizations": [{"team_id": EXT_TEAM}]}
    assert await adapter._prefilter_inbound(event, body) is None


@pytest.mark.asyncio
async def test_multi_install_dm_from_sibling_install_member_passes():
    """Grid with two installs: ``authorizations[0]`` may name the other install. A member of an
    authenticated workspace is not an external-org user."""
    adapter = _make_adapter()
    adapter._team_clients = {"T_A": MagicMock(), "T_B": MagicMock()}
    event = _dm_event("T_A", "1712345690.000100")
    body = {"team_id": "T_A", "authorizations": [{"team_id": "T_B"}]}
    assert await adapter._prefilter_inbound(event, body) is not None


@pytest.mark.asyncio
async def test_multi_install_dm_from_external_org_still_refused():
    """The sibling-install allowance never admits a team outside every install."""
    adapter = _make_adapter()
    adapter._team_clients = {"T_A": MagicMock(), "T_B": MagicMock()}
    event = _dm_event(EXT_TEAM, "1712345691.000100")
    body = {"team_id": "T_A", "authorizations": [{"team_id": "T_B"}]}
    assert await adapter._prefilter_inbound(event, body) is None


@pytest.mark.asyncio
async def test_file_share_dm_from_sibling_install_member_synthesizes(monkeypatch):
    """The file_shared sharer gate uses the same install-set rule as the DM gate."""
    adapter = _make_adapter()
    adapter._team_clients = {HOST_TEAM: _file_share_client("T_SIBLING"), "T_SIBLING": MagicMock()}
    monkeypatch.setattr(adapter, "_first_file_share", lambda file_obj, channel_id: {
        "ts": "1712345686.000100", "thread_ts": ""})
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_slack_message", handled)
    await adapter._handle_slack_file_shared(
        _file_share_event(), {"team_id": HOST_TEAM, "authorizations": [{"team_id": HOST_TEAM}]})
    assert handled.called


def test_resolver_legacy_call_without_install_set_keeps_envelope_first():
    """External/Bolt callers without an install set keep the envelope-first order."""
    adapter = _make_adapter()
    assert adapter._event_team_id(_connect_message_event(), _connect_message_body()) == HOST_TEAM


def test_team_in_install_set_filters_foreign_ids():
    """Cached/fallback ids only pass through for authenticated workspaces."""
    adapter = _make_adapter()
    assert adapter._team_in_install_set(HOST_TEAM) == HOST_TEAM
    assert adapter._team_in_install_set(EXT_TEAM) == ""
    assert adapter._team_in_install_set("") == ""


def test_assistant_metadata_fallback_does_not_readopt_foreign_team():
    """Multi-install with no candidate in the set: the cached assistant_thread.team_id
    must not undo the resolver's empty resolution (#143792)."""
    adapter = _make_adapter()
    adapter._team_clients = {"T_A": MagicMock(), "T_B": MagicMock()}
    meta = adapter._extract_assistant_thread_metadata(
        {"type": "message", "team": EXT_TEAM},
        {"assistant_thread": {"channel_id": "D_SHARED", "thread_ts": "171.000",
                               "user_id": "U_EXT", "team_id": EXT_TEAM}})
    assert meta["team_id"] == ""


# ---------------------------------------------------------------------------
# Tests: the DM gate only compares ``T…`` workspace ids (user_team can be E…)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_with_enterprise_user_team_passes_gate():
    """``user_team`` can be an enterprise ``E…`` id (Bolt); the gate falls back to the
    ``T…`` team and must not misread a host user as foreign."""
    adapter = _make_adapter()
    dm = _dm_event(HOST_TEAM, "1712345682.000100")
    dm["user_team"] = "E_GRID_ORG"
    accepted = await adapter._prefilter_inbound(dm, {"team_id": HOST_TEAM})
    assert accepted is not None


@pytest.mark.asyncio
async def test_allowlisted_external_user_still_refused(monkeypatch):
    """The gate sits ahead of the allowlist: an external-org user in
    SLACK_ALLOWED_USERS must not gain a DM path (#143792)."""
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_EXT_USER")
    adapter = _make_adapter()
    accepted = await adapter._prefilter_inbound(
        _dm_event(EXT_TEAM, "1712345683.000100"), {"team_id": HOST_TEAM})
    assert accepted is None


# ---------------------------------------------------------------------------
# Tests: the file_shared fallback cannot re-enter as a host-team DM
# (the synthesizer stamps the install team and would slip past the DM gate)
# ---------------------------------------------------------------------------


def _file_share_event(channel: str = "D_INBOX_1"):
    """Documented ``file_shared`` payload: no sender team on the event at all."""
    return {
        "type": "file_shared",
        "channel_id": channel,
        "file_id": "F_FILE_1",
        "user_id": "U_EXT_USER",
        "event_ts": "1712345684.000100",
    }


def _file_share_client(team_id: str):
    client = MagicMock()
    client.files_info = MagicMock(return_value=_async_return({
        "ok": True, "file": {"id": "F_FILE_1", "mimetype": "video/mp4",
                             "user": "U_EXT_USER"}}))
    client.users_info = MagicMock(return_value=_async_return({
        "ok": True, "user": {"team_id": team_id}}))
    return client


@pytest.mark.asyncio
async def test_file_share_dm_from_external_org_is_refused(monkeypatch):
    """The documented payload carries no sender team; the sharer's workspace is
    resolved from users.info and a foreign-org video DM never re-enters."""
    adapter = _make_adapter()
    adapter._team_clients[HOST_TEAM] = _file_share_client(EXT_TEAM)
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_slack_message", handled)
    await adapter._handle_slack_file_shared(
        _file_share_event(), {"team_id": HOST_TEAM})
    assert not handled.called


@pytest.mark.asyncio
async def test_file_share_dm_from_host_org_still_synthesizes(monkeypatch):
    """A host-org file share in a DM still synthesizes the fallback message."""
    adapter = _make_adapter()
    adapter._team_clients[HOST_TEAM] = _file_share_client(HOST_TEAM)
    monkeypatch.setattr(adapter, "_first_file_share", lambda file_obj, channel_id: {
        "ts": "1712345685.000100", "thread_ts": ""})
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_slack_message", handled)
    await adapter._handle_slack_file_shared(
        _file_share_event(), {"team_id": HOST_TEAM})
    assert handled.called


@pytest.mark.asyncio
async def test_file_share_dm_host_event_team_cannot_mask_external_sharer(monkeypatch):
    """`event.team` can name the receiving install, not the file sharer.
    A host routing hint must not skip the actual external author's users.info lookup."""
    adapter = _make_adapter()
    client = _file_share_client(EXT_TEAM)
    adapter._team_clients[HOST_TEAM] = client
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_slack_message", handled)
    event = _file_share_event()
    event["team"] = HOST_TEAM
    await adapter._handle_slack_file_shared(event, {"team_id": HOST_TEAM})
    assert client.users_info.call_count == 1
    assert not handled.called


@pytest.mark.asyncio
async def test_file_share_dm_with_injected_sender_team_refused(monkeypatch):
    """Wire shapes that DO carry a T… sender team are refused without any API call."""
    adapter = _make_adapter()
    adapter._team_clients[HOST_TEAM] = _file_share_client(EXT_TEAM)
    handled = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_slack_message", handled)
    event = _file_share_event()
    event["team"] = EXT_TEAM
    await adapter._handle_slack_file_shared(event, {"team_id": HOST_TEAM})
    assert not handled.called


def _async_return(value):
    import asyncio
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


# ---------------------------------------------------------------------------
# Tests: a reaction_triggers turn cannot re-enter an external-org DM
# (the synthetic reaction event stamps the install team, like file_shared)
# ---------------------------------------------------------------------------


def _reaction_adapter(reactor_team: str):
    adapter = _make_adapter()
    adapter.config.extra["reaction_triggers"] = "thumbsup"
    adapter._channel_team = {}
    adapter._reaction_handler = None
    client = adapter._team_clients[HOST_TEAM]
    client.conversations_replies = AsyncMock(return_value={"messages": [
        {"ts": "100.1", "user": "U_EXT_USER", "text": "secret ask"}]})
    client.users_info = AsyncMock(return_value={"user": {"is_bot": False, "team_id": reactor_team}})
    adapter._drop_bot_sender = AsyncMock(return_value=False)
    accepted = []

    async def _impl(event, payload=None):
        accepted.append(await adapter._prefilter_inbound(event, payload))

    adapter._handle_slack_message = _impl
    return adapter, accepted


def _dm_reaction():
    return {"type": "reaction_added", "user": "U_EXT_USER", "reaction": "thumbsup",
            "item_user": "U_EXT_USER", "item": {"type": "message", "channel": "D_INBOX_1",
                                                "ts": "100.1"}, "event_ts": "200.2"}


@pytest.mark.asyncio
async def test_reaction_in_dm_from_external_org_is_refused():
    """An allowlisted reaction by an external-org user in a DM must hit the same
    external-DM refusal as their messages (#143792)."""
    adapter, accepted = _reaction_adapter(EXT_TEAM)
    await adapter._handle_slack_reaction(_dm_reaction())
    assert accepted == [None]


@pytest.mark.asyncio
async def test_reaction_in_dm_from_host_org_still_routes():
    adapter, accepted = _reaction_adapter(HOST_TEAM)
    await adapter._handle_slack_reaction(_dm_reaction())
    assert len(accepted) == 1 and accepted[0] is not None


@pytest.mark.asyncio
async def test_reaction_handoff_from_external_dm_is_refused():
    """A configured handoff target replaces the DM channel; the reaction's DM source
    still decides whether the gate applies."""
    adapter, accepted = _reaction_adapter(EXT_TEAM)
    adapter.config.extra["reaction_trigger_target"] = "C_HANDOFF"
    await adapter._handle_slack_reaction(_dm_reaction())
    assert accepted == [None]


@pytest.mark.asyncio
async def test_reaction_handoff_into_dm_from_external_channel_user_is_refused():
    """A DM handoff target makes the turn a DM even when the reaction was in a channel."""
    adapter, accepted = _reaction_adapter(EXT_TEAM)
    adapter.config.extra["reaction_trigger_target"] = "D_HANDOFF"
    reaction = _dm_reaction()
    reaction["item"]["channel"] = SHARED_CHANNEL
    await adapter._handle_slack_reaction(reaction)
    assert accepted == [None]
