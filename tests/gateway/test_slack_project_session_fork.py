"""End-to-end routing contracts for Slack project session forks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _source(*, thread_id="1700000000.000001", platform=Platform.SLACK):
    return SessionSource(
        platform=platform,
        scope_id="T_WORKSPACE",
        chat_id="C_PROJECT",
        chat_type="group",
        user_id="U_USER",
        thread_id=thread_id,
    )


@pytest.mark.asyncio
async def test_marked_project_thread_forks_from_derived_channel_session():
    store = SimpleNamespace(get_or_create_session=AsyncMock())
    parent_entry = SimpleNamespace(session_id="parent-session")
    child_entry = SimpleNamespace(session_id="child-session")
    store.get_or_create_session.side_effect = [parent_entry, parent_entry, child_entry]
    runner = SimpleNamespace(async_session_store=store)
    event = SimpleNamespace(metadata={"slack_project_session_fork": True})
    source = _source()

    result = await GatewayRunner._get_or_create_inbound_session(
        runner, event, source, "child-key", 7
    )

    assert result is child_entry
    assert store.get_or_create_session.await_count == 3
    parent_source = store.get_or_create_session.await_args_list[0].args[0]
    assert parent_source.thread_id is None
    assert parent_source.chat_id == source.chat_id
    assert parent_source.scope_id == source.scope_id
    child_call = store.get_or_create_session.await_args_list[2]
    assert child_call.args[0] is source
    assert child_call.kwargs["fork_from_session_id"] == "parent-session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "source"),
    [
        (SimpleNamespace(metadata={}), _source()),
        (SimpleNamespace(metadata={"slack_project_session_fork": True}), _source(thread_id=None)),
        (
            SimpleNamespace(metadata={"slack_project_session_fork": True}),
            _source(platform=Platform.DISCORD),
        ),
    ],
)
async def test_unmarked_or_non_slack_lanes_never_fork(event, source):
    child_entry = SimpleNamespace(session_id="existing-session")
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=child_entry)
    )
    runner = SimpleNamespace(async_session_store=store)

    result = await GatewayRunner._get_or_create_inbound_session(
        runner, event, source, "child-key", 7
    )

    assert result is child_entry
    store.get_or_create_session.assert_awaited_once_with(source)


@pytest.mark.asyncio
async def test_project_fork_holds_parent_turn_lease_until_snapshot_finishes():
    store = SimpleNamespace(get_or_create_session=AsyncMock())
    parent_entry = SimpleNamespace(session_id="parent-session")
    snapshot_entry = SimpleNamespace(session_id="compressed-tip-session")
    child_entry = SimpleNamespace(session_id="child-session")
    store.get_or_create_session.side_effect = [
        parent_entry,
        snapshot_entry,
        child_entry,
    ]
    token = object()
    leases = SimpleNamespace(
        acquire=AsyncMock(return_value=token),
        release=MagicMock(),
    )
    runner = SimpleNamespace(async_session_store=store, _turn_leases=leases)
    event = SimpleNamespace(metadata={"slack_project_session_fork": True})

    result = await GatewayRunner._get_or_create_inbound_session(
        runner, event, _source(), "child-key", 7
    )

    assert result is child_entry
    leases.acquire.assert_awaited_once()
    acquire = leases.acquire.await_args
    assert acquire.args[0] == "parent-session"
    assert acquire.kwargs["generation"] == 7
    child_call = store.get_or_create_session.await_args_list[2]
    assert child_call.kwargs["fork_from_session_id"] == "compressed-tip-session"
    leases.release.assert_called_once_with(token)
