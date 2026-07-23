"""Routing tests for /branch's new-sibling-thread default (#66023).

On thread-capable platforms (Discord, Telegram, Slack), ``/branch`` clones
into a NEW sibling thread and binds that thread's session key to the clone,
leaving the ORIGIN key on its existing session so the original conversation
stays active on the current surface. ``--here`` opts out (legacy in-place),
non-thread platforms are always in-place, and a failed thread creation
degrades to in-place so the clone is never orphaned.

These tests exercise the routing decision directly through
``_handle_branch_command`` with a mocked session store + adapter — no live
platform, no real thread.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import AsyncSessionDB


def _source(platform: Platform, **overrides) -> SessionSource:
    base = dict(
        platform=platform,
        chat_id="chan1",
        chat_name="general",
        chat_type="channel",
        user_id="u1",
        user_name="tester",
    )
    base.update(overrides)
    return SessionSource(**base)


def _entry(session_id: str, source: SessionSource) -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(source: SessionSource, *, adapter=None):
    from gateway.run import GatewayRunner

    current_entry = _entry("current-session", source)
    branched_entry = _entry("branched-session", source)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {source.platform: adapter} if adapter is not None else {}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner.session_store = MagicMock()
    # Force the fallback path in _session_key_for_source (a MagicMock
    # _generate_session_key isn't a str), so keys come from build_session_key
    # with the default group/thread flags — deterministic and reproducible here.
    runner.session_store._generate_session_key = MagicMock(return_value=object())
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    runner.session_store.switch_session.return_value = branched_entry
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = "Current Work"
    runner._session_db._db.get_next_title_in_lineage.return_value = "Current Work #2"
    # Isolate the routing decision from the peripheral cleanup helpers.
    runner._clear_session_boundary_security_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    return runner


def _switched_keys(runner) -> list[str]:
    return [c.args[0] for c in runner.session_store.switch_session.call_args_list]


@pytest.mark.asyncio
async def test_branch_creates_sibling_thread_and_binds_destination_key():
    source = _source(Platform.DISCORD)
    origin_key = build_session_key(source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="tid-99")
    runner = _make_runner(source, adapter=adapter)

    result = await runner._handle_branch_command(_event("/branch alt", source))

    # A sibling thread was requested on the current channel.
    adapter.create_handoff_thread.assert_awaited_once()
    assert adapter.create_handoff_thread.await_args.args[0] == "chan1"

    # The clone is bound to the NEW thread's key, and the origin key is left
    # untouched so the original conversation stays active.
    dest_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan1",
        chat_name="general",
        chat_type="thread",
        user_id="u1",
        user_name="tester",
        thread_id="tid-99",
    )
    dest_key = build_session_key(dest_source)
    switched = _switched_keys(runner)
    assert switched == [dest_key]
    assert origin_key not in switched
    # The destination key is bound to the freshly created clone session id.
    clone_id = runner._session_db._db.create_session.call_args.kwargs["session_id"]
    assert runner.session_store.switch_session.call_args.args[1] == clone_id
    assert "New thread" in result


@pytest.mark.asyncio
async def test_branch_here_flag_stays_in_place_on_thread_platform():
    source = _source(Platform.DISCORD)
    origin_key = build_session_key(source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="tid-99")
    runner = _make_runner(source, adapter=adapter)

    result = await runner._handle_branch_command(_event("/branch --here alt", source))

    adapter.create_handoff_thread.assert_not_awaited()
    assert _switched_keys(runner) == [origin_key]
    assert "New thread" not in result


@pytest.mark.asyncio
async def test_branch_inside_thread_parents_on_channel_not_thread():
    # Issued inside an existing thread: chat_id is the thread, parent_chat_id
    # is the real channel. The new sibling thread must be parented on the
    # channel — adapters reject a thread as a thread parent.
    source = _source(
        Platform.DISCORD,
        chat_id="thread-123",
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="realchan",
    )
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="tid-77")
    runner = _make_runner(source, adapter=adapter)

    await runner._handle_branch_command(_event("/branch", source))

    adapter.create_handoff_thread.assert_awaited_once()
    assert adapter.create_handoff_thread.await_args.args[0] == "realchan"


@pytest.mark.asyncio
async def test_branch_thread_creation_failure_falls_back_in_place():
    source = _source(Platform.DISCORD)
    origin_key = build_session_key(source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value=None)  # unsupported/failed
    runner = _make_runner(source, adapter=adapter)

    result = await runner._handle_branch_command(_event("/branch", source))

    adapter.create_handoff_thread.assert_awaited_once()
    # No thread → clone must still be routable: origin key switched (in-place).
    assert _switched_keys(runner) == [origin_key]
    assert "New thread" not in result


@pytest.mark.asyncio
async def test_branch_non_thread_platform_stays_in_place():
    source = _source(Platform.WHATSAPP)
    origin_key = build_session_key(source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="tid-99")
    runner = _make_runner(source, adapter=adapter)

    result = await runner._handle_branch_command(_event("/branch", source))

    # Non-thread platform: adapter thread hook is never consulted.
    adapter.create_handoff_thread.assert_not_awaited()
    assert _switched_keys(runner) == [origin_key]
    assert "New thread" not in result


@pytest.mark.asyncio
async def test_branch_thread_parented_but_origin_session_not_switched():
    """The origin key must never be switched in thread mode — that was the
    core defect: switching it abandoned the original conversation."""
    source = _source(Platform.SLACK)
    origin_key = build_session_key(source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="ts-1700000000.1")
    runner = _make_runner(source, adapter=adapter)

    await runner._handle_branch_command(_event("/branch", source))

    assert origin_key not in _switched_keys(runner)
