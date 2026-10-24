"""Child process notifications must not replace the human's gateway conversation.

Regression for the October 12 incident: a terminal completion targeted a running
delegate, repointed Discord to its transcript, and waited on the child's lease.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "route.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    source = SessionSource(platform=Platform.DISCORD, chat_id="route-test", user_id="owner", chat_type="dm")
    entry = store.get_or_create_session(source)
    db.create_session(entry.session_id, source="discord")
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    try:
        yield runner, db, entry, source
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["source", "creation_source", "marker", "compressed"])
async def test_child_completion_cannot_change_route_or_close_parent(conversation, identity):
    runner, db, entry, _ = conversation
    parent_id = entry.session_id
    # Both immutable origin and the marker must work after transport metadata has
    # made source='discord'. Compression copies the marker from the child root.
    child = "delegate-child"
    db.create_session(child, source="subagent" if identity in {"source", "creation_source"} else "discord",
                      parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id} if identity in {"marker", "compressed"} else {})
    if identity == "creation_source":
        db._write_sql("UPDATE sessions SET source = 'discord' WHERE id = ?", (child,))
    if identity == "compressed":
        db.end_session(child, end_reason="compression")
        db.create_session("delegate-tip", source="discord", parent_session_id=child,
                          model_config={"_delegate_from": entry.session_id})
        child = "delegate-tip"
    assert db.try_acquire_session_turn_lease(child, "delegate-worker", ttl_seconds=300)
    event = {"type": "completion", "session_id": "proc-child", "parent_session_id": child,
             "session_key": entry.session_key}

    # Preflight must terminally decline before adapter acceptance, not falsely
    # acknowledge or retry forever. Already-admitted events need the same guard.
    claim = await runner._preflight_completion_delivery(event)
    resolved = (await runner._resolve_async_delegation_session(entry, child)
                if identity in {"marker", "compressed"} else None)

    assert not claim.proceed and claim.early_result is None
    # Only marker-backed final-result ownership may resolve to the parent.
    if identity in {"marker", "compressed"}:
        assert resolved is not None and resolved.session_id == parent_id
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == parent_id
    assert db.get_session(parent_id)["ended_at"] is None
    assert db.session_turn_lease_expires_at(child, "delegate-worker") is not None


@pytest.mark.asyncio
async def test_queued_child_watch_event_never_starts_foreground_turn(conversation):
    runner, db, entry, source = conversation
    db.create_session("watch-child", source="subagent", parent_session_id=entry.session_id)
    foreground_turn = AsyncMock()

    async def handle(event):
        resolved = await runner._resolve_async_delegation_session(
            entry, event.metadata["gateway_session_id"],
        )
        if resolved is not None:
            await foreground_turn(resolved)
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=handle))
    runner._build_process_event_source = MagicMock(return_value=source)
    runner._resolve_injection_adapter = MagicMock(return_value=adapter)
    event = {"type": "watch_match", "session_id": "proc-watch", "parent_session_id": "watch-child",
             "session_key": entry.session_key}

    await runner._inject_watch_notification("process ready", event)

    adapter.handle_message.assert_not_awaited()
    foreground_turn.assert_not_awaited()
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    assert db.get_session(entry.session_id)["ended_at"] is None


@pytest.mark.asyncio
async def test_parent_results_and_compression_continuations_still_deliver(conversation):
    runner, db, entry, _ = conversation
    parent_id = entry.session_id
    # Ordinary parent-owned processes and delegation results use this parent ID.
    claim = await runner._preflight_completion_delivery(
        {"type": "completion", "session_id": "handed-off", "parent_session_id": entry.session_id},
    )
    assert claim.proceed
    assert await runner._resolve_async_delegation_session(entry, entry.session_id) is entry

    db.end_session(entry.session_id, end_reason="compression")
    db.create_session("parent-tip", source="discord", parent_session_id=entry.session_id)
    assert await runner._classify_completion_target(entry.session_id) == "deliver"
    resolved = await runner._resolve_async_delegation_session(entry, entry.session_id)
    assert resolved.session_id == "parent-tip"
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == "parent-tip"
    assert db.get_session(parent_id)["end_reason"] == "compression"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["same", "compressed-root", "compressed-tip"])
async def test_explicit_resume_keeps_background_delivery(conversation, target):
    runner, db, entry, _ = conversation
    child = "explicit-child"
    db.create_session(child, source="subagent", parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id})
    resumed = runner.session_store.switch_session(entry.session_key, child, preserve_prompt_pin=False)
    target_id = child
    expected = child
    if target != "same":
        db.end_session(child, end_reason="compression")
        db.create_session("explicit-tip", source="discord", parent_session_id=child,
                          model_config={"_delegate_from": entry.session_id})
        expected = "explicit-tip"
        if target == "compressed-tip":
            target_id = expected
    claim = await runner._preflight_completion_delivery(
        {"type": "completion", "session_id": "proc-resumed", "parent_session_id": target_id,
         "session_key": entry.session_key},
    )
    resolved = await runner._resolve_async_delegation_session(resumed, target_id)
    assert claim.proceed
    assert resolved.session_id == expected
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == expected
    if target != "same":
        assert db.get_session(child)["end_reason"] == "compression"


@pytest.mark.asyncio
async def test_real_handoff_retargets_an_already_registered_watcher(conversation, monkeypatch):
    import gateway.run as gateway_run
    import tools.process_registry as processes
    from tools.delegate_tool_registry import _active_subagents
    from tools.terminal_tool_background import _register_completion_watcher

    runner, db, entry, source = conversation
    child_id, owner = "handoff-child", "sa-handoff-child"
    db.create_session(child_id, source="subagent", parent_session_id=entry.session_id)
    registry = processes.ProcessRegistry()
    proc = processes.ProcessSession(
        id="proc-handoff", command="synthetic", task_id=owner, owner_task_id=owner,
        session_key=entry.session_key, parent_session_id=child_id, watcher_platform="discord",
        watcher_chat_id=source.chat_id, watcher_user_id=source.user_id, notify_on_complete=True,
    )
    registry._running[proc.id] = proc  # No OS process or provider is launched.
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", None)
    _register_completion_watcher(registry, proc, entry.session_key)
    watcher = registry.pending_watchers[0]
    parent = SimpleNamespace(session_id=entry.session_id, _current_task_id="parent-turn")
    child = SimpleNamespace(_delegate_parent_ref=lambda: parent)
    monkeypatch.setitem(_active_subagents, owner, {"agent": child})
    monkeypatch.setattr(processes, "process_registry", registry)

    result = processes._handoff_process(proc.id, {"data": "report this synthetic result"}, owner)
    assert result["status"] == "handed_off"
    proc.exit_code = 0
    event = runner._build_process_completion_event(watcher, proc, proc.id)
    assert event["parent_session_id"] == entry.session_id
    assert (await runner._preflight_completion_delivery(event)).proceed
    assert await runner._resolve_async_delegation_session(entry, event["parent_session_id"]) is entry
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
