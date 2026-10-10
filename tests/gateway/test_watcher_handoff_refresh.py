"""Already-running watchers follow handoff without changing the chat owner (#135685)."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


def _runner(monkeypatch, tmp_path, *, profile=None, surface_child=False, mode="off"):
    import gateway.run as gateway_run
    from hermes_cli.profiles import get_profile_dir

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BASE_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "voice.json")
    (tmp_path / "config.yaml").write_text(
        f"delegation:\n  surface_child_process_notifications: {str(not surface_child if profile else surface_child).lower()}\n"
        f"display:\n  background_process_notifications: {mode}\n",
    )
    if profile:
        home = get_profile_dir(profile)
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"delegation:\n  surface_child_process_notifications: {str(surface_child).lower()}\n"
            f"display:\n  background_process_notifications: {mode}\n",
        )
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions", multiplex_profiles=bool(profile)))
    runner._completion_notification_batch_window = 0

    async def admit(event):
        event._gateway_accepted = True

    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock(side_effect=admit), _active_sessions={})
    if profile:
        runner._profile_adapters = {profile: {Platform.DISCORD: adapter}}
    else:
        runner.adapters[Platform.DISCORD] = adapter
    entry = runner.session_store.get_or_create_session(SessionSource(
        platform=Platform.DISCORD, chat_id="room", chat_type="thread", thread_id="topic",
        user_id="person", profile=profile,
    ))
    db = runner.session_store._db_for_key(entry.session_key)
    db.create_session("child-session", source="subagent", parent_session_id=entry.session_id)
    session = ProcessSession(
        id="proc_handoff", command="worker", owner_task_id="sa-child", task_id="sa-child",
        session_key=entry.session_key, parent_session_id="child-session", started_at=time.time(),
        watcher_platform="discord", watcher_chat_id="room", watcher_thread_id="topic",
        watcher_interval=5,
    )
    watcher = dict(session_id=session.id, check_interval=0, session_key=entry.session_key,
                   parent_session_id="child-session", platform="discord", chat_id="room",
                   chat_type="thread", thread_id="topic", notify_on_complete=False)
    return runner, adapter, entry, db, session, watcher


def _transfer_on_second_poll(monkeypatch, entry, session):
    import tools.process_registry as processes

    registry = ProcessRegistry()
    registry._running[session.id] = session
    polls = 0

    def poll(_session_id):
        nonlocal polls
        polls += 1
        if polls == 1:
            return session
        if polls > 2:
            return None
        assert registry.transfer_ownership(
            session.id, from_owner="sa-child", to_owner="parent-turn", to_task_id="parent-turn",
            to_session_key=entry.session_key, to_parent_session_id=entry.session_id,
            note="report worker result",
        ) is session
        session.append_output("worker completed")
        session.mark_exited(0)
        return session

    watcher_registry = SimpleNamespace(get=Mock(side_effect=poll),
                                       is_completion_consumed=registry.is_completion_consumed)
    monkeypatch.setattr(processes, "process_registry", watcher_registry)
    return watcher_registry


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_notify", [False, True])
@pytest.mark.parametrize("legacy_key", [False, True])
async def test_existing_watcher_delivers_handoff_using_current_owner(
    monkeypatch, tmp_path, initial_notify, legacy_key,
):
    runner, adapter, entry, db, session, watcher = _runner(monkeypatch, tmp_path)
    session.notify_on_complete = watcher["notify_on_complete"] = initial_notify
    if legacy_key:
        watcher["session_key"] = "child-session"
    registry = _transfer_on_second_poll(monkeypatch, entry, session)

    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["gateway_session_id"] == entry.session_id
    assert event.metadata["gateway_session_key"] == entry.session_key
    assert "report worker result" in event.text
    assert registry.get.call_count == 2
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    assert db.get_session("child-session")["source"] == "subagent"
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_pattern_handoff_preserves_existing_status_watcher_mode(monkeypatch, tmp_path):
    runner, adapter, entry, _db, session, watcher = _runner(monkeypatch, tmp_path, mode="all")
    session.watch_patterns = ["READY"]
    registry = _transfer_on_second_poll(monkeypatch, entry, session)

    await runner._run_process_watcher(watcher)

    assert session.watch_patterns == ["READY"]
    assert session.notify_on_complete is False
    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "room"
    assert registry.get.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_child", [False, True])
async def test_child_watcher_uses_owning_profile_opt_in(monkeypatch, tmp_path, surface_child):
    import tools.process_registry as processes

    runner, adapter, entry, _db, session, watcher = _runner(
        monkeypatch, tmp_path, profile="research", surface_child=surface_child,
    )
    session.notify_on_complete = watcher["notify_on_complete"] = True
    session.append_output("child completed")
    session.mark_exited(0)
    monkeypatch.setattr(processes, "process_registry", SimpleNamespace(
        get=Mock(side_effect=[session, None]), is_completion_consumed=lambda _id: False,
    ))

    await runner._run_process_watcher(watcher)

    assert adapter.handle_message.await_count == int(surface_child)
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    if surface_child:
        event = adapter.handle_message.await_args.args[0]
        assert event._completion_owner_receipt.session_id == entry.session_id
        assert event._completion_owner_receipt.pinned_session_id == "child-session"
        assert event.source.profile == "research"
    adapter.send.assert_not_awaited()
