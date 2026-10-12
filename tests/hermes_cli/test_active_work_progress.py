"""Active-work progress checks: preserve user heartbeats, deliver, then stop."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.active_work_progress import _deliverable, reconcile_active_work_watches
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.response_filters import is_intentional_silence_response
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli.active_work_progress import (
    ACTIVE_WORK_INTERVAL_SECONDS,
    ACTIVE_WORK_PROMPT,
    collect_active_session_ids,
    reconcile_active_work_result,
    ScanResult,
)
from hermes_cli.heartbeat import (
    ACTIVE_WORK_HEARTBEAT_ORIGIN,
    USER_HEARTBEAT_ORIGIN,
    HeartbeatManager,
    HeartbeatState,
    load_heartbeat,
)


@pytest.fixture
def heartbeat_db(monkeypatch):
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    goals._get_session_db()
    yield
    goals._DB_CACHE.clear()


def test_legacy_heartbeat_without_origin_is_user_authored():
    raw = json.dumps({
        "prompt": "watch the deploy", "interval_seconds": 600, "status": "active",
        "created_at": 1, "last_fired_at": 0, "fire_count": 0,
    })
    state = HeartbeatState.from_json(raw)
    assert state.origin == USER_HEARTBEAT_ORIGIN


def test_reconcile_preserves_user_heartbeat_and_clears_when_work_settles(heartbeat_db):
    HeartbeatManager("user-session").set("my own check", 600)
    first = reconcile_active_work_result(["user-session", "work-session"])
    assert first.actions["user-session"] == "preserved-user"
    assert first.actions["work-session"] == "installed"
    user = load_heartbeat("user-session")
    managed = load_heartbeat("work-session")
    assert user.origin == USER_HEARTBEAT_ORIGIN
    assert user.prompt == "my own check"
    assert managed.origin == ACTIVE_WORK_HEARTBEAT_ORIGIN
    assert managed.interval_seconds == ACTIVE_WORK_INTERVAL_SECONDS == 180
    assert "[SILENT]" in managed.prompt
    assert "child-process" in managed.prompt

    settled = reconcile_active_work_result([])
    assert "work-session" in settled.cleared
    assert load_heartbeat("work-session") is None
    assert load_heartbeat("user-session").prompt == "my own check"


def test_failed_scan_does_not_clear_an_existing_check(heartbeat_db, monkeypatch):
    reconcile_active_work_result(["work-session"])
    monkeypatch.setattr(
        "hermes_cli.active_work_progress._scan_kanban_sessions",
        lambda: ScanResult(set(), False),
    )
    ids, complete = collect_active_session_ids(
        delegation_ids=set(), background_ids=set(),
    )
    assert complete is False
    settled = reconcile_active_work_result(ids, scans_complete=complete)
    assert settled.cleared == []
    kept = load_heartbeat("work-session")
    assert kept is not None
    assert kept.origin == ACTIVE_WORK_HEARTBEAT_ORIGIN


def test_restart_reloads_managed_check_and_does_not_refire_settled_work(heartbeat_db):
    reconcile_active_work_result(["work-session"])
    reloaded = HeartbeatManager("work-session")
    reloaded.reload()
    assert reloaded.state.origin == ACTIVE_WORK_HEARTBEAT_ORIGIN
    assert reloaded.state.interval_seconds == 180
    reconcile_active_work_result([])
    assert HeartbeatManager("work-session").reload() is None


def test_collect_skips_child_handoff_and_stopped_delegations():
    ids, complete = collect_active_session_ids(
        kanban_ids={"kanban-parent"}, delegation_ids=set(), background_ids=set(),
    )
    assert ids == {"kanban-parent"}
    assert complete is True

    class _Proc:
        def __init__(self, parent, handoff=""):
            self.parent_session_id = parent
            self.handoff_note = handoff

    registry = SimpleNamespace(_running={
        "build": _Proc("human-session"),
        "child-build": _Proc("child-session", handoff="subagent handed off npm ci"),
    })
    from hermes_cli import active_work_progress as progress
    assert progress._scan_background_sessions(registry).ids == {"human-session"}


def test_delegation_scan_ignores_stopped_rows(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE async_delegations (origin_session_id TEXT, parent_session_id TEXT, "
        "state TEXT, stop_state TEXT)"
    )
    conn.execute(
        "INSERT INTO async_delegations VALUES ('live-parent', '', 'running', '')"
    )
    conn.execute(
        "INSERT INTO async_delegations VALUES ('stalling-parent', '', 'stalling', '')"
    )
    conn.execute(
        "INSERT INTO async_delegations VALUES ('finalizing-parent', '', 'finalizing', '')"
    )
    conn.execute(
        "INSERT INTO async_delegations VALUES ('stopped-parent', '', 'running', 'requested')"
    )
    conn.execute(
        "INSERT INTO async_delegations VALUES ('stopped-stalling', '', 'stalling', 'requested')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("hermes_cli.active_work_progress._state_db_path", lambda: db)
    from hermes_cli.active_work_progress import _scan_delegation_sessions
    assert _scan_delegation_sessions().ids == {
        "live-parent", "stalling-parent", "finalizing-parent",
    }


class _Adapter(BasePlatformAdapter):
    def __init__(self, config, platform):
        super().__init__(config, platform)
        self.last_send: dict = {}
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.last_send = {"chat_id": chat_id, "content": content, "metadata": metadata or {}}
        return SendResult(success=True, message_id="wire-1")


@pytest.mark.asyncio
async def test_due_check_delivers_to_originating_chat_and_silence_is_not_a_send(heartbeat_db, monkeypatch):
    from hermes_cli import heartbeat as heartbeat_mod

    clock = SimpleNamespace(now=1_000.0)
    monkeypatch.setattr(heartbeat_mod, "time", SimpleNamespace(time=lambda: clock.now))
    source = SessionSource(
        platform=Platform.SLACK, chat_id="D-origin", user_id="U1", chat_type="dm",
        thread_id="1791695152.254539",
    )
    key = build_session_key(source)
    adapter = _Adapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.SLACK)
    received = []

    async def handler(event):
        event._heartbeat_execution_started = True
        received.append(event)

    adapter.set_message_handler(handler)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda _source: adapter
    runner._run_in_executor_with_context = asyncio.to_thread
    HeartbeatManager("origin-session").ensure_managed(
        ACTIVE_WORK_PROMPT, ACTIVE_WORK_INTERVAL_SECONDS, origin=ACTIVE_WORK_HEARTBEAT_ORIGIN,
    )
    watch = {key: (source, "origin-session")}
    clock.now += ACTIVE_WORK_INTERVAL_SECONDS
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    assert len(received) == 1
    event = received[0]
    assert event.source.chat_id == "D-origin"
    assert event.source.thread_id == "1791695152.254539"
    assert event.source.platform == Platform.SLACK
    assert ACTIVE_WORK_PROMPT in event.text
    assert "[SILENT]" in event.text
    assert "nothing has changed" not in event.text
    assert "child-process" in event.text
    from gateway.platforms.base import _thread_metadata_for_source
    progress = "Build still running, next is the canary."
    assert is_intentional_silence_response("[SILENT]") is True
    assert is_intentional_silence_response(progress) is False
    sent = await adapter.send(
        event.source.chat_id, progress, metadata=_thread_metadata_for_source(event.source),
    )
    assert sent.success is True
    assert adapter.last_send["chat_id"] == "D-origin"
    assert adapter.last_send["metadata"]["thread_id"] == "1791695152.254539"


@pytest.mark.asyncio
async def test_reconcile_registers_origin_watch_and_drops_it_when_settled(heartbeat_db, monkeypatch):
    source = SessionSource(platform=Platform.SLACK, chat_id="D9", chat_type="dm", thread_id="t1")
    key = build_session_key(source)
    entry = SimpleNamespace(
        session_key=key, session_id="origin-session", origin=source, suspended=False,
    )
    runner = SimpleNamespace(
        session_store=SimpleNamespace(list_sessions=lambda: [entry], _routing_home=None),
        _heartbeat_watch={},
        _run_in_executor_with_context=asyncio.to_thread,
    )
    runner._register_heartbeat_watch = lambda quick_key, src, sid: runner._heartbeat_watch.__setitem__(
        quick_key, (src, sid)
    )
    runner._unregister_heartbeat_watch = lambda quick_key: runner._heartbeat_watch.pop(quick_key, None)
    runner._start_heartbeat_poller = lambda: None

    monkeypatch.setattr(
        "hermes_cli.active_work_progress.collect_active_session_ids",
        lambda **_kwargs: ({"origin-session"}, True),
    )
    monkeypatch.setattr(
        "gateway.run_heartbeat_restore._watched_homes",
        lambda _runner, home: [home],
    )
    await reconcile_active_work_watches(runner)
    assert key in runner._heartbeat_watch
    assert runner._heartbeat_watch[key][0].thread_id == "t1"
    assert load_heartbeat("origin-session").origin == ACTIVE_WORK_HEARTBEAT_ORIGIN

    monkeypatch.setattr(
        "hermes_cli.active_work_progress.collect_active_session_ids",
        lambda **_kwargs: (set(), True),
    )
    await reconcile_active_work_watches(runner)
    assert key not in runner._heartbeat_watch
    assert load_heartbeat("origin-session") is None


def _watch_runner(entries):
    runner = SimpleNamespace(
        session_store=SimpleNamespace(
            list_sessions=lambda: entries,
            _routing_home=None,
            peek_session_id=lambda session_key: next(
                (entry.session_id for entry in entries if entry.session_key == session_key),
                None,
            ),
        ),
        _heartbeat_watch={},
        _run_in_executor_with_context=asyncio.to_thread,
        config=SimpleNamespace(multiplex_profiles=False),
    )
    runner._register_heartbeat_watch = lambda quick_key, src, sid: runner._heartbeat_watch.__setitem__(
        quick_key, (src, sid)
    )
    runner._unregister_heartbeat_watch = lambda quick_key: runner._heartbeat_watch.pop(quick_key, None)
    runner._start_heartbeat_poller = lambda: None
    runner._restored_source = lambda entry: entry.origin
    runner._profile_scope_key_for_source = lambda _source: "default"

    from contextlib import contextmanager

    @contextmanager
    def _scope(_source):
        yield

    runner._profile_scope_for_source = _scope
    return runner


@pytest.mark.asyncio
async def test_reconcile_keeps_watch_on_compression_continuation(heartbeat_db, monkeypatch):
    source = SessionSource(platform=Platform.SLACK, chat_id="D9", chat_type="dm", thread_id="t1")
    key = build_session_key(source)
    entry = SimpleNamespace(
        session_key=key, session_id="child-session", prev_session_id="parent-session",
        origin=source, suspended=False,
    )
    runner = _watch_runner([entry])
    runner.session_store._compression_tip_for_session_id = (
        lambda sid: "child-session" if sid == "parent-session" else sid
    )
    HeartbeatManager("child-session").ensure_managed(
        ACTIVE_WORK_PROMPT, ACTIVE_WORK_INTERVAL_SECONDS, origin=ACTIVE_WORK_HEARTBEAT_ORIGIN,
    )
    monkeypatch.setattr(
        "hermes_cli.active_work_progress.collect_active_session_ids",
        lambda **_kwargs: ({"parent-session"}, True),
    )
    monkeypatch.setattr(
        "gateway.run_heartbeat_restore._watched_homes",
        lambda _runner, home: [home],
    )
    await reconcile_active_work_watches(runner)
    assert runner._heartbeat_watch[key][1] == "child-session"
    assert load_heartbeat("child-session").origin == ACTIVE_WORK_HEARTBEAT_ORIGIN
    assert load_heartbeat("parent-session") is None


@pytest.mark.asyncio
async def test_reconcile_then_restore_skips_non_deliverable_origin(heartbeat_db, monkeypatch):
    from gateway.run_heartbeat_restore import restore_heartbeat_watches

    source = SessionSource(platform=Platform.API_SERVER, chat_id="api-1", chat_type="dm")
    key = build_session_key(source)
    entry = SimpleNamespace(
        session_key=key, session_id="api-session", origin=source, suspended=False,
    )
    runner = _watch_runner([entry])
    HeartbeatManager("api-session").ensure_managed(
        ACTIVE_WORK_PROMPT, ACTIVE_WORK_INTERVAL_SECONDS, origin=ACTIVE_WORK_HEARTBEAT_ORIGIN,
    )
    monkeypatch.setattr(
        "hermes_cli.active_work_progress.collect_active_session_ids",
        lambda **_kwargs: ({"api-session"}, True),
    )
    monkeypatch.setattr(
        "gateway.run_heartbeat_restore._watched_homes",
        lambda _runner, home: [home],
    )
    monkeypatch.setattr(
        "gateway.run_idle_gates.profile_has_active_heartbeat",
        lambda _home: True,
    )
    await reconcile_active_work_watches(runner)
    await restore_heartbeat_watches(runner)
    assert key not in runner._heartbeat_watch
    assert load_heartbeat("api-session") is None


def test_rendered_due_prompt_requires_exact_silence(heartbeat_db):
    HeartbeatManager("origin-session").ensure_managed(
        ACTIVE_WORK_PROMPT, ACTIVE_WORK_INTERVAL_SECONDS, origin=ACTIVE_WORK_HEARTBEAT_ORIGIN,
    )
    state = load_heartbeat("origin-session")
    state.created_at = 0
    state.last_fired_at = 0
    from hermes_cli.heartbeat import save_heartbeat
    save_heartbeat("origin-session", state)
    prompt = HeartbeatManager("origin-session").due_prompt(now=10**12)
    assert prompt is not None
    assert "[SILENT]" in prompt
    assert "nothing has changed" not in prompt


def test_internal_platforms_are_not_progress_destinations():
    slack = SimpleNamespace(
        session_key="k", session_id="s", suspended=False,
        origin=SimpleNamespace(platform=SimpleNamespace(value="slack"), chat_id="D1"),
    )
    a2a = SimpleNamespace(
        session_key="k2", session_id="s2", suspended=False,
        origin=SimpleNamespace(platform=SimpleNamespace(value="a2a"), chat_id="ctx-1"),
    )
    assert _deliverable(slack) is True
    assert _deliverable(a2a) is False


def test_notifier_terminal_kinds_are_not_gated_on_the_progress_interval():
    import inspect
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS, _WAKE_KINDS

    assert "completed" in _WAKE_KINDS
    assert "blocked" in _WAKE_KINDS
    assert ACTIVE_WORK_INTERVAL_SECONDS == 180
    signature = inspect.signature(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert signature.parameters["interval"].default == 5.0
    notifier_source = inspect.getsource(__import__("gateway.kanban_watchers_notifier", fromlist=["*"]))
    assert "active_work_progress" not in notifier_source
    assert "ACTIVE_WORK_INTERVAL_SECONDS" not in notifier_source
    note = SimpleNamespace(head="Kanban", title="canary", task=SimpleNamespace(result="shipped"))
    completed_fmt = _EVENT_FORMATTERS["completed"]
    blocked_fmt = _EVENT_FORMATTERS["blocked"]
    completed, handoff, _review = completed_fmt(
        SimpleNamespace(kind="completed", payload={"summary": "shipped the canary"}), note,
    )
    blocked, _handoff, _review = blocked_fmt(
        SimpleNamespace(kind="blocked", payload={"reason": "needs a credential"}), note,
    )
    assert completed and "shipped" in completed
    assert handoff == "shipped the canary"
    assert blocked and "credential" in blocked
    assert "180" not in completed and "180" not in blocked
