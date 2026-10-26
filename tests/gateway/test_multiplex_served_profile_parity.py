"""Standalone-vs-served parity for goals/loops, completion notices and process recovery under
``gateway.multiplex_profiles``: every read that decides FOR a served profile must run inside that
profile's runtime scope, and every store the served profile wrote must be recovered under it.
"""

import asyncio
import json
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


class RecordingAdapter:
    def __init__(self, notice_delivery=None):
        self.config = PlatformConfig(enabled=True, extra={"notice_delivery": notice_delivery} if notice_delivery else {})
        self.platform = Platform.TELEGRAM
        self.calls = []
        self._active_sessions, self._pending_messages, self._session_tasks = {}, {}, {}

    async def send(self, chat_id, content, metadata=None, **kw):
        self.calls.append(("send", str(get_hermes_home())))
        return SimpleNamespace(success=True, error=None)

    async def send_private_notice(self, chat_id, user_id, content, metadata=None, **kw):
        self.calls.append(("send_private_notice", str(get_hermes_home())))
        return SimpleNamespace(success=True, error=None)

    async def handle_message(self, event):
        self.calls.append(("handle_message", str(get_hermes_home())))
        event._gateway_accepted = True


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Default host + served profile ``alpha``; the runner is the default multiplexer."""
    root = tmp_path / "hermes"
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\ndisplay:\n  background_process_notifications: concise\n")
    (alpha / "config.yaml").write_text("display:\n  background_process_notifications: 'off'\n")
    (root / ".env").write_text("")
    (alpha / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    set_multiplex_active(True)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[], get_notice_delivery=lambda p: "public")
    runner._running = True
    default_adapter, alpha_adapter = RecordingAdapter(), RecordingAdapter(notice_delivery="private")
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner._profile_adapters = {"alpha": {Platform.TELEGRAM: alpha_adapter}}
    runner._primary_profile_name = "default"
    runner._session_source_cache = {}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._profile_failed_platforms = {}
    try:
        yield SimpleNamespace(root=root, alpha=alpha, runner=runner, alpha_adapter=alpha_adapter,
                              default_adapter=default_adapter)
    finally:
        set_multiplex_active(False)


def _alpha_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="1001", chat_type="dm", user_id="u1", profile="alpha")


def test_platform_notice_honours_the_served_profiles_notice_delivery(served):
    """alpha's ``platforms.telegram.notice_delivery: private`` (carried by ITS adapter) wins over the
    launch profile's GatewayConfig — as a standalone alpha gateway would behave."""
    runner = served.runner
    runner._thread_metadata_for_source = lambda source: None
    with _profile_runtime_scope(served.alpha):
        asyncio.run(runner._deliver_platform_notice(_alpha_source(), "notice"))
    assert [op for op, _ in served.alpha_adapter.calls] == ["send_private_notice"]
    assert served.default_adapter.calls == []


def test_loop_completion_persists_into_the_served_profiles_store(served):
    """The post-turn /loop completion hop carries the profile contextvars: the completed tick lands
    in alpha's state.db, not the default profile's."""
    from hermes_cli.goals import _get_session_db
    from hermes_cli.loops import LoopManager

    runner = served.runner
    entry = SimpleNamespace(session_id="sess-alpha-1", session_key="agent:alpha:telegram:dm:1001")
    with _profile_runtime_scope(served.alpha):
        _get_session_db()
        mgr = LoopManager(session_id=entry.session_id)
        mgr.set("check", interval_seconds=300, route={"platform": "telegram", "chat_id": "1001", "profile": "alpha"})
        assert mgr.fire_tick()
        asyncio.run(runner._post_turn_loop_completion(
            session_entry=entry, source=_alpha_source(), final_response="done"))

    def loop_row(home):
        import sqlite3
        db = home / "state.db"
        if not db.exists():
            return None
        con = sqlite3.connect(db)
        try:
            row = con.execute("SELECT value FROM state_meta WHERE key=?", (f"loop:{entry.session_id}",)).fetchone()
        finally:
            con.close()
        return json.loads(row[0]) if row else None

    assert loop_row(served.alpha)["awaiting_response"] is False
    assert loop_row(served.root) is None


def test_watch_event_gate_uses_the_owning_profiles_mode(served):
    """alpha has notifications ``off``; its watch event is drained silently even when the shared
    queue is drained from the root context, exactly as a standalone alpha gateway would."""
    from tools.process_registry import ProcessRegistry

    runner = served.runner
    registry = ProcessRegistry()
    evt = {"type": "watch_match", "session_id": "p1", "session_key": "agent:alpha:telegram:dm:1001",
           "platform": "telegram", "chat_type": "dm", "chat_id": "1001", "pattern": "DONE",
           "output": "DONE", "command": "x"}
    registry.completion_queue.put(evt)
    asyncio.run(runner._drain_watch_notifications(registry.completion_queue))
    assert registry.completion_queue.qsize() == 0
    assert served.alpha_adapter.calls == []
    assert served.default_adapter.calls == []


def test_served_profile_process_checkpoint_is_recovered_at_startup(served, monkeypatch):
    """A background process checkpointed during alpha's turn (alpha/processes.json) is re-adopted by
    the multiplexer's startup recovery, once, with its watcher re-armed."""
    from tools.process_registry import ProcessRegistry

    runner = served.runner
    import os
    entry = {"session_id": "proc_alpha01", "pid": os.getpid(), "pid_scope": "host", "command": "sleep 1",
             "started_at": 1.0, "watcher_interval": 5, "notify_on_complete": True,
             "session_key": "agent:alpha:telegram:dm:1001"}
    (served.alpha / "processes.json").write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: True)

    recovered = registry.recover_from_checkpoint()
    recovered += runner._recover_secondary_process_checkpoints(registry)
    assert recovered == 1
    assert [w["session_id"] for w in registry.pending_watchers] == ["proc_alpha01"]
    # Idempotent across homes: the process-global registry already tracks it.
    assert runner._recover_secondary_process_checkpoints(registry) == 0
