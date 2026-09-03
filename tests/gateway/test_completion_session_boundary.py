"""Session-boundary gating for background-process completion delivery.

Plain ``type=completion`` events historically carried only ``session_key``
(chat/thread routing), so a background process spawned in session A whose
completion fired after ``/new`` was injected into the chat's NEW session.
The fix stamps the spawning conversation's session-db id on the watcher at
spawn time and routes stamped events through the SAME pre-flight policy the
async-delegation path already uses (``_classify_completion_target``):

- terminal (user boundary such as /new)  -> drop with a log
- retry (transient DB uncertainty)       -> watcher re-polls
- deliver (live / idle-ended parent)     -> proceed as today

Unstamped legacy events keep today's deliver-always behavior.
"""

import asyncio
import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


class _SessionDB:
    def __init__(self, row, tip=None):
        self._row = row
        self._tip = tip

    async def get_session(self, session_id):
        return self._row

    async def get_compression_tip(self, session_id):
        return self._tip


def _runner(adapter, *, session_db=...):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    if session_db is not ...:
        runner._session_db = session_db
    return runner


def _finished_session(registry, session_id="proc_boundary", **kwargs):
    session = ProcessSession(
        id=session_id,
        command="echo done",
        task_id="task",
        started_at=1234.5,
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
        **kwargs,
    )
    registry._finished[session.id] = session
    return session


def _watcher(session_id, parent_session_id=None):
    watcher = {
        "session_id": session_id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }
    if parent_session_id is not None:
        watcher["parent_session_id"] = parent_session_id
    return watcher


def _run_watcher(monkeypatch, runner, watcher):
    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher(watcher))


def _completion_evt(parent_session_id=None, session_id="proc_x"):
    evt = {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": 1234.5,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }
    if parent_session_id is not None:
        evt["parent_session_id"] = parent_session_id
    return evt


# ---------------------------------------------------------------------------
# The stamp is threaded from the watcher into the completion event
# ---------------------------------------------------------------------------

def test_watcher_stamps_parent_session_id_on_completion_event(
    monkeypatch, isolated_registry,
):
    _finished_session(isolated_registry)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    captured = {}

    async def _capture(_text, evt):
        captured.update(evt)
        return True

    monkeypatch.setattr(runner, "_deliver_completion_notification", _capture)
    _run_watcher(
        monkeypatch, runner, _watcher("proc_boundary", "sess-spawner"),
    )

    assert captured.get("type") == "completion"
    assert captured.get("parent_session_id") == "sess-spawner"


def test_watcher_falls_back_to_process_session_stamp(
    monkeypatch, isolated_registry,
):
    """Watchers recovered without the stamp still pick it up off the
    ProcessSession (spawn-time stamp survives checkpoint/restore there)."""
    _finished_session(
        isolated_registry, parent_session_id="sess-from-registry",
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    captured = {}

    async def _capture(_text, evt):
        captured.update(evt)
        return True

    monkeypatch.setattr(runner, "_deliver_completion_notification", _capture)
    _run_watcher(monkeypatch, runner, _watcher("proc_boundary"))

    assert captured.get("parent_session_id") == "sess-from-registry"


# ---------------------------------------------------------------------------
# Pre-flight verdicts on stamped completion events
# ---------------------------------------------------------------------------

def test_completion_from_user_closed_session_is_dropped(
    monkeypatch, isolated_registry,
):
    """/new closed the spawning session -> the stamped completion must NOT
    land in the chat's new session."""
    _finished_session(isolated_registry)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(
        adapter,
        session_db=_SessionDB(
            {"ended_at": 1786288000.0, "end_reason": "session_reset"}
        ),
    )

    _run_watcher(
        monkeypatch, runner, _watcher("proc_boundary", "sess-closed"),
    )

    adapter.handle_message.assert_not_awaited()


def test_completion_after_idle_end_still_delivers(
    monkeypatch, isolated_registry,
):
    """Idle/timeout ends are the relay-plane norm — the chat stays routable
    and the completion must deliver."""
    _finished_session(isolated_registry)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(
        adapter,
        session_db=_SessionDB(
            {"ended_at": 1786288000.0, "end_reason": "idle_timeout"}
        ),
    )

    _run_watcher(
        monkeypatch, runner, _watcher("proc_boundary", "sess-idle"),
    )

    adapter.handle_message.assert_awaited_once()


def test_completion_from_live_session_delivers(monkeypatch, isolated_registry):
    _finished_session(isolated_registry)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))

    _run_watcher(
        monkeypatch, runner, _watcher("proc_boundary", "sess-live"),
    )

    adapter.handle_message.assert_awaited_once()


def test_unstamped_legacy_completion_delivers(monkeypatch, isolated_registry):
    """Events without the spawn-time stamp keep today's behavior even when a
    session DB is present."""
    _finished_session(isolated_registry)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB(None))

    _run_watcher(monkeypatch, runner, _watcher("proc_boundary"))

    adapter.handle_message.assert_awaited_once()


def test_retry_verdict_returns_false_for_watcher_repoll():
    """No session DB yet -> transient uncertainty -> retryable False, and no
    adapter injection happens."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, session_db=None)

    result = asyncio.run(
        runner._deliver_completion_notification(
            "text", _completion_evt("sess-uncertain"),
        )
    )

    assert result is False
    adapter.handle_message.assert_not_awaited()


def test_terminal_verdict_returns_none_without_injection():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB(None))

    result = asyncio.run(
        runner._deliver_completion_notification(
            "text", _completion_evt("sess-gone"),
        )
    )

    assert result is None
    adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The async-delegation path is unaffected
# ---------------------------------------------------------------------------

def test_async_delegation_gate_unchanged():
    """A stamped async_delegation event still routes through the existing
    delegation-owned gate (terminal verdict -> None), proving the completion
    branch did not fork or shadow the delegation policy."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB(None))

    evt = {
        "type": "async_delegation",
        "delegation_id": "",
        "session_key": "agent:main:telegram:dm:12345:678",
        "parent_session_id": "sess-gone",
        "status": "completed",
    }
    result = asyncio.run(runner._deliver_completion_notification("text", evt))

    assert result is None
    adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The stamp survives checkpoint/restore
# ---------------------------------------------------------------------------

def test_parent_session_id_survives_checkpoint_recovery(tmp_path, monkeypatch):
    import tools.process_registry as pr_module

    checkpoint = tmp_path / "processes.json"
    checkpoint.write_text(json.dumps([{
        "session_id": "proc_recovered",
        "command": "sleep 999",
        "pid": 4242,
        "pid_scope": "host",
        "host_start_time": 111.0,
        "started_at": 1234.5,
        "task_id": "task",
        "session_key": "agent:main:telegram:dm:123",
        "watcher_platform": "telegram",
        "watcher_chat_id": "123",
        "watcher_interval": 5,
        "notify_on_complete": True,
        "parent_session_id": "sess-spawner",
    }]), encoding="utf-8")
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)

    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_a: True)
    monkeypatch.setattr(
        registry, "_write_checkpoint", lambda *_a, **_kw: None,
    )

    assert registry.recover_from_checkpoint() == 1
    assert registry.get("proc_recovered").parent_session_id == "sess-spawner"
    assert len(registry.pending_watchers) == 1
    assert registry.pending_watchers[0]["parent_session_id"] == "sess-spawner"


def _watch_evt(parent_session_id="sess-closed", task_id="sa-0-watch"):
    return {
        "type": "watch_match",
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:dm:123",
        "parent_session_id": parent_session_id,
        "pattern": "FAIL",
        "command": "build",
        "output": "FAIL\n",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "task_id": task_id,
    }


def test_watch_match_from_user_closed_session_is_dropped(monkeypatch):
    """Watch events must use the same /new preflight as completions."""
    import queue as queue_mod

    adapter = SimpleNamespace(handle_message=AsyncMock(), send=AsyncMock())
    runner = _runner(
        adapter,
        session_db=_SessionDB(
            {"ended_at": 1786288000.0, "end_reason": "session_reset"}
        ),
    )
    classifier = AsyncMock(return_value="terminal")
    runner._classify_completion_target = classifier
    monkeypatch.setattr(
        ProcessRegistry,
        "should_surface_process_notification",
        staticmethod(lambda evt, surface_child=None: True),
    )
    completion_queue = queue_mod.Queue()
    completion_queue.put(_watch_evt("sess-closed"))
    asyncio.run(runner._drain_watch_notifications(completion_queue))

    classifier.assert_awaited_once_with("sess-closed")
    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_not_awaited()


def test_watch_match_from_live_session_still_delivers(monkeypatch):
    import queue as queue_mod

    adapter = SimpleNamespace(handle_message=AsyncMock(), send=AsyncMock())
    runner = _runner(adapter, session_db=_SessionDB({"ended_at": None}))
    completion_queue = queue_mod.Queue()
    completion_queue.put(_watch_evt("sess-live", task_id="task"))
    monkeypatch.setattr(
        ProcessRegistry,
        "should_surface_process_notification",
        staticmethod(lambda evt, surface_child=None: True),
    )
    asyncio.run(runner._drain_watch_notifications(completion_queue))

    assert adapter.handle_message.await_count + adapter.send.await_count == 1
