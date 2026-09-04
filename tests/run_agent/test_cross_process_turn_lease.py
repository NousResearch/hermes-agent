"""AIAgent enters turns only after acquiring and reloading durable state."""

from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace

from agent import relay_runtime
from hermes_state import SessionDB
from run_agent import AIAgent


class _DB:
    def __init__(self, session_exists=True, acquire_result=True):
        self.events = []
        self.session_exists = session_exists
        self.acquire_result = acquire_result

    def get_session(self, session_id):
        return {"id": session_id} if self.session_exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None and self.acquire_result is False:
            on_wait(0.0)
        return self.acquire_result

    def resolve_resume_session_id(self, session_id):
        self.events.append(("resolve", session_id))
        return "compressed-tip"

    def get_messages_as_conversation(self, session_id, **kwargs):
        self.events.append(("reload", session_id, kwargs))
        return [{"role": "user", "content": "durable latest"}]

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


def _agent_with_db(db, *, session_id="stale-parent", platform="desktop"):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = platform
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: session_id
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None
    agent.status_callback = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    return agent


def test_run_conversation_acquires_then_reloads_latest_tip(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    status_events = []
    agent.status_callback = lambda kind, text=None: status_events.append(
        (kind, text)
    )

    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        observed["session_id"] = _agent.session_id
        return {"final_response": "ok", "messages": history, "failed": False}

    # Simulate a contended wait so the resume status path is covered.
    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None:
            on_wait(0.0)
        return True

    db.acquire_session_turn_lease = acquire_with_wait

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result["final_response"] == "ok"
    assert observed == {
        "history": [{"role": "user", "content": "durable latest"}],
        "session_id": "compressed-tip",
    }
    assert [event[0] for event in db.events] == [
        "acquire",
        "resolve",
        "reload",
        "release",
    ]
    assert db.events[2][2] == {
        "repair_alternation": True,
        "include_row_ids": True,
    }
    assert any(
        kind == "lifecycle"
        and text
        and "waiting for it to finish" in text
        for kind, text in status_events
    )
    assert any(
        kind == "lifecycle"
        and text
        and "loading the latest transcript" in text
        for kind, text in status_events
    )


def test_run_conversation_acquires_lease_when_session_probe_raises(monkeypatch):
    """A locked / non-WAL get_session must not skip the durable lease."""
    db = _DB()

    def locked_get_session(_session_id):
        raise sqlite3.OperationalError("database is locked")

    db.get_session = locked_get_session
    agent = _agent_with_db(db)

    # Simulate a contended wait so the resolve+reload path is exercised.
    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None:
            on_wait(0.0)
        return True

    db.acquire_session_turn_lease = acquire_with_wait

    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        observed["session_id"] = _agent.session_id
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result["final_response"] == "ok"
    assert observed == {
        "history": [{"role": "user", "content": "durable latest"}],
        "session_id": "compressed-tip",
    }
    assert [event[0] for event in db.events] == [
        "acquire",
        "resolve",
        "reload",
        "release",
    ]


def test_fresh_session_keeps_caller_seed_without_durable_lease(monkeypatch):
    db = _DB(session_exists=False)
    agent = _agent_with_db(db, session_id="fresh", platform="subagent")
    agent._session_db_created = False
    agent._parent_session_id = "parent"
    agent._conversation_root_id = lambda: "parent"

    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    seed = [{"role": "user", "content": "delegated context"}]

    AIAgent.run_conversation(agent, "work", conversation_history=seed)

    assert observed["history"] is seed
    assert db.events == []


def test_run_conversation_lease_timeout_returns_resend_notice(monkeypatch):
    db = _DB(acquire_result=False)
    agent = _agent_with_db(db)
    status_events = []
    agent.status_callback = lambda kind, text=None: status_events.append(
        (kind, text)
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("turn must not start without a lease")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", boom)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result["failed"] is True
    assert result["completed"] is False
    assert "session_turn_lease_timeout:" in result["error"]
    assert "send it again" in result["final_response"]
    assert [event[0] for event in db.events] == ["acquire"]
    assert any(
        kind == "lifecycle"
        and text
        and "waiting for it to finish" in text
        for kind, text in status_events
    )
    assert any(
        kind == "warn" and text and "send it again" in text
        for kind, text in status_events
    )


def test_run_conversation_lease_wait_honors_interrupt(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)

    def acquire_with_abort(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        should_abort = kwargs.get("should_abort")
        assert callable(should_abort)
        agent._interrupt_requested = True
        agent._interrupt_message = "follow-up while waiting"
        assert should_abort()
        return False

    db.acquire_session_turn_lease = acquire_with_abort

    def boom(*_args, **_kwargs):
        raise AssertionError("turn must not start when lease wait is aborted")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", boom)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result.get("interrupted") is True
    assert result.get("failed") is not True
    assert result.get("final_response")
    assert "not processed" in result["final_response"]
    assert result.get("interrupt_message") == "follow-up while waiting"
    assert "session_turn_lease_timeout" not in str(result.get("error", ""))
    assert [event[0] for event in db.events] == ["acquire"]
    assert agent._interrupt_requested is False
    assert agent._interrupt_message is None


def test_run_conversation_second_turn_after_lease_wait_abort(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    turns = {"n": 0}

    def acquire_then_succeed(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        should_abort = kwargs.get("should_abort")
        if turns["n"] == 0:
            agent._interrupt_requested = True
            agent._interrupt_message = "follow-up while waiting"
            assert should_abort()
            return False
        assert not should_abort()
        return True

    db.acquire_session_turn_lease = acquire_then_succeed

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    first = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )
    assert first.get("interrupted") is True
    turns["n"] = 1
    second = AIAgent.run_conversation(
        agent,
        "follow-up",
        conversation_history=[{"role": "user", "content": "stale"}],
    )
    assert second["final_response"] == "ok"
    assert agent._interrupt_requested is False


def test_run_conversation_interrupts_when_lease_refresh_lost(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False):
        interrupt_calls.append((message, hard_cancel))
        agent._interrupt_requested = True
        agent._interrupt_message = message

    agent.interrupt = track_interrupt

    def refresh_lost(session_id, holder, **kwargs):
        return False

    db.refresh_session_turn_lease = refresh_lost

    observed = {"started": False}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["started"] = True
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if getattr(_agent, "_interrupt_requested", False):
                return {
                    "final_response": "",
                    "messages": history,
                    "api_calls": 0,
                    "completed": False,
                    "interrupted": True,
                }
            time.sleep(0.01)
        raise AssertionError("refresh loss did not interrupt the turn")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)

    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "seed"}],
    )

    assert observed["started"] is True
    assert result.get("interrupted") is True
    assert interrupt_calls
    assert interrupt_calls[0][1] is True
    assert "lease lost" in str(interrupt_calls[0][0]).lower()


def test_run_conversation_interrupts_when_lease_refresh_errors(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False):
        interrupt_calls.append((message, hard_cancel))
        agent._interrupt_requested = True
        agent._interrupt_message = message

    agent.interrupt = track_interrupt

    def refresh_error(session_id, holder, **kwargs):
        raise OSError("database unavailable")

    db.refresh_session_turn_lease = refresh_error

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if getattr(_agent, "_interrupt_requested", False):
                return {
                    "final_response": "",
                    "messages": history,
                    "api_calls": 0,
                    "completed": False,
                    "interrupted": True,
                }
            time.sleep(0.01)
        raise AssertionError("refresh error did not interrupt the turn")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)

    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "seed"}],
    )

    assert result.get("interrupted") is True
    assert interrupt_calls
    assert interrupt_calls[0][1] is True
    assert "could not be refreshed" in str(interrupt_calls[0][0]).lower()


def test_refresh_error_after_loop_completion_does_not_poison_next_turn(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    interrupt_started = threading.Event()
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False):
        interrupt_calls.append((message, hard_cancel))
        interrupt_started.set()
        release_refresh.wait(timeout=2.0)
        agent._interrupt_requested = True
        agent._interrupt_message = message

    agent.interrupt = track_interrupt

    def delayed_refresh_error(session_id, holder, **kwargs):
        refresh_started.set()
        raise OSError("database unavailable")

    db.refresh_session_turn_lease = delayed_refresh_error

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        assert refresh_started.wait(timeout=2.0)
        assert interrupt_started.wait(timeout=2.0)
        threading.Timer(0.05, release_refresh.set).start()
        return {"final_response": "ok", "messages": history, "failed": False}

    original_finish = relay_runtime.SESSION_COORDINATOR.finish_logical_calls

    def finish_after_refresh(turn, *, outcome):
        time.sleep(0.05)
        return original_finish(turn, outcome=outcome)

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    monkeypatch.setattr(
        relay_runtime.SESSION_COORDINATOR,
        "finish_logical_calls",
        finish_after_refresh,
    )

    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "seed"}],
    )

    assert result["final_response"] == "ok"
    assert len(interrupt_calls) == 1
    assert interrupt_calls[0][1] is True
    assert agent._interrupt_requested is False
    assert agent._interrupt_message is None


def test_late_refresh_miss_after_release_does_not_interrupt(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    released = threading.Event()
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False):
        interrupt_calls.append((message, hard_cancel))
        agent._interrupt_requested = True
        agent._interrupt_message = message

    agent.interrupt = track_interrupt

    def refresh_after_release(session_id, holder, **kwargs):
        released.wait(timeout=2.0)
        return False

    db.refresh_session_turn_lease = refresh_after_release

    orig_release = db.release_session_turn_lease

    def release_and_signal(session_id, holder):
        orig_release(session_id, holder)
        released.set()

    db.release_session_turn_lease = release_and_signal

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        time.sleep(0.03)
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "seed"}],
    )

    time.sleep(0.05)
    assert result["final_response"] == "ok"
    assert interrupt_calls == []
    assert agent._interrupt_requested is False


def test_run_conversation_exposes_holder_for_fenced_flush(monkeypatch):
    """The acquired holder is visible to persist, then cleared on release."""
    db = _DB()
    captured = {}

    def append_messages_batch(session_id, messages, **kwargs):
        captured["session_id"] = session_id
        captured["turn_lease_holder"] = kwargs.get("turn_lease_holder")
        captured["count"] = len(messages)
        return len(messages)

    db.append_messages_batch = append_messages_batch
    agent = _agent_with_db(db)
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._db_flush_scan_prefix = None
    agent._pending_cli_user_message = None
    agent._session_persist_lock = None

    # Simulate a contended wait so the resolve+reload path is exercised.
    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None:
            on_wait(0.0)
        return True

    db.acquire_session_turn_lease = acquire_with_wait

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        captured["active"] = getattr(
            _agent, "_active_session_turn_lease_holder", None
        )
        ok = _agent._flush_messages_to_session_db(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "done"},
            ],
            [],
        )
        captured["flush_ok"] = ok
        return {"final_response": "done", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "durable latest"}],
    )

    assert result["final_response"] == "done"
    assert captured["flush_ok"] is True
    assert captured["active"]
    assert captured["active"].startswith("pid=")
    assert captured["turn_lease_holder"] == captured["active"]
    assert captured["session_id"] == "compressed-tip"
    assert captured["count"] == 2
    assert getattr(agent, "_active_session_turn_lease_holder", None) is None
    assert [event[0] for event in db.events] == [
        "acquire",
        "resolve",
        "reload",
        "release",
    ]


def _flush_agent(db, session_id):
    """Bind the real flush onto a stand-in so we can use a live SessionDB."""
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent


def test_flush_messages_to_session_db_fences_stale_holder_on_live_db(tmp_path):
    """A-loses / B-acquires / A-late-flush, through the real persist path."""
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    first.create_session("shared", source="test")
    stale_holder = "pid=1:turn=stale"
    next_holder = "pid=2:turn=next"
    assert first.try_acquire_session_turn_lease(
        "shared", stale_holder, ttl_seconds=5
    )

    agent = _flush_agent(first, "shared")
    agent._active_session_turn_lease_holder = stale_holder
    owned = [{"role": "user", "content": "stale-owned"}]
    assert agent._flush_messages_to_session_db(owned, []) is True
    assert [m["content"] for m in first.get_messages("shared")] == ["stale-owned"]

    first.release_session_turn_lease("shared", stale_holder)
    assert second.try_acquire_session_turn_lease(
        "shared", next_holder, ttl_seconds=5
    )

    late = [{"role": "assistant", "content": "late stale reply"}]
    assert agent._flush_messages_to_session_db(late, []) is False
    assert agent._last_persistence_error_cause == "turn_lease"
    assert [m["content"] for m in second.get_messages("shared")] == ["stale-owned"]

    agent._active_session_turn_lease_holder = next_holder
    assert agent._flush_messages_to_session_db(late, []) is True
    assert [m["content"] for m in second.get_messages("shared")] == [
        "stale-owned",
        "late stale reply",
    ]
    second.release_session_turn_lease("shared", next_holder)
    first.close()
    second.close()
