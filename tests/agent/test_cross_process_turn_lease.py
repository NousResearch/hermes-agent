"""AIAgent enters turns only after acquiring and reloading durable state."""

from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


@pytest.mark.parametrize("signal", ["on_wait", "on_contended"])
def test_run_conversation_acquires_then_reloads_latest_tip(monkeypatch, signal):
    """A lease wait and a busy-database retry both reload after admission (the busy writer may
    be the previous holder's final flush); only a real holder is announced to the user."""
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

    def resolve_relay_cwds(_agent, _task_id, session_id, _platform):
        observed["relay_cwd_session_id"] = session_id
        return "", ""

    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        kwargs[signal](*((0.0,) if signal == "on_wait" else ()))
        return True

    db.acquire_session_turn_lease = acquire_with_wait

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", resolve_relay_cwds)
    result = AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )

    assert result["final_response"] == "ok"
    assert observed == {
        "history": [{"role": "user", "content": "durable latest"}],
        "session_id": "compressed-tip",
        "relay_cwd_session_id": "compressed-tip",
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
    texts = [text or "" for _kind, text in status_events]
    for notice in ("Another Hermes process", "Session is free"):
        assert any(notice in text for text in texts) is (signal == "on_wait"), notice


def test_post_turn_review_starts_only_after_durable_lease_release(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    launches = []

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        assert _agent._active_session_turn_lease_holder is not None
        _agent._post_turn_background_review_candidate = {
            "messages_snapshot": [{"role": "assistant", "content": "done"}],
            "review_memory": True,
            "review_skills": False,
            "_review_session_id": _agent.session_id,
        }
        return {"final_response": "ok", "messages": history, "failed": False}

    def launch_review(**kwargs):
        launches.append(
            (
                kwargs,
                [event[0] for event in db.events],
                agent._active_session_turn_lease_holder,
            )
        )

    agent._spawn_background_review = launch_review
    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)

    AIAgent.run_conversation(agent, "new message", conversation_history=[])

    assert launches == [
        (
            {
                "messages_snapshot": [
                    {"role": "assistant", "content": "done"}
                ],
                "review_memory": True,
                "review_skills": False,
                "_review_session_id": "stale-parent",
            },
            ["acquire", "release"],
            None,
        )
    ]
    assert not hasattr(agent, "_post_turn_background_review_candidate")


def test_durable_resume_rebinds_foreground_review_admission(monkeypatch):
    from agent import review_admission

    db = _DB()
    agent = _agent_with_db(db)

    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        kwargs["on_wait"](0.0)
        return True

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        assert review_admission.other_live_turn(
            "compressed-tip", None, _agent._active_turn_profile_key
        )
        return {"final_response": "ok", "messages": history, "failed": False}

    db.acquire_session_turn_lease = acquire_with_wait
    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)

    AIAgent.run_conversation(agent, "new message", conversation_history=[])

    assert review_admission.other_live_turn("compressed-tip", None) is False


def test_durable_resume_waits_for_the_rotated_sessions_review_before_the_loop(monkeypatch):
    """Resuming onto a rotated tip fences THAT session's admitted review and does not enter the
    conversation loop until the fork publishes its exit — every escalation bound may elapse
    unacknowledged; the turn still parks on the unbounded wait."""
    from agent import background_review, review_admission

    monkeypatch.setattr(
        background_review, "_CANCEL_ACK_ESCALATION_SECONDS", 0.01, raising=False
    )
    db = _DB()
    agent = _agent_with_db(db)
    other = SimpleNamespace(
        session_id="compressed-tip",
        _background_review_agent=None,
        _background_review_run=None,
        _background_review_lock=threading.Lock(),
    )
    run = background_review.prepare_background_review_run(
        other,
        session_id="compressed-tip",
        profile_key=review_admission.current_profile_key(),
    )
    assert run is not None
    assert run.begin_request(object()) is True

    entered_wait = threading.Event()
    release = threading.Event()
    loop_entered = threading.Event()
    observed_timeouts = []

    class ControlledCompletion:
        def __init__(self):
            self.set_calls = 0

        def wait(self, timeout=None):
            observed_timeouts.append(timeout)
            if timeout is not None:
                return False  # an escalation bound elapsed with no acknowledgement
            entered_wait.set()
            assert release.wait(timeout=10.0)
            return True

        def set(self):
            self.set_calls += 1

        def is_set(self):
            return self.set_calls > 0

    run.request_done = ControlledCompletion()
    monkeypatch.setattr(
        background_review, "_interrupt_background_review", lambda _fork, **_kwargs: None
    )

    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        kwargs["on_wait"](0.0)
        return True

    db.acquire_session_turn_lease = acquire_with_wait

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        loop_entered.set()
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)

    outcome = {}

    def foreground():
        try:
            outcome["result"] = AIAgent.run_conversation(
                agent, "new message", conversation_history=[]
            )
        except BaseException as exc:  # surfaced by the assertions below
            outcome["error"] = exc

    turn = threading.Thread(target=foreground, daemon=True)
    turn.start()
    try:
        assert entered_wait.wait(2.0)
        assert run.cancel_requested.is_set()
        assert loop_entered.is_set() is False
    finally:
        release.set()
        turn.join(timeout=10.0)
        background_review.finish_background_review_run(other, run)

    assert not turn.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert observed_timeouts[-1] is None
    assert observed_timeouts[:-1] and all(t == 0.01 for t in observed_timeouts[:-1])
    assert loop_entered.is_set()
    assert agent.session_id == "compressed-tip"
    assert review_admission.other_live_turn("compressed-tip", None) is False


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


_LATEST = [{"role": "user", "content": "durable latest"}]


@pytest.mark.parametrize(
    "row_after, reload_rows, expect_reload",
    [
        # No row after the wait: keep the caller's seed, even if a reload would return rows.
        (False, _LATEST, False),
        # The holder created the row meanwhile: reload it, and skip the redundant create.
        (True, _LATEST, True),
        # A proven row wins even when its transcript is empty.
        (True, [], True),
        # An unreadable row that reloads nothing keeps the caller's history, carried input included.
        ("raises", [], False),
    ],
    ids=["absent-after-wait", "created-during-wait", "created-empty-during-wait", "unreadable-empty"],
)
def test_waited_admission_uses_the_row_read_after_the_lease(
    monkeypatch, row_after, reload_rows, expect_reload,
):
    """After a contended wait, the row state read under the lease decides reload and row flag."""
    from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT

    db = _DB()

    def locked_get_session(_session_id):
        raise sqlite3.OperationalError("database is locked")

    def acquire_with_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        kwargs["on_wait"](0.0)
        if row_after == "raises":
            db.get_session = locked_get_session
        else:
            db.session_exists = row_after
        return True

    db.acquire_session_turn_lease = acquire_with_wait
    db.get_messages_as_conversation = lambda *_a, **_k: [dict(m) for m in reload_rows]
    agent = _agent_with_db(db, session_id="client-id", platform="api_server")
    agent._session_db_created = False
    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        observed["row_known"] = _agent._session_db_created
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    carried = {"role": "user", "content": "carried", _PERSIST_AFTER_ADMISSION_INTERRUPT: True}
    seed = [{"role": "user", "content": "caller history"}, carried]
    AIAgent.run_conversation(agent, "work", conversation_history=seed)

    if expect_reload:
        assert observed["history"] == reload_rows + [carried]
    else:
        assert observed["history"] is seed
    assert observed["row_known"] is (row_after is True)


def test_first_turn_on_fresh_session_serializes_a_second_writer(tmp_path, monkeypatch):
    """A client-addressed session id has no row until its first turn writes one; a second
    turn arriving meanwhile must wait for that turn and see its rows, not interleave."""
    path = tmp_path / "state.db"
    db_a, db_b = SessionDB(path), SessionDB(path)
    a_in_turn, a_may_finish, a_finished = threading.Event(), threading.Event(), threading.Event()
    observed = {}

    def fake_run(_agent, message, _system, history, *_args, **_kwargs):
        if message == "first":
            _agent._session_db.create_session("client-id", source="api_server")
            _agent._session_db.append_message("client-id", "user", "first")
            a_in_turn.set()
            a_may_finish.wait(timeout=10)
            _agent._session_db.append_message("client-id", "assistant", "first answer")
            a_finished.set()  # still inside the turn, so strictly before the lease is released
        else:
            observed["a_finished"] = a_finished.is_set()
            observed["history"] = [m.get("content") for m in history]
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    agent_a = _agent_with_db(db_a, session_id="client-id", platform="api_server")
    agent_a._session_db_created = False
    agent_b = _agent_with_db(db_b, session_id="client-id", platform="api_server")
    # B either starts waiting on the lease or, unserialized, runs to the end.
    b_waiting_or_done = threading.Event()
    real_acquire = db_b.acquire_session_turn_lease

    def acquire_b(*args, on_wait=None, **kwargs):
        def waiting(elapsed):
            b_waiting_or_done.set()
            on_wait(elapsed)
        return real_acquire(*args, on_wait=waiting, **kwargs)

    db_b.acquire_session_turn_lease = acquire_b

    def run_b():
        try:
            AIAgent.run_conversation(
                agent_b, "second",
                conversation_history=db_b.get_messages_as_conversation("client-id"))
        finally:
            b_waiting_or_done.set()

    thread_a = threading.Thread(
        target=lambda: AIAgent.run_conversation(agent_a, "first", conversation_history=[]))
    thread_b = threading.Thread(target=run_b)
    try:
        thread_a.start()
        assert a_in_turn.wait(timeout=10)
        thread_b.start()
        assert b_waiting_or_done.wait(timeout=10)
        a_may_finish.set()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)
        assert not thread_a.is_alive() and not thread_b.is_alive()
    finally:
        a_may_finish.set()
        db_a.close()
        db_b.close()

    assert observed["a_finished"] is True
    assert observed["history"] == ["first", "first answer"]


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
    assert result["final_response"]
    assert [event[0] for event in db.events] == ["acquire"]
    assert any(kind == "lifecycle" and text for kind, text in status_events)
    assert any(kind == "warn" and text for kind, text in status_events)


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
        "[Monday 12:00] new message",
        conversation_history=[{"role": "user", "content": "stale"}],
        persist_user_message="new message",
        persist_user_timestamp=123.0,
        persist_user_display_kind="internal_notification",
        persist_user_display_metadata={"kind": "test"},
        persist_user_platform_id="platform-1",
    )

    assert result.get("interrupted") is True
    assert result.get("failed") is not True
    assert result.get("final_response")
    assert result.get("interrupt_message") == "follow-up while waiting"
    assert result["messages"][-1] == {
        "role": "user",
        "content": "new message",
        "api_content": "[Monday 12:00] new message",
        "timestamp": 123.0,
        "display_kind": "internal_notification",
        "display_metadata": {"kind": "test"},
        "platform_message_id": "platform-1",
        "_persist_after_admission_interrupt": True,
    }
    assert "session_turn_lease_timeout" not in str(result.get("error", ""))
    assert [event[0] for event in db.events] == ["acquire"]
    assert agent._interrupt_requested is False
    assert agent._interrupt_message is None


def test_pre_admission_user_row_in_history_is_flushed_once():
    db = MagicMock()
    db.append_messages_batch.return_value = [1, 2]
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._persist_user_message_idx = 2
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._flushed_db_message_ids = set()
    agent._last_flushed_db_idx = 0
    agent.session_id = "session"

    persisted = {"role": "assistant", "content": "old reply"}
    interrupted = {
        "role": "user",
        "content": "original",
        "_persist_after_admission_interrupt": True,
    }
    current = {"role": "user", "content": "follow-up"}
    history = [persisted, interrupted]
    messages = [*history, current]

    agent._flush_messages_to_session_db(messages, conversation_history=history)
    agent._flush_messages_to_session_db(messages, conversation_history=history)

    rows = db.append_messages_batch.call_args.kwargs["messages"]
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "original"),
        ("user", "follow-up"),
    ]
    assert db.append_messages_batch.call_count == 1


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


def test_carried_input_survives_waited_reload_on_follow_up_turn(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    turns = {"n": 0}

    def acquire_false_then_true_after_wait(session_id, holder, **kwargs):
        db.events.append(("acquire", session_id, holder))
        if turns["n"] == 0:
            agent._interrupt_requested = True
            agent._interrupt_message = "follow-up while waiting"
            return False
        kwargs["on_wait"](0.0)  # the follow-up also has to wait before admission
        return True

    db.acquire_session_turn_lease = acquire_false_then_true_after_wait
    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = history
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    first = AIAgent.run_conversation(
        agent, "original", conversation_history=[{"role": "user", "content": "stale"}]
    )
    assert first.get("interrupted") is True
    carried = first["messages"][-1]
    assert carried["_persist_after_admission_interrupt"] is True
    turns["n"] = 1
    AIAgent.run_conversation(agent, "follow-up", conversation_history=first["messages"])

    assert observed["history"] == [
        {"role": "user", "content": "durable latest"},
        carried,
    ]


def test_run_conversation_interrupts_when_lease_refresh_lost(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False, **kwargs):
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


def test_run_conversation_interrupts_when_lease_refresh_errors(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False, **kwargs):
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


def test_refresh_error_after_loop_completion_does_not_poison_next_turn(monkeypatch):
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.01
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    interrupt_started = threading.Event()
    interrupt_calls = []

    def track_interrupt(message=None, hard_cancel=False, **kwargs):
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

    def track_interrupt(message=None, hard_cancel=False, **kwargs):
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


def test_foreground_admission_preempts_a_cross_process_review(tmp_path):
    """The real waiting path: a foreground turn admitted through ``admit_durable_turn_lease``
    while another handle's automatic review holds the row is told it is waiting, the review is
    hard-interrupted from its own renewal tick, and the turn is admitted once the fork's exit
    releases the row — the foreground never waits for the review to finish its work."""
    from agent import background_review
    from agent.turn_facade_lease import admit_durable_turn_lease

    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared", source="test")
    interrupted = threading.Event()
    fork = SimpleNamespace(hard_interrupt=lambda *_a, **_k: interrupted.set())
    run = background_review._BackgroundReviewRun()
    assert run.begin_request(fork) is True
    lease, reason = background_review._try_acquire_durable_review_lease(
        SimpleNamespace(_session_db=review_db), fork, "shared", run
    )
    assert reason is None and lease is not None

    agent = _agent_with_db(foreground_db, session_id="shared", platform="cli")
    status_events = []
    agent.status_callback = lambda kind, text=None: status_events.append((kind, text))
    outcome = {}

    def foreground():
        outcome["admission"] = admit_durable_turn_lease(
            agent, session_id="shared", relay_turn_id="turn-1",
            task_context={"platform": "cli", "session_id": "shared", "task_id": "t"},
            conversation_history=[],
        )

    turn = threading.Thread(target=foreground, daemon=True)
    turn.start()
    try:
        for _ in range(400):
            lease.refresh_tick()
            if interrupted.is_set():
                break
            interrupted.wait(0.05)
        assert interrupted.is_set()
        assert run.cancel_requested.is_set()
        assert "admission" not in outcome  # the row is still the review's
    finally:
        lease.stop_refresher()
        lease.release()
        turn.join(timeout=10.0)

    admission = outcome["admission"]
    assert admission.early_result is None
    assert admission.lease is not None
    assert any(
        kind == "lifecycle" and text and "waiting for it to finish" in text
        for kind, text in status_events
    )
    admission.lease.release()
