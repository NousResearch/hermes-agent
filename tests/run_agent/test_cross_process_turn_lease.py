"""AIAgent enters turns only after acquiring and reloading durable state."""

from __future__ import annotations

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
