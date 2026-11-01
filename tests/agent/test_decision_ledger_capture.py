from types import SimpleNamespace

from agent.interrupt_control import InterruptControlMixin


def test_redirect_records_accepted_user_correction():
    recorded = []
    agent = SimpleNamespace(_executing_tools=False, _model_request_active=SimpleNamespace(is_set=lambda: True), _pending_redirect_lock=None, _pending_redirect=None, _interrupt_requested=False, _execution_thread_id=None, _interrupt_thread_signal_pending=False, _active_request_abort=None, api_mode="chat_completions", session_id="session", _current_turn_id="turn", _session_db=SimpleNamespace(append_decision_ledger_entry=lambda *args, **kwargs: recorded.append((args, kwargs))))
    assert InterruptControlMixin.redirect(agent, "Use Postgres instead.") is True
    assert recorded == [(("session", "correction", "Use Postgres instead."), {"turn_id": "turn"})]


def test_tool_active_redirect_records_a_correction_not_a_preference():
    recorded = []
    agent = type("ToolActiveAgent", (InterruptControlMixin,), {})()
    agent._executing_tools = True
    agent._pending_steer_lock = None
    agent._pending_steer = None
    agent._turn_user_intervened = False
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent.session_id = "session"
    agent._current_turn_id = "turn"
    agent._session_db = SimpleNamespace(
        append_decision_ledger_entry=lambda *args, **kwargs: recorded.append((args, kwargs))
    )

    assert InterruptControlMixin.redirect(agent, "Use Postgres instead.") is True

    assert recorded == [(("session", "correction", "Use Postgres instead."), {"turn_id": "turn"})]


def test_native_redirect_records_an_accepted_correction():
    recorded = []
    agent = SimpleNamespace(
        api_mode="codex_app_server",
        _codex_session=SimpleNamespace(request_steer=lambda _text: True),
        _pending_redirect_lock=None,
        _interrupt_requested=False,
        _turn_user_intervened=False,
        session_id="session",
        _current_turn_id="turn",
        _session_db=SimpleNamespace(
            append_decision_ledger_entry=lambda *args, **kwargs: recorded.append((args, kwargs))
        ),
    )

    assert InterruptControlMixin.redirect(agent, "Use Postgres instead.") is True

    assert recorded == [(("session", "correction", "Use Postgres instead."), {"turn_id": "turn"})]


def test_steer_does_not_create_a_durable_preference():
    recorded = []
    agent = SimpleNamespace(
        _pending_steer_lock=None,
        _pending_steer=None,
        _turn_user_intervened=False,
        session_id="session",
        _current_turn_id="turn",
        _session_db=SimpleNamespace(
            append_decision_ledger_entry=lambda *args, **kwargs: recorded.append((args, kwargs))
        ),
    )

    assert InterruptControlMixin.steer(agent, "Skip this check once.") is True

    assert recorded == []
