from types import SimpleNamespace
from agent.interrupt_control import InterruptControlMixin


def test_redirect_records_accepted_user_correction():
    recorded = []
    agent = SimpleNamespace(_executing_tools=False, _model_request_active=SimpleNamespace(is_set=lambda: True), _pending_redirect_lock=None, _pending_redirect=None, _interrupt_requested=False, _execution_thread_id=None, _interrupt_thread_signal_pending=False, _active_request_abort=None, api_mode="chat_completions", session_id="session", _current_turn_id="turn", _session_db=SimpleNamespace(append_decision_ledger_entry=lambda *args, **kwargs: recorded.append((args, kwargs))))
    assert InterruptControlMixin.redirect(agent, "Use Postgres instead.") is True
    assert recorded == [(("session", "correction", "Use Postgres instead."), {"turn_id": "turn"})]
