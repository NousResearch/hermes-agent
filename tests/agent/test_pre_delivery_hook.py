from types import SimpleNamespace

from agent import turn_stop_gates


def _agent():
    return SimpleNamespace(
        _pre_delivery_nudges=0,
        _session_messages=[],
        _interim_content_was_streamed=lambda _value: False,
    )


def test_reset_retries_without_emitting_candidate(monkeypatch):
    agent = _agent()
    messages = [{"role": "user", "content": "Keep the beacon blue."}]
    monkeypatch.setattr(
        turn_stop_gates, "_pre_delivery_decision",
        lambda *_args: {"action": "RESET", "message": "Correct the color."})
    verdict = turn_stop_gates.apply_stop_gates(
        agent, {"role": "assistant", "content": "It is red."},
        final_response="It is red.", messages=messages, conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    assert verdict.continue_turn is True
    assert verdict.pending_verification_response is None
    assert agent._pre_delivery_nudges == 1
    assert messages[-2]["_pre_delivery_synthetic"] is True
    assert messages[-1]["content"] == "Correct the color."


def test_exhausted_withholds_candidate(monkeypatch):
    agent = _agent()
    messages = [{"role": "user", "content": "Keep the beacon blue."}]
    monkeypatch.setattr(
        turn_stop_gates, "_pre_delivery_decision",
        lambda *_args: {"action": "EXHAUSTED", "message": "Response withheld."})
    monkeypatch.setattr(turn_stop_gates, "_verify_on_stop_nudge", lambda *_args: None)
    monkeypatch.setattr(turn_stop_gates, "_pre_verify_nudge", lambda *_args: None)
    monkeypatch.setattr(turn_stop_gates, "_kanban_stop_nudge", lambda *_args: None)
    final = {"role": "assistant", "content": "It is red."}
    verdict = turn_stop_gates.apply_stop_gates(
        agent, final, final_response="It is red.", messages=messages,
        conversation_history=[], pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    assert verdict.continue_turn is False
    assert verdict.final_response == "Response withheld."
    assert final["content"] == "Response withheld."
