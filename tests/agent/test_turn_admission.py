"""The shared turn entry point refuses policy-blocked work before preflight."""

from types import SimpleNamespace

from agent import conversation_loop
from hermes_cli import lifecycle


def test_pre_turn_admission_returns_a_failed_turn_without_starting_the_loop(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda *_a, **_kw: [{"action": "block", "message": "policy refused"}])
    agent = SimpleNamespace(session_id="session-id", platform="cli", model="test", provider="test")
    history = [{"role": "user", "content": "earlier"}]

    result = conversation_loop.run_conversation(agent, "new prompt", conversation_history=history)

    assert result == {
        "failed": True, "completed": False, "error": "policy refused", "messages": history}
