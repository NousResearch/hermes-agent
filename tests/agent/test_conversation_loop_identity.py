from types import SimpleNamespace

from agent import conversation_loop


def test_on_session_start_hook_receives_agent_identity(monkeypatch):
    captured = {}

    def fake_invoke_hook(event, **kwargs):
        captured["event"] = event
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    monkeypatch.setattr(
        "agent.credits_tracker.seed_credits_at_session_start",
        lambda _agent: None,
    )
    agent = SimpleNamespace(
        _session_db=None,
        _user_id="alice",
        _user_name="Alice Example",
        model="gpt-5.5",
        platform="tui",
        provider="openai-codex",
        session_id="session-1",
        _build_system_prompt=lambda system_message=None: "system prompt",
    )

    conversation_loop._restore_or_build_system_prompt(
        agent,
        system_message=None,
        conversation_history=[],
    )

    assert captured["event"] == "on_session_start"
    assert captured["kwargs"]["session_id"] == "session-1"
    assert captured["kwargs"]["platform"] == "tui"
    assert captured["kwargs"]["user_id"] == "alice"
    assert captured["kwargs"]["user_name"] == "Alice Example"
