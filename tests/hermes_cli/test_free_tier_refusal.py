"""Classic CLI admission precedes prompt staging and image/auxiliary work."""
from types import SimpleNamespace
from unittest.mock import Mock


def test_cli_refuses_before_staging(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_cli_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    for _ in range(10):
        usage.record_completed_tool(usage.current_identity())
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1")
    cli = SimpleNamespace(agent=agent, conversation_history=[], _secret_capture_callback=None,
        _ensure_runtime_credentials=lambda: True, _active_agent_route_signature="route",
        _resolve_turn_agent_config=lambda _: {"signature": "route", "model": "nous/welcome", "runtime": {}},
        _init_agent=lambda **kw: True, _chat_route_images=Mock(side_effect=AssertionError("before aux work")))
    assert CLIChatTurnMixin.chat(cli, "not staged") == usage.LIMIT_NOTICE
    assert cli.conversation_history == []
    cli._chat_route_images.assert_not_called()
    assert cli._last_turn_result["refusal_reason"] == "free_tier_limit"
    assert cli._pending_input.get_nowait() == ("not staged", [])


def test_raced_cli_refusal_removes_only_staged_input(monkeypatch):
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from cli import _ChatTurn
    import hermes_cli.cli_chat_turn_mixin as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "answer"}]
    agent = SimpleNamespace(_session_messages=history)
    from cli import _SeededQueryMessage
    envelope = _SeededQueryMessage("exact @file:before expansion", ["image.png"])
    cli = SimpleNamespace(conversation_history=history, _prompt_start_time=None,
                          agent=agent, _flush_stream=lambda: None,
                          _turn_input=("expanded", []), _submitted_input=envelope)
    CLIChatTurnMixin._chat_stage_user_message(cli, agent, "expanded")
    turn = _ChatTurn()
    turn.result = {"messages": history, "refusal_reason": "free_tier_limit", "free_tier": {"capped": True}}
    CLIChatTurnMixin._chat_settle_turn(cli, turn)
    assert cli.conversation_history is history
    assert [m["content"] for m in history] == ["before", "answer"]
    assert agent._pending_cli_user_message is None
    assert cli._pending_input.get_nowait() is envelope
