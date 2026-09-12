"""Classic CLI triggers use real skill loading without changing durable input."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import skill_commands
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


@pytest.fixture
def trigger_turn(tmp_path, monkeypatch):
    import tools.skills_tool as skills_tool

    skills = tmp_path / "skills"
    skill = skills / "market-watch"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: market-watch\ndescription: Market analysis.\n"
        "metadata:\n  hermes:\n    triggers: [market, EUR/USD]\n---\n"
        "Use verified market sources.\n", encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    skill_commands.scan_skill_commands()
    shell = SimpleNamespace(
        session_id="trigger-test", agent=SimpleNamespace(run_conversation=MagicMock(return_value={})),
        conversation_history=[], _sudo_password_callback=None, _approval_callback=None,
        _secret_capture_callback=None, _flush_credit_notices=lambda: None,
    )
    return shell, SimpleNamespace(voice_prefix="", stream_callback=None, result=None)


@pytest.mark.parametrize("voice_prefix", ["", "[Voice input] "])
def test_trigger_expansion_preserves_original_turn(trigger_turn, voice_prefix):
    shell, turn = trigger_turn
    turn.voice_prefix = voice_prefix
    message = "What is happening with eur/usd today?"
    prior = {"role": "assistant", "content": "Previous response."}
    staged = {"role": "user", "content": message}
    shell.conversation_history = [prior, staged]

    CLIChatTurnMixin._chat_run_agent(shell, turn, message)

    kwargs = shell.agent.run_conversation.call_args.kwargs
    expanded = kwargs["user_message"]
    assert "Use verified market sources." in expanded
    assert expanded.startswith(voice_prefix + "[IMPORTANT:")
    assert skill_commands.extract_user_instruction_from_skill_message(
        expanded.removeprefix(voice_prefix)
    ) == message
    assert kwargs["persist_user_message"] == message
    assert kwargs["conversation_history"] == [prior]
    assert shell.conversation_history == [prior, staged]


@pytest.mark.parametrize("message", [
    "Tell me a joke.", "This supermarket is busy.", "/market-watch market",
    "", [{"type": "text", "text": "market"}],
])
def test_non_trigger_turns_pass_through(trigger_turn, message):
    shell, turn = trigger_turn
    shell.conversation_history = [{"role": "user", "content": message}]

    CLIChatTurnMixin._chat_run_agent(shell, turn, message)

    kwargs = shell.agent.run_conversation.call_args.kwargs
    assert kwargs["user_message"] == message
    assert kwargs["persist_user_message"] is None
