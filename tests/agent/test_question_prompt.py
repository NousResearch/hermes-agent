from agent.question_prompt import build_question_prompt


def test_question_command_is_registered_for_cli_and_gateway():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("/question")

    assert cmd is not None
    assert cmd.name == "question"
    assert cmd.args_hint == "[topic or request]"
    assert "question" in GATEWAY_KNOWN_COMMANDS


def test_question_prompt_uses_clarify_until_actionable():
    prompt = build_question_prompt("build me a dashboard")

    assert "You are in /question mode" in prompt
    assert "build me a dashboard" in prompt
    assert "Use the clarify tool" in prompt
    assert "one focused question at a time" in prompt
    assert "Continue until" in prompt
    assert "Do not do the actual task yet" in prompt


def test_question_prompt_without_topic_asks_for_topic_first():
    prompt = build_question_prompt()

    assert "without an initial topic" in prompt
    assert "First ask what they want clarified" in prompt


def test_cli_question_command_sets_pending_agent_seed(capsys):
    from types import SimpleNamespace

    from cli import HermesCLI
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = SimpleNamespace(
        _pending_agent_seed=None,
        _pending_resume_sessions=None,
    )
    cli.process_command = HermesCLI.process_command.__get__(cli, type(cli))
    cli._handle_question_command = CLICommandsMixin._handle_question_command.__get__(
        cli,
        type(cli),
    )

    assert cli.process_command("/question choose a database") is True
    assert cli._pending_agent_seed is not None
    assert "choose a database" in cli._pending_agent_seed
    assert "Use the clarify tool" in cli._pending_agent_seed
    assert "Question mode started" in capsys.readouterr().out
