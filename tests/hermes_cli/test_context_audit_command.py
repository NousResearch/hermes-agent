from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from agent.context_audit import collect_context_audit


def _make_cli(agent):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = agent
    cli._pending_resume_sessions = None
    cli._console_output = []
    cli._console_print = lambda message, **_kwargs: cli._console_output.append(message)
    return cli


def test_cli_context_audit_command_renders_redacted_report():
    report = collect_context_audit(
        SimpleNamespace(tools=[], model="gpt-test", provider="test"),
        prompt_parts={"stable": "RAW SOUL SHOULD NOT LEAK", "context": "", "volatile": "SECRET MEMORY BODY"},
    )
    agent = SimpleNamespace(_context_audit_report=report, _context_audit_report_path="/tmp/audit.json")
    cli = _make_cli(agent)

    assert cli.process_command("/context-audit") is True

    output = "\n".join(cli._console_output)
    assert "Context audit" in output
    assert "Debug file: /tmp/audit.json" in output
    assert "RAW SOUL SHOULD NOT LEAK" not in output
    assert "SECRET MEMORY BODY" not in output


def test_cli_context_audit_command_reports_unavailable():
    cli = _make_cli(SimpleNamespace(_context_audit_report=None))

    assert cli.process_command("/context_audit") is True

    output = "\n".join(cli._console_output)
    assert "Context audit is not available" in output
    assert "agent.startup_context_audit" in output


def test_cli_resume_then_context_audit_does_not_show_previous_session(tmp_path):
    from run_agent import AIAgent

    agent = MagicMock()
    agent._memory_store = None
    agent._startup_context_audit_mode = "debug_file"
    agent._context_audit_report = collect_context_audit(
        SimpleNamespace(tools=[], model="gpt-test", provider="test"),
        prompt_parts={"stable": "A" * 1000},
    )
    agent._context_audit_report_path = str(tmp_path / "session_a.json")
    agent._invalidate_system_prompt = MethodType(AIAgent._invalidate_system_prompt, agent)
    cli = _make_cli(agent)
    cli.session_id = "session_a"
    cli._resumed = False
    cli._pending_title = None
    cli.conversation_history = []
    cli.resume_display = "minimal"
    cli._session_db = MagicMock()
    cli._session_db.get_session.return_value = {"id": "session_b", "title": "B"}
    cli._session_db.resolve_resume_session_id.return_value = "session_b"
    history = [{"role": "user", "content": "hi"}]
    cli._session_db.get_resume_conversations.return_value = (history, history)

    with (
        patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="session_b"),
        patch("cli._cprint"),
    ):
        assert cli.process_command("/resume session_b") is True
    assert cli.session_id == "session_b"
    assert cli.process_command("/context-audit") is True

    output = "\n".join(cli._console_output)
    assert "1,000 chars" not in output
    assert "session_a.json" not in output
    assert "after the next message" in output
