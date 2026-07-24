from types import SimpleNamespace

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
