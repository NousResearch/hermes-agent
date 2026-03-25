from io import StringIO

from rich.console import Console

from hermes_cli.admission import do_audit, do_list


def test_do_list_renders_admission_records(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.security.admission import quarantine_mcp_server

    quarantine_mcp_server(
        "demo",
        {"url": "https://example.com/mcp"},
        [("search", "Search docs")],
        hermes_home=tmp_path,
    )

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(console=console)
    output = sink.getvalue()

    assert "Admission Records" in output
    assert "demo" in output
    assert "Revision" in output


def test_do_audit_runs_mcp_and_skill_audits(monkeypatch):
    import hermes_cli.admission as admission_cli

    calls = []

    monkeypatch.setattr("hermes_cli.mcp_config.audit_mcp_integrity", lambda: ["mcp drift"])
    monkeypatch.setattr("hermes_cli.skills_hub.do_audit", lambda console=None: calls.append("skills"))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_audit(console=console)
    output = sink.getvalue()

    assert "Admission Audit" in output
    assert "mcp drift" in output
    assert calls == ["skills"]
