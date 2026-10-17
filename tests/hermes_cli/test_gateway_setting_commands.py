"""Per-session settings typed in the classic CLI run the gateway's handler (slash.exec) when the
session is on the gateway; client-enforced state (/busy) is mirrored after the owner accepts."""
from types import SimpleNamespace

from hermes_cli import gateway_setting_commands


class _Conn:
    closed = False

    def __init__(self):
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))
        return {"type": "exec", "output": "Busy input mode: queue"}


def test_busy_runs_on_the_gateway_and_mirrors_the_mode(monkeypatch, capsys):
    conn = _Conn()
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    cli = SimpleNamespace(session_id="local-s", busy_input_mode="interrupt")
    assert gateway_setting_commands.run(cli, "busy", "/busy queue") is True
    assert conn.calls == [("slash.exec", {"session_id": "local-s", "command": "busy queue"})]
    assert cli.busy_input_mode == "queue"
    assert gateway_setting_commands.run(cli, "history", "/history") is False  # a local read stays local


def test_refine_and_review_run_on_the_gateway_sessions_agent(monkeypatch, capsys):
    """/refine and /review (#134889 M3, M4/CLI-6): the CLI's agent is a gateway stand-in, so the
    gateway's handler runs the reviewer beside the session's agent (as messaging does) and the CLI
    prints its reply; no reviewer runs in the CLI process."""
    conn = _Conn()
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)

    def local_review(*_a, **_k):
        raise AssertionError("the reviewer must not run in the CLI process")
    monkeypatch.setattr("agent.review_engine.start_review", local_review)
    cli = SimpleNamespace(session_id="local-s")
    assert gateway_setting_commands.run(cli, "refine", "/refine naming") is True
    assert gateway_setting_commands.run(cli, "review", "/review the diff") is True
    assert conn.calls == [("slash.exec", {"session_id": "local-s", "command": "refine naming"}),
                          ("slash.exec", {"session_id": "local-s", "command": "review the diff"})]
    assert "Busy input mode: queue" in capsys.readouterr().out  # the owner's reply is printed
