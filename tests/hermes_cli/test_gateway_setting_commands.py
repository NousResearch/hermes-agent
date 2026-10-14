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
