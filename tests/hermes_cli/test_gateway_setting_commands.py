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


def test_cron_writes_run_on_the_gateway_not_in_the_cli(monkeypatch, capsys):
    """T1 (#134889): /cron add|pause|remove go to the gateway's handler (one writer to the profile's
    cron store), with no answer deadline (a /cron run lasts as long as its job); nothing writes here."""
    calls = []

    class _UntilConn(_Conn):
        def call_until(self, method, wake, **params):
            calls.append((method, params, wake.is_set()))
            return {"type": "exec", "output": "(^_^)b Paused job: probe (probe)"}
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: _UntilConn())

    def local_write(**_k):
        raise AssertionError("the CLI process must not write the cron store")
    monkeypatch.setattr("tools.cronjob_tools.cronjob", local_write)
    cli = SimpleNamespace(session_id="local-s", _gateway_command_wait=None)
    assert gateway_setting_commands.run(cli, "cron", "/cron pause probe") is True
    assert calls == [("slash.exec", {"session_id": "local-s", "command": "cron pause probe"}, False)]
    assert "Paused job: probe" in capsys.readouterr().out and cli._gateway_command_wait is None


def test_ctrl_c_during_a_cron_wait_prints_the_stopped_line_and_propagates(monkeypatch, capsys):
    """Ctrl+C while /cron waits on the gateway (a /cron run lasts as long as its job): the same stopped line
    as a plugin command, the wait handle cleared, and the interrupt still reaches the CLI."""
    import pytest

    class _InterruptedConn(_Conn):
        def call_until(self, method, wake, **params):
            raise KeyboardInterrupt
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: _InterruptedConn())
    cli = SimpleNamespace(session_id="local-s", _gateway_command_wait=None)
    with pytest.raises(KeyboardInterrupt):
        gateway_setting_commands.run(cli, "cron", "/cron run probe")
    assert "/cron: stopped waiting" in capsys.readouterr().out and cli._gateway_command_wait is None


def test_cron_reads_keep_the_answer_deadline(monkeypatch, capsys):
    """Only a /cron write waits with no deadline (a /cron run lasts as long as its job): bare /cron and
    /cron list answer on the session's connection with its deadline, as /fast does, so a frozen gateway
    gives up instead of hanging until Ctrl+C."""
    class _BoundedConn(_Conn):
        def call_until(self, method, wake, **params):
            raise AssertionError("a /cron read must keep the answer deadline")
    conn = _BoundedConn()
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    cli = SimpleNamespace(session_id="local-s", _gateway_command_wait=None)
    for typed in ("/cron", "/cron list", "/cron LIST -a", '/cron add "every 2h'):
        assert gateway_setting_commands.run(cli, "cron", typed) is True
    assert [params["command"] for _m, params in conn.calls] == [
        "cron", "cron list", "cron LIST -a", 'cron add "every 2h']
