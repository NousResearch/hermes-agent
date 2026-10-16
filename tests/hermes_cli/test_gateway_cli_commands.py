"""Classic CLI commands whose state lives in the gateway session are answered there (tracker #134889
CLI-8, CLI-17, CLI-21): ``/diff session`` (the gateway tools' checkpoint baseline), ``/reload``
(the gateway's ``.env``) and ``/handoff`` (no gateway verb yet: refused, never silent). Driven
through ``HermesCLI.process_command`` with a scripted gateway connection."""
from types import SimpleNamespace

import pytest

from cli import HermesCLI


class _Conn:
    closed = False

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))
        return self.replies.get(method, {})


@pytest.fixture
def gateway_cli(monkeypatch):
    def make(**replies):
        conn = _Conn(replies)
        monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id = "local-s"
        # The gateway-backed agent has no checkpoint manager of its own (GatewayAgent._checkpoint_mgr).
        cli.agent = SimpleNamespace(_checkpoint_mgr=None)
        return cli, conn
    return make


def test_diff_session_renders_the_gateway_sessions_checkpoint_diff(gateway_cli, capsys):
    """The gateway session's checkpoint diff comes back as data and gets main's terminal renderer:
    400 body lines, not the chat handler's 60 lines / 3000 chars (Greptile 4277259989)."""
    body = "\n".join(f"+edited {i}" for i in range(450))
    cli, conn = gateway_cli(**{"rollback.session_diff": {
        "enabled": True, "success": True, "stat": " a.py | 450 +", "diff": body, "diff_lines": 450}})
    assert cli.process_command("/diff session") is True
    assert conn.calls == [("rollback.session_diff", {"session_id": "local-s"})]
    out = capsys.readouterr().out
    assert "a.py | 450 +" in out and "+edited 399" in out and "+edited 400" not in out
    assert "50 more lines" in out and "```" not in out
    cli.process_command("/diff session --stat")
    out = capsys.readouterr().out
    assert "a.py | 450 +" in out and "+edited 0" not in out


def test_diff_session_reports_a_session_without_checkpoints(gateway_cli, capsys):
    cli, _conn = gateway_cli(**{"rollback.session_diff": {"enabled": False}})
    assert cli.process_command("/diff session") is True
    assert "Checkpoints are not enabled" in capsys.readouterr().out


def test_diff_of_the_working_tree_stays_a_local_git_read(gateway_cli, monkeypatch):
    cli, conn = gateway_cli()
    seen = []
    monkeypatch.setattr("tools.working_diff.collect_working_diff",
                        lambda cwd, mode, paths=None: seen.append(mode) or {"success": True, "empty": True})
    cli.process_command("/diff staged")
    assert seen == ["staged"] and conn.calls == []


def test_reload_rereads_env_in_the_gateway_and_reports_its_count(gateway_cli, monkeypatch, capsys):
    cli, conn = gateway_cli(**{"reload.env": {"updated": 3}})
    monkeypatch.setattr("hermes_cli.config.reload_env", lambda: 0)
    assert cli.process_command("/reload") is True
    assert conn.calls == [("reload.env", {"session_id": "local-s"})]
    assert "3" in capsys.readouterr().out


def test_handoff_is_refused_with_a_message_on_the_gateway(gateway_cli, capsys):
    cli, conn = gateway_cli()
    assert cli.process_command("/handoff telegram") is True  # the session stays
    assert "/handoff is not available on the shared gateway yet" in capsys.readouterr().out
    assert conn.calls == []
