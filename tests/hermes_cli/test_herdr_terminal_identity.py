"""The terminal, not its shared gateway, owns Herdr conversation identity."""
import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import terminal_breadcrumbs as tb


@pytest.fixture
def herdr(monkeypatch):
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_PANE_ID", "w2:pE")
    monkeypatch.setenv("HERDR_BIN_PATH", "/Applications/Herdr/bin/herdr")
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("HERMES_TUI_SIDECAR_URL", raising=False)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    return run


def test_reports_exact_identity_from_each_frontend_environment(herdr, monkeypatch):
    monkeypatch.setattr(tb.time, "time_ns", lambda: 100)
    monkeypatch.setattr(tb, "_herdr_last_seq", 0)
    tb.report_agent_session("durable-one")
    monkeypatch.setenv("HERDR_PANE_ID", "w2:pF")
    tb.report_agent_session("durable-two")
    assert herdr.call_count == 2
    for call, pane, sid, seq in zip(herdr.call_args_list, ("w2:pE", "w2:pF"),
                                    ("durable-one", "durable-two"), ("100", "101")):
        assert call.args[0] == ["/Applications/Herdr/bin/herdr", "pane", "report-agent-session", pane,
                                "--source", "herdr:hermes", "--agent", "hermes", "--seq", seq,
                                "--agent-session-id", sid, "--session-start-source", "resume"]
        assert call.kwargs == {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                               "stderr": subprocess.DEVNULL, "timeout": 1, "check": False}


@pytest.mark.parametrize("variable,value", [("HERDR_ENV", "0"), ("HERDR_PANE_ID", ""),
    ("HERMES_TUI_GATEWAY_URL", "wss://remote.example/ws"),
    ("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:7777/ws"),
    ("HERMES_TUI_SIDECAR_URL", "https://remote.example"),
    ("HERMES_TUI_SIDECAR_URL", "http://127.0.0.1:7777")])
def test_does_not_bind_without_local_herdr_owner(herdr, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    tb.report_agent_session("durable")
    herdr.assert_not_called()


def test_empty_identity_is_not_reported(herdr):
    tb.report_agent_session("")
    herdr.assert_not_called()


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("herdr", 1)])
def test_missing_or_slow_binary_is_silent_best_effort(herdr, failure, capsys):
    herdr.side_effect = failure
    tb.report_agent_session("durable")
    assert herdr.call_args.kwargs["timeout"] == 1
    assert capsys.readouterr() == ("", "")


def test_classic_binding_reports_when_breadcrumbs_disabled(herdr, monkeypatch):
    from hermes_cli.cli_session_mixin import CLISessionMixin
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    cli = SimpleNamespace(session_id="durable", _session_db=None)
    CLISessionMixin._write_terminal_breadcrumb(cli)
    assert herdr.call_args.args[0][-3] == "durable"


@pytest.mark.parametrize("reason", ["resume", "branch"])
def test_classic_switch_reports_even_without_agent(herdr, monkeypatch, reason):
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.cli_session_switch import _sync_agent_to_session
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    cli = SimpleNamespace(agent=None, session_id="target", _session_db=None)
    cli._write_terminal_breadcrumb = lambda: CLISessionMixin._write_terminal_breadcrumb(cli)
    _sync_agent_to_session(cli, "target", parent_session_id="old", reason=reason)
    assert herdr.call_args.args[0][-3] == "target"


def test_gateway_adoption_reports_new_resume_and_branch_with_breadcrumbs_disabled(herdr, monkeypatch):
    from hermes_cli.gateway_chat_view import GatewayChatView
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    view = GatewayChatView(SimpleNamespace(), {"stored_session_id": "initial"})
    for sid in ("new", "resumed", "branch"):
        asyncio.run(view.adopt({"stored_session_id": sid, "execution_generation": 0}))
        assert herdr.call_args.args[0][-3] == sid


def test_finite_gateway_adoption_still_reports(herdr):
    from hermes_cli.gateway_chat_view import GatewayChatView
    view = GatewayChatView(SimpleNamespace(), {"stored_session_id": "initial"})
    view.finite = True
    asyncio.run(view.adopt({"stored_session_id": "new-durable", "execution_generation": 0}))
    assert herdr.call_args.args[0][-3] == "new-durable"


def test_classic_provisional_gateway_id_is_not_reported(herdr, monkeypatch):
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.gateway_session_store import GatewaySessionStore
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    cli = SimpleNamespace(session_id="provisional", _session_db=Mock(spec=GatewaySessionStore),
                          _gateway_session_open=False)
    CLISessionMixin._write_terminal_breadcrumb(cli)
    herdr.assert_not_called()
    cli._gateway_session_open = True
    cli.session_id = "owner-durable"
    CLISessionMixin._write_terminal_breadcrumb(cli)
    assert herdr.call_args.args[0][-3] == "owner-durable"


def test_single_query_compression_reports_changed_durable_identity(herdr, monkeypatch):
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.cli_single_query import _sync_cli_session_id_from_agent
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    cli = SimpleNamespace(session_id="old", agent=SimpleNamespace(session_id="compressed"),
                          _session_db=None)
    cli._write_terminal_breadcrumb = lambda: CLISessionMixin._write_terminal_breadcrumb(cli)
    _sync_cli_session_id_from_agent(cli)
    assert herdr.call_args.args[0][-3] == "compressed"


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [None, "requested-id"])
async def test_gateway_initial_create_and_resume_report_owner_id(herdr, monkeypatch, resume):
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": ["cwd"]}}
            return {"stored_session_id": "owner-durable"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(gateway_chat, "check_resume_policy", lambda *args: None)
    monkeypatch.setattr(gateway_chat, "env_launch_flags", dict)
    monkeypatch.setattr(gateway_chat, "client_env_policy", lambda parameters: {})
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.setattr(tb, "is_enabled", lambda: False)
    args = SimpleNamespace(query="hello", quiet=True, resume=resume)
    assert await gateway_chat.run_gateway_chat(args) == 0
    assert ("session.resume" if resume else "session.create") in calls
    assert herdr.call_args.args[0][-3] == "owner-durable"
