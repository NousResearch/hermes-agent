"""Classic CLI commands whose state lives in the gateway session are answered there (tracker #134889
CLI-8, CLI-17, CLI-21): ``/diff session`` (the gateway tools' checkpoint baseline), ``/reload``
(the gateway's ``.env``) and ``/handoff`` (the owner's ``handoff.*`` verbs, #143608). Driven
through ``HermesCLI.process_command`` with a scripted gateway connection."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from hermes_agent.cli import HermesCLI


class _Conn:
    closed = final = False

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))
        reply = self.replies.get(method, {})
        if isinstance(reply, Exception):
            raise reply
        return reply


def _observed(name, arg):
    """The ``command.observe`` notice a command the CLI runs itself sends first (``pre_command``)."""
    return ("command.observe", {"session_id": "local-s", "name": name, "arg": arg})


@pytest.fixture
def gateway_cli(monkeypatch):
    def make(**replies):
        conn = _Conn(replies)
        monkeypatch.setattr("hermes_agent.hermes_cli.gateway_side.connection_for", lambda cli: conn)
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
    assert conn.calls == [_observed("diff", "session"), ("rollback.session_diff", {"session_id": "local-s"})]
    out = capsys.readouterr().out
    assert "a.py | 450 +" in out and "+edited 399" in out and "+edited 400" not in out
    assert "50 more lines" in out and "```" not in out
    cli.process_command("/diff session --stat")
    out = capsys.readouterr().out
    assert "a.py | 450 +" in out and "+edited 0" not in out


def test_diff_session_counts_lines_the_gateway_cut_to_fit_its_frame(gateway_cli, capsys):
    """CLI-26: a body the gateway cut below 400 lines to fit one frame names every line not shown."""
    body = "\n".join(f"+edited {i}" for i in range(100))
    cli, _conn = gateway_cli(**{"rollback.session_diff": {
        "enabled": True, "success": True, "stat": " a.py | 5000 +", "diff": body, "diff_lines": 5000}})
    cli.process_command("/diff session")
    out = capsys.readouterr().out
    assert "+edited 99" in out and "4900 more lines" in out
    # One line too long for any frame leaves no body; the cut is still named.
    cli, _conn = gateway_cli(**{"rollback.session_diff": {
        "enabled": True, "success": True, "stat": " a.py | 1 +", "diff": "", "diff_lines": 3}})
    cli.process_command("/diff session")
    assert "3 more lines" in capsys.readouterr().out


def test_diff_session_reports_a_session_without_checkpoints(gateway_cli, capsys):
    cli, _conn = gateway_cli(**{"rollback.session_diff": {"enabled": False}})
    assert cli.process_command("/diff session") is True
    assert "Checkpoints are not enabled" in capsys.readouterr().out


def test_diff_of_the_working_tree_stays_a_local_git_read(gateway_cli, monkeypatch):
    cli, conn = gateway_cli()
    seen = []
    monkeypatch.setattr("hermes_agent.tools.working_diff.collect_working_diff",
                        lambda cwd, mode, paths=None: seen.append(mode) or {"success": True, "empty": True})
    cli.process_command("/diff staged")
    assert seen == ["staged"] and conn.calls == [_observed("diff", "staged")]


def test_reload_rereads_env_in_the_gateway_and_reports_its_count(gateway_cli, monkeypatch, capsys):
    cli, conn = gateway_cli(**{"reload.env": {"updated": 3}})
    monkeypatch.setattr("hermes_agent.hermes_cli.config.reload_env", lambda: 0)
    assert cli.process_command("/reload") is True
    assert conn.calls == [_observed("reload", ""), ("reload.env", {"session_id": "local-s"})]
    assert "3" in capsys.readouterr().out


class _HandoffConn(_Conn):
    """``handoff.state`` answers its scripted states in order (the watcher's progress)."""

    def __init__(self, states, **replies):
        super().__init__(replies)
        self.states = list(states)

    def call(self, method, **params):
        if method == "handoff.state":
            self.calls.append((method, params))
            return {"state": self.states.pop(0), "platform": "telegram", "error": "boom"}
        return super().call(method, **params)


@pytest.fixture
def handoff_cli(monkeypatch):
    def make(states, **replies):
        conn = _HandoffConn(states, **{"handoff.request": {"queued": True, "session_key": "local-s",
                                                          "platform": "telegram", "home_name": "Phone"}, **replies})
        monkeypatch.setattr("hermes_agent.hermes_cli.gateway_side.connection_for", lambda cli: conn)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id, cli._agent_running, cli._should_exit = "local-s", False, False
        cli._pending_agent_seed = None
        cli._app = SimpleNamespace(is_running=True, exited=[], exit=lambda: cli._app.exited.append(True))
        return cli, conn
    return make


def test_handoff_to_telegram_is_queued_on_the_gateway_and_the_cli_exits_once_it_lands(handoff_cli, capsys):
    """#143608: the pending row is written in the owner's store (where its watcher claims it), never
    this process's, and a completed transfer ends the CLI session like /quit."""
    from hermes_agent.cli import _handed_off_session_ids
    cli, conn = handoff_cli(["pending", "running", "completed"])
    # The TUI's slash dispatch: a session-ending answer closes the running app, as /quit does.
    assert cli._tui_run_slash_input("/handoff Telegram") is None
    assert cli._app.exited == [True]
    assert ("handoff.request", {"session_id": "local-s", "platform": "telegram"}) in conn.calls
    assert not [call for call in conn.calls if call[0] == "handoff.fail"]
    assert cli._should_exit is True and "local-s" in _handed_off_session_ids
    out = capsys.readouterr().out
    assert "Phone" in out and "Handoff complete" in out
    _handed_off_session_ids.discard("local-s")


def test_handoff_failure_keeps_the_cli_session(handoff_cli, capsys):
    cli, _conn = handoff_cli(["running", "failed"])
    assert cli.process_command("/handoff telegram") is True
    assert cli._should_exit is False and cli._app.exited == []
    assert "Handoff failed: boom" in capsys.readouterr().out


def test_handoff_refusal_from_the_gateway_is_shown_and_nothing_waits(handoff_cli, capsys):
    from hermes_agent.hermes_cli.gateway_client import GatewayClientError
    cli, conn = handoff_cli([], **{"handoff.request": GatewayClientError("no home channel configured for telegram")})
    assert cli.process_command("/handoff telegram") is True
    assert "no home channel configured for telegram" in capsys.readouterr().out
    assert not [call for call in conn.calls if call[0] == "handoff.state"]


def test_a_handoff_refusal_names_what_to_fix_not_its_code(handoff_cli, capsys):
    """The owner's refusal code (``invalid_params``) is printed with its detail, the actionable part."""
    from hermes_agent.hermes_cli.gateway_client import GatewayRPCError
    detail = "no home channel configured for telegram; set one with /sethome on the destination chat first"
    cli, _conn = handoff_cli([], **{"handoff.request": GatewayRPCError(
        "invalid_params", code=4001, data={"reason": "invalid_params", "field": "platform", "detail": detail})})
    assert cli.process_command("/handoff telegram") is True
    assert f"/handoff: {detail}" in capsys.readouterr().out


def _save_cli(conn, tmp_path, monkeypatch):
    """A gateway-backed CLI with one said line in its mirror, run from ``tmp_path/work``."""
    from hermes_agent.hermes_cli.gateway_session_store import GatewaySessionStore
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id, cli.conversation_history = "local-s", [{"role": "user", "content": "in-memory only"}]
    cli.model, cli.session_start = "fake-model", datetime(2026, 10, 18)
    cli._session_db = GatewaySessionStore(conn)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    return cli


def test_save_writes_the_gateway_owners_export_on_this_machine(tmp_path, monkeypatch, capsys):
    """CLI-19: ``/save`` asks the owner (``session.export``) for its canonical export, never writes the
    in-memory fallback (no title, no metadata, no archived rows) that ``import_sessions`` can't restore.
    The CLI writes the content itself, so a remote owner's disk is never taken for this one."""
    from hermes_agent.hermes_constants import get_hermes_home
    conn = _Conn({"session.export": {"format": "md", "filename": "hermes_session_s.md", "content": "# owner export"}})
    cli = _save_cli(conn, tmp_path, monkeypatch)
    work = tmp_path / "work"
    cli.save_conversation("/save markdown notes.md redact")
    assert conn.calls == [("session.export", {"session_id": "local-s", "format": "md", "redact": True})]
    assert (work / "notes.md").read_text(encoding="utf-8") == "# owner export"
    assert str(work / "notes.md") in capsys.readouterr().out
    cli.save_conversation("/save md")  # no name: main's local folder and name, under this CLI's home
    [default] = (get_hermes_home() / "sessions" / "saved").glob("hermes_conversation_*.md")
    assert default.read_text(encoding="utf-8") == "# owner export" and str(default) in capsys.readouterr().out
    conn.replies["session.export"] = {}  # an owner answer without content is reported, nothing written
    cli.save_conversation("/save json x.json")
    assert "Failed to save" in capsys.readouterr().out
    cli.conversation_history = []  # nothing said yet: nothing asked of the owner, nothing written
    cli.save_conversation("/save json")
    assert "No conversation to save" in capsys.readouterr().out and len(conn.calls) == 3
    assert [p.name for p in work.iterdir()] == ["notes.md"]


def test_save_on_an_older_gateway_falls_back_only_for_a_plain_json_save(tmp_path, monkeypatch, capsys):
    """An owner without ``session.export`` (-32601): its ``session.save`` writes default-named, unredacted
    JSON on the gateway host, so only ``/save json`` uses it, reported as on the gateway host; a
    format, name or ``redact`` is refused with a message, never silently dropped."""
    from hermes_agent.hermes_cli.gateway_client import GatewayRPCError
    conn = _Conn({"session.export": GatewayRPCError("unknown_method", code=-32601),
                  "session.save": {"file": "/srv/gw/sessions/saved/hermes_conversation_x.json"}})
    cli = _save_cli(conn, tmp_path, monkeypatch)
    for cmd in ("/save md", "/save json mine.json", "/save json redact"):
        cli.save_conversation(cmd)
        assert "predates this CLI" in capsys.readouterr().out, cmd
    assert [m for m, _ in conn.calls] == ["session.export"] * 3
    cli.save_conversation("/save json")
    assert conn.calls[-1] == ("session.save", {"session_id": "local-s"})
    assert "saved on the gateway host: /srv/gw/sessions/saved/hermes_conversation_x.json" in capsys.readouterr().out
    assert not any((tmp_path / "work").iterdir())


def test_an_unclaimed_handoff_is_failed_on_the_gateway_so_it_can_be_retried(handoff_cli, monkeypatch, capsys):
    cli, conn = handoff_cli(["pending"] * 3, **{"handoff.fail": {"failed": True, "state": "failed"}})
    clock = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr("time.time", lambda: next(clock, 100.0))
    assert cli.process_command("/handoff telegram") is True
    assert ("handoff.fail", {"session_id": "local-s", "error": "timed out waiting for gateway"}) in conn.calls
    assert "Timed out waiting for the gateway" in capsys.readouterr().out


def test_a_handoff_claimed_as_the_pending_timeout_fires_still_completes(handoff_cli, monkeypatch):
    """The CAS-fail loses to a claim that finishes first: the CLI reports what the row says, not a timeout."""
    from hermes_agent.cli import _handed_off_session_ids
    cli, conn = handoff_cli(["pending", "completed", "completed"],
                            **{"handoff.fail": {"failed": False, "state": "completed"}})
    clock = iter([0.0, 100.0])
    monkeypatch.setattr("time.time", lambda: next(clock, 100.0))
    assert cli.process_command("/handoff telegram") is False  # the session ended, as /quit
    assert ("handoff.fail", {"session_id": "local-s", "error": "timed out waiting for gateway"}) in conn.calls
    assert cli._should_exit is True and "local-s" in _handed_off_session_ids
    _handed_off_session_ids.discard("local-s")


def test_a_handoff_claimed_at_the_timeout_then_unreadable_still_ends(handoff_cli, monkeypatch, capsys):
    """The read after the pending timeout's CAS-fail sees ``running``, then the gateway goes away: the
    wait still ends (an unreadable row is the timeout), never polls forever."""
    from hermes_agent.hermes_cli.gateway_client import GatewayClientError
    cli, conn = handoff_cli(["pending", "running"], **{"handoff.fail": {"failed": False, "state": "running"}})
    clock = iter([0.0, 100.0])
    monkeypatch.setattr("time.time", lambda: next(clock, 100.0))
    sleeps = []

    def sleep(_s):
        sleeps.append(_s)
        if len(sleeps) > 50:
            pytest.fail("the handoff wait never ended")
    monkeypatch.setattr("time.sleep", sleep)
    scripted = conn.call

    def flaky(method, **params):
        if method == "handoff.state" and not conn.states:
            raise GatewayClientError("gateway went away")
        return scripted(method, **params)
    conn.call = flaky
    assert cli.process_command("/handoff telegram") is True
    out = capsys.readouterr().out
    assert "picked it up" in out and "Timed out waiting for the gateway" in out
