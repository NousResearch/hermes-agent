"""/rollback on the shared gateway: the checkpoints are the gateway's, so list/diff/restore go to
the owner's ``rollback.*`` verbs and print main's text (the CLI process has no manager)."""
import asyncio
from contextlib import asynccontextmanager

from hermes_cli import gateway_rollback
from hermes_cli.gateway_connection import GatewayConnection


class _Client:
    def __init__(self, answers):
        self.events = asyncio.Queue()
        self.calls = []
        self.answers = answers

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        return self.answers[method]


def _connection(client):
    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    return conn


class _Cli:
    session_id = "local-rb"
    conversation_history = [{"role": "user", "content": "old"}]


def test_rollback_lists_and_restores_through_the_gateway(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    client = _Client({
        "rollback.list": {"enabled": True, "checkpoints": [
            {"hash": "3101637abcdef", "timestamp": "2026-10-13T11:25:00+00:00", "message": "before write_file"}]},
        "rollback.restore": {"success": True, "restored_to": "3101637", "reason": "before write_file",
                             "history_removed": 2},
        "session.resume": {"revision": 3, "execution_generation": 1, "messages": []}})
    conn = _connection(client)
    cli = _Cli()
    try:
        gateway_rollback.rollback(cli, conn, "/rollback")
        out = capsys.readouterr().out
        assert "1. 3101637" in out and "before write_file" in out, out
        gateway_rollback.rollback(cli, conn, "/rollback 1")
        assert ("rollback.restore", {"session_id": "local-rb", "hash": "1"}) in client.calls
        assert cli.conversation_history == []  # the owner's rewound transcript
    finally:
        conn.close()


def test_rollback_all_reaches_the_gateway_and_kept_edits_are_shown(capsys, monkeypatch, tmp_path):
    """The owner's whole-tree restore keeps hand-edits by default: the CLI says which, and its
    ``--all``/``--force`` override travels as ``all`` instead of being dropped."""
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    client = _Client({
        "rollback.restore": {"success": True, "restored_to": "3101637", "reason": "before write_file",
                             "skipped_user_edits": ["notes.md"]}})
    conn = _connection(client)
    cli = _Cli()
    try:
        gateway_rollback.rollback(cli, conn, "/rollback 1")
        assert client.calls[-1] == ("rollback.restore", {"session_id": "local-rb", "hash": "1"})
        out = capsys.readouterr().out
        assert "notes.md" in out and "--all" in out, out
        gateway_rollback.rollback(cli, conn, "/rollback 1 --all")
        assert client.calls[-1] == ("rollback.restore", {"session_id": "local-rb", "hash": "1", "all": True})
        gateway_rollback.rollback(cli, conn, "/rollback 1 a.txt --force")
        assert client.calls[-1] == ("rollback.restore", {"session_id": "local-rb", "hash": "1", "file_path": "a.txt"})
    finally:
        conn.close()
