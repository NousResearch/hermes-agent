"""``/rollback`` quotes the prompt a checkpoint was taken for, on every route a list takes.

* the gateway's ``rollback.list`` (what the classic CLI and the TUI read) carries ``prompt_preview`` for a
  linked checkpoint whose prompt is stored, and keeps the row exactly as it was for an unlinked one;
* the classic CLI, which has no manager and may not share the gateway's database, prints the quote the
  payload carries instead of looking for one.
"""
import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from hermes_cli import gateway_rollback
from hermes_cli.gateway_connection import GatewayConnection
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner, settled
from tests.gateway.test_session_tui_ops import _git_rpc


def _take(home, cwd, reason, session_id=None, prompt_uid=None):
    """A checkpoint in the owner profile's store, linked to a prompt when ``prompt_uid`` is given."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager

    token = set_hermes_home_override(str(home))
    try:
        mgr = CheckpointManager(enabled=True)
        mgr.new_turn(session_id=session_id, prompt_uid=prompt_uid)
        assert mgr.ensure_checkpoint(str(cwd), reason)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("linux")
def test_gateway_rollback_list_carries_the_prompt_preview_only_for_a_linked_checkpoint(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("original\n")

        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, "session.create", request_id="rb", source="tui", toolsets=[],
                                 cwd=str(work), checkpoints=True))["result"]["session_id"]
                await rpc(ws, "prompt.submit", session_id=sid, input_id="first", text="FIRST PROMPT")
                await settled(home)
                db = sqlite3.connect(f"file:{Path(home) / 'state.db'}?mode=ro", uri=True)
                try:
                    db_session, uid = db.execute(
                        "SELECT session_id, message_uid FROM messages WHERE role = 'user' AND content = 'FIRST PROMPT'"
                    ).fetchone()
                finally:
                    db.close()
                await asyncio.to_thread(_take, home, work, "before terminal: rm x")        # unlinked
                (work / "a.txt").write_text("edited\n")
                await asyncio.to_thread(_take, home, work, "before write_file", db_session, uid)  # linked
                listed = (await _git_rpc(ws, "rollback.list", session_id=sid))["result"]
                return listed["checkpoints"]

        newest, older = asyncio.run(probe())

    assert newest["message"] == "before write_file" and newest["prompt_preview"] == "FIRST PROMPT"
    assert older["message"] == "before terminal: rm x" and "prompt_preview" not in older  # row shape unchanged
    assert set(older) == {"hash", "timestamp", "message"}


class _Client:
    def __init__(self, answers):
        self.events = asyncio.Queue()
        self.answers = answers

    async def rpc(self, method, *, _timeout=None, **params):
        return self.answers[method]


def test_the_cli_prints_the_quote_the_gateway_sent_without_a_database_of_its_own(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    assert not (Path(os.environ["HERMES_HOME"]) / "state.db").exists()  # nothing here to read a quote from
    client = _Client({"rollback.list": {"enabled": True, "checkpoints": [
        {"hash": "aaaaaaa1", "timestamp": "2026-11-04T10:01:00+00:00", "message": "before write_file",
         "prompt_preview": "fix the login bug"},
        {"hash": "bbbbbbb2", "timestamp": "2026-11-04T10:00:00+00:00", "message": "before terminal: rm x"}]}})

    @asynccontextmanager
    async def connect():
        yield client

    conn = GatewayConnection(connect=connect)
    conn.start(5)
    try:
        class Cli:
            session_id = "local-rb"
        gateway_rollback.rollback(Cli(), conn, "/rollback")
    finally:
        conn.close()
    lines = capsys.readouterr().out.splitlines()

    quoted = [i for i, line in enumerate(lines) if "prompt:" in line]
    assert len(quoted) == 1 and "fix the login bug" in lines[quoted[0]]
    assert "before write_file" in lines[quoted[0] - 1]


@pytest.mark.asyncio
async def test_the_messaging_gateway_quotes_the_prompt_in_its_rollback_reply(tmp_path, monkeypatch):
    """A chat platform's ``/rollback`` goes through the same formatter, reading the quote from this
    process's own session database."""
    from gateway import run as gateway_run
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from tools.checkpoint_manager import CheckpointManager

    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("v0\n")
    monkeypatch.setenv("TERMINAL_CWD", str(work))

    state_db = Path(os.environ["HERMES_HOME"]) / "state.db"
    db = sqlite3.connect(state_db)
    db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
               "message_uid TEXT, active INTEGER DEFAULT 1)")
    db.execute("INSERT INTO messages VALUES (1, 'chat-session', 'user', 'fix the login bug', 'uid-1', 1)")
    db.commit()
    db.close()

    manager = CheckpointManager(enabled=True)
    manager.new_turn(session_id="chat-session", prompt_uid="uid-1")
    assert manager.ensure_checkpoint(str(work), "before write_file")
    (work / "a.txt").write_text("v1\n")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), "before terminal: rm x")  # a checkpoint with no link

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._checkpoint_manager = lambda source=None: manager
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c", user_name="t", chat_type="dm")

    reply = await runner._handle_rollback_command(MessageEvent(text="/rollback", source=source))

    lines = reply.splitlines()
    quoted = [i for i, line in enumerate(lines) if "prompt:" in line]
    assert len(quoted) == 1 and "fix the login bug" in lines[quoted[0]]
    assert "before write_file" in lines[quoted[0] - 1]
