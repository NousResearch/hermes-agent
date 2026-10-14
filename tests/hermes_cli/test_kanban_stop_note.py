"""Stop notes cancel a live worker; created-blocked cards need --force to unblock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_stop import (
    looks_like_close_note,
    looks_like_ship_command,
    looks_like_stop_note,
    stop_note_holds,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running(conn, monkeypatch, title="live"):
    tid = kb.create_task(conn, title=title, assignee="a")
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:worker")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot:1|777")
    kbd._set_worker_pid(conn, tid, 54321)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    return tid


def test_looks_like_stop_note_matches_operator_phrases():
    assert looks_like_stop_note("HARD STOP")
    assert looks_like_stop_note("HARD STOP — Laura said no")
    assert looks_like_stop_note("CLOSE")
    assert looks_like_stop_note("stop work here")
    assert looks_like_stop_note("do not merge")
    assert looks_like_stop_note("do not deploy")
    assert looks_like_stop_note("Please stop the run now")
    assert not looks_like_stop_note("use the 2026 schema")
    assert not looks_like_stop_note("closed the file and kept going")
    assert not looks_like_stop_note("")
    assert not looks_like_stop_note(None)
    assert looks_like_close_note("CLOSE")
    assert looks_like_close_note("CLOSE — Laura 10:58, do NOT merge")
    assert not looks_like_close_note("HARD STOP. Do not merge.")
    assert not looks_like_close_note("Please CLOSE the dropdown")


def test_looks_like_ship_command_is_narrow():
    assert looks_like_ship_command("gh pr merge 12")
    assert looks_like_ship_command("gh merge 12")
    assert looks_like_ship_command("./lab deploy backend")
    assert looks_like_ship_command("lab deploy frontend-leads")
    assert looks_like_ship_command("git push origin main")
    assert looks_like_ship_command("git push origin HEAD:main")
    assert looks_like_ship_command("echo x && gh pr merge --auto 9")
    assert not looks_like_ship_command("git push origin feat/stop-note")
    assert not looks_like_ship_command("git push origin feat/main-page")
    assert not looks_like_ship_command("git push")
    assert not looks_like_ship_command("gh pr create --fill")
    assert not looks_like_ship_command("echo do not merge")


def test_stop_comment_on_running_task_terminates_worker(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _running(conn, monkeypatch)
        signalled = []
        kb.add_comment(
            conn, tid, "ops", "HARD STOP. Do not merge.",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        )

        assert signalled and signalled[0][0] == 54321
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.status != "ready"
        bodies = [c.body for c in kb.list_comments(conn, tid)]
        assert "HARD STOP. Do not merge." in bodies
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'stop_worker_termination'",
            (tid,),
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["prev_pid"] == 54321
        assert payload["terminated"] is True
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run["outcome"] == "cancelled"


def test_close_comment_archives_running_task(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _running(conn, monkeypatch)
        signalled = []
        kb.add_comment(
            conn, tid, "ops", "CLOSE — stop the run, do NOT merge",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        )
        assert signalled and signalled[0][0] == 54321
        assert kb.get_task(conn, tid).status == "archived"
        bodies = [c.body for c in kb.list_comments(conn, tid)]
        assert any(b.startswith("CLOSE") for b in bodies)


def test_ordinary_comment_does_not_kill_or_block(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _running(conn, monkeypatch)
        signalled = []
        kb.add_comment(
            conn, tid, "ops", "use the 2026 schema",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        )
        assert signalled == []
        assert kb.get_task(conn, tid).status == "running"


def test_claim_refuses_while_stop_note_holds_even_if_status_is_ready(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _running(conn, monkeypatch)
        kb.add_comment(conn, tid, "ops", "HARD STOP")

        # Bypass unblock_task the way a raw status write would.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        assert stop_note_holds(conn, tid) is True
        assert kb.claim_task(conn, tid) is None
        rejected = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'claim_rejected' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert json.loads(rejected["payload"])["reason"] == "stop_note"

        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.unblock_task(conn, tid) is False
        assert kb.unblock_task(conn, tid, force=True) is True
        assert stop_note_holds(conn, tid) is False
        assert kb.claim_task(conn, tid) is not None


def test_bare_unblock_refuses_while_stop_note_holds(kanban_home, monkeypatch):
    """CB3 10:14: empty unblock after 'Stop work here' must not return the card to ready."""
    with kbc.connect() as conn:
        tid = _running(conn, monkeypatch)
        kb.add_comment(conn, tid, "ops", "Stop work here.")
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid) is False
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid) is None


def test_initial_status_blocked_refuses_bare_unblock(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", assignee="a", initial_status="blocked",
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid) is False
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid, force=True) is True
        assert kb.get_task(conn, tid).status == "ready"


def test_worker_block_still_unblocks_without_force(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="live", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:worker")
        assert kb.block_task(conn, tid, reason="need a human") is True
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"


def test_stop_comment_on_ready_task_parks_without_signalling(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="queued", assignee="a")
        signalled = []
        kb.add_comment(
            conn, tid, "ops", "HARD STOP",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        )
        assert signalled == []
        assert kb.get_task(conn, tid).status == "blocked"
        row = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'stop_worker_termination'",
            (tid,),
        ).fetchone()
        assert row is None
