"""The armed-board transition gate: a release file is the only authority, and one use each."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_transition_guard as ktg

ARMED = "fi-plus"


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    """Throwaway kanban home with an existing (armed) ``fi-plus`` board."""
    home = tmp_path / ".hermes"
    (home / "kanban" / "boards" / ARMED).mkdir(parents=True)
    (home / "kanban" / "boards" / ARMED / "board.json").write_text(
        json.dumps({"slug": ARMED, "name": ARMED}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", ARMED)
    db_path = kb.kanban_db_path(board=ARMED)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board=ARMED)
    yield home
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))


@pytest.fixture()
def armed_config(monkeypatch):
    """Arm exactly the fixture board; never read the operator's real config.yaml."""
    monkeypatch.setattr(ktg, "armed_boards", lambda config=None: (ARMED,))


@pytest.fixture()
def conn(kanban_home):
    with kbc.connect(board=ARMED) as c:
        yield c


def _blocked(conn) -> str:
    """A card sitting in ``blocked`` — the state the guard has to defend."""
    tid = kb.create_task(conn, title="FI+ integration card", created_by="test")
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
    conn.commit()
    return tid


def _release(home, *, task_id, scope="unblock", board=ARMED, request_id="1234567",
             requested_at=None, name="r1.release.json") -> Path:
    inbox = home / "kanban" / "boards" / board / "controls" / "releases"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(json.dumps({
        "task_id": task_id,
        "request_id": request_id,
        "board": board,
        "scope": scope,
        "requested_at": str(int(time.time()) if requested_at is None else requested_at),
        "requested_by": "fi-plus-owner",
    }), encoding="utf-8")
    return path


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]


def test_blocked_card_is_refused_without_a_release(conn, armed_config):
    tid = _blocked(conn)
    with pytest.raises(ktg.BoardTransitionRefused) as excinfo:
        kb.unblock_task(conn, tid)
    assert "no valid release" in str(excinfo.value)
    assert _status(conn, tid) == "blocked"


def test_release_authorizes_exactly_one_unblock(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    release = _release(kanban_home, task_id=tid)
    assert kb.unblock_task(conn, tid) is True
    assert _status(conn, tid) != "blocked"
    assert not release.exists(), "the release must be spent"
    assert list((release.parent / "consumed").iterdir())

    # Re-block, then try the same approval again: it must not carry over.
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
    conn.commit()
    with pytest.raises(ktg.BoardTransitionRefused):
        kb.unblock_task(conn, tid)
    assert _status(conn, tid) == "blocked"


def test_release_is_bound_to_task_board_and_scope(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    _release(kanban_home, task_id="t_00000000")
    _release(kanban_home, task_id=tid, scope="promote", name="r2.release.json")
    _release(kanban_home, task_id=tid, board="other-board", name="r3.release.json")
    with pytest.raises(ktg.BoardTransitionRefused) as excinfo:
        kb.unblock_task(conn, tid)
    message = str(excinfo.value)
    assert "names task" in message
    assert "does not grant unblock" in message


def test_stale_and_abbreviated_request_codes_are_not_authority(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    _release(kanban_home, task_id=tid, request_id="4821")  # the legacy request id shape
    _release(kanban_home, task_id=tid, requested_at=int(time.time()) - ktg.RELEASE_WINDOW_SECONDS - 60,
             name="expired.release.json")
    with pytest.raises(ktg.BoardTransitionRefused) as excinfo:
        kb.unblock_task(conn, tid)
    assert "seven-digit" in str(excinfo.value)
    assert "expired" in str(excinfo.value)


def test_read_only_audit_document_is_never_a_release(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    inbox = kanban_home / "kanban" / "boards" / ARMED / "controls" / "releases"
    inbox.mkdir(parents=True, exist_ok=True)
    # An audit artifact naming the card and an approval, but no owner request code.
    (inbox / "audit.release.json").write_text(json.dumps({
        "task_id": tid, "board": ARMED, "scope": "unblock", "approved": True,
        "requested_by": "audit", "requested_at": str(int(time.time())),
    }), encoding="utf-8")
    with pytest.raises(ktg.BoardTransitionRefused):
        kb.unblock_task(conn, tid)


def test_boards_setting_shapes():
    assert ktg.armed_boards({"kanban": {}}) == ()
    assert ktg.armed_boards({"kanban": {"transition_guard": {"boards": ["fi-plus"]}}}) == ("fi-plus",)
    # A malformed value arms everything rather than silently opening the gate.
    assert ktg.armed_boards({"kanban": {"transition_guard": {"boards": "fi-plus"}}}) == ktg.UNREADABLE
    assert ktg.armed_boards({"kanban": {"transition_guard": {"boards": ["fi-plus", ""]}}}) == ktg.UNREADABLE
    assert ktg.board_is_armed("anything", ktg.UNREADABLE) is True


def test_unarmed_board_keeps_working_without_a_release(conn, monkeypatch):
    monkeypatch.setattr(ktg, "armed_boards", lambda config=None: ())
    tid = _blocked(conn)
    assert kb.unblock_task(conn, tid) is True
    assert _status(conn, tid) != "blocked"


def test_promote_is_gated_too(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    with pytest.raises(ktg.BoardTransitionRefused):
        kb.promote_task(conn, tid, actor="nemo", reason="try again")
    assert _status(conn, tid) == "blocked"
    _release(kanban_home, task_id=tid, scope="promote")
    ok, err = kb.promote_task(conn, tid, actor="nemo", reason="approved")
    assert ok, err
    assert _status(conn, tid) == "ready"


def test_consumed_release_is_recorded_on_the_task(conn, kanban_home, armed_config):
    tid = _blocked(conn)
    _release(kanban_home, task_id=tid)
    kb.unblock_task(conn, tid)
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'transition_released'",
        (tid,)).fetchone()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["request_id"] == "1234567"
    assert payload["board"] == ARMED
