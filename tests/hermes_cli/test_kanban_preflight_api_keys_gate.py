"""Tests für den Pre-Flight API-Key Gate in claim_task (kanban_preflight_api_keys).

Abdeckung:
- Gate blockt einen ready-Task bei fehlendem Key (Failure-Path am Gate).
- Gate lässt das Claimen zu, wenn der Key erfüllt ist.
- Gate fragt die tatsächlich geöffnete Board-DB ab, nicht eine aus
  HERMES_HOME neu abgeleitete (Bypass-Verschluss: HERMES_KANBAN_DB-Pinning
  von Profil-Dispatchern muss respektiert werden).
- Fail-open bei defektem Check: Claim läuft weiter, Fehler wird geloggt,
  statt silently zu passieren (kein bypass).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_preflight_api_keys as kpf

GATE_KEY = "TEST_PREFLIGHT_GATE_KEY"


class _StubPassCache:
    """Neutralisiert den pass-Store (kein Subprocess in Tests)."""

    def has_key(self, key_name: str) -> bool:
        return False


@pytest.fixture
def gate_key_missing(monkeypatch):
    """Test-Toolset 'testtool' -> GATE_KEY; Key fehlt überall."""
    monkeypatch.setitem(kpf.BODY_TOOLSET_HINTS, "testtool", ["testtool"])
    monkeypatch.setitem(kpf.TOOLSET_REQUIRED_KEYS, "testtool", [[GATE_KEY]])
    monkeypatch.setattr(kpf, "_PASS_CACHE", _StubPassCache())
    monkeypatch.delenv(GATE_KEY, raising=False)
    return GATE_KEY


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Kein Board-Pinning im Test (außer dort, wo ein Test es bewusst setzt)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    return home


def _make_ready_task(conn, title: str = "t", body: str = "testtool use") -> str:
    tid = kb.create_task(conn, title=title, assignee="worker-a", body=body)
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    conn.commit()
    return tid


def test_gate_blocks_ready_task_when_key_missing(kanban_home, gate_key_missing):
    conn = kbc.connect()
    try:
        tid = _make_ready_task(conn, body="body mentioning testtool")
        claimed = kb.claim_task(conn, tid)
        assert claimed is None
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert row["status"] == "blocked"
        assert row["block_kind"] == "missing-api-keys"
        events = [
            str(r["payload"]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? "
                "AND kind='preflight_block'", (tid,),
            )
        ]
        # Klar lesbarer, profil-spezifischer Grund mit fehlendem Key-Namen
        assert any(
            GATE_KEY in e and "worker-a" in e for e in events
        ), events
    finally:
        conn.close()


def test_gate_passes_when_key_present(kanban_home, gate_key_missing, monkeypatch):
    monkeypatch.setenv(GATE_KEY, "present")
    conn = kbc.connect()
    try:
        tid = _make_ready_task(conn)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "running"
    finally:
        conn.close()


def test_gate_uses_pinned_board_db_not_hermes_home(
    kanban_home, gate_key_missing, monkeypatch
):
    """Bypass-Verschluss: trotz abweichendem HERMES_HOME muss das Gate den
    Task in der gepinnten Board-DB blockieren (HERMES_KANBAN_DB)."""
    other_home = kanban_home.parent / "dispatcher-home"
    other_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    board = kb.kanban_db_path()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board))
    conn = kbc.connect()
    try:
        tid = _make_ready_task(conn)
        claimed = kb.claim_task(conn, tid)
        assert claimed is None
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "blocked"
    finally:
        conn.close()


def test_gate_fail_open_logs_not_silent(
    kanban_home, gate_key_missing, monkeypatch, caplog
):
    """Defekter Check -> Claim läuft weiter (fail-open), aber der Fehler ist
    sichtbar geloggt statt verschluckt."""

    def broken(board_path, task_id):
        raise RuntimeError("preflight check exploded")

    conn = kbc.connect()
    try:
        tid = _make_ready_task(conn)
        # claim_task importiert check_single_task lazy aus diesem Modul —
        # daher am Modul-Attribut patchen.
        monkeypatch.setattr(kpf, "check_single_task", broken)
        with caplog.at_level("ERROR", logger="hermes_cli.kanban_db"):
            claimed = kb.claim_task(conn, tid)
        assert claimed is not None  # fail-open
        assert any("preflight-gate" in r.message for r in caplog.records), \
            caplog.records
    finally:
        conn.close()


def test_check_single_task_missing_row_returns_false(kanban_home, gate_key_missing):
    board = kb.kanban_db_path()
    should_block, reason = kpf.check_single_task(board, "t_does_not_exist")
    assert should_block is False
    assert reason == ""


def test_check_single_task_ignores_non_ready_task(kanban_home, gate_key_missing):
    conn = kbc.connect()
    try:
        # status 'running' (default), kein 'ready' -> kein Gate-Greifen
        tid = kb.create_task(
            conn, title="t", assignee="worker-a", body="testtool use"
        )
        # Parentloser Task wird sofort 'ready' — für diesen Test explizit
        # einen Nicht-ready-Status setzen.
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        conn.commit()
        board = kb.kanban_db_path()
        should_block, reason = kpf.check_single_task(board, tid)
        assert should_block is False
        assert reason == ""
    finally:
        conn.close()


def _make_review_task(conn, title: str = "t", body: str = "testtool use") -> str:
    tid = kb.create_task(conn, title=title, assignee="worker-a", body=body)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
    conn.commit()
    return tid


def test_review_claim_blocked_when_key_missing(kanban_home, gate_key_missing):
    """Bypass-Verschluss: review -> running (claim_review_task) läuft durch
    dasselbe Gate — fehlender Key blockt auch den Reviewer-Claim."""
    conn = kbc.connect()
    try:
        tid = _make_review_task(conn, body="body mentioning testtool")
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is None
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert row["status"] == "blocked"
        assert row["block_kind"] == "missing-api-keys"
        events = [
            str(r["payload"]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? "
                "AND kind='preflight_block'", (tid,),
            )
        ]
        assert any(GATE_KEY in e for e in events)
        assert any("worker-a" in e for e in events)
    finally:
        conn.close()


def test_review_claim_passes_when_key_present(kanban_home, gate_key_missing, monkeypatch):
    monkeypatch.setenv(GATE_KEY, "present")
    conn = kbc.connect()
    try:
        tid = _make_review_task(conn)
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "running"
    finally:
        conn.close()


def test_gate_review_status_ignored_for_ready_default(kanban_home, gate_key_missing):
    """check_single_task ohne status-Arg prüft nur 'ready' — ein Review-Task
    wird vom Scan-Pfad (ready-only) nicht doppelt geblockt."""
    board = kb.kanban_db_path()
    conn = kbc.connect()
    try:
        tid = _make_review_task(conn, body="body mentioning testtool")
    finally:
        conn.close()
    should_block, reason = kpf.check_single_task(board, tid)
    assert should_block is False
    assert reason == ""