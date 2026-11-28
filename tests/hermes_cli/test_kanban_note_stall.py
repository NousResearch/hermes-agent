"""ADR 0020 §2.2/§3.2/§3.3/§3.6 — note-stall watcher unit tests (build card t_68d5b5a4).

Time is a fixture, not a sleep: every test steps one fake clock that the kernel writers
(``claim_task``/``heartbeat_worker``/``add_comment``) and the watcher both read, so "40 seconds
of work, then a note, then 55 more seconds" is reproducible in milliseconds and no test ever
waits on a window.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_note_stall as nsw


class FakeClock:
    """One wall clock for a whole scenario (kernel writes and watcher verdicts agree)."""

    def __init__(self, start: int) -> None:
        self.t = int(start)

    def now(self) -> int:
        return self.t

    def advance(self, seconds: int) -> int:
        self.t += int(seconds)
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock(1_800_000_000)
    monkeypatch.setattr(time, "time", lambda: float(c.t))
    return c


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Throwaway HERMES_HOME with an initialized board DB (repo isolation pattern)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _stub_config(monkeypatch, kanban_cfg: dict) -> None:
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {"kanban": kanban_cfg})


def _spawn(conn, *, title="long job", assignee="worker", window=60):
    """ready -> running + technical (note-less) heartbeat + the replay's short window."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    assert kbd.heartbeat_worker(conn, tid) is True
    nsw.set_note_stall_window(conn, tid, window)
    return tid


def _started_at(conn, tid) -> int:
    task = kb.get_task(conn, tid)
    assert task is not None and task.started_at
    return int(task.started_at)


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


# --- AC 1: heartbeats, no note, window elapses ⇒ exactly one stalled_suspect at ~window ---

def test_stalled_suspect_fires_exactly_once_when_the_window_elapses(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        t0 = _started_at(conn, tid)
        assert t0 == clock.now()

        clock.advance(54)  # 10 % short of the window: not yet (§5.3's ±10 % budget)
        assert nsw.check_running_notes(conn) == []
        assert _events(conn, tid, "stalled_suspect") == []

        clock.advance(6)
        verdicts = nsw.check_running_notes(conn, board="default")
        assert [v["task_id"] for v in verdicts] == [tid]
        assert verdicts[0]["window_s"] == 60 and verdicts[0]["silence_s"] == 60
        events = _events(conn, tid, "stalled_suspect")
        assert len(events) == 1
        payload = events[0].payload or {}
        assert payload["window_s"] == 60 and payload["window_epoch"] == 1
        assert payload["last_note_at"] is None and payload["evaluated_at"] == t0 + 60
        assert payload["dedup_key"] == f"stalled:{tid}:{events[0].run_id}:1"

        # Repeated ticks inside the same window never double-alert.
        clock.advance(1)
        assert nsw.check_running_notes(conn) == []
        clock.advance(29)
        assert nsw.check_running_notes(conn) == []
        assert len(_events(conn, tid, "stalled_suspect")) == 1
    finally:
        conn.close()


def test_ping_repeats_once_per_window_for_a_long_silent_run(kanban_home, clock):
    """SPECIMEN-14 shape (75 min of note-less heartbeats) with a 60 s window."""
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(60)
        kbd.heartbeat_worker(conn, tid)
        nsw.check_running_notes(conn)
        clock.advance(65)
        kbd.heartbeat_worker(conn, tid)
        nsw.check_running_notes(conn)
        clock.advance(10)  # inside the second window's ping shadow
        assert nsw.check_running_notes(conn) == []
        events = _events(conn, tid, "stalled_suspect")
        assert [(e.payload or {}).get("window_epoch") for e in events] == [1, 2]
    finally:
        conn.close()


# --- AC 2: a worker posting normal milestone notes is never flagged ---

@pytest.mark.parametrize("prefix", list(nsw.NOTE_PREFIXES))
def test_milestone_note_prefixes_reset_the_window(kanban_home, clock, prefix):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(40)
        kb.add_comment(conn, tid, "worker", f"{prefix} still working — step 2 of 3\nNEXT: the report")
        kbd.heartbeat_worker(conn, tid)
        clock.advance(55)  # 95 s into the run: the ORIGINAL window expired at 60...
        assert nsw.check_running_notes(conn) == []  # ...the note bought a fresh window
        assert _events(conn, tid, "stalled_suspect") == []
        clock.advance(10)  # 65 s of silence measured from the note
        assert len(nsw.check_running_notes(conn)) == 1
    finally:
        conn.close()


def test_plain_chatter_is_not_a_milestone_note(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(40)
        kb.add_comment(conn, tid, "worker", "just thinking about the parser")
        kbd.heartbeat_worker(conn, tid)
        clock.advance(21)
        assert len(nsw.check_running_notes(conn)) == 1
        assert len(_events(conn, tid, "stalled_suspect")) == 1
    finally:
        conn.close()


def test_noted_heartbeat_counts_as_progress(kanban_home, clock):
    """``kanban_heartbeat(note=...)`` is the documented milestone channel — it must silence us."""
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(40)
        kbd.heartbeat_worker(conn, tid, note="finished the extraction, writing tests next")
        clock.advance(55)
        assert nsw.check_running_notes(conn) == []
        assert _events(conn, tid, "stalled_suspect") == []
    finally:
        conn.close()


def test_note_after_a_ping_defers_the_next_one(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(60)
        assert len(nsw.check_running_notes(conn)) == 1
        assert len(_events(conn, tid, "stalled_suspect")) == 1

        clock.advance(10)
        kb.add_comment(conn, tid, "worker", "NOTE: still working — half done")
        clock.advance(55)  # silence restarted at the note
        assert nsw.check_running_notes(conn) == []
        assert len(_events(conn, tid, "stalled_suspect")) == 1

        clock.advance(10)  # 65 s from the note, and a full window past the last ping
        assert len(nsw.check_running_notes(conn)) == 1
        assert len(_events(conn, tid, "stalled_suspect")) == 2
    finally:
        conn.close()


def test_stall_receipt_comment_does_not_silence_its_own_watcher(kanban_home, clock):
    """The ``NOTE-STALL:`` receipt must not read as a ``NOTE:`` milestone."""
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(60)
        nsw.check_running_notes(conn)
        receipts = [
            c for c in kb.list_comments(conn, tid) if c.body.startswith(nsw.STALL_RECEIPT_PREFIX)
        ]
        assert len(receipts) == 1
        assert not nsw.is_milestone_note(receipts[0].body)
        clock.advance(65)  # the receipt did not buy a window; the next one pings
        assert len(nsw.check_running_notes(conn)) == 1
        assert len(_events(conn, tid, "stalled_suspect")) == 2
    finally:
        conn.close()


# --- Dedup with the estate stall-sentinel watchdog (ADR §3.3) ---

def test_sentinel_receipt_owns_the_episode(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        kb.add_comment(conn, tid, "stall-sentinel", "STALL-SENTINEL: QUIET — nothing on this card for 4h")
        clock.advance(300)
        assert nsw.check_running_notes(conn) == []
        assert _events(conn, tid, "stalled_suspect") == []
    finally:
        conn.close()


# --- Heartbeat liveness stays a separate signal (ADR §3.3 timescale separation) ---

def test_card_without_any_heartbeat_is_left_to_claim_liveness(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="no heartbeat", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        nsw.set_note_stall_window(conn, tid, 60)
        clock.advance(600)
        assert nsw.check_running_notes(conn) == []
        assert _events(conn, tid, "stalled_suspect") == []
        # ...and the same card is reportable once it starts heart-beating silently.
        kbd.heartbeat_worker(conn, tid)
        assert len(nsw.check_running_notes(conn)) == 1
    finally:
        conn.close()


# --- AC 5: the config knob really moves the window ---

def test_global_window_knob_changes_the_window(kanban_home, clock, monkeypatch):
    conn = kbc.connect()
    try:
        _stub_config(monkeypatch, {"note_stall": {"window_seconds": 120}})
        other = kb.create_task(conn, title="config-driven", assignee="worker")
        assert kb.claim_task(conn, other) is not None
        assert kbd.heartbeat_worker(conn, other) is True
        clock.advance(90)
        assert nsw.check_running_notes(conn) == []
        clock.advance(31)
        verdicts = nsw.check_running_notes(conn)
        assert [v["window_s"] for v in verdicts] == [120]

        # Shrinking the knob mid-episode shortens the next ping's wait too (the cadence guard
        # reads the same window), so one more window of silence is what earns the next verdict.
        _stub_config(monkeypatch, {"note_stall": {"window_seconds": 30}})
        clock.advance(31)
        verdicts = nsw.check_running_notes(conn, board="other-board", emit=False)
        assert [v["window_s"] for v in verdicts] == [30]
    finally:
        conn.close()


def test_per_card_override_beats_config(kanban_home, clock, monkeypatch):
    """A card that declares its own window is trusted over whatever the board/global says."""
    conn = kbc.connect()
    try:
        _stub_config(monkeypatch, {"note_stall": {"window_seconds": 30, "by_board": {"default": 45}}})
        tid = _spawn(conn, window=600)
        clock.advance(400)
        assert nsw.check_running_notes(conn) == []
        clock.advance(201)
        verdicts = nsw.check_running_notes(conn)
        assert [v["window_s"] for v in verdicts] == [600]
        assert [v["task_id"] for v in verdicts] == [tid]
    finally:
        conn.close()


def test_by_board_override_beats_global(kanban_home, clock, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="slow phase", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kbd.heartbeat_worker(conn, tid) is True
        _stub_config(monkeypatch, {
            "note_stall": {"window_seconds": 60, "by_board": {"slow-board": 300}},
        })
        clock.advance(200)
        assert nsw.check_running_notes(conn, board="slow-board") == []
        # Same silence judged under a board with no override uses the global window (one DB is
        # one board), so it is reportable there — reported, not written, to keep this read-only.
        verdicts = nsw.check_running_notes(conn, board="other-board", emit=False)
        assert [v["window_s"] for v in verdicts] == [60]
        clock.advance(101)
        verdicts = nsw.check_running_notes(conn, board="slow-board")
        assert [v["window_s"] for v in verdicts] == [300]
    finally:
        conn.close()


def test_enabled_false_disables_the_watcher(kanban_home, clock, monkeypatch):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=10)
        _stub_config(monkeypatch, {"note_stall": {"enabled": False}})
        clock.advance(600)
        assert nsw.check_running_notes(conn) == []
        assert not nsw.note_stall_enabled()
        assert _events(conn, tid, "stalled_suspect") == []
    finally:
        conn.close()


def test_window_resolution_precedence(kanban_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        task = kb.get_task(conn, tid)
        assert task is not None
        run_id = task.current_run_id
        _stub_config(monkeypatch, {
            "note_stall": {"window_seconds": 1800, "by_board": {"default": 900}},
        })
        # Per-card metadata override (already 60) wins over board 900 and global 1800.
        assert nsw.resolve_window_seconds(conn, task_id=tid, run_id=run_id, board="default") == 60
        nsw.set_note_stall_window(conn, tid, 45, run_id=run_id)
        assert nsw.resolve_window_seconds(conn, task_id=tid, run_id=run_id, board="default") == 45
        # An explicit caller override wins over the card.
        assert nsw.resolve_window_seconds(
            conn, task_id=tid, run_id=run_id, board="default", override=7
        ) == 7
        # Garbage values fall through instead of arming on window=0.
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            ('{"note_stall_window_seconds": -5}', run_id),
        )
        assert nsw.resolve_window_seconds(conn, task_id=tid, run_id=run_id, board="default") == 900
        with pytest.raises(ValueError):
            nsw.set_note_stall_window(conn, tid, 0)
        assert nsw.set_note_stall_window(conn, "t_nonexistent_run", 30) is False
    finally:
        conn.close()


def test_default_window_is_the_adr_shipped_default(kanban_home, monkeypatch):
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {})
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="no knobs", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        task = kb.get_task(conn, tid)
        assert task is not None
        assert nsw.DEFAULT_WINDOW_SECONDS == 1800
        assert nsw.resolve_window_seconds(
            conn, task_id=tid, run_id=task.current_run_id, board="default"
        ) == 1800
    finally:
        conn.close()


def test_emit_false_reports_without_writing(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(61)
        verdicts = nsw.check_running_notes(conn, emit=False)
        assert verdicts and verdicts[0]["emitted"] is False
        assert _events(conn, tid, "stalled_suspect") == []
        assert kb.list_comments(conn, tid) == []
    finally:
        conn.close()


def test_closed_episodes_are_never_reported(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        task = kb.get_task(conn, tid)
        assert task is not None
        kb.block_task(conn, tid, reason="waiting", kind="needs_input", expected_run_id=task.current_run_id)
        clock.advance(600)
        assert nsw.check_running_notes(conn) == []
        assert _events(conn, tid, "stalled_suspect") == []
    finally:
        conn.close()


def test_event_is_stamped_on_the_run_that_earned_it(kanban_home, clock):
    conn = kbc.connect()
    try:
        tid = _spawn(conn, window=60)
        clock.advance(61)
        nsw.check_running_notes(conn)
        events = _events(conn, tid, "stalled_suspect")
        assert len(events) == 1
        task = kb.get_task(conn, tid)
        assert task is not None
        assert events[0].run_id == task.current_run_id
        assert events[0].created_at == clock.now()
    finally:
        conn.close()
