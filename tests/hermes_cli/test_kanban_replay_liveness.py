"""ADR 0020 §5.3 — synthetic replay of the 2026-11-21 incidents (build card t_68d5b5a4).

One file, one board DB, the two operator-visible failures rebuilt end to end with kernel
functions only (``create_task`` / ``claim_task`` / ``heartbeat_worker`` / ``add_comment`` /
``block_task`` / ``complete_task`` / ``list_events``). The stall window is compressed with the
per-card knob and the clock is a fixture: the replay covers a "75-minute" silence in
milliseconds, never by sleeping, and no live board is opened. The container pin is printed so
the run's receipt names the image it ran under.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_board_liveness as bl
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_note_stall as nsw

BOARD = "default"
REPLAY_WINDOW_S = 8  # §5.3's "8 s window knob", never a real 30-minute wait


class FakeClock:
    def __init__(self, start: int) -> None:
        self.t = int(start)

    def now(self) -> int:
        return self.t

    def advance(self, seconds: int) -> int:
        self.t += int(seconds)
        return self.t


@pytest.fixture
def clock(monkeypatch):
    import time as time_mod

    c = FakeClock(1_800_000_000)
    monkeypatch.setattr(time_mod, "time", lambda: float(c.t))
    return c


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    _print_container_receipt()
    return home


def _print_container_receipt() -> None:
    """A tested claim must name its container (CI-CONTAINERS-GUIDANCE §Receipts)."""
    status = Path(__file__).resolve().parents[2] / "ci" / "hermes-upstream" / "status.json"
    try:
        raw = status.read_text()
    except OSError:
        print("[replay-receipt] ci/hermes-upstream/status.json absent — mounted-tree run "
              "(the image pin is recorded in the card's completion metadata)")
        return
    try:
        pin = json.loads(raw)
    except ValueError:
        print(f"[replay-receipt] unparseable status.json: {raw[:200]}")
        return
    print(
        "[replay-receipt] container=hermes-upstream tag=%s upstream=%s result=%s tests=%s/%s"
        % (pin.get("dated_tag"), pin.get("upstream_short"), pin.get("result"),
           pin.get("tests_passed"), pin.get("tests_total"))
    )


def _events(conn, task_id, kind):
    return [e for e in kb.list_events(conn, task_id) if e.kind == kind]


class _NotifyView:
    """Only the fields the liveness renderers read off ``_KanbanNotification``."""

    def __init__(self, task_id: str, title: str) -> None:
        self.task_id = task_id
        self.title = title
        self.board_slug = BOARD
        self.head = f"[{BOARD}] {task_id}"


# --- Incident 2 (SPECIMEN-14): heartbeats, zero notes, nobody noticed for 49 minutes ---

def test_replay_silent_stall_alerts_at_the_window_and_reaches_the_notifier(kanban_home, clock):
    from gateway.kanban_watchers_notifier import (
        TERMINAL_KINDS, _EVENT_FORMATTERS, _WAKE_KINDS, diagnostic_event,
    )

    conn = kbc.connect()
    try:
        # The board's operator route: the stream the notifier tick converges on — the ops chat
        # is auto-subscribed to every open card (2026-10-24 ruling), exactly like completions.
        ops_card = kb.create_task(conn, title="board heartbeat card", assignee="worker")
        tid = kb.create_task(conn, title="big bisect", assignee="worker")
        assert kbn.auto_subscribe_stream(
            conn, platform="telegram", chat_id="ops-chat", notifier_profile="swco-butler",
        ) == 2
        assert kb.get_task(conn, ops_card) is not None  # the route card stays on the board

        # The SPECIMEN-14 worker: claims, beats every ~60 s, writes nothing a human can read.
        assert kb.claim_task(conn, tid) is not None
        nsw.set_note_stall_window(conn, tid, REPLAY_WINDOW_S)
        task = kb.get_task(conn, tid)
        assert task is not None
        t0 = int(task.started_at or 0)

        for _ in range(3):  # technical heartbeats, no notes, all inside the 8 s window
            clock.advance(2)
            kbd.heartbeat_worker(conn, tid)
            assert nsw.check_running_notes(conn, board=BOARD) == []
        assert clock.now() == t0 + 6
        assert _events(conn, tid, "stalled_suspect") == []

        # AC 1: exactly one event at ~window.
        clock.advance(2)
        verdicts = nsw.check_running_notes(conn, board=BOARD)
        assert [v["task_id"] for v in verdicts] == [tid]
        events = _events(conn, tid, "stalled_suspect")
        assert len(events) == 1
        event = events[0]
        payload = event.payload or {}
        assert payload["window_s"] == REPLAY_WINDOW_S and payload["silence_s"] == REPLAY_WINDOW_S
        assert "stalled_suspect" in TERMINAL_KINDS
        assert not diagnostic_event(event), "liveness alerts are primary attention"
        assert "stalled_suspect" not in _WAKE_KINDS, "a stall is a question, not a wake"

        # Notifier receipt: the existing per-subscription cursor delivers it (no new path)...
        _old, _new, delivered = kbn.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="ops-chat", kinds=TERMINAL_KINDS,
        )
        assert [e.kind for e in delivered] == ["stalled_suspect"]

        # ...and the renderer prints operator-actionable text naming the mitigation.
        text, wake_handoff, _detail = _EVENT_FORMATTERS["stalled_suspect"](
            event, _NotifyView(tid, "big bisect")
        )
        assert "no progress note" in text and "NOTE: still working" in text
        assert wake_handoff is None
        print(f"[replay-receipt] stalled_suspect notification: {text}")

        # AC 2: the same worker posting a milestone note is not pinged again in its window.
        clock.advance(4)
        kb.add_comment(conn, tid, "worker", "NOTE: still working — bisect step 3 of 7")
        kbd.heartbeat_worker(conn, tid)
        assert nsw.check_running_notes(conn, board=BOARD) == []
        assert len(_events(conn, tid, "stalled_suspect")) == 1

        # The episode ends with the run: a completed card can never be re-flagged.
        kb.complete_task(conn, tid, summary="done", result="done",
                         expected_run_id=(task.current_run_id if task else None), force=True)
        clock.advance(10_000)
        assert nsw.check_running_notes(conn, board=BOARD) == []
        assert len(_events(conn, tid, "stalled_suspect")) == 1
    finally:
        conn.close()


# --- Incident 1's aftermath: the board holds open goals and nothing can advance them ---

def test_replay_dead_board_alerts_once_and_quiet_boards_stay_quiet(kanban_home):
    from gateway.kanban_watchers_notifier import (
        TERMINAL_KINDS, _EVENT_FORMATTERS, _WAKE_KINDS,
    )

    conn = kbc.connect()
    try:
        ops_card = kb.create_task(conn, title="board heartbeat card", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=ops_card, platform="telegram", chat_id="ops-chat",
            notifier_profile="swco-butler",
        )

        # AC 3: open work, zero running, zero promotable (the assignee is not a live profile —
        # exactly how gargantua01 looked on 2026-11-21 after the chain broke).
        stalled = kb.create_task(conn, title="downstream work", assignee="ghost-profile")
        dead = bl.check_and_emit_board_liveness(conn, board=BOARD, assignee_live=lambda _n: False)
        assert dead.state == bl.STATE_DEAD and dead.counts["open_goals"] == 2

        sentinel = bl.sentinel_id(BOARD)
        events = _events(conn, sentinel, "board_dead_quiet")
        assert len(events) == 1
        payload = events[0].payload or {}
        assert payload["state"] == "DEAD" and payload["running"] == 0 and payload["ready"] == 2
        assert "board_dead_quiet" in TERMINAL_KINDS and "board_dead_quiet" in _WAKE_KINDS

        _old, _new, delivered = kbn.claim_unseen_events_for_sub(
            conn, task_id=sentinel, platform="telegram", chat_id="ops-chat", kinds=TERMINAL_KINDS,
        )
        assert [e.kind for e in delivered] == ["board_dead_quiet"], "the butler route receives it"

        text, handoff, _ = _EVENT_FORMATTERS["board_dead_quiet"](
            events[0], _NotifyView(sentinel, "[system] board liveness sentinel")
        )
        assert "dead-quiet" in text and handoff
        print(f"[replay-receipt] board_dead_quiet notification: {text}")

        # The 300 s beat keeps evaluating and never re-pings on the same UTC day.
        for _ in range(3):
            assert bl.check_and_emit_board_liveness(
                conn, board=BOARD, assignee_live=lambda _n: False
            ).state == bl.STATE_DEAD
        assert len(_events(conn, sentinel, "board_dead_quiet")) == 1

        # The fix lands: somebody takes the work and parks it on the operator ⇒ AC 4 — the board
        # is quiet *by design*, so it must never be labelled DEAD again.
        conn.execute("UPDATE tasks SET assignee = 'dev' WHERE id = ?", (stalled,))
        conn.execute("UPDATE tasks SET assignee = 'dev' WHERE id = ?", (ops_card,))
        kb.complete_task(conn, ops_card, summary="route card retired", result="retired", force=True)
        assert kb.claim_task(conn, stalled) is not None
        assert bl.check_board_liveness(
            conn, board=BOARD, assignee_live=lambda _n: True
        ).state == bl.STATE_HEALTHY

        task = kb.get_task(conn, stalled)
        assert task is not None
        kb.block_task(conn, stalled, reason="operator decision needed", kind="needs_input",
                      expected_run_id=task.current_run_id)
        quiet = bl.check_and_emit_board_liveness(conn, board=BOARD, assignee_live=lambda _n: True)
        assert quiet.state == bl.STATE_QUIET_BY_DESIGN
        assert len(_events(conn, sentinel, "board_dead_quiet")) == 1  # never re-labelled DEAD
        quiet_events = _events(conn, sentinel, "board_quiet_by_design")
        assert len(quiet_events) == 1
        assert "board_quiet_by_design" not in _WAKE_KINDS

        text, handoff, _ = _EVENT_FORMATTERS["board_quiet_by_design"](
            quiet_events[0], _NotifyView(sentinel, "[system] board liveness sentinel")
        )
        assert "idle by design" in text and handoff is None
        print(f"[replay-receipt] board_quiet_by_design notification: {text}")
    finally:
        conn.close()


def test_replay_window_knob_moves_the_alert_without_wall_clock(kanban_home, clock):
    """AC 5 on the replay board: the same silence, two windows, two outcomes."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="long compile", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kbd.heartbeat_worker(conn, tid) is True

        nsw.set_note_stall_window(conn, tid, 120)
        clock.advance(60)
        assert nsw.check_running_notes(conn, board=BOARD, emit=False) == []

        # The knob alone flips the verdict — no code change, no sleep.
        nsw.set_note_stall_window(conn, tid, 30)
        verdicts = nsw.check_running_notes(conn, board=BOARD, emit=False)
        assert [v["window_s"] for v in verdicts] == [30]
        assert _events(conn, tid, "stalled_suspect") == []  # emit=False wrote nothing

        assert len(nsw.check_running_notes(conn, board=BOARD)) == 1
        assert len(_events(conn, tid, "stalled_suspect")) == 1
    finally:
        conn.close()
