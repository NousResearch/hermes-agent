"""t_f7f07208: a card must not become un-workable by accumulating history.

The incident: `t_2c12aac4` sat in `review` and four consecutive dispatcher spawns
exited rc=1 with no session row. Every one of them reported the SAME "Worker's last
output" — the tail of a comment written by an *earlier* attempt — so triage chased a
load spike that did not exist. The real diagnostic (`Error: Unknown skill(s):
sdlc-review`) had been written to the same append-mode log, *below* that earlier
attempt's exit summary, and `_worker_final_output` dropped everything from the marker
on:

    cut = raw.rfind(_EXIT_SUMMARY_MARKER)      # LAST marker in the window
    raw = raw[:cut]                            # ...which killed the newest output

Two contracts are pinned here:

* the crash diagnostic carries the dead worker's own newest output, and
* a card carrying a padded (15 KB+) comment thread is still claimable, reviewable and
  rendered under the `_CTX_MAX_*` caps — thread size is never fatal.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


# The shape of a real per-task log after a success followed by a startup abort. Both
# halves are byte-real: the summary/footer is what `hermes chat -q` prints when the
# session ends, the second half is `hermes … --skills <name> chat -q` failing to
# resolve a pinned skill before the session exists.
_ATTEMPT_1 = (
    "Two things a human should look at: the alert transport is genuinely down.\n"
    "STALE-COMMENT-TAIL: this text belongs to the PREVIOUS attempt.\n"
    "\n"
    "Resume this session with:\n"
    "  hermes --resume 20260921_143631_c0e3c2\n"
    '  hermes -c "Digest hygiene: SEC-11 duplicate findings"\n'
    "\n"
    "Session:        20260921_143631_c0e3c2\n"
    "Title:          Digest hygiene: SEC-11 duplicate findings\n"
    "Duration:       1h 19m 42s\n"
    "Messages:       162 (2 user, 159 tool calls)\n"
)
_CRASH_LINE = "Error: Unknown skill(s): sdlc-review"


def _attempt_2_crash(task_id: str) -> str:
    return (
        "Warning: Unknown toolsets: a2a, brave-search\n"
        f"Query: work kanban task {task_id}\n"
        "Initializing agent...\r\n"
        f"{_CRASH_LINE}\n"
        f"{KANBAN_WORKER_EXIT_TRAILER}1\n"
    )


def _write_log(tid: str, text: str) -> None:
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(text)


def _dead_worker(conn, tid: str, pid: int) -> None:
    """Claim ``tid`` for a worker that already exited and will never be reaped here."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# The diagnostic: newest output wins, even when an older summary follows it
# ---------------------------------------------------------------------------


def test_final_output_reports_the_newest_attempt_not_the_previous_summary(kanban_home: Path) -> None:
    """An append-mode log holds both attempts; the crash line must survive the read."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="grew a thread", assignee="a")
        raw = _ATTEMPT_1 + _attempt_2_crash(tid)
        _write_log(tid, raw)

    # The old behaviour, spelled out: cutting at the LAST marker discards the crash.
    cut = raw.rfind(kbd._EXIT_SUMMARY_MARKER)
    assert cut != -1 and _CRASH_LINE not in raw[:cut]

    out = kbd._worker_final_output(tid)
    assert _CRASH_LINE in out
    assert out.endswith(_CRASH_LINE), out
    # Older prose may stay as context, but it can never be the LAST thing again.
    assert "STALE-COMMENT-TAIL" in out
    assert out.index("STALE-COMMENT-TAIL") < out.index(_CRASH_LINE)


def test_final_output_keeps_the_model_prose_of_a_single_clean_attempt(kanban_home: Path) -> None:
    """The trim still works for the case it was written for: one attempt, no trailer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="clean", assignee="a")
        _write_log(tid, "I could not comply because the provider rejected the model.\n" + _ATTEMPT_1)

    out = kbd._worker_final_output(tid)
    assert "I could not comply because the provider rejected the model." in out
    assert kbd._EXIT_SUMMARY_MARKER not in out
    assert "Session:" not in out and "Duration:" not in out


def test_crashed_run_error_carries_the_workers_own_output(kanban_home: Path) -> None:
    """End to end through the reap: the run's `error` is the worker's stderr, not card text."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="crash", assignee="a")
        _dead_worker(conn, tid, 74001)
        _write_log(tid, _ATTEMPT_1 + _attempt_2_crash(tid))

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
    assert run["outcome"] == "crashed"
    # The dispatcher's own preamble first, the WORKER's own last line last
    # (the tail is quoted by ``_classify_dead_worker``).
    assert (run["error"] or "").rstrip().endswith(_CRASH_LINE + "'")
    assert "STALE-COMMENT-TAIL" in (run["error"] or "")


# ---------------------------------------------------------------------------
# Thread size is not fatal: a padded card is still claimable and renderable
# ---------------------------------------------------------------------------


def _pad_thread(conn, tid: str, *, comments: int = 40, body_chars: int = 3_000) -> int:
    """Append a throwaway thread; returns the raw bytes of comment text written."""
    written = 0
    for i in range(comments):
        body = f"comment {i} " + ("x" * body_chars)
        kb.add_comment(conn, tid, author="tester", body=body)
        written += len(body)
    return written


def test_padded_comment_thread_is_bounded_in_the_worker_context(kanban_home: Path) -> None:
    """~120 KB of thread renders under the caps, with the omission stated."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="padded", assignee="a")
        raw_bytes = _pad_thread(conn, tid)
        ctx = kb.build_worker_context(conn, tid)

    assert raw_bytes > 15_000  # the card's floor: a 15 KB+ thread must be workable
    assert "earlier comment" in ctx and "omitted; showing most recent" in ctx
    # Bounded by the caps (30 comments x 2 KB), not by the 120 KB on disk.
    assert len(ctx) < 70_000, len(ctx)
    assert "truncated" in ctx  # the per-comment cap left its marker


def test_padded_comment_thread_card_is_claimed_and_dispatched_for_review(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance case: a padded card can be claimed, worked and handed to review."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    spawned: list[tuple[str, list[str]]] = []

    def spawn(task, workspace):
        spawned.append((task.id, list(task.skills or [])))
        return 4321

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="padded review", assignee="a")
        raw_bytes = _pad_thread(conn, tid, comments=12, body_chars=1_300)
        assert raw_bytes > 15_000

        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn, tid, summary="padded card ready",
            expected_run_id=implementation.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid in [s[0] for s in result.spawned]
        run = conn.execute(
            "SELECT status, profile FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
    assert run["status"] == "running"
    assert [s[0] for s in spawned] == [tid]
