"""Bounded worker-context regression suite (t_3fd6545b).

The t_c2404f92 amplification: 121 comments x ~965 chars + 20 done parents x
~4.2KB handoffs produced a ~121KB worker context that blew the prompt budget.
These tests pin the fix's contract:

* the render is hard-bounded by ``kb._CTX_MAX_TOTAL_BYTES``;
* the render is deterministic (same DB state -> byte-identical text);
* building the context NEVER deletes or mutates historical DB records;
* the task body's leading mandatory markers (``task_type:``) survive even a
  fail-closed body truncation, which carries a retrieval pointer;
* parent handoffs beyond the cap are omitted by explicit count + ids, never
  silently dropped;
* omitted comments/attempts/attachments report a count and a retrieval
  pointer (fail-closed, not lossy);
* small tasks render unchanged (no regression on the common path).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _closed_run(conn, task_id: str, *, profile: str, outcome: str = "completed",
                summary: str | None = None, metadata: dict | None = None,
                error: str | None = None, started: int | None = None,
                ended: int | None = None) -> int:
    """Insert a closed run row directly (historical attempts / parent handoffs)."""
    now = int(time.time())
    started = started if started is not None else now
    ended = ended if ended is not None else now
    import json as _json
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, summary, error, "
        "metadata, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, profile, outcome, outcome, summary, error,
         _json.dumps(metadata) if metadata else None, started, ended),
    )
    return int(cur.lastrowid)


def _mark_done(conn, task_id: str) -> None:
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
        (int(time.time()), task_id),
    )


# ---------------------------------------------------------------------------
# The t_c2404f92 amplification scenario, bounded
# ---------------------------------------------------------------------------


def test_amplification_scenario_is_hard_bounded(kanban_home):
    """121 long comments + 20 fat parent handoffs: the old renderer produced
    ~121KB. The new one must stay under the ceiling AND keep the newest
    content + explicit omission accounting."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="long-running", body="task_type: code\n\ndo the work",
                           assignee="builder")
        # 20 done parents, each with a ~4.2KB completed-run summary. Timestamps
        # are strictly increasing so "newest handoffs survive" is deterministic
        # (equal ended_at would fall back to the random parent_id tiebreak).
        base = int(time.time())
        for i in range(20):
            p = kb.create_task(conn, title=f"parent {i}")
            kb.link_tasks(conn, p, t)
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
                (base - (20 - i), p),
            )
            _closed_run(conn, p, profile="researcher",
                        summary=f"parent-{i}-handoff " + ("x" * 4200),
                        started=base - (20 - i) - 60, ended=base - (20 - i))
        # 121 comments of ~965 chars.
        for i in range(121):
            kb.add_comment(conn, t, f"worker-{i % 5}", f"comment {i}: " + ("y" * 950))
        # Make comment order deterministic (same-second burst -> unique ts).
        conn.execute(
            "UPDATE task_comments SET created_at = 1000000 + id WHERE task_id = ?", (t,)
        )

        ctx = kb.build_worker_context(conn, t)

        # Hard bound.
        assert len(ctx.encode("utf-8")) <= kb._CTX_MAX_TOTAL_BYTES, len(ctx.encode("utf-8"))

        # The leading task_type marker and body survive.
        assert "task_type: code" in ctx

        # Newest comment is present; the oldest is omitted — but counted and
        # pointed at, never silently dropped.
        assert "comment 120:" in ctx
        assert "comment 0:" not in ctx
        assert "91 earlier comments omitted" in ctx
        assert "kanban_show" in ctx

        # Parent handoffs: cap honoured, omitted parents listed by id.
        assert "12 earlier parent handoffs omitted" in ctx
        assert "retrieve with kanban_show on the parent id" in ctx
        # The newest parent's full handoff still renders.
        assert "parent-19-handoff" in ctx

        # Determinism: same DB state -> identical render.
        assert kb.build_worker_context(conn, t) == ctx


def test_building_context_never_mutates_history(kanban_home):
    """build_worker_context is read-only: every comment/run row is intact in
    the DB after the bounded render (no history deletion to fit)."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="audit", body="task_type: ops\n\nkeep records")
        for i in range(130):
            kb.add_comment(conn, t, "worker", f"record {i}: " + ("z" * 900))
        for i in range(15):
            _closed_run(conn, t, profile="ops", outcome="failed",
                        summary=f"attempt {i}", error=f"boom {i}")

        before_comments = conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (t,)).fetchone()[0]
        before_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (t,)).fetchone()[0]
        before_body = kb.get_task(conn, t).body

        kb.build_worker_context(conn, t)
        kb.build_worker_context(conn, t)  # twice: repeated builds also mutate nothing

        after_comments = conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (t,)).fetchone()[0]
        after_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (t,)).fetchone()[0]

        assert (before_comments, before_runs) == (130, 15)
        assert (after_comments, after_runs) == (130, 15)
        assert kb.get_task(conn, t).body == before_body


def test_omitted_attempts_reported_with_pointer(kanban_home):
    """Prior attempts beyond the cap: omitted with count + `hermes kanban
    runs` pointer; the newest attempts render in full."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="retry-heavy", body="task_type: code\n\nfix it")
        for i in range(14):
            _closed_run(conn, t, profile="builder", outcome="failed",
                        summary=f"attempt {i} did things", error=f"err {i}",
                        started=int(time.time()) - (14 - i) * 60,
                        ended=int(time.time()) - (14 - i) * 60 + 30)

        ctx = kb.build_worker_context(conn, t)
        assert f"4 earlier attempts omitted" in ctx
        assert f"hermes kanban runs {t}" in ctx
        # Newest shown (attempt 13), oldest omitted (attempt 0).
        assert "attempt 13 did things" in ctx
        assert "attempt 0 did things" not in ctx
        # All 14 run rows intact.
        n = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL",
            (t,)).fetchone()[0]
        assert n == 14


def test_body_marker_survives_fail_closed_body_cap(kanban_home):
    """A pathological >32K body still fails closed, but head truncation keeps
    the leading task_type marker and the note points at the full text."""
    with kbc.connect() as conn:
        body = "task_type: code\n\n" + ("spec " * 20000)  # ~100KB body
        t = kb.create_task(conn, title="huge body", body=body)

        ctx = kb.build_worker_context(conn, t)
        assert "task_type: code" in ctx
        assert "chars omitted" in ctx
        assert f"hermes kanban show {t}" in ctx
        assert len(ctx.encode("utf-8")) <= kb._CTX_MAX_TOTAL_BYTES
        # Body intact in DB.
        assert kb.get_task(conn, t).body == body


def test_many_attachments_omitted_with_count(kanban_home):
    """Attachments beyond the cap: newest shown, older omitted by explicit
    count with the kanban_attachments pointer; rows never deleted."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="file-heavy", body="task_type: ops\n\nfiles")
        for i in range(25):
            kb.add_attachment(
                conn, t, filename=f"f{i}.bin", stored_path=f"/tmp/f{i}.bin",
                content_type="application/octet-stream", size=1024,
            )
        ctx = kb.build_worker_context(conn, t)
        assert "5 earlier attachments omitted" in ctx
        assert "kanban_attachments" in ctx
        assert "f24.bin" in ctx
        assert "f0.bin" not in ctx
        n = conn.execute(
            "SELECT COUNT(*) FROM task_attachments WHERE task_id = ?", (t,)).fetchone()[0]
        assert n == 25


# ---------------------------------------------------------------------------
# No regression on the common (small-task) path
# ---------------------------------------------------------------------------


def test_small_task_renders_full_context_without_omission_notes(kanban_home):
    """A normal task (short body, few comments, one parent, few attempts)
    renders everything in full — no omission notes, no reduced-budget notice,
    markers and parent handoff intact."""
    with kbc.connect() as conn:
        p = kb.create_task(conn, title="research parent")
        _mark_done(conn, p)
        _closed_run(conn, p, profile="researcher", summary="findings: A, B, C",
                    metadata={"sources": 3})
        t = kb.create_task(
            conn, title="child work", body="task_type: code\n\nBuild the thing.\nLine 3.",
            assignee="builder", parents=(p,),
        )
        kb.add_comment(conn, t, "operator", "please hurry")
        kb.add_comment(conn, t, "builder", "on it")
        _closed_run(conn, t, profile="builder", outcome="crashed", error="oom")

        ctx = kb.build_worker_context(conn, t)

        assert "# Kanban task" in ctx
        assert "task_type: code" in ctx
        assert "Build the thing." in ctx
        assert "Line 3." in ctx
        assert "## Parent task results" in ctx
        assert "findings: A, B, C" in ctx
        assert '"sources": 3' in ctx
        assert "please hurry" in ctx
        assert "on it" in ctx
        assert "_error_: oom" in ctx
        # Nothing omitted -> no omission accounting at all.
        assert "omitted" not in ctx
        assert "reduced budget" not in ctx
        # No truncation ellipsis on this small task.
        assert "[truncated" not in ctx


def test_render_reduced_budget_notice_only_on_rebuild(kanban_home, monkeypatch):
    """When the full render exceeds the ceiling, the reduced rebuild says so
    explicitly; forcing a tiny ceiling proves the reduced path is reachable and
    still bounded + marker-preserving."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="forced reduce", body="task_type: code\n\ngo")
        for i in range(40):
            kb.add_comment(conn, t, "w", f"c{i}: " + ("q" * 500))

        # Force the ceiling below the full-budget render size.
        full_len = len(
            kb._render_worker_context(conn, t, kb._CTX_BUDGET_FULL).encode("utf-8")
        )
        monkeypatch.setattr(kb, "_CTX_MAX_TOTAL_BYTES", full_len - 1)

        ctx = kb.build_worker_context(conn, t)
        assert "reduced budget" in ctx
        assert "Nothing was deleted" in ctx
        assert "task_type: code" in ctx
        assert len(ctx.encode("utf-8")) <= full_len
