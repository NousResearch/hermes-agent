"""Shared Kanban archive policy: protect exactly ``ready``/``running``/``review``.

Operator ruling (Randy, 2026-09-24): "don't refuse unknown, just protect the
known." A direct archive must succeed for every status outside the protected
set — legacy/unrecognized raw values such as ``completed`` included — and must
fail without a single mutation for the protected ones, telling the caller to
block/stop the task first. This file exercises the operator CLI surface and the
block-first / fence / verified-stop contracts the shared DB lifecycle enforces
for every caller (CLI, dashboard, agent tool).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


PROTECTED = ("ready", "running", "review")
# Representative known non-protected statuses + a legacy raw value + an
# arbitrary unrecognized one ("don't refuse unknown").
UNPROTECTED = ("triage", "todo", "scheduled", "blocked", "done", "completed", "wibble")


def _set_status(conn, task_id, status):
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()


def _event_kinds(conn, task_id):
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))]


# ---------------------------------------------------------------------------
# 1. Boundary: what archives directly, what is refused
# ---------------------------------------------------------------------------


def test_cli_archive_refuses_protected_statuses_and_names_the_way_out(kanban_home):
    """No legacy CLI bypass: the operator command quotes the shared instruction
    and leaves the card, its runs and its events untouched."""
    for status in PROTECTED:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title=f"cli {status}", assignee="a")
            _set_status(conn, tid, status)
            before_kinds = _event_kinds(conn, tid)

        out = kc.run_slash(f"archive {tid}")

        assert f"cannot archive {tid}" in out, out
        assert "block" in out and "Nothing changed" in out, out
        with kbc.connect() as conn:
            row = kb.get_task(conn, tid)
            assert row is not None and row.status == status
            assert row.block_kind is None, "a refusal must never park the card"
            assert _event_kinds(conn, tid) == before_kinds, "a refusal must not write events"


def test_cli_archive_allows_every_non_protected_status(kanban_home):
    for status in UNPROTECTED:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title=f"cli {status}", assignee="a", triage=True)
            _set_status(conn, tid, status)

        out = kc.run_slash(f"archive {tid}")

        assert f"Archived {tid}" in out, out
        with kbc.connect() as conn:
            task = kb.get_task(conn, tid)
            assert task is not None and task.status == "archived"
            assert _event_kinds(conn, tid).count("archived") == 1


def test_cli_block_then_archive_is_the_supported_path(kanban_home):
    """The instruction a refusal gives is one that actually works."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="two-step", assignee="a")
        assert kb.get_task(conn, tid).status == "ready"

    refused = kc.run_slash(f"archive {tid}")
    assert f"cannot archive {tid}" in refused

    blocked = kc.run_slash(f"block {tid} waiting on credentials")
    assert f"Blocked {tid}" in blocked, blocked

    archived = kc.run_slash(f"archive {tid}")
    assert f"Archived {tid}" in archived, archived
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "archived"


# ---------------------------------------------------------------------------
# 2. Block-first: ready fencing, review path, verified worker stop
# ---------------------------------------------------------------------------


def test_archive_loses_to_a_dispatcher_claim_that_lands_before_the_write(kanban_home, monkeypatch):
    """The archive's status pre-read passes, then the dispatcher promotes the
    card to ``ready`` and claims it before the archive writes: the claim wins
    and the archive changes nothing (races fail closed), so a ``ready`` card is
    never archived out from under a worker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="racer", assignee="a")
        # Archivable now, so the diagnostic pre-read passes; the dispatcher's
        # promote + claim land after it.
        _set_status(conn, tid, "blocked")
        parked = kb.get_task(conn, tid)
        assert parked is not None and parked.status == "blocked"

    real_pre_read = kb._archive_row
    seen = {"n": 0}

    def _claiming_pre_read(conn, task_id):
        # The diagnostic pre-read returns the archivable snapshot, then the
        # dispatcher's promote + claim land on the SAME connection before
        # archive_task's guarded UPDATE.
        row = real_pre_read(conn, task_id)
        if seen["n"] == 0 and task_id == tid:
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'blocked'", (tid,))
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'peer:1', claim_expires = 0 "
                "WHERE id = ? AND status = 'ready' AND claim_lock IS NULL", (tid,))
        seen["n"] += 1
        return row

    monkeypatch.setattr(kb, "_archive_row", _claiming_pre_read)
    with kbc.connect() as conn:
        ok, why = kb.archive_task(conn, tid, with_reason=True)
    monkeypatch.setattr(kb, "_archive_row", real_pre_read)

    assert seen["n"] >= 1, "the stale pre-read never happened"
    assert ok is False and "changed concurrently" in why, why
    with kbc.connect() as conn:
        row = kb.get_task(conn, tid)
        assert row is not None and row.status == "running", "the claim must win"
        assert _event_kinds(conn, tid).count("archived") == 0

    # The other order: archive first, then the claim has nothing left to take.
    with kbc.connect() as conn:
        other = kb.create_task(conn, title="archive first", assignee="a", triage=True)
        assert kb.archive_task(conn, other) is True
        assert kb.claim_task(conn, other) is None
        task = kb.get_task(conn, other)
        assert task is not None and task.status == "archived"


def test_running_is_archivable_only_after_the_worker_is_stopped_and_verified(kanban_home, monkeypatch):
    """``running`` never archives directly; the block path stops the worker with
    its canonical identity (pid + start fingerprint), closes the run truthfully
    and only then lets the archive through."""
    import signal as signal_mod

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="live", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        assert kb.claim_task(conn, tid, claimer=f"{host}:worker") is not None
        monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot:1|777")
        kbd._set_worker_pid(conn, tid, 4242)

        ok, why = kb.archive_task(conn, tid, with_reason=True)
        assert ok is False and "'running'" in why
        assert kb.get_task(conn, tid).status == "running"

        signalled = []
        assert kb.block_task(
            conn, tid, reason="operator stop",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)), with_reason=True,
        ) == (True, None)
        assert signalled and signalled[0] == (4242, signal_mod.SIGTERM)
        run = kb.latest_run(conn, tid)
        assert run is not None and run.outcome == "blocked", "run must close truthfully"
        blocked = kb.get_task(conn, tid)
        assert blocked is not None and blocked.worker_pid is None

    with kbc.connect() as conn:
        assert kb.archive_task(conn, tid) is True


def test_review_card_reaches_archive_only_through_its_own_block_lane(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="subject", assignee="builder")
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer") is True

        refused = kb.archive_task(conn, tid, with_reason=True)
        assert refused[0] is False and "stop the review" in refused[1]

        assert kb.block_task(conn, tid, reason="waiting on decision") is True
        events = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        assert events[-1].payload["source_status"] == "review"
        assert kb.unblock_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "review", "review history must survive"

        assert kb.block_task(conn, tid, reason="still waiting") is True
        assert kb.archive_task(conn, tid) is True


def test_cli_block_surfaces_a_worker_that_could_not_be_stopped(kanban_home, monkeypatch):
    """The operator sees the stop blocker at block time, and the later archive
    refuses for exactly that reason until the process is gone."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wedged", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        assert kb.claim_task(conn, tid, claimer=f"{host}:worker") is not None
        monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot:1|777")
        kbd._set_worker_pid(conn, tid, 4242)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *a, **k: False)

    out = kc.run_slash(f"block {tid} operator stop")
    assert f"Blocked {tid}" in out and "[!]" in out and "cannot be proven stopped" in out, out

    with kbc.connect() as conn:
        row = kb.get_task(conn, tid)
        assert row is not None and row.status == "blocked"
        assert row.worker_pid == 4242, "the identity must survive an unproven stop"
        refused = kb.archive_task(conn, tid, with_reason=True)
        assert refused[0] is False and "live worker (pid 4242)" in refused[1]
