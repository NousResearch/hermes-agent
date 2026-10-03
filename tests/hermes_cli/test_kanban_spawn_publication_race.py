"""Spawn/PID-publication race: a child can be alive before any row records it.

The dispatcher claims a card, starts the child, and only THEN publishes the PID
(``kanban_db_dispatch._set_worker_pid``). An operator who blocked or archived
inside that window used to see ``worker_pid IS NULL`` and read it as "nothing is
running": the run closed as if the work had stopped, the card archived, and the
late publication welded a live PID onto the blocked/archived row — violating
stop-before-archive.

These tests park the dispatcher deterministically between "child started" and
"PID published" (a barrier wrapped around ``_set_worker_pid``) and race
block/archive against the publication in BOTH orders. Invariants asserted:

* no card ever reads back as ``archived`` while that worker is live;
* a losing publication never attaches its PID to a state its claim no longer
  owns (blocked, archived, or a successor run) — it stops the child with the
  canonical ``(pid, start fingerprint)``, verifies it gone, and records it on
  the spawn's own run plus a ``spawn_discarded`` event;
* a child that cannot be proven gone keeps the card unarchivable with its
  identity retained;
* a successor claim never takes a fence an earlier spawn still owns — with or
  without a published PID — never starts a second child beside it, and is
  unwound instead of being left owning a ``running`` card with nothing behind
  it;
* run outcome and event provenance stay truthful at every step.

Everything runs against a real child process and a real dispatcher tick
(``dispatch_once``), so the stop/verify path is the production one.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and a spawnable profile."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kbc.init_db()
    return home


@pytest.fixture
def procs():
    """Every child this test started — reaped no matter how the test ends."""
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


def _pid_running(pid: int) -> bool:
    """True while ``pid`` is a live (non-zombie) process on this host."""
    try:
        import psutil  # type: ignore
    except ImportError:  # pragma: no cover - psutil-free fallback
        try:
            fields = Path(f"/proc/{int(pid)}/stat").read_text().split()
        except (OSError, IndexError):
            return False
        return len(fields) > 2 and fields[2] != "Z"
    if not psutil.pid_exists(int(pid)):
        return False
    try:
        return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _wait_until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _child_spawner(procs: list):
    """A ``spawn_fn`` that starts a real, long-lived child and remembers it."""

    def _spawn(task, workspace, *, board=None):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        procs.append(proc)
        return proc.pid

    return _spawn


class _PublicationBarrier:
    """Parks the dispatcher AFTER the child exists and BEFORE its PID is written.

    Wraps ``_set_worker_pid`` so the interval the defect lives in becomes a
    deterministic, test-controlled window instead of a timing accident."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        self.pid: int | None = None
        self.kwargs: dict = {}

    def wrap(self, real):
        def _parked(conn, task_id, pid, **kwargs):
            self.pid = int(pid)
            self.kwargs = kwargs
            self.reached.set()
            if not self.release.wait(timeout=120):
                raise AssertionError("publication barrier was never released")
            return real(conn, task_id, pid, **kwargs)

        return _parked

    def wait_parked(self) -> int:
        assert self.reached.wait(timeout=60), "the dispatcher never reached publication"
        assert self.pid, "the spawn reported no pid"
        return self.pid


def _dispatch_async(spawn_fn):
    """Run one real dispatcher tick on its own thread + connection."""
    box: dict = {}

    def _run() -> None:
        try:
            with kbc.connect_closing() as conn:
                box["res"] = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        except BaseException as exc:  # pragma: no cover - surfaced by the test
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, box


def _join(thread, box, timeout: float = 60.0) -> None:
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "the dispatcher never finished"
    assert "error" not in box, box.get("error")


def _events(conn, task_id):
    return [e for e in kb.list_events(conn, task_id)]


def _event(conn, task_id, kind):
    matches = [e for e in _events(conn, task_id) if e.kind == kind]
    return matches[-1] if matches else None


def _latest_run_row(conn, task_id):
    return conn.execute(
        "SELECT id, status, outcome, worker_pid, worker_started_at, metadata "
        "FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()


# ---------------------------------------------------------------------------
# 1. block/archive land FIRST, the publication loses
# ---------------------------------------------------------------------------


def test_block_and_archive_wait_out_a_parked_publication(board, monkeypatch, procs):
    """Order A: the operator blocks and tries to archive while the child is alive
    and its PID is unpublished. The block must not report the work stopped, the
    archive must refuse with the exact hold, and only after the fenced-out
    publication has stopped and verified the child may the card archive."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="racing", assignee="worker")
        assert kb.get_task(conn, tid).status == "ready"

    barrier = _PublicationBarrier()
    monkeypatch.setattr(kbd, "_set_worker_pid", barrier.wrap(kbd._set_worker_pid))
    thread, box = _dispatch_async(_child_spawner(procs))

    pid = barrier.wait_parked()
    assert _pid_running(pid), "the spawned child must be alive while unpublished"

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence, current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()
        # The fence is what makes "no PID recorded" honest during the window.
        assert row["status"] == "running"
        assert row["worker_pid"] is None
        fence = kb._json_dict(row["spawn_fence"])
        assert fence.get("run") == row["current_run_id"]
        claimed_run = row["current_run_id"]

        # --- archive FIRST: refused outright while the work is in flight
        # (protected status) and the child is live, so nothing changes and no
        # event is written — the archive can never win this interleaving.
        events_before_block = [e.kind for e in _events(conn, tid)]
        refused_direct, why_direct = kb.archive_task(conn, tid, with_reason=True)
        assert refused_direct is False and "'running'" in why_direct, why_direct
        assert kb.get_task(conn, tid).status == "running"
        assert [e.kind for e in _events(conn, tid)] == events_before_block, \
            "a refusal writes nothing"
        assert _pid_running(pid), "the child is still live while unpublished"

        # --- block FIRST: it lands, but it does not claim the work stopped.
        ok, why = kb.block_task(conn, tid, reason="operator stop", with_reason=True)
        assert ok is True
        assert why and "has not published a PID yet" in why, why

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.worker_pid is None, "a PID that was never published must not appear"

        # Run/event provenance records the in-flight spawn instead of a stop.
        run = kb.latest_run(conn, tid)
        assert run is not None and run.outcome == "blocked"
        assert run.id == claimed_run
        assert run.metadata["spawn_fence"]["run"] == claimed_run
        blocked_event = _event(conn, tid, "blocked")
        assert blocked_event.payload["spawn_fence"]["run"] == claimed_run
        assert _event(conn, tid, "block_worker_termination") is None, \
            "nothing was stopped yet, so no stop may be claimed"

        # --- archive FIRST: refused for exactly that reason, no mutation.
        before = [e.kind for e in _events(conn, tid)]
        refused, why_archive = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False and "nothing was archived" in why_archive, why_archive
        assert "has not published a worker identity yet" in why_archive, why_archive
        assert "run" in why_archive and str(claimed_run) in why_archive, why_archive
        assert kb.get_task(conn, tid).status == "blocked"
        assert [e.kind for e in _events(conn, tid)] == before, "a refusal writes nothing"
        assert _pid_running(pid), "the child is still live, so the card must be held"

    # --- publication lands LAST: its claim is gone, so it is fenced out.
    barrier.release.set()
    _join(thread, box)

    assert _wait_until(lambda: not _pid_running(pid)), \
        "the fenced-out publication must stop the child it started"
    assert len(box["res"].discarded_spawns) == 1
    assert box["res"].discarded_spawns[0] == (tid, pid)
    assert box["res"].spawned == [], "a discarded spawn must never be reported as spawned"

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "blocked"
        assert row["worker_pid"] is None, "the late PID must never land on the blocked row"
        assert row["spawn_fence"] is None, "the hold is released only once the child is gone"

        # Truthful provenance: no `spawned`, one `spawn_discarded` for THIS pid.
        assert _event(conn, tid, "spawned") is None
        discarded = _event(conn, tid, "spawn_discarded")
        assert discarded is not None
        assert discarded.payload["pid"] == pid
        assert discarded.payload["stopped"] is True
        assert discarded.payload["claim"] == fence["claim"]
        assert discarded.payload["run"] == claimed_run
        assert discarded.run_id == claimed_run

        # The identity stays on the spawn's own run row (historical truth).
        run_row = _latest_run_row(conn, tid)
        assert run_row["id"] == claimed_run
        assert int(run_row["worker_pid"]) == pid
        assert run_row["worker_started_at"], "the fingerprint travels with the run's identity"

        # Only NOW is the card archivable — and the child is provably gone.
        assert kb.archive_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "archived"
        assert not _pid_running(pid)


# ---------------------------------------------------------------------------
# 2. the publication lands FIRST: the ordinary block/stop/archive path
# ---------------------------------------------------------------------------


def test_publication_before_block_leaves_a_stoppable_worker(board, procs):
    """Order B: the other interleaving. The PID is published while the claim is
    intact, so the block finds the worker, stops it, verifies it, and the
    archive then goes through — never the other way round."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="normal", assignee="worker")

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_child_spawner(procs))
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]
    assert res.discarded_spawns == []

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"
        pid = int(row["worker_pid"])
        assert row["spawn_fence"] is None, "publication settles its own fence"
        spawned = _event(conn, tid, "spawned")
        assert spawned is not None and spawned.payload["pid"] == pid
        assert _pid_running(pid)

        # Archive is still refused while the work is in flight.
        refused, why = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False and "'running'" in why
        assert _pid_running(pid), "a refusal must not stop anything"

        # Block stops it with pid + fingerprint and reports the truth.
        ok, why = kb.block_task(conn, tid, reason="operator stop", with_reason=True)
        assert ok is True and why is None, why
        term = _event(conn, tid, "block_worker_termination")
        assert term is not None and term.payload["stopped"] is True
        assert term.payload["worker_pid"] == pid
        assert _wait_until(lambda: not _pid_running(pid))

        task = kb.get_task(conn, tid)
        assert task.status == "blocked" and task.worker_pid is None
        assert kb.archive_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "archived"
        assert not _pid_running(pid)


# ---------------------------------------------------------------------------
# 3. a late spawn must not reattach to a successor run
# ---------------------------------------------------------------------------


def test_a_late_spawn_never_reattaches_to_a_successor_run(board, monkeypatch, procs):
    """The claim is replaced (block -> unblock -> re-claim) while the child is
    starting. The publication belongs to the OLD claim, so it must not touch the
    successor row or its run: the child is stopped, the successor keeps a clean
    row, and the discarded spawn is recorded against the run that started it."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="successor", assignee="worker")

    barrier = _PublicationBarrier()
    monkeypatch.setattr(kbd, "_set_worker_pid", barrier.wrap(kbd._set_worker_pid))
    thread, box = _dispatch_async(_child_spawner(procs))

    pid = barrier.wait_parked()
    with kbc.connect_closing() as conn:
        first_run = kb.get_task(conn, tid).current_run_id
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
        assert kb.unblock_task(conn, tid) is True
        successor = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:successor")
        assert successor is not None
        successor_run = successor.current_run_id
        assert successor_run != first_run

    barrier.release.set()
    _join(thread, box)
    assert _wait_until(lambda: not _pid_running(pid)), "the old spawn's child must be stopped"

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, current_run_id, claim_lock, spawn_fence "
            "FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"
        assert row["current_run_id"] == successor_run, "the successor run must be untouched"
        assert row["worker_pid"] is None, "a late PID must never reattach to the successor row"
        assert row["claim_lock"].endswith(":successor")
        # The successor's own fence decision is unaffected: whatever marker is
        # left names the OLD spawn, never a claim that no longer exists.
        fence = kb._json_dict(row["spawn_fence"]) or None
        if fence is not None:
            assert fence["run"] == first_run

        discarded = _event(conn, tid, "spawn_discarded")
        assert discarded is not None and discarded.payload["pid"] == pid
        assert discarded.payload["stopped"] is True
        assert discarded.run_id == first_run
        assert _event(conn, tid, "spawned") is None

        # The discarded child's identity lives on the run that started it.
        first_run_row = conn.execute(
            "SELECT worker_pid FROM task_runs WHERE id = ?", (first_run,)).fetchone()
        assert int(first_run_row["worker_pid"]) == pid
        successor_run_row = conn.execute(
            "SELECT worker_pid FROM task_runs WHERE id = ?", (successor_run,)).fetchone()
        assert successor_run_row["worker_pid"] is None

        assert kb.block_task(conn, tid, reason="stop the successor", with_reason=True)[0] is True


# ---------------------------------------------------------------------------
# 4. a worker that cannot be proven gone keeps the card unarchivable
# ---------------------------------------------------------------------------


def test_an_unproven_fenced_worker_keeps_the_card_unarchivable(board, monkeypatch, procs):
    """If the stop cannot prove the child gone, the late PID must not be welded
    onto ``worker_pid`` — but it must be retained somewhere the archive reads,
    so the card stays held with the exact identity instead of archiving beside a
    live process."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="survivor", assignee="worker")

    barrier = _PublicationBarrier()
    monkeypatch.setattr(kbd, "_set_worker_pid", barrier.wrap(kbd._set_worker_pid))
    # The stop path refuses to signal and the probe insists the child is live.
    monkeypatch.setattr(
        kbd, "_terminate_reclaimed_worker",
        lambda pid, claim_lock, **kw: {"termination_attempted": False, "terminated": False},
    )
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, started_at: True)
    thread, box = _dispatch_async(_child_spawner(procs))

    pid = barrier.wait_parked()
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True

    barrier.release.set()
    _join(thread, box)

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "blocked"
        assert row["worker_pid"] is None, "an unproven PID never becomes the row's worker"

        fence = kb._json_dict(row["spawn_fence"])
        assert fence["pid"] == pid, "the surviving identity must be retained"
        assert fence["started_at"], "the fingerprint travels with the pid"
        assert fence["claim"] and fence["run"]

        discarded = _event(conn, tid, "spawn_discarded")
        assert discarded.payload["pid"] == pid
        assert discarded.payload["stopped"] is False
        assert "blocker" in discarded.payload

        refused, why = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False, "the card must stay unarchivable"
        assert str(pid) in why and "has a spawned worker" in why, why
        assert kb.get_task(conn, tid).status == "blocked"

    # Restore the real probes so the fixture's teardown can reap the child.
    monkeypatch.undo()
    with contextlib.suppress(Exception):
        for proc in procs:
            proc.kill()


# ---------------------------------------------------------------------------
# 5. a claim that moves before the spawn starts a child at all
# ---------------------------------------------------------------------------


def test_a_claim_lost_before_the_spawn_starts_no_child(board):
    """The fence is armed under a claim CAS: if the claim has already moved on,
    the dispatcher must not start a worker for a card it no longer owns."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="lost claim", assignee="worker")
        claimed = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:one")
        assert claimed is not None
        lock, run = claimed.claim_lock, claimed.current_run_id

        assert kb._hold_spawn_fence(conn, tid, lock, run) is True
        assert kb.spawn_fence(conn, tid)["run"] == run

        # The operator blocks mid-dispatch: the claim is gone.
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
        assert kb.spawn_fence(conn, tid) is not None, "the fence survives the flip"
        assert kb._hold_spawn_fence(conn, tid, lock, run) is False, \
            "a lost claim must never arm a spawn (and never clobber the hold)"


# ---------------------------------------------------------------------------
# 6. a retained identity that later proves gone releases the hold
# ---------------------------------------------------------------------------


def test_a_fenced_worker_that_dies_releases_the_hold_and_archives(board, monkeypatch, procs):
    """Recovery for the held case: the stop could not prove the child gone, so
    its identity stays on the fence. While that exact ``(pid, start
    fingerprint)`` is alive the card stays unarchivable; once it can no longer
    be observed alive, the hold releases in the SAME guarded UPDATE as the
    archive (never a blind clear) and the release is recorded truthfully — the
    card is never wedged forever beside a dead process."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="held then released", assignee="worker")

    barrier = _PublicationBarrier()
    real_set_pid, real_terminate, real_alive = (
        kbd._set_worker_pid, kbd._terminate_reclaimed_worker, kbd._worker_alive,
    )
    monkeypatch.setattr(kbd, "_set_worker_pid", barrier.wrap(kbd._set_worker_pid))
    # The stop path cannot prove the child gone (as in test 4).
    monkeypatch.setattr(
        kbd, "_terminate_reclaimed_worker",
        lambda pid, claim_lock, **kw: {"termination_attempted": False, "terminated": False},
    )
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, started_at: True)
    thread, box = _dispatch_async(_child_spawner(procs))

    pid = barrier.wait_parked()
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
    barrier.release.set()
    _join(thread, box)

    with kbc.connect_closing() as conn:
        fence = kb.spawn_fence(conn, tid)
        assert fence is not None and fence["pid"] == pid, "the identity must be retained"
        refused, why = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False and str(pid) in why, why

    # Back to reality: the real probes decide (the fixture's own env patches stay
    # in place — only the three dispatcher probes above are restored).
    monkeypatch.setattr(kbd, "_set_worker_pid", real_set_pid)
    monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", real_terminate)
    monkeypatch.setattr(kbd, "_worker_alive", real_alive)
    assert _pid_running(pid)
    with kbc.connect_closing() as conn:
        refused, why = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False and "has a spawned worker" in why, why
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.spawn_fence(conn, tid)["pid"] == pid, "a refusal must keep the hold"

    # The operator reaps the retained identity; only now is the hold releasable.
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=10)
    assert _wait_until(lambda: not _pid_running(pid))

    with kbc.connect_closing() as conn:
        ok, why = kb.archive_task(conn, tid, with_reason=True)
        assert ok is True, why
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["status"] == "archived"
        assert row["spawn_fence"] is None, "the hold released with the archive"
        assert row["worker_pid"] is None
        release = _event(conn, tid, "spawn_fence_released")
        assert release is not None
        assert release.payload["pid"] == pid
        assert release.payload["started_at"] and release.payload["claim"]
        assert release.payload["run"]
        assert _event(conn, tid, "archived") is not None
        assert _event(conn, tid, "spawned") is None, "the late PID never became a spawn"
        assert kb.get_task(conn, tid).status == "archived"


# ---------------------------------------------------------------------------
# 7. a successor dispatch must not overwrite an unresolved (no-PID) fence
# ---------------------------------------------------------------------------


class _SpawnBarrier:
    """Parks the dispatcher INSIDE the spawn call: the fence is armed, the card
    claims it, and no child exists yet — the exact window in which a successor
    used to be able to take the hold over."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()

    def wrap(self, real_spawn):
        def _spawn(task, workspace, *, board=None):
            self.reached.set()
            if not self.release.wait(timeout=120):
                raise AssertionError("spawn barrier was never released")
            return real_spawn(task, workspace, board=board)

        return _spawn

    def wait_parked(self) -> None:
        assert self.reached.wait(timeout=60), "the dispatcher never reached the spawn call"


def test_a_successor_dispatch_never_overwrites_an_unresolved_fence(board, monkeypatch, procs):
    """THE uncovered interleaving: spawn A arms its fence and parks before it
    publishes, the operator blocks then unblocks the card, and a successor
    dispatch tries to start B. The successor must not take the hold, must not
    start a second child, and must not be left owning a ``running`` card with
    nothing behind it. Once A's own spawn has settled (fenced out, stopped,
    verified) the card dispatches again normally."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="held by an unresolved fence", assignee="worker")

    barrier = _SpawnBarrier()
    thread, box = _dispatch_async(barrier.wrap(_child_spawner(procs)))
    barrier.wait_parked()

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id, spawn_fence FROM tasks WHERE id = ?",
            (tid,)).fetchone()
        assert row["status"] == "running"
        fence_a_text = row["spawn_fence"]
        run_a, lock_a = row["current_run_id"], row["claim_lock"]
        assert (kb._json_dict(fence_a_text) or {}).get("pid") is None, "A has not published yet"

        # Block (the fence deliberately survives) then unblock: the card is
        # claimable again while A's spawn is still unresolved.
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
        assert kb.spawn_fence(conn, tid) is not None, "the fence survives the flip"
        assert kb.unblock_task(conn, tid) is True

    # Successor tick. The board's ``.dispatch.lock`` is a same-host
    # single-writer optimization that dies with the process holding it — the
    # fence is the correctness barrier — so the tick body is driven directly to
    # reach it instead of being skipped as a lock loser.
    spawn_b_calls: list[str] = []
    spawner_b = _child_spawner(procs)

    def spawn_b(task, workspace, *, board=None):
        spawn_b_calls.append(task.id)
        return spawner_b(task, workspace, board=board)

    with kbc.connect_closing() as conn:
        res = kbd._dispatch_once_locked(conn, spawn_fn=spawn_b)

    assert spawn_b_calls == [], "no second child may start beside a held fence"
    assert len(procs) == 0, "and no child exists at all before A publishes"
    assert res.spawned == [] and res.discarded_spawns == [] and res.auto_blocked == []
    assert (tid, "spawn_fence_hold") in res.respawn_guarded
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id, spawn_fence FROM tasks WHERE id = ?",
            (tid,)).fetchone()
        assert row["status"] == "ready", "the card must not be left running"
        assert row["claim_lock"] is None and row["current_run_id"] is None, \
            "no unowned claim/run may survive a refused spawn"
        assert row["spawn_fence"] == fence_a_text, "the predecessor's hold must not be overwritten"
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0]
        assert runs == 1, "a refused successor must not open a run it cannot own"
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'running'").fetchone()[0] == 0
        assert _event(conn, tid, "spawned") is None
        guarded = _event(conn, tid, "respawn_guarded")
        assert guarded is not None and guarded.payload["reason"] == "spawn_fence_hold"

    # A's spawn resumes: its publication loses the claim, so the child it did
    # start is fenced out, stopped and verified, and only then is the hold gone.
    barrier.release.set()
    _join(thread, box)
    assert len(procs) == 1, "exactly one child exists — A's own"
    pid = procs[0].pid
    assert _wait_until(lambda: not _pid_running(pid))
    assert box["res"].discarded_spawns == [(tid, pid)]
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready"
        assert row["worker_pid"] is None
        assert row["spawn_fence"] is None, "the hold releases only once A's child is gone"
        discarded = _event(conn, tid, "spawn_discarded")
        assert discarded is not None and discarded.payload["pid"] == pid
        assert discarded.payload["stopped"] is True and discarded.run_id == run_a
        assert _event(conn, tid, "spawned") is None, "the discarded spawn never became a spawn"
        assert lock_a and run_a

    # Recovery: the predecessor's spawn is settled, so the next real tick claims
    # and spawns exactly one child — no stale fence blocks the card forever.
    spawner_c = _child_spawner(procs)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawner_c)
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]
    assert len(procs) == 2
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"
        assert int(row["worker_pid"]) == procs[1].pid
        assert row["spawn_fence"] is None
        assert _event(conn, tid, "spawned").payload["pid"] == procs[1].pid
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: not _pid_running(procs[1].pid))


# ---------------------------------------------------------------------------
# 8. a successor dispatch must wait out a retained (live) identity
# ---------------------------------------------------------------------------


def test_a_successor_dispatch_waits_out_a_retained_identity_then_recovers(
    board, monkeypatch, procs,
):
    """The retained-PID side: A's late publication could not prove its child
    gone, so the identity stays on the fence and the archive refuses. A
    successor dispatch must hold that identity in place — not overwrite it, not
    spawn beside it — and the card must dispatch only once the exact
    ``(pid, start fingerprint)`` is proven gone, with the release recorded."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="held by a retained identity", assignee="worker")

    barrier = _PublicationBarrier()
    real_set_pid, real_terminate = kbd._set_worker_pid, kbd._terminate_reclaimed_worker
    monkeypatch.setattr(kbd, "_set_worker_pid", barrier.wrap(kbd._set_worker_pid))
    # The stop path cannot prove the child gone, so its identity is retained.
    monkeypatch.setattr(
        kbd, "_terminate_reclaimed_worker",
        lambda pid, claim_lock, **kw: {"termination_attempted": False, "terminated": False},
    )
    thread, box = _dispatch_async(_child_spawner(procs))
    pid = barrier.wait_parked()
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
    barrier.release.set()
    _join(thread, box)
    assert box["res"].discarded_spawns == [(tid, pid)]
    assert _pid_running(pid), "the survivor must be alive for this case"

    with kbc.connect_closing() as conn:
        fence = kb.spawn_fence(conn, tid)
        assert fence is not None and fence["pid"] == pid and fence["started_at"]
        run_a = fence["run"]
        refused, why = kb.archive_task(conn, tid, with_reason=True)
        assert refused is False and str(pid) in why and "has a spawned worker" in why, why
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid) is True
        assert kb.spawn_fence(conn, tid)["pid"] == pid, \
            "unblock must not drop the retained identity"

    # Restore the real probes: the successor's own spawn must publish normally.
    monkeypatch.setattr(kbd, "_set_worker_pid", real_set_pid)
    monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", real_terminate)

    spawn_b_calls: list[str] = []
    spawner_b = _child_spawner(procs)

    def spawn_b(task, workspace, *, board=None):
        spawn_b_calls.append(task.id)
        return spawner_b(task, workspace, board=board)

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn_b)
    assert spawn_b_calls == [], "a live retained identity must hold the successor back"
    assert (tid, "spawn_fence_hold") in res.respawn_guarded
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, claim_lock, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready" and row["claim_lock"] is None
        fence = kb._json_dict(row["spawn_fence"])
        assert fence["pid"] == pid, "the old identity must stay attached to the fence"

    # The exact fingerprint proves gone: the hold releases truthfully and the
    # successor finally dispatches.
    procs[0].kill()
    procs[0].wait(timeout=10)
    assert _wait_until(lambda: not _pid_running(pid))
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn_b)
    assert spawn_b_calls == [tid], "exactly one successor spawn, and only after the release"
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"
        new_pid = int(row["worker_pid"])
        assert new_pid == procs[1].pid
        assert row["spawn_fence"] is None, "the successor's own fence settled on publication"

        release = _event(conn, tid, "spawn_fence_released")
        assert release is not None
        assert release.payload["pid"] == pid
        assert release.payload["started_at"] and release.payload["claim"]
        assert release.payload["run"] == run_a
        # Not silently cleared: the old identity still names its own run row.
        run_row = conn.execute(
            "SELECT worker_pid, worker_started_at FROM task_runs WHERE id = ?", (run_a,),
        ).fetchone()
        assert int(run_row["worker_pid"]) == pid and run_row["worker_started_at"]
        assert _event(conn, tid, "spawn_discarded").payload["pid"] == pid
        assert _event(conn, tid, "spawned").payload["pid"] == new_pid
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: not _pid_running(new_pid))


# ---------------------------------------------------------------------------
# 9. a claim that cannot arm its fence is unwound, never stranded
# ---------------------------------------------------------------------------


def test_a_claim_that_cannot_arm_its_fence_is_unwound(board, monkeypatch, procs):
    """When the hold appears in the window BETWEEN a successor's claim and its
    fence arm, the claim must not be left owning a ``running`` row with no
    child: it is unwound atomically back to ``ready`` with truthful
    provenance, the predecessor's hold stays untouched, and the card spawns
    only once that hold has settled."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="unwind the unarmed claim", assignee="worker")

    real_claim, vanished_claim, vanished_run = kb.claim_task, "vanished:1", 424242
    injected = {"done": False}

    def claim_then_arm_predecessor(conn, task_id, **kw):
        claimed = real_claim(conn, task_id, **kw)
        if claimed is not None and task_id == tid and not injected["done"]:
            # A predecessor arms its hold in the claim -> fence-arm window
            # (once: the later, legitimate claim must find a settled card).
            injected["done"] = True
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET spawn_fence = ? WHERE id = ?", (
                    kb._spawn_fence_payload(vanished_claim, vanished_run), tid))
        return claimed

    monkeypatch.setattr(kb, "claim_task", claim_then_arm_predecessor)

    calls: list[str] = []
    spawner = _child_spawner(procs)

    def spawn_fn(task, workspace, *, board=None):
        calls.append(task.id)
        return spawner(task, workspace, board=board)

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn)

        assert calls == [], "the refused spawn must start no child"
        assert res.spawned == [] and res.auto_blocked == [] and res.discarded_spawns == []
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id, spawn_fence, "
            "       consecutive_failures, last_failure_error "
            "FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready", "no unowned `running` card may be left behind"
        assert row["claim_lock"] is None and row["current_run_id"] is None
        fence = kb._json_dict(row["spawn_fence"])
        assert fence["claim"] == vanished_claim and fence.get("pid") is None, \
            "the predecessor's hold must be untouched"
        assert row["consecutive_failures"] == 0 and row["last_failure_error"] is None, \
            "nothing ran, so the card's retry budget and error are untouched"
        run_row = conn.execute(
            "SELECT id, outcome, ended_at, worker_pid FROM task_runs "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        assert run_row["outcome"] == "spawn_refused" and run_row["ended_at"] is not None
        assert run_row["worker_pid"] is None
        refused = _event(conn, tid, "spawn_refused")
        assert refused is not None and refused.run_id == run_row["id"]
        assert refused.payload["spawn_fence"]["claim"] == vanished_claim
        assert refused.payload["retry_status"] == "ready"
        assert "no child was started" in refused.payload["error"]
        assert _event(conn, tid, "spawned") is None
        assert _event(conn, tid, "gave_up") is None

        # Still held on the next tick — visibly, and without a second claim.
        res2 = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert (tid, "spawn_fence_hold") in res2.respawn_guarded
        assert calls == []
        assert kb.get_task(conn, tid).status == "ready"

        # The predecessor's own spawn settles (it never got a PID), released by
        # ITS claim — then the card dispatches normally.
        assert kb._release_spawn_fence(conn, tid, vanished_claim, vanished_run) is True
        res3 = kbd.dispatch_once(conn, spawn_fn=spawn_fn)

    assert [(t, a) for t, a, _ws in res3.spawned] == [(tid, "worker")]
    assert len(procs) == 1
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"
        assert int(row["worker_pid"]) == procs[0].pid
        assert row["spawn_fence"] is None
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: not _pid_running(procs[0].pid))


# ---------------------------------------------------------------------------
# 10. the primitive: every foreign fence is an exclusive hold
# ---------------------------------------------------------------------------


def test_hold_spawn_fence_is_exclusive_for_every_foreign_fence(board, procs):
    """``_hold_spawn_fence`` itself: a fence naming ANOTHER claim/run is
    exclusive whether or not it has published a PID, it is never overwritten,
    and it is cleared only when its retained identity can be proven gone —
    recorded, never dropped silently."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="exclusive hold", assignee="worker")
        claimer = kb._claimer_id().split(":", 1)[0]
        first = kb.claim_task(conn, tid, claimer=f"{claimer}:one")
        lock_a, run_a = first.claim_lock, first.current_run_id
        assert kb._hold_spawn_fence(conn, tid, lock_a, run_a) is True
        armed = conn.execute("SELECT spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()[0]

        assert kb.block_task(conn, tid, reason="operator stop", with_reason=True)[0] is True
        assert kb.unblock_task(conn, tid) is True
        second = kb.claim_task(conn, tid, claimer=f"{claimer}:two")
        lock_b, run_b = second.claim_lock, second.current_run_id
        assert (lock_b, run_b) != (lock_a, run_a)

        # (a) unresolved — no PID published: exclusive, never overwritten.
        assert kb._hold_spawn_fence(conn, tid, lock_b, run_b) is False
        assert conn.execute(
            "SELECT spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == armed

        # (b) retained identity still alive: exclusive, and not released either.
        own_fingerprint = kbd._process_fingerprint(os.getpid())
        assert own_fingerprint
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET spawn_fence = ? WHERE id = ?", (
                kb._spawn_fence_payload(lock_a, run_a, pid=os.getpid(),
                                        started_at=own_fingerprint), tid))
        held = conn.execute("SELECT spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert kb._hold_spawn_fence(conn, tid, lock_b, run_b) is False
        assert conn.execute(
            "SELECT spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == held

        # (c) that identity proven gone: released truthfully, and only then may
        # the successor arm its own hold.
        dead = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        procs.append(dead)
        dead_fingerprint = kbd._process_fingerprint(dead.pid)
        dead.kill()
        dead.wait(timeout=10)
        assert not _pid_running(dead.pid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET spawn_fence = ? WHERE id = ?", (
                kb._spawn_fence_payload(lock_a, run_a, pid=dead.pid,
                                        started_at=dead_fingerprint), tid))
        assert kb._hold_spawn_fence(conn, tid, lock_b, run_b) is True
        now_fence = kb.spawn_fence(conn, tid)
        assert now_fence["claim"] == lock_b and now_fence["run"] == run_b
        released = _event(conn, tid, "spawn_fence_released")
        assert released is not None
        assert released.payload["pid"] == dead.pid
        assert released.payload["started_at"] == dead_fingerprint
        assert released.payload["claim"] == lock_a and released.payload["run"] == run_a
        assert released.payload["reason"] and "gone" in released.payload["reason"]


# ---------------------------------------------------------------------------
# 10. worker-side self-registration (upstream ``adopt_worker_pid``) settles the fence
# ---------------------------------------------------------------------------


def _armed_spawn(board, monkeypatch, *, foreign_claim=None):
    """A ``running`` card whose dispatcher armed an in-flight spawn fence and then
    died before publishing the PID — the exact upstream scenario
    ``adopt_worker_pid`` exists for. Returns ``(tid, claim_lock, run_id, pid)``."""
    host = kb._claimer_id().split(":", 1)[0]
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="self-registered", assignee="worker")
        kb.claim_task(conn, tid, claimer=f"{host}:worker")
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        claim, run_id = task.claim_lock, int(task.current_run_id)
        assert claim and claim.startswith(kb._host_prefix())
        assert kb._hold_spawn_fence(conn, tid, claim, run_id) is True
        if foreign_claim is not None:
            # Simulate a predecessor's hold that this spawn must never clobber.
            conn.execute("UPDATE tasks SET spawn_fence = ? WHERE id = ?",
                         (kb._spawn_fence_payload(foreign_claim, run_id - 1), tid))
            conn.commit()
        assert kb.spawn_fence(conn, tid) is not None
    # A pid that is definitely not this process and not alive.
    pid = 2 ** 22 - 1
    assert not kbd._worker_alive(pid, None)
    return tid, claim, run_id, pid


def test_adopt_worker_pid_settles_its_own_spawn_fence(board, monkeypatch):
    """Upstream lets a worker register its own PID when the dispatcher died
    between spawn and ``_set_worker_pid``. That self-registration IS this spawn's
    publication, so it must also settle the in-flight spawn fence the dispatcher
    armed before starting the child — otherwise the hold stays armed forever and
    the card can never be archived or re-dispatched."""
    tid, claim, run_id, pid = _armed_spawn(board, monkeypatch)

    with kbc.connect_closing() as conn:
        assert kbd.adopt_worker_pid(conn, tid, run_id, pid) is True
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["worker_pid"] == pid, "the self-registration must still land"
        assert row["spawn_fence"] is None, "publication settles the in-flight hold"
        assert _event(conn, tid, "worker_registered") is not None
        assert kb.spawn_fence(conn, tid) is None

    # The card is no longer held: a later block + archive runs the ordinary path.
    with kbc.connect_closing() as conn:
        ok, why = kb.block_task(conn, tid, reason="operator stop", with_reason=True)
        assert ok is True, why
        ok, why = kb.archive_task(conn, tid, with_reason=True)
        assert ok is True, why
        assert kb.get_task(conn, tid).status == "archived"


def test_adopt_worker_pid_never_clears_a_foreign_spawn_fence(board, monkeypatch):
    """The self-registration only settles a fence naming ITS claim/run: a
    predecessor's unresolved hold keeps the card unarchivable, exactly as the
    dispatcher-side rules require."""
    foreign = "some-other-claim-token"
    tid, claim, run_id, pid = _armed_spawn(board, monkeypatch, foreign_claim=foreign)

    with kbc.connect_closing() as conn:
        assert kbd.adopt_worker_pid(conn, tid, run_id, pid) is True
        fence = kb.spawn_fence(conn, tid)
        assert fence is not None and fence["claim"] == foreign, fence
        row = conn.execute(
            "SELECT worker_pid FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["worker_pid"] == pid

    with kbc.connect_closing() as conn:
        refused, why = kb.archive_task(conn, tid, with_reason=True)
        # Status guard first (the card is still ``running``); the foreign hold
        # itself is what keeps the card held after the worker is stopped.
        assert refused is False, why
        assert kb.get_task(conn, tid).status == "running"
        assert kb.spawn_fence(conn, tid)["claim"] == foreign


# ---------------------------------------------------------------------------
# 11. the spawner died before it could publish: settle on evidence, not on age
# ---------------------------------------------------------------------------

_ARM_AND_CLAIM = """
import sys, time
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

tid, ttl = sys.argv[1], int(sys.argv[2])
with kbc.connect_closing() as conn:
    claimed = kb.claim_task(conn, tid, ttl_seconds=ttl)
    if claimed is None:
        sys.exit("claim failed")
    if not kb._hold_spawn_fence(conn, tid, claimed.claim_lock, claimed.current_run_id):
        sys.exit("fence arm failed")
    print("ARMED %s %s" % (claimed.claim_lock, claimed.current_run_id), flush=True)
time.sleep(600)
"""


def _arm_fence_then_die(procs: list, tid: str, *, ttl: int = 1) -> tuple[subprocess.Popen, str, int]:
    """Claim ``tid`` and arm its spawn fence in a REAL child process, then SIGKILL
    that process exactly where a dispatcher dies: after ``_hold_spawn_fence`` and
    before any child exists or a PID could be published.

    The claim gets a 1 s TTL so the next real tick reclaims it the way
    ``release_stale_claims`` does in production — the fence deliberately survives
    that reclaim, which is the stuck ``ready``-with-an-armed-fence state the
    recovery has to clear. Returns ``(killed process, claim_lock, run_id)``."""
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-c", _ARM_AND_CLAIM, tid, str(ttl)],
        cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    procs.append(proc)
    # ``readline`` blocks; run it on a thread so a child that dies before it
    # prints (closing the pipe) ends the wait instead of hanging the test.
    captured: dict = {}
    reader = threading.Thread(target=lambda: captured.update(line=proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(timeout=60)
    line = (captured.get("line") or "").strip()
    if not line.startswith("ARMED "):
        proc.kill()
        proc.wait(timeout=30)
        raise AssertionError(
            f"the arming child never armed its fence: {line!r} "
            f"stderr={proc.stderr.read()[:4000]!r}")
    _, claim, run_id = line.split()
    proc.kill()
    proc.wait(timeout=30)
    assert not _pid_running(proc.pid), "the spawner must be gone"
    return proc, claim, int(run_id)


def _claim_expired(tid: str) -> bool:
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (tid,)).fetchone()
    return (
        row is not None and row["claim_expires"] is not None
        and int(row["claim_expires"]) < int(time.time())
    )


def _worker_pid_of(tid: str) -> int:
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT worker_pid FROM tasks WHERE id = ?", (tid,)).fetchone()
    return int(row["worker_pid"]) if row and row["worker_pid"] else 0


def test_identityless_fence_verdicts(board):
    """The decision an armed fence with NO published identity may take, asserted
    directly: a live spawner means the spawn is still in flight (hold — its child
    may not exist yet), a dead spawner with no observable child releases, a dead
    spawner whose child IS observable adopts that child's identity, and both
    "this host cannot prove absence" and "this fence predates spawner
    identities" fail closed. Age is deliberately absent from every branch."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="verdicts", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb._hold_spawn_fence(conn, tid, claimed.claim_lock, claimed.current_run_id)
        fence = kb.spawn_fence(conn, tid)
    assert fence["spawned_by"]["pid"] == os.getpid()
    assert kb._identityless_fence_action(fence)[0] == "hold", "our own spawn is still in flight"

    dead_pid = 2 ** 22 - 1
    assert kbd._worker_alive(dead_pid, None) is False
    dead_spawner = dict(fence, spawned_by={"pid": dead_pid, "started_at": None})
    assert kb._identityless_fence_action(dead_spawner)[0] == "release", \
        "no child of a dead spawner exists"

    with pytest.MonkeyPatch.context() as probe_off:
        probe_off.setattr(kb, "_observe_spawn_child", lambda f: (False, None))
        assert kb._identityless_fence_action(dead_spawner)[0] == "hold", \
            "without a probe, absence proves nothing"

    # ``claim_lock`` is the CLAIMER (host:pid) and is shared by every card this
    # dispatcher claimed: a sibling spawn carrying our claim but another run is
    # NOT our child and must never be adopted — only the run makes it ours.
    sibling = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, "HERMES_KANBAN_CLAIM_LOCK": fence["claim"],
             "HERMES_KANBAN_RUN_ID": str(int(fence["run"]) + 1)})
    try:
        assert kb._identityless_fence_action(dead_spawner)[0] == "release", \
            "a sibling spawn of the same claimer is not this spawn's child"
    finally:
        sibling.kill()
        sibling.wait(timeout=30)

    # A live child of that dead spawner is what keeps the hold and gets adopted.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, "HERMES_KANBAN_CLAIM_LOCK": fence["claim"],
             "HERMES_KANBAN_RUN_ID": str(fence["run"])})
    try:
        action, found = kb._identityless_fence_action(dead_spawner)
        assert (action, found[0] if found else None) == ("adopt", child.pid), (action, found)
    finally:
        child.kill()
        child.wait(timeout=30)

    legacy = {"claim": "a-claim-from-before-spawner-identities", "run": 1, "at": int(time.time())}
    assert kb._identityless_fence_action(legacy)[0] == "hold", \
        "an unknown spawner is never assumed gone"

    with kbc.connect_closing() as conn:
        assert kb._release_spawn_fence(conn, tid, claimed.claim_lock, claimed.current_run_id)
        assert kb.reclaim_task(conn, tid) is True
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
        assert kb.archive_task(conn, tid, with_reason=True)[0] is True


def test_a_killed_spawner_releases_its_identityless_fence_and_recovers_one_worker(
    board, procs,
):
    """THE demonstrated liveness defect: the dispatcher SIGKILLed after arming the
    fence and before publishing a PID used to leave the card permanently ``ready``
    behind a NULL-PID fence — every later tick reported ``respawn_guarded`` /
    ``spawn_fence_hold`` and nothing ever settled the hold. Recovery must come
    from evidence (spawner provably gone, no live child of that spawn) rather
    than from an age threshold, and must yield exactly ONE worker."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="killed before publication", assignee="worker")

    killed, claim, run_id = _arm_fence_then_die(procs, tid, ttl=1)

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        fence = kb._json_dict(row["spawn_fence"])
        assert row["status"] == "running" and row["worker_pid"] is None
        assert fence.get("pid") is None, "the dispatcher died before publishing"
        assert fence["claim"] == claim and fence["run"] == run_id
        assert fence["spawned_by"]["pid"] == killed.pid, "the fence names the spawner that armed it"
        assert kb._identityless_fence_action(fence)[0] == "release"

    spawn_calls: list[str] = []
    spawner = _child_spawner(procs)

    def spawn(task, workspace, *, board=None):
        spawn_calls.append(task.id)
        return spawner(task, workspace, board=board)

    assert _wait_until(lambda: _claim_expired(tid), 30), "the dead spawner's claim never expired"
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert res.reclaimed >= 1, "the dead spawner's claim must be reclaimed first"
    assert spawn_calls == [tid], f"the card must dispatch exactly once, got {spawn_calls}"
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]
    assert (tid, "spawn_fence_hold") not in res.respawn_guarded, res.respawn_guarded
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, worker_pid, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running" and row["spawn_fence"] is None
        assert int(row["worker_pid"]) == procs[-1].pid
        released = _event(conn, tid, "spawn_fence_released")
        assert released is not None, "the settlement is recorded, never a silent clear"
        assert "spawner died" in released.payload["reason"], released.payload
        assert released.payload["spawner"]["pid"] == killed.pid
        assert "pid" not in released.payload, "nothing was ever held but the spawner's identity"
        assert released.payload["claim"] == claim and released.payload["run"] == run_id
        assert len([e for e in _events(conn, tid) if e.kind == "spawned"]) == 1

    # Later ticks must not start a second worker beside the live one.
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=spawn)
        kbd.dispatch_once(conn, spawn_fn=spawn)
    assert spawn_calls == [tid], f"a duplicate worker was spawned: {spawn_calls}"
    assert _worker_pid_of(tid) == procs[-1].pid

    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: all(not _pid_running(p.pid) for p in procs), 30), "workers left running"


def test_a_live_unreported_child_keeps_the_hold_and_prevents_a_duplicate_spawn(
    board, procs,
):
    """The other half of the same window: the spawner died AFTER it started a
    child, which is alive but has not published its PID yet. Age cannot tell that
    apart from "no child was ever started", so the recovery must look: the child
    is found on this host, its identity is attached to the fence, and the card
    keeps holding instead of spawning a second worker beside it. Only once that
    child is gone may the hold release and exactly one successor dispatch."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="live child, dead spawner", assignee="worker")

    killed, claim, run_id = _arm_fence_then_die(procs, tid, ttl=1)
    # The child this spawn DID start: alive, unreported (never published a PID).
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, "HERMES_KANBAN_CLAIM_LOCK": claim,
             "HERMES_KANBAN_RUN_ID": str(run_id)},
    )
    procs.append(child)
    assert _pid_running(child.pid)

    with kbc.connect_closing() as conn:
        fence = kb.spawn_fence(conn, tid)
        assert fence.get("pid") is None
    action, found = kb._identityless_fence_action(fence)
    assert (action, found[0] if found else None) == ("adopt", child.pid), (action, found)

    spawn_calls: list[str] = []
    spawner = _child_spawner(procs)

    def spawn(task, workspace, *, board=None):
        spawn_calls.append(task.id)
        return spawner(task, workspace, board=board)

    assert _wait_until(lambda: _claim_expired(tid), 30), "the dead spawner's claim never expired"
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert spawn_calls == [], f"a live unreported child must never be spawned beside: {spawn_calls}"
    assert (tid, "spawn_fence_hold") in res.respawn_guarded
    with kbc.connect_closing() as conn:
        fence = kb.spawn_fence(conn, tid)
        assert fence["pid"] == child.pid, "the live child's identity is now what holds the card"
        assert fence["started_at"] and fence["claim"] == claim and fence["run"] == run_id
        assert fence["spawned_by"]["pid"] == killed.pid
        assert _event(conn, tid, "spawn_fence_released") is None, "nothing was released yet"

    # The unreported child goes away: only now does the hold settle and the card
    # dispatch — once.
    child.kill()
    child.wait(timeout=30)
    assert _wait_until(lambda: not _pid_running(child.pid), 15)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert spawn_calls == [tid], f"exactly one successor spawn, got {spawn_calls}"
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]
    with kbc.connect_closing() as conn:
        released = _event(conn, tid, "spawn_fence_released")
        assert released is not None
        assert released.payload["pid"] == child.pid, released.payload
        assert released.payload["started_at"] and "gone" in released.payload["reason"]
        assert _worker_pid_of(tid) == procs[-1].pid
        assert kb.spawn_fence(conn, tid) is None
        assert len([e for e in _events(conn, tid) if e.kind == "spawned"]) == 1
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: all(not _pid_running(p.pid) for p in procs), 30), "workers left running"


def test_a_reclaim_alone_never_frees_a_fence_whose_spawner_is_still_alive(
    board, procs,
):
    """The "not yet started child" side: the claim behind an armed fence is
    reclaimed (or expires) while the dispatcher that armed it is still alive and
    has not reached its spawn call. Nothing may settle that hold — there is no
    child to find and the spawn can still create one — so the card stays
    ``ready`` under ``spawn_fence_hold`` instead of racing a second child onto
    it; only the owner's own settlement frees the card."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="spawner still spawning", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb._hold_spawn_fence(conn, tid, claimed.claim_lock, claimed.current_run_id)
        fence_before = conn.execute(
            "SELECT spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert kb._json_dict(fence_before)["spawned_by"]["pid"] == os.getpid()
        # Reclaim the claim only — the fence deliberately outlives it.
        assert kb.reclaim_task(conn, tid) is True
        row = conn.execute(
            "SELECT status, claim_lock, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready" and row["claim_lock"] is None
        assert row["spawn_fence"] == fence_before, "reclaim must not drop the hold"

    spawn_calls: list[str] = []

    def spawn(task, workspace, *, board=None):
        spawn_calls.append(task.id)
        return _child_spawner(procs)(task, workspace, board=board)

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert spawn_calls == [], f"the spawn in flight must not be raced: {spawn_calls}"
    assert (tid, "spawn_fence_hold") in res.respawn_guarded
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, spawn_fence FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready" and row["spawn_fence"] == fence_before
        # The owner settles its own never-started spawn, then the card dispatches.
        assert kb._release_spawn_fence(conn, tid, claimed.claim_lock, claimed.current_run_id) is True
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert spawn_calls == [tid], f"exactly one spawn after the owner settled: {spawn_calls}"
    assert [(t, a) for t, a, _ws in res.spawned] == [(tid, "worker")]
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="teardown", with_reason=True)[0] is True
    assert _wait_until(lambda: all(not _pid_running(p.pid) for p in procs), 30), "workers left running"
