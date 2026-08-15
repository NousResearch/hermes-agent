"""Regression tests for claim-owned kanban completion."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_complete_task_rejects_matching_claim_lock_in_triage(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        assert not kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"


def test_complete_task_matching_claim_lock_preserves_non_completable_state(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="human-routed work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET status = 'todo' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        # A matching lock must not bypass the set of completable task states.
        assert not kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "todo"
        assert task.result is None
        assert task.claim_lock == "worker:current"


def test_complete_task_rejects_stale_claim_lock(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="reassigned work", assignee="worker")
        first = kb.claim_task(conn, task_id, claimer="worker:first")
        assert first is not None
        assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_args: None)
        second = kb.claim_task(conn, task_id, claimer="worker:second")
        assert second is not None

        assert not kb.complete_task(
            conn,
            task_id,
            result="stale result",
            expected_claim_lock="worker:first",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.result is None
        assert task.claim_lock == "worker:second"
        assert task.current_run_id == second.current_run_id


def test_complete_task_matching_claim_lock_retry_is_idempotent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="retry completion", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:retry")
        assert claimed is not None

        assert kb.complete_task(
            conn,
            task_id,
            result="original",
            summary="original summary",
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="worker:retry",
        )
        assert kb.complete_task(
            conn,
            task_id,
            result="replacement",
            summary="replacement summary",
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="worker:retry",
        )

        task = kb.get_task(conn, task_id)
        completed = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ]
        assert task is not None
        assert task.result == "original"
        assert len(completed) == 1


def test_complete_task_without_claim_lock_preserves_legacy_behavior(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="manual completion")

        assert kb.complete_task(conn, task_id, result="done")

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "done"


@pytest.mark.parametrize("status", ["running", "ready", "blocked"])
def test_complete_task_accepts_matching_claim_lock_on_completable_status(
    kanban_home: Path, status: str
) -> None:
    """The happy path of the fenced branch: right lock, every completable status.

    The rejection tests only pin what the fence rejects. Parametrized over the
    full IN list because a single positive case (say ``running``) would stay
    green if the predicate silently dropped ``ready`` or ``blocked``.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()

        assert kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "finished"
        assert task.claim_lock is None


def test_complete_task_rejects_inherited_claim_lock_from_another_pid(
    kanban_home: Path,
) -> None:
    """A nested CLI inherits the claim lock but not the worker's identity.

    Reported on NousResearch/hermes-agent#71175: a nested Hermes CLI inherits
    HERMES_KANBAN_TASK and HERMES_KANBAN_CLAIM_LOCK from its parent, so it
    presents a claim lock that matches and completes the parent's card from a
    different pid. Every other component of the identity travels down to a
    child the same way, which is why the pid of the calling process is the one
    that can tell them apart.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="worker task", assignee="worker")
        claim = kb.claim_task(conn, task_id, claimer="worker:live")
        assert claim is not None

        # claim_task does not stamp the pid; the dispatcher does it separately
        # via _set_worker_pid, so a test that wants a live worker has to say so.
        live_pid = 424242
        kb._set_worker_pid(conn, task_id, live_pid)

        assert not kb.complete_task(
            conn,
            task_id,
            result="completed by a nested CLI",
            expected_claim_lock="worker:live",
            expected_worker_pid=live_pid + 1,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.result is None
        assert task.claim_lock == "worker:live"
        assert task.worker_pid == live_pid


def test_complete_task_accepts_matching_worker_pid(kanban_home: Path) -> None:
    """The real worker still completes its own card.

    The bypass test above passes for a gate that refuses everything, so this is
    the half that keeps it honest.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="worker task", assignee="worker")
        claim = kb.claim_task(conn, task_id, claimer="worker:live")
        assert claim is not None

        # claim_task does not stamp the pid; the dispatcher does it separately
        # via _set_worker_pid, so a test that wants a live worker has to say so.
        live_pid = 424242
        kb._set_worker_pid(conn, task_id, live_pid)

        assert kb.complete_task(
            conn,
            task_id,
            result="completed by its own worker",
            expected_claim_lock="worker:live",
            expected_worker_pid=live_pid,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "completed by its own worker"


def test_tool_complete_rejects_inherited_claim_lock_from_another_pid(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent tool path must carry the caller's pid, not just the claim lock.

    This is the path the bypass on #71175 travels: a nested CLI inherits
    HERMES_KANBAN_TASK and HERMES_KANBAN_CLAIM_LOCK and completes its parent's
    card. A database that refuses a mismatched pid proves nothing if the tool
    never sends one, so this pins the wiring rather than the check.
    """
    from tools import kanban_tools

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="worker task", assignee="worker")
        claim = kb.claim_task(conn, task_id, claimer="worker:live")
        assert claim is not None
        live_pid = 424242
        kb._set_worker_pid(conn, task_id, live_pid)

    # The nested process inherits the parent's task and claim lock verbatim.
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "worker:live")
    # ...but runs under its own pid, which is the one thing it cannot inherit.
    monkeypatch.setattr(kanban_tools.os, "getpid", lambda: live_pid + 1)

    kanban_tools._handle_complete({"id": task_id, "result": "from a nested CLI"})

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert task.result is None
    assert task.worker_pid == live_pid


def test_complete_task_allows_worker_whose_spawn_reported_no_pid(
    kanban_home: Path,
) -> None:
    """A deployment that never stamps a pid must keep working.

    Reporting a pid from spawn_fn is a crash-detection nicety rather than a
    contract, so a custom spawn that returns none leaves worker_pid NULL
    forever. Tightening the fence to refuse those rejected the legitimate
    worker finishing its own task — caught in review after the first attempt
    did exactly that. Where no pid was recorded this fence is no weaker than
    before; where one was, it is strictly stronger.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="pidless worker", assignee="worker")
        claim = kb.claim_task(conn, task_id, claimer="worker:live")
        assert claim is not None
        assert kb.get_task(conn, task_id).worker_pid is None

        assert kb.complete_task(
            conn,
            task_id,
            result="completed without a recorded pid",
            expected_claim_lock="worker:live",
            expected_worker_pid=4242,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
