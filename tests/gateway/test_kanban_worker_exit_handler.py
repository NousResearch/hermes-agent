"""Run-side exit handler: a clean rc=0 worker exit with no terminal kanban call books one.

Campaign memory-architecture card t_626fecff: the worker runtime synthesizes the missing
terminal call on a clean exit — completion (artifacts listed) if the run left evidence of
work, one blocked event (exact reason string) if it produced nothing — instead of leaving
the dispatcher to burn three retry slots classifying protocol violations.
"""
import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_worker_exit as kexit
from hermes_cli.kanban_db_connect import connect_closing


def _make_claimed_task(tmp_path, monkeypatch, *, workspace_kind="scratch",
                       completion_contract="local-only"):
    """Fresh HERMES_HOME with one claimed (running) task; returns (db_path, context)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(
            conn, title="exit-handler fixture", assignee="default",
            workspace_kind=workspace_kind, completion_contract=completion_contract,
        )
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        assert task is not None
        # Scratch workspaces live at <workspaces_root>/<task_id>/ (resolve_workspace
        # convention); the DB workspace_path column is NULL for scratch tasks.
        ws_root = kb.workspaces_root()
        workspace = ws_root / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        context = {
            "db": str(path),
            "task_id": task_id,
            "run_id": task.current_run_id,
            "claim_lock": task.claim_lock,
            "workspace": str(workspace),
        }
    return path, context


def _task_row(path, task_id):
    with connect_closing(path) as conn:
        row = conn.execute(
            "SELECT status, current_run_id, result FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return dict(row) if row else None


def _block_events(path, task_id):
    with connect_closing(path) as conn:
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (task_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]


def test_scratch_artifacts_trigger_completion(tmp_path, monkeypatch):
    """A clean rc=0 exit after real work (non-empty scratch file) books a completion with artifacts."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    artifact = os.path.join(context["workspace"], "report.md")
    with open(artifact, "w") as f:
        f.write("# evidence of work\n")

    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "completed"

    row = _task_row(path, context["task_id"])
    assert row["status"] == "done"
    assert row["current_run_id"] is None  # fence released the claim
    assert "report.md" in (row["result"] or "")
    assert "[exit-handler]" in (row["result"] or "")


def test_no_evidence_blocks_once_with_exact_reason(tmp_path, monkeypatch):
    """A clean rc=0 exit with no work books ONE blocked event with the card's exact reason string."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    # scratch workspace stays empty -> no artifact evidence

    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "blocked"

    row = _task_row(path, context["task_id"])
    assert row["status"] == "blocked"
    events = _block_events(path, context["task_id"])
    assert len(events) == 1
    assert events[0]["reason"] == kexit.BLOCK_REASON_NO_ARTIFACTS


def test_double_fire_books_once(tmp_path, monkeypatch):
    """Idempotency: the exit handler fired twice (managed finally + one-shot exit) books one result."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    artifact = os.path.join(context["workspace"], "out.txt")
    with open(artifact, "w") as f:
        f.write("x\n")

    first = kexit.synthesize_missing_terminal_call(context, 0)
    second = kexit.synthesize_missing_terminal_call(context, 0)
    assert first == "completed"
    assert second == "skipped_superseded"  # run no longer current/running after the fence

    row = _task_row(path, context["task_id"])
    assert row["status"] == "done"


def test_real_terminal_call_makes_synthesis_noop(tmp_path, monkeypatch):
    """If the worker already called kanban_complete, the exit handler is a no-op (SQL fence)."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    with connect_closing(path) as conn:
        kb.complete_task(
            conn, context["task_id"], result="worker completed normally",
            summary="done", expected_run_id=context["run_id"],
        )

    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "skipped_superseded"

    row = _task_row(path, context["task_id"])
    assert row["status"] == "done"
    assert row["result"] == "worker completed normally"  # untouched


def test_contract_cards_are_never_synthesized(tmp_path, monkeypatch):
    """A completion_contract card (github-pr acceptance) is left to its contract path."""
    path, context = _make_claimed_task(
        tmp_path, monkeypatch, workspace_kind="worktree", completion_contract="octocat/hello-world",
    )
    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "skipped_contract"
    row = _task_row(path, context["task_id"])
    assert row["status"] == "running"  # untouched


def test_nonzero_exit_is_skipped(tmp_path, monkeypatch):
    """Non-zero exits belong to the dispatcher's failure machinery, never this handler."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    outcome = kexit.synthesize_missing_terminal_call(context, 1)
    assert outcome == "skipped_nonzero"
    row = _task_row(path, context["task_id"])
    assert row["status"] == "running"  # untouched


def test_goal_steps_count_as_evidence(tmp_path, monkeypatch):
    """A goal run with completed_steps in task_runs metadata completes even with no workspace files."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    with connect_closing(path) as conn, kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"completed_steps": ["step 1: read schema", "step 2: wrote design"]}),
             context["run_id"]),
        )

    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "completed"
    row = _task_row(path, context["task_id"])
    assert row["status"] == "done"
    assert "step 1" in (row["result"] or "")


def test_env_variant_reads_documented_contract(tmp_path, monkeypatch):
    """synthesize_from_env builds the context from HERMES_KANBAN_* env vars."""
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="scratch")
    artifact = os.path.join(context["workspace"], "env-artifact.bin")
    with open(artifact, "wb") as f:
        f.write(b"\x00" * 8)
    monkeypatch.setenv("HERMES_KANBAN_TASK", context["task_id"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(context["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", context["workspace"])

    outcome = kexit.synthesize_from_env(0)
    assert outcome == "completed"
    row = _task_row(path, context["task_id"])
    assert row["status"] == "done"
    assert "env-artifact.bin" in (row["result"] or "")


def test_env_variant_without_contract_vars_is_skipped(tmp_path, monkeypatch):
    """No HERMES_KANBAN_TASK/RUN_ID -> the one-shot path is a silent no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert kexit.synthesize_from_env(0) == "skipped_env"


def test_worktree_pristine_checkout_is_not_evidence(tmp_path, monkeypatch):
    """A worktree with no uncommitted changes is NOT evidence of work -> blocked, not completed."""
    import subprocess
    path, context = _make_claimed_task(tmp_path, monkeypatch, workspace_kind="worktree")
    # Initialize a pristine git checkout inside the worktree.
    ws = context["workspace"]
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, check=True)
    with open(os.path.join(ws, "tracked.txt"), "w") as f:
        f.write("initial\n")
    subprocess.run(["git", "add", "."], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=ws, check=True)

    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "blocked"  # pristine checkout = no uncommitted evidence

    # Now add an uncommitted edit -> evidence of work. Reset the row to a fresh
    # running state (fixture shortcut: after block the card is 'blocked', and
    # claim_task only claims 'ready'; we just need a live current_run_id fence).
    task_id = context["task_id"]
    with connect_closing(path) as conn, kb.write_txn(conn, allow_nested=True):
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, started_at) "
            "VALUES (?, 'default', 'running', 'reclaim', 9999999999, strftime('%s','now'))",
            (task_id,),
        )
        new_run = cur.lastrowid
        conn.execute(
            "UPDATE tasks SET status='running', current_run_id=?, claim_lock='reclaim' WHERE id=?",
            (new_run, task_id),
        )
    context = {**context, "run_id": new_run, "claim_lock": "reclaim"}
    with open(os.path.join(ws, "edit.txt"), "w") as f:
        f.write("uncommitted work\n")
    outcome = kexit.synthesize_missing_terminal_call(context, 0)
    assert outcome == "completed"
    row = _task_row(path, task_id)
    assert row["status"] == "done"
    assert "edit.txt" in (row["result"] or "")
