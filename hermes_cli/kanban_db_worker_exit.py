"""Run-side half of the terminal-call contract: synthesize the missing call on a clean exit.

Dispatch-side classification of ``rc=0`` exits with no terminal kanban call already
exists upstream (``protocol_violation`` outcome + gave_up payload in
``kanban_db_dispatch``); this module is the *other* direction from the campaign
card — the worker runtime itself fires an exit handler so a run that did real
work but never called ``kanban_complete``/``kanban_block`` still lands as
``done`` (artifacts listed) instead of burning retry slots as a protocol
violation, and a run that produced nothing lands as ``blocked`` once instead
of three crashed retries.

Idempotency is structural: both ``complete_task`` and ``block_task`` fence on
``current_run_id`` (``expected_run_id``) and only fire from ``running``, so a
real terminal call that already happened makes the synthesis a no-op, and a
double fire (managed ``finally`` + one-shot exit) can only ever book once.

Evidence rules (deliberately conservative — never invent work):
- workspace ``scratch``: any non-empty file under the workspace (it starts empty).
- workspace ``worktree``: ``git status --porcelain`` non-empty (edits); a pristine
  checkout is NOT evidence.
- goal runs: ``task_runs.metadata['completed_steps']`` non-empty.
Cards with a ``completion_contract`` (github-pr acceptance) are never synthesized —
those must complete through the PR acceptance path, never an exit handler.

Wire points: the managed owner path (``_run_worker_turns`` finally) and the quiet
one-shot CLI (``exit_single_query``); a contract card or a superseded run is left
for the dispatcher's existing failure machinery.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli import kanban_db as kb

# Exact reason string from the campaign card (memory-architecture t_626fecff).
BLOCK_REASON_NO_ARTIFACTS = (
    "rc=0 exit with no terminal kanban call and no output artifacts -- worker produced nothing"
)

_LISTING_CAP = 50
_SCAN_CAP = 500


def _run_row(conn, run_id: int):
    return conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ?", (int(run_id),)
    ).fetchone()


def _goal_steps(run_row) -> list[str]:
    """Completed steps recorded by the goal loop, if any."""
    if run_row is None:
        return []
    raw = run_row["metadata"]
    if not raw:
        return []
    try:
        meta = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):
        return []
    steps = meta.get("completed_steps") if isinstance(meta, dict) else None
    if isinstance(steps, str):
        return [steps] if steps.strip() else []
    if isinstance(steps, (list, tuple)):
        return [str(s) for s in steps if str(s).strip()]
    return []


def _scratch_artifacts(workspace: Path) -> list[tuple[str, int]]:
    """Non-empty files under a scratch workspace (it starts empty, so any file is work)."""
    found: list[tuple[str, int]] = []
    scanned = 0
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part == ".git" for part in path.relative_to(workspace).parts):
            continue
        scanned += 1
        if scanned > _SCAN_CAP:
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 0:
            found.append((str(path.relative_to(workspace)), size))
    return found


def _worktree_artifacts(workspace: Path) -> list[tuple[str, int]]:
    """Uncommitted work in a worktree checkout (a pristine checkout is NOT evidence)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    found: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip()
        if " -> " in rel:  # rename: take the new path
            rel = rel.split(" -> ", 1)[1]
        rel = rel.strip('"')
        if not rel:
            continue
        target = workspace / rel
        try:
            size = target.stat().st_size if target.is_file() else 1
        except OSError:
            size = 1  # deleted/renamed-away path still proves the run touched the tree
        found.append((rel, size))
        if len(found) >= _LISTING_CAP:
            break
    return found


def _resolve_workspace(context: dict[str, Any], task_row) -> Optional[Path]:
    """Resolved workspace dir: the worker context's path wins (the dispatcher
    already ran ``resolve_workspace``); scratch tasks have NULL
    ``workspace_path`` in the DB, so fall back to the workspaces_root
    convention (``<root>/<task_id>``) matching ``resolve_workspace``."""
    ctx_ws = context.get("workspace")
    if ctx_ws:
        p = Path(ctx_ws)
        return p if p.is_dir() else None
    db_path = task_row["workspace_path"] if task_row is not None else None
    if db_path:
        p = Path(db_path)
        return p if p.is_dir() else None
    kind = task_row["workspace_kind"] if task_row is not None else None
    if (kind or "scratch") == "scratch" and task_row is not None:
        from hermes_cli import kanban_db as _kb
        p = _kb.workspaces_root() / task_row["id"]
        return p if p.is_dir() else None
    return None


def _collect_evidence(conn, task_row, run_id: int, workspace: Optional[Path]) -> tuple[list[tuple[str, int]], list[str]]:
    """(artifact (path, size) list, completed goal steps) proving the run did work."""
    steps = _goal_steps(_run_row(conn, run_id))
    artifacts: list[tuple[str, int]] = []
    if workspace is not None and workspace.is_dir():
        kind = task_row["workspace_kind"] if task_row is not None else None
        if kind == "worktree":
            artifacts = _worktree_artifacts(workspace)
        else:
            artifacts = _scratch_artifacts(workspace)
    return artifacts, steps


def _completion_body(artifacts: list[tuple[str, int]], steps: list[str]) -> str:
    lines = [
        "[exit-handler] Terminal kanban call auto-emitted: the worker exited rc=0 "
        "without calling kanban_complete or kanban_block; this completion was "
        "synthesized by the worker exit handler because the run left evidence of work.",
    ]
    if artifacts:
        lines.append("Artifacts found:")
        lines.extend(f"- {path} ({size} bytes)" for path, size in artifacts[:_LISTING_CAP])
        if len(artifacts) > _LISTING_CAP:
            lines.append(f"- ... and {len(artifacts) - _LISTING_CAP} more")
    if steps:
        lines.append(f"Recorded completed steps: {len(steps)}")
        lines.extend(f"  * {step}" for step in steps[:10])
    return "\n".join(lines)


def synthesize_missing_terminal_call(context: dict[str, Any], code: int) -> str:
    """Fire the exit handler for a clean (rc=0) worker exit; idempotent by DB fence.

    ``context`` is the worker context dict (``db``, ``task_id``, ``run_id``,
    ``workspace``); ``code`` is the process exit code. Returns a short outcome
    tag for logging: ``skipped_*`` / ``completed`` / ``blocked``. Never raises —
    a synthesis failure must not break the worker's own exit path.
    """
    try:
        return _synthesize(context, code)
    except Exception as exc:  # noqa: BLE001 - exit path must never propagate
        print(f"kanban exit-handler synthesis failed: {exc}", flush=True)
        return "skipped_error"


def synthesize_from_env(code: int = 0) -> str:
    """One-shot CLI variant: build the worker context from the documented env contract."""
    task_id = os.environ.get("HERMES_KANBAN_TASK")
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not task_id or not run_id:
        return "skipped_env"
    try:
        run_id_int = int(run_id)
    except ValueError:
        return "skipped_env"
    from hermes_cli.kanban_db import kanban_db_path

    db = os.environ.get("HERMES_KANBAN_DB") or str(kanban_db_path())
    context = {
        "db": db,
        "task_id": task_id,
        "run_id": run_id_int,
        "workspace": os.environ.get("HERMES_KANBAN_WORKSPACE"),
    }
    return synthesize_missing_terminal_call(context, code)


def _synthesize(context: dict[str, Any], code: int) -> str:
    if code != 0:
        # Non-zero exits belong to the dispatcher's failure machinery
        # (retry / rate-limit wall / terminal provider), never to this handler.
        return "skipped_nonzero"

    task_id = context.get("task_id")
    run_id = context.get("run_id")
    db = context.get("db")
    if not task_id or run_id is None or not db:
        return "skipped_context"
    try:
        run_id = int(run_id)
    except (TypeError, ValueError):
        return "skipped_context"

    with connect_closing(Path(db)) as conn:
        task_row = conn.execute(
            "SELECT id, status, current_run_id, workspace_kind, workspace_path, "
            "completion_contract FROM tasks WHERE id = ?",
            (str(task_id),),
        ).fetchone()
        if task_row is None:
            return "skipped_missing"
        # Idempotency pre-check (the SQL fence inside complete/block is the real
        # guarantee; this avoids even opening a txn on the common no-op paths).
        if task_row["status"] != "running" or task_row["current_run_id"] != run_id:
            return "skipped_superseded"
        contract = task_row["completion_contract"]
        if contract and contract != "local-only":
            # PR-acceptance cards must complete through their contract, never synthesized.
            return "skipped_contract"

        workspace = _resolve_workspace(context, task_row)
        artifacts, steps = _collect_evidence(conn, task_row, run_id, workspace)
        metadata: Optional[dict[str, Any]] = None
        if artifacts:
            metadata = {
                "artifacts": [path for path, _size in artifacts],
                "auto_emitted_by": "exit_handler",
            }

        # complete_task / block_task each open their own write_txn; do not wrap.
        if artifacts or steps:
            body = _completion_body(artifacts, steps)
            summary = (
                f"[exit-handler] auto-completed after clean exit without terminal call: "
                f"{len(artifacts)} artifact(s), {len(steps)} step(s)"
            )
            ok = kb.complete_task(
                conn, str(task_id), result=body, summary=summary,
                metadata=metadata, expected_run_id=run_id,
            )
            return "completed" if ok else "skipped_fence"

        ok = kb.block_task(
            conn, str(task_id), reason=BLOCK_REASON_NO_ARTIFACTS,
            expected_run_id=run_id,
        )
        return "blocked" if ok else "skipped_fence"
