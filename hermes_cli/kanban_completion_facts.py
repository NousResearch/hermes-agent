"""Git-backed completion facts for dispatcher-owned worktree runs.

The dispatcher snapshots HEAD after materialising a worktree and before the
worker starts. Completion probes happen outside SQLite transactions; the
receipt is persisted only after rechecking run ownership under the write lock.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Optional

_ACTIVE_STATUSES = {"running", "ready", "blocked", "review"}
_IMPLEMENTATION_TITLE = re.compile(
    r"^(?:fix|feat|perf|refactor)(?:\([^)]*\))?:|\b(?:implement|build|add|create|fix|refactor)\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_HANDOFF = re.compile(
    r"\b(?:implemented|built|added|created|fixed|refactored|changed files?|"
    r"completed implementation|implementation (?:is )?(?:complete|done))\b",
    re.IGNORECASE,
)
_PUSH_CLAIM = re.compile(r"\b(?:push(?:ed|es|ing)?|published)\b", re.IGNORECASE)
_SHA = re.compile(r"[0-9a-f]{40}")


def _run_git(path: str, *args: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    return subprocess.run(
        ["git", "-C", path, *args], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=False, env=env,
    )


def _git_output(path: str, *args: str, timeout: int = 15) -> Optional[str]:
    try:
        proc = _run_git(path, *args, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def record_workspace_baseline(conn: sqlite3.Connection, task_id: str, workspace: Path | str) -> bool:
    """Record the active run's starting HEAD before its worker process starts."""
    row = conn.execute(
        "SELECT current_run_id, status, workspace_kind FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if not row or row["status"] != "running" or row["workspace_kind"] != "worktree" \
            or row["current_run_id"] is None:
        return False
    run_id = int(row["current_run_id"])
    path = str(workspace)
    head = _git_output(path, "rev-parse", "--verify", "HEAD")
    branch = _git_output(path, "symbolic-ref", "--quiet", "--short", "HEAD")
    if not head or not _SHA.fullmatch(head):
        return False
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        current = conn.execute(
            "SELECT current_run_id, status FROM tasks WHERE id=?", (task_id,),
        ).fetchone()
        if not current or current["status"] != "running" or current["current_run_id"] != run_id:
            return False
        kb._append_event(
            conn, task_id, "workspace_baseline",
            {"head_sha": head, "branch": branch, "workspace": path}, run_id=run_id,
        )
    return True


def _snapshot(conn: sqlite3.Connection, task_id: str) -> Optional[tuple]:
    row = conn.execute(
        "SELECT current_run_id, status, workspace_kind, workspace_path, branch_name, title "
        "FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    return tuple(row) if row else None


def _baseline(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[str]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
        "AND kind='workspace_baseline' ORDER BY id DESC LIMIT 1",
        (task_id, run_id),
    ).fetchone()
    if not row:
        return None
    try:
        value = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return None
    head = value.get("head_sha") if isinstance(value, dict) else None
    return head if isinstance(head, str) and _SHA.fullmatch(head) else None


def _dirty_count(path: str) -> Optional[int]:
    try:
        proc = _run_git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    fields = proc.stdout.split("\0")
    count = index = 0
    while index < len(fields) and fields[index]:
        record = fields[index]
        count += 1
        index += 2 if len(record) >= 2 and (record[0] in "RC" or record[1] in "RC") else 1
    return count


def _claims_implementation(title: str, summary: Optional[str], result: Optional[str], metadata: Any) -> bool:
    if isinstance(metadata, dict):
        if metadata.get("commit") or metadata.get("commits") or metadata.get("changed_files"):
            return True
        git_claim = metadata.get("git")
        if isinstance(git_claim, dict) and git_claim.get("implementation") is True:
            return True
    handoff = "\n".join(x for x in (summary, result) if isinstance(x, str))
    return bool(_IMPLEMENTATION_TITLE.search(title or "") or _IMPLEMENTATION_HANDOFF.search(handoff))


def _claims_push(summary: Optional[str], result: Optional[str], metadata: Any) -> bool:
    if isinstance(metadata, dict):
        if metadata.get("pushed") is True or metadata.get("git_refs_pushed"):
            return True
    handoff = "\n".join(x for x in (summary, result) if isinstance(x, str))
    return bool(_PUSH_CLAIM.search(handoff))


def _verify_pushed_refs(path: str, metadata: Any) -> tuple[list[dict], Optional[str]]:
    claims = metadata.get("git_refs_pushed") if isinstance(metadata, dict) else None
    if not isinstance(claims, list) or not claims:
        return [], (
            "A push/publication was claimed without metadata.git_refs_pushed. Retry with "
            "[{\"remote\":\"origin\",\"ref\":\"refs/heads/<branch>\",\"sha\":\"<40-hex>\"}]."
        )
    receipts: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict):
            return receipts, "Each metadata.git_refs_pushed entry must be an object."
        remote, ref, sha = (claim.get(key) for key in ("remote", "ref", "sha"))
        if not isinstance(remote, str) or not remote.strip() or not isinstance(ref, str) \
                or not ref.startswith("refs/heads/") or not isinstance(sha, str) or not _SHA.fullmatch(sha):
            return receipts, (
                "Each pushed-ref claim requires remote, refs/heads/<branch>, and an exact 40-hex sha."
            )
        try:
            proc = _run_git(path, "ls-remote", "--heads", remote, ref, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return receipts, f"Remote-ref evidence unavailable for {remote}:{ref}; retry or block on infrastructure."
        fields = proc.stdout.strip().split()
        actual = fields[0] if proc.returncode == 0 and len(fields) >= 2 and fields[1] == ref else None
        receipt = {"remote": remote, "ref": ref, "claimed_sha": sha, "remote_sha": actual}
        receipts.append(receipt)
        if actual != sha:
            return receipts, f"Remote ref {remote}:{ref} does not resolve to claimed sha {sha}."
    return receipts, None


def prepare_completion_facts(
    conn: sqlite3.Connection, task_id: str, expected_run_id: Optional[int],
    metadata: Any, summary: Optional[str], result: Optional[str], override_reason: Optional[str],
):
    """Collect a run-bound receipt, or None for non-dispatched/non-worktree completion."""
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, kind, path, branch, title = snapshot
    if status not in _ACTIVE_STATUSES or (expected_run_id is not None and run_id != expected_run_id):
        return False
    if kind != "worktree" or run_id is None:
        return None
    receipt = {
        "ok": False, "workspace": path, "branch": branch, "baseline_sha": None,
        "head_sha": None, "commits_ahead": None, "dirty_file_count": None,
        "implementation_claimed": _claims_implementation(title, summary, result, metadata),
        "push_claimed": _claims_push(summary, result, metadata), "pushed_refs": [],
        "override_reason": override_reason or None,
    }
    error = None
    baseline = _baseline(conn, task_id, int(run_id)) if run_id is not None else None
    receipt["baseline_sha"] = baseline
    if not path or not baseline:
        error = "Worktree baseline evidence is missing; block on infrastructure or ask an operator to override."
    elif not Path(path).is_dir():
        error = "Worktree path is missing; block on infrastructure or ask an operator to override."
    else:
        head = _git_output(path, "rev-parse", "--verify", "HEAD")
        dirty = _dirty_count(path)
        receipt.update(head_sha=head, dirty_file_count=dirty)
        if not head or not _SHA.fullmatch(head) or dirty is None:
            error = "Git completion evidence is unavailable; retry or block on infrastructure."
        else:
            try:
                ancestor = _run_git(path, "merge-base", "--is-ancestor", baseline, head)
                ahead = _git_output(path, "rev-list", "--count", f"{baseline}..{head}")
            except (OSError, subprocess.SubprocessError):
                ancestor, ahead = None, None
            receipt["commits_ahead"] = int(ahead) if ahead and ahead.isdigit() else None
            if ancestor is None or ancestor.returncode != 0 or receipt["commits_ahead"] is None:
                error = "Current HEAD does not descend from the recorded run baseline."
            elif dirty:
                error = f"Worktree has {dirty} uncommitted file(s); commit or discard them before completion."
            elif receipt["implementation_claimed"] and receipt["commits_ahead"] == 0:
                error = "Implementation was claimed but HEAD has zero commits over the run baseline."
            elif receipt["push_claimed"]:
                pushed, error = _verify_pushed_refs(path, metadata)
                receipt["pushed_refs"] = pushed
    receipt["error"] = error
    receipt["ok"] = error is None or bool(override_reason)
    return snapshot, receipt


def attach_receipt(metadata: Any, prepared: Any) -> Any:
    if not prepared or prepared is False:
        return metadata
    receipt = prepared[1]
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    merged["completion_facts"] = receipt
    return merged


def record_completion_facts(conn: sqlite3.Connection, task_id: str, prepared: Any) -> bool:
    """Persist the receipt under run ownership; false leaves the task in-flight."""
    if prepared is None:
        return True
    if prepared is False:
        return False
    snapshot, receipt = prepared
    if _snapshot(conn, task_id) != snapshot:
        return False
    from hermes_cli import kanban_db as kb
    run_id = snapshot[0]
    kb._append_event(conn, task_id, "completion_facts", receipt, run_id=run_id)
    if not receipt["ok"]:
        recovery = (
            f"Completion fact check failed: {receipt['error']} The task remains in-flight. "
            "Fix the workspace facts and retry, use kanban_block for unfinished/blocked work, or have an "
            "operator use `hermes kanban complete --override-git-facts <reason>` for an intentional no-op."
        )
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (recovery, task_id))
    return bool(receipt["ok"])
