"""``hermes kanban boards doctor`` - read-only health audits for registered boards.

Every check is observational: nothing here writes a board DB, prunes a worktree
or deletes a branch. ``git`` runs with ``check=False`` and a 30s timeout so a
wedged repo (NFS stall, giant dirty tree) surfaces as a ``fail`` check instead
of hanging the CLI. A raised check is downgraded to ``fail`` with the exception
message rather than dropping the check, because consumers index ``checks[]``
by name.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_workspace import _git_current_branch, _git_toplevel

_SCHEMA = "boards-doctor/v1"
_CHECKS = ("git_repo_valid", "clean_main", "worktree_hygiene", "stale_branches")
_CLEAN_BRANCHES = frozenset({"main", "master"})
# ``ok``/``skip``/``warn`` are non-fatal; ``fail`` makes the board unhealthy.
_FATAL_STATUS = "fail"
_NOT_A_REPO = "skipped because default_workdir is not a valid git repository"


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def _check(
    name: str,
    status: str,
    *,
    message: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"check": name, "status": status}
    if message is not None:
        result["message"] = message
    if detail is not None:
        result["detail"] = detail
    return result


def _guard(name: str, run_check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one check; a raised audit becomes this check's ``fail``."""
    try:
        return run_check()
    except Exception as exc:  # a broken repo must not kill the whole report
        return _check(name, _FATAL_STATUS, message=f"audit raised: {exc}")


def _task_rows(slug: str) -> list[dict[str, Any]]:
    """Worktree-kind tasks on ``slug``; ``[]`` when the board DB cannot be read."""
    try:
        with kbc.connect_closing(board=slug) as conn:
            rows = conn.execute(
                "SELECT id, status, workspace_path, workspace_kind, branch_name FROM tasks "
                "WHERE workspace_kind='worktree'"
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []


def _worktree_blocks(repo: Path) -> list[dict[str, str]]:
    result = _git(repo, "worktree", "list", "--porcelain")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git worktree list failed").strip())
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line:
            if current.get("path"):
                blocks.append(current)
            current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "detached":
            current["branch"] = "(detached)"
    if current.get("path"):
        blocks.append(current)
    return blocks


def _task_id_for_worktree(path: str, branch: str, rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        task_id = str(row.get("id") or "")
        if task_id and (task_id in path or row.get("workspace_path") == path or row.get("branch_name") == branch):
            return task_id
    return None


def _audit_clean_main(path: Path) -> dict[str, Any]:
    status = _git(path, "status", "--porcelain")
    branch = _git_current_branch(path)
    dirty_count = len(status.stdout.splitlines()) if status.returncode == 0 else 0
    clean = status.returncode == 0 and dirty_count == 0 and branch in _CLEAN_BRANCHES
    message = None if clean else f"{dirty_count} uncommitted entries on branch '{branch or '(unknown)'}'"
    return _check(
        "clean_main",
        "ok" if clean else _FATAL_STATUS,
        message=message,
        detail={"branch": branch, "dirty_count": dirty_count},
    )


def _audit_worktrees(slug: str, repo: Path) -> dict[str, Any]:
    blocks = _worktree_blocks(repo)
    rows = _task_rows(slug)
    stale: list[dict[str, str]] = []
    for block in blocks:
        path = Path(block["path"]).resolve(strict=False)
        if path == repo.resolve(strict=False) or ".worktrees" not in path.parts:
            continue
        task_id = _task_id_for_worktree(str(path), block.get("branch", ""), rows)
        row = next((item for item in rows if str(item.get("id")) == task_id), None)
        if row and row.get("status") == "done":
            stale.append({"task_id": task_id or "unknown", "branch": block.get("branch", ""), "path": str(path)})
    detail = {"active_worktrees": len(blocks), "stale_worktrees": stale}
    message = (
        f"{len(stale)} stale worktrees for done tasks: {', '.join(x['task_id'] for x in stale)}"
        if stale
        else None
    )
    return _check("worktree_hygiene", "warn" if stale else "ok", message=message, detail=detail)


def _audit_stale_branches(slug: str, repo: Path) -> dict[str, Any]:
    rows = _task_rows(slug)
    done = {
        str(row["branch_name"]): str(row["id"])
        for row in rows
        if row.get("status") == "done" and row.get("branch_name")
    }
    live = {item.get("branch", "") for item in _worktree_blocks(repo)}
    result = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads")
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "git branch listing failed").strip())
    stale = [
        {"task_id": done[branch], "branch": branch}
        for branch in result.stdout.splitlines()
        if branch in done and branch not in live
    ]
    detail = {"stale_branches": stale}
    message = f"{len(stale)} stale branches for done tasks" if stale else None
    return _check("stale_branches", "warn" if stale else "ok", message=message, detail=detail)


def _collect_checks(slug: str, path: Path) -> list[dict[str, Any]]:
    """The four canonical checks, in order, always with a status from the contract."""
    repo_root = _git_toplevel(path) if path.exists() else None
    if repo_root is None:
        return [
            _check("git_repo_valid", _FATAL_STATUS, message=f"not a git repository: {path}"),
            *(_check(name, "skip", message=_NOT_A_REPO) for name in _CHECKS[1:]),
        ]
    return [
        _check("git_repo_valid", "ok", detail={"repo_root": str(repo_root)}),
        _guard("clean_main", lambda: _audit_clean_main(path)),
        _guard("worktree_hygiene", lambda: _audit_worktrees(slug, repo_root)),
        _guard("stale_branches", lambda: _audit_stale_branches(slug, repo_root)),
    ]


def audit_board(board: dict[str, Any]) -> dict[str, Any]:
    slug = str(board.get("slug") or kb.DEFAULT_BOARD)
    workdir = board.get("default_workdir")
    result: dict[str, Any] = {
        "slug": slug,
        "name": board.get("name") or slug,
        "default_workdir": str(workdir) if workdir else None,
        "archived": bool(board.get("archived", False)),
    }
    if not workdir:
        result.update(
            {
                "healthy": True,
                "checks": [_check("git_repo_valid", "skip", message="board has no default_workdir; nothing to audit")],
            }
        )
        return result
    checks = _collect_checks(slug, Path(str(workdir)).expanduser())
    result.update(
        {"healthy": all(item["status"] != _FATAL_STATUS for item in checks), "checks": checks}
    )
    return result


def build_report() -> dict[str, Any]:
    boards = [audit_board(board) for board in kb.list_boards(include_archived=False)]
    skipped = sum(board["default_workdir"] is None for board in boards)
    audited = [board for board in boards if board["default_workdir"] is not None]
    unhealthy = sum(not board["healthy"] for board in audited)
    return {
        "schema": _SCHEMA,
        "generated_at": int(time.time()),
        "healthy": unhealthy == 0,
        "summary": {
            "boards_total": len(boards),
            "boards_healthy": len(audited) - unhealthy,
            "boards_unhealthy": unhealthy,
            "boards_skipped": skipped,
        },
        "boards": boards,
    }


def render_human(report: dict[str, Any]) -> None:
    print(f"  {'SLUG':24s} {'HEALTH':8s} {'DIRTY':6s} {'STALE-WT':9s} NOTES")
    for board in report["boards"]:
        checks = {item["check"]: item for item in board["checks"]}
        clean = checks.get("clean_main", {})
        hygiene = checks.get("worktree_hygiene", {})
        dirty = clean.get("detail", {}).get("dirty_count", "-")
        stale_items = hygiene.get("detail", {}).get("stale_worktrees", [])
        stale = len(stale_items) if hygiene else "-"
        notes: list[str] = []
        if board["default_workdir"] is None:
            notes.append("no default_workdir")
        if clean.get("status") == _FATAL_STATUS:
            notes.append("dirty main")
        if stale not in {"-", 0}:
            notes.append(f"{stale} stale worktrees ({', '.join(x['task_id'] for x in stale_items)})")
        print(
            f"  {board['slug']:24s} {'ok' if board['healthy'] else 'FAIL':8s} "
            f"{dirty!s:6s} {stale!s:9s} {'; '.join(notes) or board['default_workdir'] or ''}"
        )
    count = report["summary"]["boards_unhealthy"]
    if count:
        print(f"\n{count} unhealthy board. Run with --json for full diagnostics.")


def _emit_enumeration_failure(as_json: bool, exc: Exception) -> int:
    """Board enumeration itself failed: keep stdout parseable for shell-out consumers."""
    message = f"boards doctor failed: {exc}"
    if as_json:
        print(
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "generated_at": int(time.time()),
                    "healthy": False,
                    "summary": {
                        "boards_total": 0,
                        "boards_healthy": 0,
                        "boards_unhealthy": 0,
                        "boards_skipped": 0,
                    },
                    "boards": [],
                    "error": message,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(message, file=sys.stderr)
    return 1


def run(args: Any) -> int:
    as_json = bool(getattr(args, "json", False))
    try:
        report = build_report()
    except Exception as exc:
        return _emit_enumeration_failure(as_json, exc)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        render_human(report)
    return 0 if report["healthy"] else 1
