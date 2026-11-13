#!/usr/bin/env python3
"""Live probe: do reused kanban task worktrees start from the fetched remote default branch?

Usage: ./venv/bin/python3 scripts/kanban_worktree_base_probe.py <scratch-repo>

Builds a bare origin + clone under <scratch-repo> (must be absent or empty) with
``origin/main`` one commit ahead of the parked task branches (pushed from a second
clone, so the dispatched clone is unfetched), then drives
``resolve_workspace`` for a new branch, an existing branch, an existing linked
worktree and a dirty worktree. Exits non-zero if any case is not as expected.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=Probe", "-c", "user.email=probe@example.com",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout


def commit(repo: Path, name: str) -> None:
    (repo / name).write_text(name, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-m", name)


def build_repo(scratch: Path) -> Path:
    origin = scratch / "origin.git"
    clone = scratch / "clone"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(origin), str(clone)], check=True, capture_output=True)
    git(clone, "checkout", "-b", "main")
    commit(clone, "base.txt")
    git(clone, "push", "-u", "origin", "main")
    for branch in ("wt/existing-branch", "wt/existing-tree", "wt/dirty-tree"):
        git(clone, "branch", branch)
    # A second clone moves origin ahead; ``clone`` is never fetched, so its
    # origin/main lags the true remote tip and only a fetch can see the new commit.
    pusher = scratch / "pusher"
    subprocess.run(["git", "clone", str(origin), str(pusher)], check=True, capture_output=True)
    commit(pusher, "ahead.txt")
    git(pusher, "push", "origin", "main")
    return clone


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    scratch = Path(argv[1]).expanduser().resolve()
    if scratch.exists() and any(scratch.iterdir()):
        print(f"refusing: {scratch} exists and is not empty")
        return 2
    scratch.mkdir(parents=True, exist_ok=True)
    # An ambient kanban DB/workspace override or delegate fence would point the
    # probe at the live board (or make it read-only); drop them before importing.
    for key in [k for k in os.environ if k.startswith("HERMES_KANBAN_") or k == "HERMES_DELEGATED_CHILD_CONTEXT"]:
        del os.environ[key]
    # ``kanban_home()`` (hermes_cli/kanban_db.py:400) reads ``HERMES_KANBAN_HOME``,
    # NOT ``HERMES_HOME``: scrubbing the KANBAN vars and setting only HERMES_HOME
    # left ``kanban_db_path()`` on the live board, so each run created 4 real cards.
    kanban_root = scratch / "hermes-home"
    kanban_root.mkdir()
    os.environ["HERMES_KANBAN_HOME"] = str(kanban_root)
    os.environ["HERMES_HOME"] = str(kanban_root)
    sys.path.insert(0, str(REPO_ROOT))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workspace as kbw

    # Print the resolved roots and abort BEFORE any task exists: a silent escape
    # here writes cards onto the live board.
    board_db = Path(kb.kanban_db_path()).resolve()
    workspaces = Path(kb.workspaces_root()).resolve()
    print(f"board db: {board_db}")
    print(f"workspaces root: {workspaces}")
    for label, path in (("board db", board_db), ("workspaces root", workspaces)):
        if not path.is_relative_to(scratch):
            print(f"FATAL: {label} {path} is not under the scratch dir {scratch}; refusing to write to a live board")
            return 3

    clone = build_repo(scratch)
    kb.init_db()
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(clone))
    ref = "origin/main"

    def dispatch(branch: str, path: Path | None = None) -> Path:
        with kbc.connect() as conn:
            tid = kb.create_task(
                conn, title=branch, workspace_kind="worktree",
                workspace_path=str(path) if path else None, branch_name=branch,
            )
            task = kb.get_task(conn, tid)
        return kbw.resolve_workspace(task)

    def behind(tree: Path) -> int:
        # Truth is the remote's tip, not whatever this clone last saw.
        git(tree, "fetch", "origin", "--quiet")
        return int(git(tree, "rev-list", "--count", f"HEAD..{ref}").strip())

    ok = True

    def report(label: str, tree: Path) -> None:
        nonlocal ok
        n = behind(tree)
        ok &= n == 0
        print(f"case {label} -> behind {ref}: {n}")

    report("A new branch     ", dispatch("wt/new-branch"))
    report("B existing branch", dispatch("wt/existing-branch"))

    tree_c = clone / ".worktrees" / "existing-tree"
    git(clone, "worktree", "add", str(tree_c), "wt/existing-tree")
    report("C existing tree  ", dispatch("wt/existing-tree", tree_c))

    tree_d = clone / ".worktrees" / "dirty-tree"
    git(clone, "worktree", "add", str(tree_d), "wt/dirty-tree")
    (tree_d / "wip.txt").write_text("uncommitted", encoding="utf-8")
    status_before = git(tree_d, "status", "--porcelain")
    try:
        dispatch("wt/dirty-tree", tree_d)
    except RuntimeError as exc:
        refused = str(tree_d) in str(exc)
        ok &= refused
        print(f"case D dirty tree      -> REFUSED: {exc}")
    else:
        ok = False
        print("case D dirty tree      -> NOT REFUSED (dispatched stale)")
    unchanged = git(tree_d, "status", "--porcelain") == status_before
    ok &= unchanged
    print(f"case D git status --porcelain unchanged: {unchanged}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
