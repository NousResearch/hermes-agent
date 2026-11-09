#!/usr/bin/env python3
"""Carry the current branch across the module moves that landed on its target.

Each move chunk on the target carries the trailer ``Hermes-Move: NN-<name>`` on two commits:
commit A renames files (``git mv`` only) and commit B rewrites the imports. For every chunk
the branch has not merged yet, in landing order, this script

1. merges the commit before A, so ordinary conflicts surface apart from the move;
2. merges A with directory rename detection, so the branch's edits follow the moved files;
3. runs ``scripts/move/rewrite_imports.py --from-commit A`` on the Python files the branch
   changed and commits the result for those files only;
4. merges B, keeping the branch's version of the files step 3 rewrote (B is the same rewrite
   applied to the target's version of them).

It then merges the target itself, which brings in the chunk's hand fixes. It stops on any other
conflict; resolve it, then run ``--continue``, or ``--abort`` to put the branch back where it
started. Sites the import rewrite flags are printed and left for the compat finder and lints. Merges only, no
history rewrite. The remote is never touched unless ``--push`` is given.

Usage:
    python scripts/move/migrate_branch.py [--onto origin/main] [--push]
    python scripts/move/migrate_branch.py --continue
    python scripts/move/migrate_branch.py --abort
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

TRAILER = "Hermes-Move"
CODEMOD = "scripts/move/rewrite_imports.py"
GIT_TIMEOUT_S = 600
CODEMOD_TIMEOUT_S = 1800


class Stop(Exception):
    """A step needs the user; the message says what to do."""


@dataclass(frozen=True)
class Chunk:
    name: str
    rename: str  # commit A
    rewrite: str  # commit B, the chunk's last commit carrying the trailer


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=GIT_TIMEOUT_S)
    if check and result.returncode != 0:
        raise Stop(f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def out(*args: str) -> str:
    return git(*args).stdout.strip()


def is_ancestor(commit: str, of: str = "HEAD") -> bool:
    return git("merge-base", "--is-ancestor", commit, of, check=False).returncode == 0


def state_dir() -> Path:
    return Path(out("rev-parse", "--git-path", "migrate-branch")).resolve()


def load_state() -> dict | None:
    path = state_dir() / "state.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def save_state(state: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def find_chunks(base: str, onto: str) -> list[Chunk]:
    """Move chunks between base and onto, in the order they landed on onto."""
    log = out("log", "--reverse", "--topo-order", "--no-merges",
              f"--format=%H%x09%(trailers:key={TRAILER},valueonly,separator=%x2C)", f"{base}..{onto}")
    commits: dict[str, list[str]] = {}
    for line in log.splitlines():
        sha, _, name = line.partition("\t")
        if name.strip():
            commits.setdefault(name.strip(), []).append(sha)
    mainline = out("rev-list", "--first-parent", "--reverse", f"{base}..{onto}").splitlines()
    chunks = [Chunk(name, shas[0], shas[-1]) for name, shas in commits.items() if _present(shas[0], onto)]
    return sorted(chunks, key=lambda chunk: mainline.index(_landing_commit(chunk.rewrite, mainline, onto)))


def _landing_commit(last: str, mainline: list[str], onto: str) -> str:
    descendants = set(out("rev-list", "--ancestry-path", f"{last}..{onto}").splitlines())
    return next(sha for sha in mainline if sha == last or sha in descendants)


def _present(rename: str, onto: str) -> bool:
    """False for a reverted chunk: onto has every old path of commit A back and none of the new ones."""
    status = [line.split("\t") for line in out("diff", "-M", "--name-status", f"{rename}^", rename).splitlines()]
    renames = [paths for kind, *paths in status if kind.startswith("R")]
    if not renames or any(kind.startswith("M") for kind, *_ in status):
        raise Stop(f"Commit {rename[:12]} is the first commit carrying {TRAILER} but is not a rename; "
                   "the first commit of a move chunk must be `git mv` only.")
    old_paths, new_paths = zip(*renames)
    def held(paths: tuple[str, ...]) -> int:
        return len(out("ls-tree", "-r", "--name-only", onto, "--", *paths).splitlines())
    return not (held(old_paths) == len(old_paths) and held(new_paths) == 0)


def rename_config(commit: str) -> list[str]:
    limit = len(out("ls-tree", "-r", "--name-only", commit).splitlines()) + 1
    return ["-c", "merge.renames=true", "-c", "merge.directoryRenames=true",
            "-c", f"merge.renameLimit={limit}", "-c", f"diff.renameLimit={limit}",
            "-c", "merge.conflictStyle=diff3"]


def merge(commit: str, message: str, resolvable: set[str] = frozenset()) -> None:
    if is_ancestor(commit):
        return
    result = git(*rename_config(commit), "merge", "--no-edit", "-m", message, commit, check=False)
    if result.returncode == 0:
        return
    unmerged = out("diff", "--name-only", "--diff-filter=U").splitlines()
    if not unmerged:
        raise Stop(f"Merge of {commit[:12]} failed without conflicts:\n{result.stdout}{result.stderr}")
    left = [path for path in unmerged if not (path in resolvable and keep_branch_side(path))]
    if left:
        raise Stop(f"{message}: conflicts in\n  " + "\n  ".join(left)
                   + "\nResolve them, `git add` the files, then run --continue (or --abort).")
    git("commit", "--no-edit")


def keep_branch_side(path: str) -> bool:
    """Resolve a conflict between the branch's rewritten file and B's rewrite of the same file."""
    stages = {line.split()[2] for line in out("ls-files", "-u", "--", path).splitlines()}
    if not {"2", "3"} <= stages:
        return False
    git("checkout", "--ours", "--", path)
    git("add", "--", path)
    return True


def branch_python_files(rename: str) -> list[str]:
    changed = out("diff", "--name-only", "--diff-filter=AMR", rename, "HEAD").splitlines()
    return [path for path in changed if path.endswith(".py")]


def rewrite(chunk: Chunk, state: dict) -> None:
    if chunk.rename in state["rewritten"]:
        return
    paths = branch_python_files(chunk.rename)
    if paths:
        run_codemod(chunk, paths, state)
        commit_changes(paths, f"migrate-branch: rewrite imports for move {chunk.name}")
        git("checkout", "--", ".")  # the target's own commit B carries every other rewrite
    state["rewritten"].append(chunk.rename)
    save_state(state)


def run_codemod(chunk: Chunk, paths: list[str], state: dict) -> None:
    tool = state_dir() / "codemod"
    env = {**os.environ, "PYTHONPATH": str(tool)}
    result = subprocess.run([sys.executable, str(tool / CODEMOD), "--from-commit", chunk.rename, "--paths", *paths],
                            check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=env, timeout=CODEMOD_TIMEOUT_S)
    print(result.stdout + result.stderr, end="")
    summary = re.search(r"^rewritten=\d+ flagged=\d+ failed=(\d+)$", result.stdout, re.MULTILINE)
    if summary is None:
        raise Stop(f"The import rewrite for move {chunk.name} did not run (output above). "
                   "Fix the cause, then run --continue to run it again.")
    if int(summary.group(1)):
        state["pending_rewrite"] = chunk.rename
        save_state(state)
        raise Stop(f"The import rewrite for move {chunk.name} could not parse some files (above). Rewrite "
                   "their imports by hand, then run --continue (it commits the changes to tracked files).")


def commit_changes(paths: list[str], message: str) -> None:
    git("add", "--", *paths) if paths else git("add", "-u")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-q", "-m", message)


def extract_codemod(onto: str) -> None:
    tool = state_dir() / "codemod"
    shutil.rmtree(tool, ignore_errors=True)
    archive = subprocess.run(["git", "archive", "--format=tar", onto, "scripts/move"], check=False,
                             capture_output=True, timeout=GIT_TIMEOUT_S)
    if archive.returncode != 0:
        raise Stop(f"{onto} has no scripts/move/: {archive.stderr.decode()}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(tool, filter="data")


def carry(state: dict) -> None:
    branch = state["branch"]
    for chunk in find_chunks(state["base"], state["onto"]):
        if is_ancestor(chunk.rewrite):
            continue
        merge(f"{chunk.rename}^", f"migrate-branch: merge the base of move {chunk.name} into {branch}")
        merge(chunk.rename, f"migrate-branch: merge the renames of move {chunk.name} into {branch}")
        rewrite(chunk, state)
        merge(chunk.rewrite, f"migrate-branch: merge the import rewrite of move {chunk.name} into {branch}",
              resolvable=set(branch_python_files(chunk.rename)))
    merge(state["onto"], f"migrate-branch: merge {state['onto_ref']} into {branch}")


def start(onto_ref: str, push: bool) -> dict:
    if load_state() is not None:
        raise Stop("A migration is in progress: run --continue or --abort.")
    if out("status", "--porcelain", "--untracked-files=no"):
        raise Stop("The working tree has uncommitted changes; commit or stash them first.")
    onto = out("rev-parse", "--verify", f"{onto_ref}^{{commit}}")
    branch = git("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip() or "HEAD"
    state = {"onto": onto, "onto_ref": onto_ref, "base": out("merge-base", "HEAD", onto),
             "orig_head": out("rev-parse", "HEAD"), "branch": branch, "push": push, "rewritten": []}
    if find_chunks(state["base"], onto):
        extract_codemod(onto)
    save_state(state)
    return state


def load_own_state() -> dict:
    state = load_state()
    if state is None:
        raise Stop("No migration in progress.")
    branch = git("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip() or "HEAD"
    if branch != state["branch"] or not is_ancestor(state["orig_head"]):
        raise Stop(f"The migration in progress is for {state['branch']}; check it out first.")
    return state


def resume() -> dict:
    state = load_own_state()
    if out("diff", "--name-only", "--diff-filter=U"):
        raise Stop("Unmerged files remain; resolve them and `git add` them first.")
    if git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        git("commit", "--no-edit")
    elif pending := state.pop("pending_rewrite", None):
        commit_changes([], "migrate-branch: rewrite imports (completed by hand)")
        state["rewritten"].append(pending)
        save_state(state)
    return state


def abort() -> None:
    state = load_own_state()
    if git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        git("merge", "--abort")
    git("reset", "-q", "--hard", state["orig_head"])
    shutil.rmtree(state_dir())
    print(f"Restored {state['branch']} to {state['orig_head'][:12]}.")


def finish(state: dict) -> None:
    applied = [chunk.name for chunk in find_chunks(state["base"], state["onto"])
               if not is_ancestor(chunk.rewrite, state["orig_head"])]
    shutil.rmtree(state_dir())
    moved = ", ".join(applied) or "none"
    print(f"{state['branch']} is merged with {state['onto_ref']}. Moves applied: {moved}.")
    if state["push"]:
        git("push")
        print("Pushed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onto", default="origin/main", help="target that holds the moves (default origin/main)")
    parser.add_argument("--push", action="store_true", help="push the branch when done (never forced)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--continue", dest="resume", action="store_true", help="resume after resolving a stop")
    action.add_argument("--abort", action="store_true", help="return the branch to where it started")
    args = parser.parse_args(argv)
    os.chdir(out("rev-parse", "--show-toplevel"))
    try:
        if args.abort:
            abort()
            return 0
        state = resume() if args.resume else start(args.onto, args.push)
        carry(state)
        finish(state)
    except Stop as stop:
        print(stop, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
