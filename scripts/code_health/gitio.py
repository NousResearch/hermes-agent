"""Git plumbing: which files changed between two trees, and their contents on either side.

A tree is a revision (``"abc123"``) or ``None`` for the working tree. Both sides are read
through the same functions so base and head are measured identically.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").strip())


def resolve_rev(repo: Path, rev: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def resolve_tree(repo: Path, rev: str) -> str:
    """A head may be any tree-ish: a commit, or the index's tree from ``git write-tree``."""
    return git(repo, "rev-parse", "--verify", f"{rev}^{{tree}}").strip()


def commit_or_none(repo: Path, rev: str) -> str | None:
    """``rev`` as a commit sha, or None when it does not name a commit (a tree, a missing ref)."""
    out = git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
    return out.strip() or None


def default_base(repo: Path, tip: str = "HEAD") -> str:
    """Merge-base of ``tip`` with origin/main (local runs); CI passes ``--base`` explicitly."""
    for ref in ("origin/main", "main"):
        out = git(repo, "merge-base", tip, ref, check=False).strip()
        if out:
            return out
    raise RuntimeError("no merge-base with origin/main or main; pass --base")


@dataclass(frozen=True)
class Change:
    status: str  # A, M, D, R
    old: str | None
    new: str | None


def _parse_name_status(out: str) -> list[Change]:
    fields = out.split("\0")
    changes: list[Change] = []
    i = 0
    while i < len(fields) and fields[i]:
        code = fields[i][0]
        if code in "RC":
            old, new = fields[i + 1], fields[i + 2]
            changes.append(Change("R" if code == "R" else "A", old if code == "R" else None, new))
            i += 3
            continue
        path = fields[i + 1]
        if code == "D":
            changes.append(Change("D", path, None))
        elif code == "A":
            changes.append(Change("A", None, path))
        else:
            changes.append(Change("M", path, path))
        i += 2
    return changes


def changed_files(repo: Path, base: str, head: str | None) -> list[Change]:
    args = ["diff", "--name-status", "-M", "-z", "--no-color", base]
    if head is not None:
        args.append(head)
    changes = _parse_name_status(git(repo, *args))
    if head is None:
        untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z")
        changes.extend(Change("A", None, p) for p in untracked.split("\0") if p)
    return changes


def read_file(repo: Path, tree: str | None, path: str) -> str | None:
    if tree is None:
        try:
            return (repo / path).read_text(encoding="utf-8-sig", errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            return None
    proc = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        return None
    # Same decoding as the working-tree read: a BOM must not make the two sides disagree.
    return proc.stdout.decode("utf-8-sig", errors="replace")


def tracked_files(repo: Path, tree: str | None) -> list[str]:
    if tree is None:
        return [p for p in git(repo, "ls-files", "-z").split("\0") if p]
    return [p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", tree).split("\0") if p]


def read_many(repo: Path, tree: str | None, paths: list[str]) -> dict[str, str]:
    """Contents of many files at once: the working tree is read directly, a revision through a
    single ``git cat-file --batch`` (a ``git show`` per file is ~2.3k processes on a whole-tree
    pass, which the import graph makes on every run)."""
    if not paths:
        return {}
    if tree is None:
        out: dict[str, str] = {}
        for path in paths:
            try:
                out[path] = (repo / path).read_text(encoding="utf-8-sig", errors="replace")
            except (FileNotFoundError, IsADirectoryError):
                continue
        return out
    specs = "".join(f"{tree}:{path}\n" for path in paths).encode()
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input=specs,
                          capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError("git cat-file --batch failed: "
                           + proc.stderr.decode("utf-8", "replace").strip())
    out = {}
    data, pos = proc.stdout, 0
    for path in paths:
        end = data.find(b"\n", pos)
        if end < 0:
            break
        header = data[pos:end].decode("utf-8", "replace").split()
        pos = end + 1
        if len(header) != 3:  # "<spec> missing"
            continue
        size = int(header[2])
        out[path] = data[pos:pos + size].decode("utf-8-sig", "replace")
        pos += size + 1  # the newline cat-file prints after each object
    return out


def untracked_files(repo: Path) -> list[str]:
    return [p for p in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split("\0") if p]


def known_env_names(repo: Path, tree: str | None) -> set[str]:
    """Every HERMES_* literal in non-test Python of a tree (the HX002 allow set)."""
    args = ["grep", "-h", "-o", "-I", "-E", r"HERMES_[A-Z0-9_]+"]
    if tree is not None:
        args.append(tree)
    args += ["--", "*.py", ":!tests/**", ":!**/tests/**"]
    out = git(repo, *args, check=False)
    names = set()
    for line in out.splitlines():
        # `git grep <tree>` prefixes "<tree>:" even with -h on some versions.
        names.add(line.rsplit(":", 1)[-1])
    return names
