"""Module renames recorded by a move commit: the paths ``git mv`` changed, as dotted module names."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


def current_name(name: str, renames: dict[str, str]) -> str | None:
    """The new name for ``name`` by its longest dotted prefix that is a key of ``renames``."""
    parts = name.split(".")
    for size in range(len(parts), 0, -1):
        old = ".".join(parts[:size])
        if old in renames:
            return renames[old] + name[len(old):]
    return None


def renamed_paths(repo: Path, commit: str) -> list[tuple[str, str]]:
    """Old -> new paths of every file ``commit`` renamed, ``__init__.py`` files paired by their directory.

    Git pairs renames by content, so two empty or identical ``__init__.py`` files can be paired across
    packages, and git never pairs empty files at all. Each one is paired through the move of its
    directory's other files instead; git's own pair is kept only when that directory has no other file.
    """
    status = git(repo, "diff", "-M", "--name-status", f"{commit}^", commit)
    moved, inits, deleted, added = [], [], [], set()
    for line in status.splitlines():
        kind, *paths = line.split("\t")
        if kind.startswith("R") and PurePosixPath(paths[0]).name == "__init__.py":
            inits.append((paths[0], paths[1]))
            deleted.append(paths[0])
            added.add(paths[1])
        elif kind.startswith("R"):
            moved.append((paths[0], paths[1]))
        elif kind == "D":
            deleted.append(paths[0])
        elif kind == "A":
            added.add(paths[0])
    directories = {}
    for old, new in moved:
        old_dir, new_dir = PurePosixPath(old).parent, PurePosixPath(new).parent
        while str(old_dir) not in (".", str(new_dir)) and old_dir.name == new_dir.name:
            directories.setdefault(str(old_dir), str(new_dir))
            old_dir, new_dir = old_dir.parent, new_dir.parent
    git_pairs = dict(inits)
    for old in deleted:
        if PurePosixPath(old).name != "__init__.py":
            continue
        old_dir = str(PurePosixPath(old).parent)
        new = f"{directories[old_dir]}/__init__.py" if old_dir in directories else git_pairs.get(old)
        if new in added:
            moved.append((old, new))
    return moved


def moved_modules(repo: Path, commit: str) -> dict[str, tuple[str, str]]:
    """Old dotted name -> (new dotted name, new path) of every ``.py`` file ``commit`` renamed."""
    old_packages = _packages(repo, f"{commit}^")
    new_packages = _packages(repo, commit)
    modules = {}
    for old_path, new_path in renamed_paths(repo, commit):
        if old_path.endswith(".py") and new_path.endswith(".py"):
            old, new = _module_name(old_path, old_packages), _module_name(new_path, new_packages)
            if old != new:
                modules[old] = (new, new_path)
    return modules


def module_renames(repo: Path, commit: str) -> dict[str, str]:
    """Old -> new dotted module names for the ``.py`` files ``commit`` renamed.

    A moved package maps once by its prefix; its modules that kept their place under it add no entry.
    """
    renames = {old: new for old, (new, _) in moved_modules(repo, commit).items()}
    for old in sorted(renames, key=len, reverse=True):
        others = {key: value for key, value in renames.items() if key != old}
        if current_name(old, others) == renames[old]:
            del renames[old]
    return renames


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, timeout=120).stdout


def _packages(repo: Path, commit: str) -> set[str]:
    files = git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
    return {str(PurePosixPath(f).parent) for f in files if PurePosixPath(f).name == "__init__.py"}


def _module_name(path: str, packages: set[str]) -> str:
    """Dotted name of a ``.py`` path; the import root is the nearest parent directory without ``__init__.py``."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    root = len(parts) - 1
    while root > 0 and "/".join(parts[:root]) in packages:
        root -= 1
    return ".".join(parts[root:])
