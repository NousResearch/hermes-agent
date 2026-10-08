"""Read-only directory overlay for native skill discovery before manifest mutations.

Directory symlinks retain their lexical paths, including currently dangling aliases
whose targets the proposed write would create. Native discovery still owns pruning.
"""

import errno
from pathlib import Path



def _future_dirs(real: Path, changes: dict[Path, bool]) -> list[str]:
    names = []
    for target, present in changes.items():
        if present and target.parent.is_relative_to(real):
            parts = target.parent.relative_to(real).parts
            if parts and parts[0] not in names:
                names.append(parts[0])
    return names


def _entries(path: Path, changes: dict[Path, bool], *, sort=True, is_dir=None):
    real = path.resolve()
    dirs, files = [], []
    try:
        entries = list(path.iterdir())
    except FileNotFoundError:
        entries = []  # Proposed writes may create a directory that is absent on disk.
    for child in entries:
        child_real = child.resolve()
        future_directory = is_dir is None and child.is_symlink() and any(
            present and target.parent.is_relative_to(child_real) for target, present in changes.items())
        if (is_dir(child) if is_dir else child.is_dir()) or future_directory:
            dirs.append(child.name)
        else:
            files.append(child.name)
    dirs.extend(name for name in _future_dirs(real, changes)
                if name not in dirs and (is_dir is None or is_dir(path / name)))
    manifest = real / "SKILL.md"
    if manifest in changes:
        files = [name for name in files if name != "SKILL.md"]
        if changes[manifest]:
            dirs = [name for name in dirs if name != "SKILL.md"]
            files.append("SKILL.md")
    return (sorted(dirs), sorted(files)) if sort else (dirs, files)


def walk_proposed_manifests(root, changes: dict[Path, bool], *, followlinks=True, sort=True, is_dir=None):
    """Yield os.walk-shaped nodes; respect the caller's in-place directory pruning."""
    stack = [(Path(root), frozenset())]
    while stack:
        path, ancestors = stack.pop()
        real = path.resolve()
        if real in ancestors:
            raise OSError(errno.ELOOP, "Cyclic directory symlink in skill discovery", str(path))
        dirs, files = _entries(path, changes, sort=sort, is_dir=is_dir)
        yield str(path), dirs, files
        ancestors = ancestors | {real}
        stack.extend((path / name, ancestors) for name in reversed(dirs)
                     if followlinks or not (path / name).is_symlink())
