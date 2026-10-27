"""What a checkout's own ``.gitignore`` excludes, split the way a disk walk needs it.

Two whole-tree guards walk the checkout and judge everything they find:
``scripts/check_no_tmp_literals.py`` and ``scripts/ci/check_profile_archive_boundary.py``. A
gitignored directory holds another tool's local state - a treehouse worktree slot, the desktop
packager's unpacked build inputs, a virtualenv - never source, so both must skip it. They cannot
skip it by name: the rules live in ``.gitignore`` files that differ per checkout and change over
time.

Git reports an embedded repository as **one directory entry** and never as its files, which is
why the prune has to happen on directories. A guard that tests only the file paths it is about
to read matches nothing inside such a tree: it prunes no directory, so it walks a whole foreign
checkout as if it were source (observed: 570 ``/tmp`` literals from a single ``.treehouse`` slot).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IgnoredPaths:
    """Ignore-rule exclusions under a checkout root, in the two shapes a walk needs.

    ``directories`` are whole subtrees git ignores, and a caller MUST prune them from its
    ``os.walk``: nothing below one is listed individually (``--directory`` collapses an ignored
    tree to a single entry, which is also why this costs one entry instead of one per file).
    An embedded repository is never listed as its files either - it arrives as a directory
    entry, its own boundary or an ignored ancestor that swallowed the whole tree around it.

    ``files`` are single ignored entries outside those subtrees - a runner artifact such as
    ``test_durations.json``, or a pattern match such as ``*.tar.gz``. Skip these by membership
    when reading files; they are not directories to prune.
    """

    directories: frozenset[Path]
    files: frozenset[Path]


def ignored_paths(root: Path) -> IgnoredPaths:
    """Ignore-rule exclusions under *root*. Both sets are empty when *root* is not a checkout.

    Only UNTRACKED entries are reported (``--others``), so the set is scoped by what git does
    not track rather than by what a path looks like: ``git add -f`` on a path the rules would
    otherwise ignore puts it in the index, out of this set, and a guard still sees it. That is
    the bypass an ignore file cannot close, and no guard may hide it.

    Paths carry *root* exactly as passed, so pass the root you intend to walk instead of
    resolving it separately here - a symlinked root would otherwise produce keys that never
    match the walk's own paths.

    A missing git, a directory outside any checkout, and a git that never answers all yield the
    empty result: the caller then walks the whole tree. That is the loud direction - a guard
    reports more, never less - so one environment failure cannot silently hide a violation.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard",
             "--directory", "-z"],
            capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return IgnoredPaths(frozenset(), frozenset())

    directories: set[Path] = set()
    files: set[Path] = set()
    for rel in out.split("\0"):
        if not rel:
            continue
        # A directory entry is spelled `dir/`; Path drops the separator, so the stored path
        # compares equal to the one a walk produces for that directory.
        (directories if rel.endswith("/") else files).add(root / rel)
    return IgnoredPaths(frozenset(directories), frozenset(files))
