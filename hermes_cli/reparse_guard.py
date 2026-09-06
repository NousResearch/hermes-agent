"""Delete a directory tree without ever traversing a Windows reparse point.

**Why this exists** (measured 2026-09-06, real data loss on a user's box):

A kanban worker needed node dependencies inside its git worktree, so it made
``<worktree>/app/node_modules`` a Windows **junction** pointing at the primary
checkout's ``node_modules``. The task finished; the dispatcher called
``git worktree remove``; git's recursive delete **followed the junction as if it
were an ordinary directory**, walked into the primary ``node_modules``, and from
there into ``@real3d/stage`` and ``@real3d/atoms`` — which npm materialises as
*further junctions* for ``file:`` dependencies, pointing at two sibling source
repositories. Both were emptied. They were only recoverable because they had
been pushed.

The failure is silent by construction. It surfaced as
``git worktree remove failed ...: Invalid argument`` — a warning about a
*worktree*, saying nothing about the two repositories it had just walked
through and deleted.

**Why nothing caught it.** ``os.path.islink()`` returns **False** for a
junction: only NTFS symlinks (tag ``IO_REPARSE_TAG_SYMLINK``) set the flag
Python reports. A junction (tag ``IO_REPARSE_TAG_MOUNT_POINT``) answers
``islink() == False`` and ``isdir() == True``, so every ordinary "don't follow
links" guard — in git, in ``shutil.rmtree``, in our own code — sees a plain
directory and descends. The only reliable test is the raw
``FILE_ATTRIBUTE_REPARSE_POINT`` bit, which is what this module reads.

## The safety model

Three states, not two. Every edge encountered during a removal is classified
``ordinary | reparse | unknown``:

- **ordinary** — a real directory or file; recurse into it / unlink it;
- **reparse** — a junction, symlink or other reparse point; unlink *the link
  itself* and never look at what it points to. ``os.rmdir`` on a directory
  reparse point detaches the link without recursing;
- **unknown** — anything we could not classify (``os.lstat``/``os.scandir``
  raised). This **aborts the whole removal**. A metadata or access failure is
  exactly how the junction class this module exists to stop would hide, so an
  unclassified edge can never be treated as ordinary.

``remove_tree`` performs the deletion itself rather than classifying a tree and
then handing it to an external deleter. That is deliberate: a scan followed by
``git worktree remove`` is two generations, and a reparse point created in
between is still followable by the second one. Owning the walk means every edge
is traversed in the same step that classified it, so no *other* process gets a
turn between the check and the use.

The residual window is in-process and one edge wide: between ``lstat`` and the
``scandir`` of that same directory. Closing it completely needs handle-based
``O_NOFOLLOW``-style traversal, which CPython does not expose portably on
Windows; this module does not claim to have closed it, and the callers'
preconditions (the worktree is finished, its owning pid is dead) are what make
it acceptable in practice.

On non-Windows this is still correct but uninteresting: ``islink`` is honest
there and no deleter follows symlinks.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import List, Tuple

_log = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"


class UnknownEdge(Exception):
    """An edge could not be classified, so the tree must not be deleted.

    Carries the path that could not be classified and the underlying OS error,
    because "we refused to delete" is only actionable if it names what stopped
    it.
    """

    def __init__(self, path: os.PathLike | str, cause: BaseException | None = None):
        self.path = Path(path)
        self.cause = cause
        super().__init__(f"cannot classify {self.path}: {cause}")


def is_reparse_point(path: os.PathLike | str) -> bool:
    """True if *path* is a junction, symlink, or any other reparse point.

    Deliberately NOT ``os.path.islink``: that answers False for a junction,
    which is the exact blind spot that cost two repositories.

    Raises ``UnknownEdge`` when the entry cannot be classified — callers on a
    deletion path must treat that as "refuse", never as "ordinary". A missing
    entry is not unknown: it is nothing, and answers False.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise UnknownEdge(path, exc) from exc
    if not IS_WINDOWS:
        return stat.S_ISLNK(st.st_mode)
    return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _unlink_edge(path: Path) -> None:
    """Remove one entry: the LINK if it is a link, the file if it is a file.

    Never recurses and never touches a link's target. Raises ``OSError`` when
    the entry survives, which the caller turns into a refusal.
    """
    try:
        os.remove(path)
        return
    except (IsADirectoryError, PermissionError, OSError):
        # A directory (or a directory reparse point) needs rmdir; on Windows
        # os.remove refuses one with PermissionError rather than IsADirectory.
        os.rmdir(path)


def remove_tree(root: os.PathLike | str) -> int:
    """Delete *root* recursively, never traversing a reparse point.

    Returns the number of entries removed. Raises ``UnknownEdge`` — having
    deleted nothing further — when any edge cannot be classified.

    A reparse point AT *root* is unlinked, not followed: asking to delete a
    junction deletes the junction.
    """
    root_path = Path(root)
    if is_reparse_point(root_path):
        _unlink_edge(root_path)
        _log.info("Unlinked reparse point instead of descending: %s", root_path)
        return 1
    if not root_path.exists():
        return 0

    removed = 0
    # Post-order: children before their parent, iteratively so a deep tree
    # cannot blow the stack.
    stack: List[Tuple[Path, bool]] = [(root_path, False)]
    while stack:
        current, children_done = stack.pop()
        if children_done:
            _unlink_edge(current)
            removed += 1
            continue

        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise UnknownEdge(current, exc) from exc

        stack.append((current, True))
        for entry in entries:
            entry_path = Path(entry.path)
            # Classify FIRST — is_reparse_point raises UnknownEdge rather than
            # letting an unreadable entry be mistaken for an ordinary one.
            if is_reparse_point(entry_path):
                _unlink_edge(entry_path)          # the link, never its target
                removed += 1
                _log.info("Unlinked reparse point during removal: %s", entry_path)
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise UnknownEdge(entry_path, exc) from exc
            if is_dir:
                stack.append((entry_path, False))
            else:
                _unlink_edge(entry_path)
                removed += 1
    return removed


def try_remove_tree(root: os.PathLike | str) -> Tuple[bool, str]:
    """``remove_tree`` for best-effort cleanup paths.

    Returns ``(removed, reason)``. ``reason`` is empty on success and otherwise
    names what stopped the removal, so the caller can log a sentence that says
    which path refused — the thing the original incident's warning never did.
    """
    try:
        remove_tree(root)
        return (True, "")
    except UnknownEdge as exc:
        return (False, f"unclassifiable entry {exc.path} ({exc.cause})")
    except OSError as exc:
        return (False, f"{type(exc).__name__}: {exc}")
