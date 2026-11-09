"""Repo paths through the same ``MOVES`` table the import finder reads.

Guards, lint baselines and scan lists name files by the path they had before a move. ``current_path``
says where such a path lives now (to find files); ``original_path`` says what a file's path was
before any move (to judge a file against keys written in the old paths). A data file follows its
directory. Both leave a path alone when nothing moved it, so with ``MOVES`` empty they are the identity.
"""
from __future__ import annotations

from fnmatch import fnmatch

from hermes_agent.compat.finder import current_name
from hermes_agent.compat.moves import MOVES


def _directory(path: str, table: dict[str, str]) -> str:
    parts = path.split("/")
    if not all(parts) or any("." in part for part in parts):
        return path
    mapped = current_name(".".join(parts), table)
    return path if mapped is None else mapped.replace(".", "/")


def _map(path: str, table: dict[str, str]) -> str:
    if not table:
        return path
    if path.endswith("/"):
        return _directory(path[:-1], table) + "/"
    head, _, last = path.rpartition("/")
    if last == "__init__.py":
        return _directory(head, table) + "/__init__.py" if head else path
    if last.endswith(".py"):
        return _directory(f"{head}/{last[:-3]}" if head else last[:-3], table) + ".py"
    if "." in last:
        return f"{_directory(head, table)}/{last}" if head else path
    return _directory(path, table)


class PathMoves:
    def __init__(self, moves: dict[str, str]) -> None:
        self.moves = moves
        self.inverse: dict[str, str] = {}
        for old, new in moves.items():
            self.inverse.setdefault(new, old)

    def current(self, path: str) -> str:
        return _map(path, self.moves)

    def original(self, path: str) -> str:
        return _map(path, self.inverse)

    def module(self, name: str) -> str:
        return current_name(name, self.moves) or name

    def original_module(self, name: str) -> str:
        return current_name(name, self.inverse) or name

    def current_globs(self, pattern: str) -> tuple[str, ...]:
        """``pattern`` and the patterns that find its moved matches, for a glob over old paths."""
        if not self.moves or not frozenset("*?[") & set(pattern):
            return (self.current(pattern),)
        head, _, tail = pattern.rpartition("/")
        found = {pattern}
        if head and not frozenset("*?[") & set(head):
            found.add(f"{self.current(head)}/{tail}")
        for old in self.moves:
            candidate = old.replace(".", "/") + ".py"
            if fnmatch(candidate, pattern):
                new = self.current(candidate)
                same_name = new.rpartition("/")[2] == candidate.rpartition("/")[2]
                found.add(f"{new.rpartition('/')[0]}/{tail}" if same_name else new)
        return (pattern, *sorted(found - {pattern}))


_DEFAULT = PathMoves(MOVES)
current_path = _DEFAULT.current
original_path = _DEFAULT.original
current_module = _DEFAULT.module
original_module = _DEFAULT.original_module
current_globs = _DEFAULT.current_globs
