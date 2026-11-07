"""Serve a moved module under each of its old names as the one real module object."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util


def current_name(name: str, moves: dict[str, str]) -> str | None:
    """The current name for ``name`` by its longest dotted prefix that is a key of ``moves``."""
    parts = name.split(".")
    for size in range(len(parts), 0, -1):
        old = ".".join(parts[:size])
        if old in moves:
            return moves[old] + name[len(old):]
    return None


class _Alias(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # The import system has just set the alias spec on the real module; reload() needs the real one.
        module.__spec__ = self.spec


class Finder(importlib.abc.MetaPathFinder):
    """Alias old names to current ones; with ``strict``, refuse an old name and name the new one."""

    def __init__(self, moves: dict[str, str], *, strict: bool = False) -> None:
        self.moves = moves
        self.strict = strict

    def find_spec(self, fullname, path=None, target=None):
        new = current_name(fullname, self.moves)
        if new is None:
            return None
        if self.strict:
            raise ImportError(f"{fullname} has moved: import {new}", name=fullname)
        return importlib.util.spec_from_loader(fullname, _Alias(new))
