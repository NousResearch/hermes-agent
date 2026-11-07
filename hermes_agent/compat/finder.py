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


def _with_child_names(moves: dict[str, str]) -> dict[str, str]:
    """Add the name ``from <old package> import <child>`` asks for when the child moved apart from its package.

    That form imports ``<package.__name__>.<child>``: the child's name under the package's CURRENT name.
    """
    derived = {}
    for old, new in moves.items():
        parent, _, child = old.rpartition(".")
        moved_parent = current_name(parent, moves) if parent else None
        if moved_parent is not None and f"{moved_parent}.{child}" != new:
            derived[f"{moved_parent}.{child}"] = new
    return {**derived, **moves}


def _canonical(spec):
    # A concurrent alias import of the same module, between its two steps, leaves its alias spec on the
    # module; that alias's loader holds the real one.
    while isinstance(spec.loader, _Alias):
        spec = spec.loader.spec
    return spec


class _Alias(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.spec = _canonical(module.__spec__)
        return module

    def exec_module(self, module) -> None:
        # The import system has just set the alias spec on the real module; reload() needs the real one.
        module.__spec__ = self.spec

    def get_code(self, fullname):
        return _canonical(importlib.util.find_spec(self.target)).loader.get_code(self.target)


class Finder(importlib.abc.MetaPathFinder):
    """Alias old names to current ones; with ``strict``, refuse an old name and name the new one."""

    def __init__(self, moves: dict[str, str], *, strict: bool = False) -> None:
        self.moves = _with_child_names(moves)
        self.strict = strict

    def find_spec(self, fullname, path=None, target=None):
        new = current_name(fullname, self.moves)
        if new is None:
            return None
        if self.strict:
            raise ImportError(f"{fullname} has moved: import {new}", name=fullname)
        spec = importlib.util.find_spec(new)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {new!r}", name=new)
        # The real origin keeps ``__file__``; the package flag lets ``runpy`` run an old package's ``__main__``.
        return importlib.util.spec_from_loader(
            fullname, _Alias(new), origin=spec.origin, is_package=spec.submodule_search_locations is not None)
