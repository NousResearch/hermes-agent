"""Serve a moved module under each of its old names as the one real module object."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util


def current_name(name: str, moves: dict[str, str]) -> str | None:
    """The current name for ``name`` by its longest dotted prefix that is a key of ``moves``."""
    parts = name.split(".")
    for size in range(len(parts), 0, -1):
        old = ".".join(parts[:size])
        if old in moves:
            return moves[old] + name[len(old):]
    return None


def _child_names(moves: dict[str, str]) -> dict[str, str]:
    """The name ``from <old package> import <child>`` asks for when the child moved apart from its package.

    That form imports ``<package.__name__>.<child>``: the child's name under the package's CURRENT name.
    """
    derived = {}
    for old, new in moves.items():
        parent, _, child = old.rpartition(".")
        moved_parent = current_name(parent, moves) if parent else None
        name = f"{moved_parent}.{child}" if moved_parent is not None else None
        if name is not None and name != new and name not in moves:
            derived[name] = new
    return derived


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


class _AliasSpec(importlib.machinery.ModuleSpec):
    def __init__(self, name: str, target: str, spec) -> None:
        # The real origin keeps ``__file__``; the package flag lets ``runpy`` run an old package's ``__main__``.
        super().__init__(name, _Alias(target), origin=spec.origin,
                         is_package=spec.submodule_search_locations is not None)
        self._current_parent = spec.parent

    @property
    def parent(self):
        # runpy executes the current code, whose relative imports need the current package.
        return self._current_parent


class Finder(importlib.abc.MetaPathFinder):
    """Alias old names to current ones; with ``strict``, refuse an old name and name the new one."""

    def __init__(self, moves: dict[str, str], *, strict: bool = False) -> None:
        self.moves = moves
        self.child_names = _child_names(moves)
        self.strict = strict

    def find_spec(self, fullname, path=None, target=None):
        new = current_name(fullname, self.moves)
        if new is None and fullname in self.child_names:
            # A derived name is a current-package spelling: a module that really exists there wins.
            if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
                return None
            new = self.child_names[fullname]
        if new is None:
            return None
        if self.strict:
            raise ImportError(f"{fullname} has moved: import {new}", name=fullname)
        parent, _, child = fullname.rpartition(".")
        moved_parent = current_name(parent, self.moves) if parent else None
        displaced = f"{moved_parent}.{child}" if moved_parent is not None else None
        # Old and current names share one parent package object, so binding the old child name would
        # replace the real module of that name on it. That table entry cannot be served; say why.
        if displaced not in (None, new) and importlib.machinery.PathFinder.find_spec(displaced, path) is not None:
            raise ImportError(f"{fullname} cannot be served as {new}: the shared package {moved_parent} already "
                              f"has a module {displaced}", name=fullname)
        try:
            spec = importlib.util.find_spec(new)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            # A process that outlived a return to the old layout (a repaired or rolled-back update)
            # still has this finder: the module is at its old path again, and the ordinary finders load it.
            if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
                return None
            raise ModuleNotFoundError(f"No module named {new!r}", name=new)
        return _AliasSpec(fullname, new, _canonical(spec))
