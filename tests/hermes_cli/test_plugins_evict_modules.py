"""_evict_modules must not iterate the live ``sys.modules`` (#123926).

The loader evicts a plugin's cached modules before importing it. When the eviction
iterated ``sys.modules`` from Python bytecode, a concurrent import on another thread
(every ``import`` in the process touches the dict) changed its size mid-iteration; the
``RuntimeError`` surfaced to the loader, which disposed the plugin's registrations and
silently dropped it — a different random subset of plugins on every boot.
"""

from __future__ import annotations

import sys

from hermes_cli.plugins_loader import _evict_modules


class _MutatingModules(dict):
    """A ``sys.modules`` stand-in that imports something new on its first iteration.

    Reproduces the concurrent-import window deterministically: the real dict iterator
    raises ``RuntimeError`` on the next ``__next__`` after the size changed.
    """

    def __iter__(self):
        it = super().__iter__()
        first = next(it)
        dict.__setitem__(self, "__concurrent_import__", object())
        yield first
        yield from it


class _VanishingModules(dict):
    """A ``sys.modules`` stand-in whose snapshot contains a name the live dict lacks.

    Another thread can evict the same module between the snapshot and the delete.
    """

    def copy(self):
        snapshot = dict(self)
        snapshot["hermes_plugins.ghost"] = None
        return snapshot


def test_eviction_survives_a_concurrent_import(monkeypatch):
    fake = _MutatingModules({
        "hermes_plugins": None,
        "hermes_plugins.demo": None,
        "keepme": None,
    })
    monkeypatch.setattr(sys, "modules", fake)

    _evict_modules("hermes_plugins.demo")

    assert "hermes_plugins.demo" not in fake
    assert "hermes_plugins" in fake
    assert "keepme" in fake


def test_eviction_removes_package_and_submodules(monkeypatch):
    fake = {
        "hermes_plugins": None,
        "hermes_plugins.demo": None,
        "hermes_plugins.demo.helpers": None,
        "hermes_plugins.other": None,
        "keepme": None,
    }
    monkeypatch.setattr(sys, "modules", fake)

    _evict_modules("hermes_plugins.demo")

    assert "hermes_plugins.demo" not in fake
    assert "hermes_plugins.demo.helpers" not in fake
    assert "hermes_plugins" in fake
    assert "hermes_plugins.other" in fake
    assert "keepme" in fake


def test_eviction_tolerates_a_name_removed_after_the_snapshot(monkeypatch):
    fake = _VanishingModules({"hermes_plugins": None})
    monkeypatch.setattr(sys, "modules", fake)

    _evict_modules("hermes_plugins.ghost")  # must not raise KeyError

    assert fake == {"hermes_plugins": None}
