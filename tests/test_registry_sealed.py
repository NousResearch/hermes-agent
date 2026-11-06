"""A sealed registry container refuses every inherited way of changing it while frozen.

Guarding the obvious mutators is not enough: ``dict.__init__`` re-run on a live container replaces
its contents, so ``registry.get_schema(name).__init__({...})`` would rewrite a frozen schema every
tenant sees. Copies and pickles stay plain writable containers, and reads are unchanged.
"""

from __future__ import annotations

import copy
import pickle

import pytest

import registry_freeze
from registry_freeze import RegistryFrozen
from registry_sealed import SealedDict, SealedList, SealedSet, seal


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    sealed = seal({"d": {"k": "v"}, "l": ["a"], "s": {"x"}})
    registry_freeze.freeze_registries()
    return sealed


def _rewrites(sealed):
    return {
        "dict __init__": lambda: sealed.__init__({"k": "after"}),
        "dict __init__ kwargs": lambda: sealed["d"].__init__(k="after"),
        "dict __ior__": lambda: sealed["d"].__ior__({"k": "after"}),
        "list __init__": lambda: sealed["l"].__init__(["after"]),
        "set __init__": lambda: sealed["s"].__init__({"after"}),
        "class swap": lambda: object.__setattr__(sealed, "__class__", dict),
        "new attribute": lambda: object.__setattr__(sealed, "extra", 1),
    }


@pytest.mark.parametrize("path", list(_rewrites({"d": {}, "l": [], "s": set()})))
def test_sealed_containers_refuse_reinitialisation(frozen, path):
    with pytest.raises((RegistryFrozen, TypeError, AttributeError)):
        _rewrites(frozen)[path]()
    assert type(frozen) is SealedDict
    assert frozen == {"d": {"k": "v"}, "l": ["a"], "s": {"x"}}


def test_reinitialisation_is_refused_by_the_freeze_itself(frozen):
    """``__init__`` refuses with the registry's own error, not an incidental TypeError."""
    for container, value in ((frozen, {}), (frozen["l"], []), (frozen["s"], set())):
        with pytest.raises(RegistryFrozen):
            container.__init__(value)


def test_copies_and_pickles_are_plain_and_leave_the_original(frozen):
    for clone in (copy.copy(frozen), copy.deepcopy(frozen), pickle.loads(pickle.dumps(frozen))):
        assert type(clone) is dict
        clone["d"] = {"k": "after"}
    deep = copy.deepcopy(frozen)
    assert type(deep["l"]) is list and type(deep["s"]) is set
    deep["l"].append("b")
    assert frozen["l"] == ["a"] and frozen["d"] == {"k": "v"}
    assert isinstance(frozen["l"], SealedList) and isinstance(frozen["s"], SealedSet)
