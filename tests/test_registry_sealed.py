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


def test_tuples_keep_their_shape_and_seal_their_contents(monkeypatch):
    from collections import namedtuple
    Pair = namedtuple("Pair", "a b")
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    inner = {"k": "v"}
    plain = ("x", 1)
    sealed = seal({"t": (inner, [1], ("y", {"z"})), "n": Pair({"k": 1}, 2), "p": plain})
    registry_freeze.freeze_registries()
    assert type(sealed["t"]) is tuple and type(sealed["n"]) is Pair
    assert sealed["p"] is plain  # nothing mutable inside: kept as is
    assert isinstance(sealed["t"][0], SealedDict) and sealed["t"][0] is not inner
    assert isinstance(sealed["t"][1], SealedList) and isinstance(sealed["t"][2][1], SealedSet)
    assert isinstance(sealed["n"].a, SealedDict)
    with pytest.raises(RegistryFrozen):
        sealed["t"][0]["k"] = "after"
    deep = copy.deepcopy(sealed)
    assert type(deep["t"]) is tuple and type(deep["t"][0]) is dict
    deep["t"][0]["k"] = "after"
    assert sealed["t"][0] == {"k": "v"}


def test_copies_and_pickles_are_plain_and_leave_the_original(frozen):
    for clone in (copy.copy(frozen), copy.deepcopy(frozen), pickle.loads(pickle.dumps(frozen))):
        assert type(clone) is dict
        clone["d"] = {"k": "after"}
    deep = copy.deepcopy(frozen)
    assert type(deep["l"]) is list and type(deep["s"]) is set
    deep["l"].append("b")
    assert frozen["l"] == ["a"] and frozen["d"] == {"k": "v"}
    assert isinstance(frozen["l"], SealedList) and isinstance(frozen["s"], SealedSet)
