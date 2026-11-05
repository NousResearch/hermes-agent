"""The legacy store for the tests that compare the backend with it, read from a pinned main checkout.

``HERMES_STATE_ORACLE_TREE`` names the checkout. Inside ``legacy_modules()`` every ``hermes_state`` and
``hermes_state_*`` import, main's hosted-room gateway modules (``gateway.hosted_room*``), and the tree modules the
legacy store imports that this tree has changed since, resolve to that checkout's files and nowhere else; on leaving it the names go back to what they were. A ``hermes_state*``
module whose file this tree carries byte for byte as the checkout has it is main's already: it stays this tree's, so the
legacy store and the backend share it, as they do when the backend imports it. Without the variable a test that needs
the checkout is skipped.
"""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_TREE = os.environ.get("HERMES_STATE_ORACLE_TREE")
ORACLE_TREE = Path(_TREE).resolve() if _TREE else None
SKIP_REASON = "compares with the legacy store: set HERMES_STATE_ORACLE_TREE to a pinned main checkout"

# Imported by the checkout's legacy store and changed in this tree.
_CHANGED_IMPORTS = frozenset({"agent.transcript_repair"})
# Main's hosted-room gateway modules, which this tree moves onto the hosted-room repositories.
_GATEWAY_PREFIX = "gateway.hosted_room"
# The checkout's modules once imported, kept between scopes so each is imported once per process.
_checkout_modules: dict[str, object] = {}
_OWN_TREE = Path(__file__).resolve().parents[2]


def oracle_tree() -> Path:
    """The pinned checkout; skips the test when none is named."""
    if ORACLE_TREE is None:
        pytest.skip(SKIP_REASON)
    return ORACLE_TREE


@functools.cache
def _same_as_checkout(name: str) -> bool:
    """This tree's file for module *name* is byte for byte the checkout's."""
    relative = Path(*name.split(".")).with_suffix(".py")
    own, main = _OWN_TREE / relative, ORACLE_TREE / relative
    return own.is_file() and main.is_file() and own.read_bytes() == main.read_bytes()


def _from_checkout(name: str) -> bool:
    top = name.partition(".")[0]
    if top == "hermes_state" or top.startswith("hermes_state_"):
        return not _same_as_checkout(name)
    return name in _CHANGED_IMPORTS or name.startswith(_GATEWAY_PREFIX)


class _CheckoutFinder:
    """Finds the checkout's modules in the checkout, and refuses one the checkout does not have."""

    def find_spec(self, name, path=None, target=None):
        if not _from_checkout(name):
            return None
        package, _, _ = name.rpartition(".")
        spec = importlib.machinery.PathFinder.find_spec(name, [str(ORACLE_TREE.joinpath(*package.split(".")))])
        if spec is None:
            raise ModuleNotFoundError(f"{name} is not in {ORACLE_TREE}", name=name)
        return spec


def _bind(modules: dict[str, object]) -> None:
    """Put *modules* in ``sys.modules`` and, for a submodule, on its package."""
    sys.modules.update(modules)
    for name, module in modules.items():
        package, _, leaf = name.rpartition(".")
        if package in sys.modules:
            setattr(sys.modules[package], leaf, module)


def _unbind() -> dict[str, object]:
    """Take the checkout's names out of ``sys.modules`` and off their packages; returns what was there."""
    taken = {name: sys.modules.pop(name) for name in [name for name in sys.modules if _from_checkout(name)]}
    for name in taken:
        package, _, leaf = name.rpartition(".")
        if package in sys.modules:
            vars(sys.modules[package]).pop(leaf, None)
    return taken


@contextmanager
def legacy_modules():
    """Import the legacy store from the pinned checkout for the block; skips the test when none is named."""
    oracle_tree()
    own = _unbind()
    _bind(_checkout_modules)
    finder = _CheckoutFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        _checkout_modules.update(_unbind())
        _bind(own)


@contextmanager
def legacy_scope():
    """``legacy_modules()``, or nothing when a scope is already active: a helper that a test holding the scope calls
    keeps that scope's modules."""
    if any(isinstance(finder, _CheckoutFinder) for finder in sys.meta_path):
        yield
    else:
        with legacy_modules():
            yield


def legacy_import(name: str):
    """The pinned checkout's module *name*, the one object every scope of this process binds."""
    with legacy_scope():
        return importlib.import_module(name)


@pytest.fixture
def legacy_store():
    """The pinned checkout's legacy store importable for the whole test, its setup and teardown included."""
    with legacy_modules():
        yield
