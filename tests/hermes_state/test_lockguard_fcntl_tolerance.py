"""A partial or lookalike ``fcntl`` must disable the WAL lock guard, never kill the process.

Regression for #118026: the module gated only on ``ImportError``, so a Windows install whose
venv has some other module importable as ``fcntl`` (stock CPython for Windows ships none) hit an
unguarded ``fcntl.F_RDLCK`` read at import time. ``hermes_state`` imports this module at module
level, so the ``AttributeError`` killed every importer — ``hermes serve`` (Desktop backend
"exited before port announcement (1)"), ``hermes doctor``, the gateway, cron — before
``supported()`` existed to report the guard as unavailable.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest


def _reimport_with_fcntl(monkeypatch, stub: types.ModuleType | None):
    """Import the guard fresh with ``fcntl`` replaced (or absent), leaving sys.modules clean."""
    monkeypatch.delitem(sys.modules, "hermes_state_lockguard", raising=False)

    if stub is None:
        monkeypatch.setitem(sys.modules, "fcntl", None)  # import fcntl -> ImportError
    else:
        monkeypatch.setitem(sys.modules, "fcntl", stub)

    return importlib.import_module("hermes_state_lockguard")


def _windows_fcntl_lookalike() -> types.ModuleType:
    """A stand-in exposing only flock/LOCK_*, as reported on the affected Windows installs."""
    stub = types.ModuleType("fcntl")
    stub.flock = lambda *args, **kwargs: None
    stub.LOCK_EX = 2
    stub.LOCK_SH = 1
    stub.LOCK_UN = 8
    return stub


def _constants_only_fcntl() -> types.ModuleType:
    """The other partial shape: every lock constant present, no ``fcntl()`` callable. Passing the
    constant checks alone would report ``supported()`` and then raise from the first hold()."""
    stub = types.ModuleType("fcntl")
    stub.F_OFD_SETLK = 37
    stub.F_RDLCK = 0
    stub.F_UNLCK = 2
    return stub


@pytest.mark.parametrize(
    "stub",
    [
        pytest.param(None, id="absent"),
        pytest.param(_windows_fcntl_lookalike(), id="partial"),
        pytest.param(_constants_only_fcntl(), id="constants-only"),
    ],
)
def test_guard_degrades_to_a_no_op_instead_of_killing_every_importer(monkeypatch, tmp_path, stub):
    guard = _reimport_with_fcntl(monkeypatch, stub)

    # An open descriptor on the file is what makes hold() reach fcntl.fcntl(); without one the
    # constants-only shape would pass on a vacuous loop.
    db = tmp_path / "state.db"
    fd = os.open(db, os.O_RDWR | os.O_CREAT)
    try:
        # The contract the crash violated: the module imports, reports itself unavailable, and
        # hold() hands back an empty guard so callers take the ordinary unguarded path.
        assert guard.supported() is False
        assert guard.hold(db) == {}
        guard.release({})
    finally:
        os.close(fd)


@pytest.mark.windows_only
def test_windows_never_arms_the_guard_even_with_an_importable_fcntl(monkeypatch):
    # Stock CPython for Windows has no fcntl, so the lookalike is the only way this branch can
    # see an importable module; the platform gate must decide before the import does.
    guard = _reimport_with_fcntl(monkeypatch, _windows_fcntl_lookalike())

    assert guard.fcntl is None
    assert guard.supported() is False
    assert guard.hold("state.db") == {}


def test_a_usable_fcntl_still_arms_the_guard(monkeypatch):
    real_fcntl = pytest.importorskip("fcntl", reason="POSIX-only: nothing to arm on Windows")

    guard = _reimport_with_fcntl(monkeypatch, real_fcntl)

    # The tolerance above must not have disabled the guard everywhere.
    assert guard.supported() is True
