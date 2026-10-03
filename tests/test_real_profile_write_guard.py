"""Canary for the real profile-registry write guard in ``tests/conftest.py``.

This file is the mutation probe: it performs, deliberately, exactly the leak
the guard exists to stop — creating a profile directory under the operator's
real ``~/.hermes/profiles`` — and asserts the guard refuses it. Weaken the
guard and these tests stop raising, which is a failure here.

Why the guard exists: a full-suite run once left eight fixture directories in a
live ``~/.hermes/profiles`` (``builder-auth``, ``work``, ``worker``,
``coder``, ``demo``, ``ops``, ``secondary``, ``yangyang``). Every test passed.
But ``hermes profile list`` enumerates that directory, so the fixtures joined
the operator's fleet as real profiles, and the per-profile watchdogs began
reporting on agents that never existed.

FAIL-CLOSED, like ``test_live_system_guard_self_test.py``: the primitives
below are REAL ``mkdir``/``open`` calls aimed at the real home. In any
collection context where this file is present but its home ``conftest.py`` is
not (published sdists that ship ``tests/`` without ``conftest.py``, trees
assembled by copying ``test*.py`` — that glob does not match ``conftest.py`` —
``pytest --noconftest``, a foreign rootdir), the guard never loaded and the
probe would BE the leak. So every test refuses to run unless the guard is
provably installed.
"""
from __future__ import annotations

import builtins
import io
import os
import types
from pathlib import Path

import pytest

#: Names used by the probes. Distinctive on purpose: if the guard ever fails
#: open, `ls ~/.hermes/profiles` names the culprit file immediately.
_PROBE_DIR = "zz-write-guard-probe-should-never-exist"
_PROBE_FILE = "zz-write-guard-probe-should-never-exist.yaml"


def _guard_is_active() -> bool:
    """True iff conftest's guard has patched the write primitives.

    Detected structurally, not by name: the guard replaces the C builtins with
    plain Python functions carrying its marker attribute. An unpatched
    ``os.mkdir`` is a ``types.BuiltinFunctionType``.
    """
    for primitive in (os.mkdir, io.open, builtins.open):
        if isinstance(primitive, types.BuiltinFunctionType):
            return False
        if not getattr(primitive, "_hermes_real_profile_guard", False):
            return False
    return True


@pytest.fixture(autouse=True)
def _require_guard():
    if not _guard_is_active():
        pytest.fail(
            "REFUSING TO RUN: the real-profile write guard from "
            "tests/conftest.py is not installed, so the probes below would "
            "actually create directories in the operator's ~/.hermes/profiles "
            "— they ARE the leak they test for. Run this file from the repo "
            "rootdir with its conftest.py, without --noconftest."
        )


@pytest.fixture
def real_profiles_root() -> Path:
    """The real registry the guard protects, read from the guard itself.

    Taken from conftest rather than recomputed here: a probe that recomputed
    the path would keep passing after the guard started watching a different
    (or empty) set of roots.
    """
    from tests.conftest import _REAL_PROFILE_ROOTS

    if not _REAL_PROFILE_ROOTS:
        pytest.fail(
            "the guard captured no profile roots — it is watching nothing"
        )
    return _REAL_PROFILE_ROOTS[0]


def test_guard_watches_the_pre_sandbox_home_not_the_test_sandbox():
    """The deny-list must point at the operator's home, not the tempdir.

    This is the kanban guard's #69385 lesson restated: capture the real root
    BEFORE the session sandbox rewires ``HERMES_HOME``, or the guard silently
    protects a throwaway directory and stops protecting anything that matters.
    """
    from tests.conftest import _REAL_PROFILE_ROOTS

    expected = (Path.home() / ".hermes" / "profiles").resolve()
    assert expected in _REAL_PROFILE_ROOTS
    # And the sandbox HERMES_HOME — the thing a naive capture would grab — is
    # NOT what the guard is watching.
    sandbox = Path(os.environ["HERMES_HOME"]).resolve() / "profiles"
    assert sandbox not in _REAL_PROFILE_ROOTS


def test_mkdir_of_a_fixture_profile_is_refused(real_profiles_root: Path):
    """The leak itself: ``mkdir`` of a new profile under the real registry."""
    victim = real_profiles_root / _PROBE_DIR
    with pytest.raises(RuntimeError, match="real_profile_write_guard"):
        victim.mkdir(parents=True, exist_ok=True)
    assert not victim.exists(), f"guard failed open: {victim} was created"


def test_makedirs_of_a_nested_fixture_path_is_refused(real_profiles_root: Path):
    """``os.makedirs`` is the other spelling and must be covered too.

    It is not a separate patch — ``os.makedirs`` resolves ``mkdir`` on the
    ``os`` module at call time — but that is an implementation detail of
    CPython, and this test is what notices if it ever stops being true.
    """
    victim = real_profiles_root / _PROBE_DIR / "skills" / "leak"
    with pytest.raises(RuntimeError, match="real_profile_write_guard"):
        os.makedirs(victim, exist_ok=True)
    assert not (real_profiles_root / _PROBE_DIR).exists()


def test_writing_a_config_into_the_real_registry_is_refused(
    real_profiles_root: Path,
):
    """A file write is the same leak one level down.

    The leaked directories were not all empty: ``work`` carried a
    ``config.yaml`` with ``mcp_servers`` pointing at example.com. Writing into
    an EXISTING real profile needs no ``mkdir`` at all.
    """
    victim = real_profiles_root / _PROBE_FILE
    with pytest.raises(RuntimeError, match="real_profile_write_guard"):
        with open(victim, "w", encoding="utf-8") as fh:
            fh.write("mcp_servers: {}\n")
    assert not victim.exists()

    # pathlib reaches the same primitive through ``io.open``; patching only
    # ``builtins.open`` would leave this half open.
    with pytest.raises(RuntimeError, match="real_profile_write_guard"):
        victim.write_text("mcp_servers: {}\n", encoding="utf-8")
    assert not victim.exists()


def test_the_refusal_names_the_offending_test(real_profiles_root: Path):
    """A guard that fires anonymously turns a 2-second fix into a bisect."""
    with pytest.raises(RuntimeError) as excinfo:
        (real_profiles_root / _PROBE_DIR).mkdir()
    message = str(excinfo.value)
    assert "test_the_refusal_names_the_offending_test" in message
    assert str(real_profiles_root) in message


def test_reads_of_the_real_registry_still_work(real_profiles_root: Path):
    """Deny-list, not a wall: the guard must not break reads.

    Plenty of legitimate test code stats or lists paths under the real home;
    only writes are the leak.
    """
    real_profiles_root.exists()
    if real_profiles_root.is_dir():
        list(real_profiles_root.iterdir())
    with pytest.raises((FileNotFoundError, IsADirectoryError, PermissionError)):
        # Opening a non-existent path for READING must fail the ordinary way,
        # not with the guard's RuntimeError.
        open(real_profiles_root / _PROBE_FILE, encoding="utf-8").close()


def test_hermetic_writes_are_untouched(tmp_path: Path):
    """The other half of "deny-list": ordinary tempdir work stays free.

    Including a path that mentions ``profiles`` — the guard's cheap pre-filter
    keys on that word, and a filter that turned into the verdict would reject
    every hermetic profile fixture in the suite.
    """
    hermetic = tmp_path / "hermes_home" / "profiles" / "work"
    hermetic.mkdir(parents=True)
    (hermetic / "config.yaml").write_text("mcp_servers: {}\n", encoding="utf-8")
    assert (hermetic / "config.yaml").read_text(encoding="utf-8")

    nested = tmp_path / "deep"
    os.makedirs(nested / "profiles" / "other")
    assert (nested / "profiles" / "other").is_dir()
