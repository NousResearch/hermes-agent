"""pytest's basetemp must never sit inside the operator's platform-native Rabbit home.

Every per-test sandbox is ``<basetemp>/.../rabbit_test`` and ``get_default_rabbit_root()``
prefers the platform-native home whenever ``RABBIT_HOME`` sits *under* it — so a basetemp
inside the home silently turns the sandbox back into the live install (#111101).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rabbit_constants
from tests import conftest as suite_conftest


@pytest.fixture(autouse=True)
def _sibling_root_only(monkeypatch):
    """Relocate into the sibling-of-native root, never the host's shared disk root: these
    tests use a fake native home under tmp_path and must not create or prune anything real."""
    import os

    real_isdir = os.path.isdir
    monkeypatch.setattr(suite_conftest.os.path, "isdir", lambda p: False if p == "/var/tmp" else real_isdir(p))  # no-tmp: ok — disables the disk-root branch


def _config_with_basetemp(given: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        _tmp_path_factory=SimpleNamespace(_given_basetemp=given),
        option=SimpleNamespace(basetemp=str(given) if given else None),
    )


def test_basetemp_inside_the_native_home_is_relocated_outside_it(tmp_path, monkeypatch):
    native = tmp_path / "native-home"
    native.mkdir()
    monkeypatch.setattr(rabbit_constants, "_get_platform_default_rabbit_home", lambda: native)
    config = _config_with_basetemp(native / ".repro")

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    relocated = config._tmp_path_factory._given_basetemp
    assert relocated is not None and not relocated.resolve().is_relative_to(native.resolve())
    assert config.option.basetemp == str(relocated)
    # One shared, prunable root (never a loose dir in the operator's $HOME) and gone when
    # this pytest exits; a run killed before that is swept as soon as the root is idle.
    assert relocated.parent.name == "rabbit-pytest" and relocated.parent != Path.home()
    suite_conftest.pytest_unconfigure(config)
    assert not relocated.exists()
    # The sandbox derived from it no longer resolves to the native root.
    monkeypatch.setenv("RABBIT_HOME", str(relocated / "t0" / "rabbit_test"))
    assert rabbit_constants.get_default_rabbit_root() == relocated / "t0" / "rabbit_test"


def test_basetemp_outside_the_native_home_is_left_alone(tmp_path, monkeypatch):
    native = tmp_path / "native-home"
    native.mkdir()
    monkeypatch.setattr(rabbit_constants, "_get_platform_default_rabbit_home", lambda: native)
    given = tmp_path / "elsewhere"
    config = _config_with_basetemp(given)

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    assert config._tmp_path_factory._given_basetemp == given
    assert config.option.basetemp == str(given)


def test_fallback_root_escapes_a_repo_checked_out_inside_the_native_home(tmp_path, monkeypatch):
    # Default install: repo at ~/.rabbit/rabbit-agent and TEMP under the home (Windows).
    native = tmp_path / "native-home"
    (native / "rabbit-agent").mkdir(parents=True)
    monkeypatch.setattr(rabbit_constants, "_get_platform_default_rabbit_home", lambda: native)
    monkeypatch.setattr(suite_conftest, "PROJECT_ROOT", native / "rabbit-agent")
    monkeypatch.setattr(suite_conftest.tempfile, "gettempdir", lambda: str(native / "tmp"))
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)
    config = _config_with_basetemp(None)

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    relocated = config._tmp_path_factory._given_basetemp
    assert relocated is not None and not relocated.resolve().is_relative_to(native.resolve())


def test_relocation_root_sweeps_basetemps_of_killed_runs_and_keeps_live_ones(tmp_path, monkeypatch):
    import os
    import time

    native = tmp_path / "native-home"
    native.mkdir()
    root = native.parent / "rabbit-pytest"
    dead, live = root / "b-dead", root / "b-live"
    dead.mkdir(parents=True)
    live.mkdir()
    (dead / "f").write_text("x", encoding="utf-8")
    (live / "f").write_text("x", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for p in (dead, dead / "f", live):
        os.utime(p, (ancient, ancient))
    assert suite_conftest._pytest_disk_temp_root(native) == root
    assert not dead.exists() and live.exists()
