"""pytest's basetemp must never sit inside the operator's platform-native Hermes home.

Every per-test sandbox is ``<basetemp>/.../hermes_test`` and ``get_default_hermes_root()``
prefers the platform-native home whenever ``HERMES_HOME`` sits *under* it — so a basetemp
inside the home silently turns the sandbox back into the live install (#111101).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from tests import conftest as suite_conftest


def _config_with_basetemp(given: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        _tmp_path_factory=SimpleNamespace(_given_basetemp=given),
        option=SimpleNamespace(basetemp=str(given) if given else None),
    )


def test_basetemp_inside_the_native_home_is_relocated_outside_it(tmp_path, monkeypatch):
    native = tmp_path / "native-home"
    native.mkdir()
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    config = _config_with_basetemp(native / ".repro")

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    relocated = config._tmp_path_factory._given_basetemp
    assert relocated is not None and not relocated.resolve().is_relative_to(native.resolve())
    assert config.option.basetemp == str(relocated)
    # The sandbox derived from it no longer resolves to the native root.
    monkeypatch.setenv("HERMES_HOME", str(relocated / "t0" / "hermes_test"))
    assert hermes_constants.get_default_hermes_root() == relocated / "t0" / "hermes_test"


def test_basetemp_outside_the_native_home_is_left_alone(tmp_path, monkeypatch):
    native = tmp_path / "native-home"
    native.mkdir()
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    given = tmp_path / "elsewhere"
    config = _config_with_basetemp(given)

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    assert config._tmp_path_factory._given_basetemp == given
    assert config.option.basetemp == str(given)


def test_fallback_root_escapes_a_repo_checked_out_inside_the_native_home(tmp_path, monkeypatch):
    # Default install: repo at ~/.hermes/hermes-agent and TEMP under the home (Windows).
    native = tmp_path / "native-home"
    (native / "hermes-agent").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.setattr(suite_conftest, "PROJECT_ROOT", native / "hermes-agent")
    monkeypatch.setattr(suite_conftest.tempfile, "gettempdir", lambda: str(native / "tmp"))
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)
    config = _config_with_basetemp(None)

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    relocated = config._tmp_path_factory._given_basetemp
    assert relocated is not None and not relocated.resolve().is_relative_to(native.resolve())


def _relocating_config(tmp_path, monkeypatch):
    native = tmp_path / "native-home"
    native.mkdir()
    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: native
    )
    monkeypatch.setattr(
        suite_conftest.tempfile, "gettempdir", lambda: str(native / "tmp")
    )
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)
    return native, _config_with_basetemp(native / ".repro")


def test_successful_sessionfinish_removes_the_relocated_basetemp(tmp_path, monkeypatch):
    native, config = _relocating_config(tmp_path, monkeypatch)
    suite_conftest._RELOCATED_BASETEMPS.clear()

    suite_conftest._relocate_basetemp_outside_operator_home(config)
    relocated = config._tmp_path_factory._given_basetemp
    assert relocated is not None and relocated.exists()

    suite_conftest.pytest_sessionfinish(None, exitstatus=0)

    assert not relocated.exists()
    assert suite_conftest._RELOCATED_BASETEMPS == []


def test_failed_sessionfinish_keeps_the_relocated_basetemp(tmp_path, monkeypatch):
    native, config = _relocating_config(tmp_path, monkeypatch)
    suite_conftest._RELOCATED_BASETEMPS.clear()

    suite_conftest._relocate_basetemp_outside_operator_home(config)
    relocated = config._tmp_path_factory._given_basetemp

    suite_conftest.pytest_sessionfinish(None, exitstatus=1)

    assert relocated is not None and relocated.exists()
    shutil.rmtree(relocated, ignore_errors=True)
    suite_conftest._RELOCATED_BASETEMPS.clear()


def test_next_session_sweeps_only_long_abandoned_basetemps(tmp_path, monkeypatch):
    native, config = _relocating_config(tmp_path, monkeypatch)
    suite_conftest._RELOCATED_BASETEMPS.clear()
    parent = native.parent
    stale = parent / "hermes-pytest-basetemp-stale"
    stale.mkdir()
    fresh = parent / "hermes-pytest-basetemp-fresh"
    fresh.mkdir()
    abandoned = suite_conftest._STALE_BASETEMP_RETENTION_S + 600
    os.utime(stale, (time.time() - abandoned,) * 2)

    suite_conftest._relocate_basetemp_outside_operator_home(config)

    assert not stale.exists()
    assert fresh.exists()
    suite_conftest._RELOCATED_BASETEMPS.clear()
