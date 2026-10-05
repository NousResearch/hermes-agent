"""tests/_interpreter_guard.py: a run under the wrong interpreter stops before collection.

The decision is by environment identity (``sys.prefix`` against the checkout's selected
pm.testenv venv), never by interpreter name or version, and a checkout with no test
environment (CI lanes with their own interpreter, Nix, a fresh clone) is not judged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _interpreter_guard as guard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATION = "gen-" + "a" * 32


def _selected_test_environment(home: Path, monkeypatch) -> Path:
    """Lay out a selected pm.testenv generation for this checkout under *home*."""
    from pm.environments import install_state_dir

    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(home))
        root = install_state_dir(PROJECT_ROOT) / "test-environment"
    venv = root / GENERATION / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    (root / "active.json").write_text(json.dumps({"generation": GENERATION}), encoding="utf-8")
    return venv


def test_the_checkouts_own_test_environment_passes(monkeypatch):
    monkeypatch.setattr(guard, "_expected_test_python", lambda root: Path(sys.prefix) / "bin" / "python")
    assert guard.foreign_interpreter_message(PROJECT_ROOT, environ={}) is None


def test_a_checkout_without_a_test_environment_is_not_judged(monkeypatch):
    monkeypatch.setattr(guard, "_expected_test_python", lambda root: None)
    assert guard.foreign_interpreter_message(PROJECT_ROOT, environ={}) is None


def test_another_interpreter_is_refused_with_the_remedy(monkeypatch, tmp_path):
    expected = tmp_path / "test-environment" / "venv" / "bin" / "python"
    monkeypatch.setattr(guard, "_expected_test_python", lambda root: expected)
    message = guard.foreign_interpreter_message(PROJECT_ROOT, environ={})
    assert message is not None
    assert "Hermes tests must run inside the activated source environment" in message
    assert "source ./activate" in message and "scripts/run_tests.sh" in message
    assert sys.executable in message and str(expected) in message


def test_an_interpreter_that_cannot_import_pm_is_refused(monkeypatch):
    def unimportable(root):
        raise TypeError("unsupported operand type(s) for |")
    monkeypatch.setattr(guard, "_expected_test_python", unimportable)
    message = guard.foreign_interpreter_message(PROJECT_ROOT, environ={})
    assert message is not None and "cannot import the Hermes package manager" in message


def test_explicit_opt_out_is_honored(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "_expected_test_python", lambda root: tmp_path / "venv" / "bin" / "python")
    assert guard.foreign_interpreter_message(PROJECT_ROOT, environ={guard.OPT_OUT_ENV: "1"}) is None


def _collect(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "tests/test_interpreter_guard.py"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env, timeout=180,
    )


def test_pytest_stops_before_collection_under_a_foreign_interpreter(tmp_path, monkeypatch):
    """End to end through the real conftest: this checkout's selected test environment is
    some other venv, so the running interpreter is foreign and nothing is collected."""
    home = tmp_path / "home"
    _selected_test_environment(home, monkeypatch)
    env = {key: value for key, value in os.environ.items() if key != guard.OPT_OUT_ENV}
    env["HERMES_HOME"] = str(home)

    refused = _collect(env)
    assert refused.returncode == pytest.ExitCode.USAGE_ERROR, refused.stdout[-2000:] + refused.stderr[-2000:]
    output = refused.stdout + refused.stderr
    assert "Hermes tests must run inside the activated source environment" in output
    assert "test_explicit_opt_out_is_honored" not in output  # stopped before collection

    env[guard.OPT_OUT_ENV] = "1"
    allowed = _collect(env)
    assert allowed.returncode == 0, allowed.stdout[-2000:] + allowed.stderr[-2000:]
    assert "test_explicit_opt_out_is_honored" in allowed.stdout
