"""Tests for the Hermes-safe pytest wrapper."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hermes_safe_pytest.py"
_SPEC = importlib.util.spec_from_file_location("hermes_safe_pytest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
hermes_safe_pytest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hermes_safe_pytest
_SPEC.loader.exec_module(hermes_safe_pytest)


def test_build_pytest_plan_adds_bounded_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_SAFE_PYTEST_WORKERS", raising=False)
    monkeypatch.setenv("HERMES_SAFE_PYTEST_PYTHON", hermes_safe_pytest.sys.executable)
    plan = hermes_safe_pytest.build_pytest_plan(["tests/agent"], tmp_root=tmp_path)

    command = plan.command
    joined = " ".join(command)

    assert command[:3] == [hermes_safe_pytest._pytest_python(), "-m", "pytest"]
    assert "tests/agent" in command
    assert any(arg.startswith("--basetemp=") for arg in command)
    assert "-o" in command and "addopts=" in command
    assert "-p" in command and "no:cacheprovider" in command
    assert "--maxfail=1" in command
    assert "--tb=short" in command
    assert "-n" in command and "0" in command
    assert str(tmp_path) in joined


def test_build_pytest_plan_preserves_explicit_basetemp_and_can_keep_workers(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_SAFE_PYTEST_WORKERS", "keep")
    explicit = tmp_path / "explicit-base"

    plan = hermes_safe_pytest.build_pytest_plan(
        ["-q", "--basetemp", str(explicit), "-n", "2"],
        tmp_root=tmp_path,
    )

    assert plan.basetemp == explicit
    assert plan.command.count("-n") == 1
    assert "2" in plan.command
    assert f"--basetemp={explicit}" not in plan.command


def test_cleanup_known_pytest_artifacts_removes_only_matching_paths(tmp_path):
    doomed_dir = tmp_path / "pytest-of-root"
    doomed_dir.mkdir()
    (doomed_dir / "big.tmp").write_bytes(b"x" * 10)
    doomed_file = tmp_path / "hermes-pytest-log"
    doomed_file.write_text("log")
    keep = tmp_path / "important"
    keep.mkdir()
    (keep / "data").write_text("keep")

    removed = hermes_safe_pytest.cleanup_known_pytest_artifacts(tmp_path)

    assert removed >= 13
    assert not doomed_dir.exists()
    assert not doomed_file.exists()
    assert keep.exists()
    assert (keep / "data").read_text() == "keep"


def test_preflight_disk_refuses_when_cleanup_cannot_restore_space(monkeypatch, tmp_path):
    monkeypatch.setattr(hermes_safe_pytest, "_free_gb", lambda _path: 1.0)
    monkeypatch.setattr(hermes_safe_pytest, "cleanup_known_pytest_artifacts", lambda _tmp: 0)

    with pytest.raises(SystemExit, match="refusing to run"):
        hermes_safe_pytest.preflight_disk(8.0, path=tmp_path, tmp_root=tmp_path)
