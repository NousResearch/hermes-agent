"""Seam tests for the R3-S1 npm toolchain extraction (main.py god-file slice).

Covers the re-export seam between ``hermes_cli.main`` and the new sibling
module ``hermes_cli/npm_toolchain.py`` (epic #78647, target #78631):
object identity through the re-export, plus aggressive behavioral cases on
the moved npm command-building and toolchain-resolution paths.
"""

import os
import subprocess
import sys as _sys

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import npm_toolchain

MOVED_NAMES = (
    "_run_with_idle_timeout",
    "_nixos_build_env",
    "_run_npm_install_deterministic",
    "_run_npm_watching_for_engine_failure",
)


def test_reexport_preserves_object_identity():
    """Every moved name resolves on hermes_cli.main to the same object."""
    for name in MOVED_NAMES:
        assert getattr(hermes_main, name) is getattr(npm_toolchain, name), name


def test_reexport_names_are_callable():
    for name in MOVED_NAMES:
        assert callable(getattr(hermes_main, name)), name


def test_npm_command_building_prefers_ci_then_falls_back_to_install(tmp_path, monkeypatch):
    """Deterministic npm install: npm ci when a lockfile exists, install otherwise.

    subprocess.run is patched so the exact argv built by the moved
    command-construction logic is recorded (no real npm needed).
    """
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        # ci fails (lockfile out of sync) -> install fallback succeeds
        rc = 1 if cmd[1] == "ci" else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

    import hermes_cli.npm_engine as npm_engine  # noqa: PLC0415

    monkeypatch.setattr(npm_engine, "maybe_repair_npm_engine", lambda npm, combined: None)
    monkeypatch.setattr(npm_toolchain.subprocess, "run", fake_run)
    monkeypatch.setattr(npm_toolchain.subprocess, "Popen", pytest.fail)

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "package-lock.json").touch()

    result = npm_toolchain._run_npm_install_deterministic("npm", cwd=web_dir)
    assert result.returncode == 0
    assert recorded == [
        ["npm", "ci", "--include=dev"],
        ["npm", "install", "--no-save", "--include=dev"],
    ]


def test_npm_engine_failure_retries_once_with_repaired_npm(tmp_path, monkeypatch):
    """EBADENGINE output triggers exactly one repair retry with the managed npm."""
    calls = []

    def fake_repair(npm, combined):
        calls.append((npm, combined))
        return "repaired-npm"

    import hermes_cli.npm_engine as npm_engine  # noqa: PLC0415

    monkeypatch.setattr(npm_engine, "maybe_repair_npm_engine", fake_repair)

    def fake_run(cmd, **kwargs):
        # The original npm fails both ci and install with EBADENGINE; the
        # repaired npm succeeds on its first ci.
        if cmd[0] == "npm":
            return subprocess.CompletedProcess(
                cmd, 1, stdout="npm error code EBADENGINE", stderr=""
            )
        assert cmd[0] == "repaired-npm"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(npm_toolchain.subprocess, "run", fake_run)
    monkeypatch.setattr(npm_toolchain.subprocess, "Popen", pytest.fail)

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "package-lock.json").touch()

    result = npm_toolchain._run_npm_install_deterministic("npm", cwd=web_dir)
    assert result.returncode == 0
    assert len(calls) == 1
    assert calls[0][0] == "npm"
    assert "EBADENGINE" in calls[0][1]


def test_nixos_build_env_resolves_venv_python_via_lazy_project_root(tmp_path, monkeypatch):
    """_nixos_build_env reaches PROJECT_ROOT through the lazy _m() seam.

    Simulates a NixOS host (ID=nixos) with python3 absent from PATH but a
    hermes venv present; asserts the PYTHON env var points into PROJECT_ROOT.
    Also proves a monkeypatch on ``hermes_cli.main.PROJECT_ROOT`` is observed
    by the moved module (patch-transparency through _m()).
    """
    monkeypatch.setattr(npm_toolchain.Path, "read_text", lambda self, encoding="utf-8": "ID=nixos\n")
    monkeypatch.setattr(npm_toolchain.shutil, "which", lambda name: None)

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python3 = venv_bin / "python3"
    python3.touch()

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)

    env = npm_toolchain._nixos_build_env()
    assert env is not None
    assert env["PYTHON"] == str(python3)


def test_run_npm_watching_for_engine_failure_tees_stderr(tmp_path):
    """capture_output=False still accumulates stderr for EBADENGINE detection."""
    script = tmp_path / "stderr.py"
    script.write_text("import sys; sys.stderr.write('boom\\n'); sys.exit(3)\n", encoding="utf-8")
    result = npm_toolchain._run_npm_watching_for_engine_failure(
        [_sys.executable, str(script)],
        cwd=tmp_path,
        env={**os.environ},
        capture_output=False,
    )
    assert result.returncode == 3
    assert "boom" in (result.stderr or "")
