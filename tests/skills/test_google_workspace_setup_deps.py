"""OAuth must not run against an unavailable or newly selected environment."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pm
import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.mark.parametrize("command", ["--check", "--check-live", "--auth-url", "--auth-code", "--revoke"])
def test_oauth_stops_at_pm_restart_boundary(command, monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("TOKEN_PATH", "CLIENT_SECRET_PATH", "PENDING_AUTH_PATH"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"state": "pending-state", "code_verifier": "verifier"}))
        monkeypatch.setattr(module, name, path)
    before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
    ensure = Mock(side_effect=pm.InstallError("venv", "google installed; restart Hermes to activate"))
    monkeypatch.setattr(pm, "ensure_import", ensure)
    monkeypatch.setattr("subprocess.check_call", Mock(side_effect=AssertionError("ambient install")))
    monkeypatch.setattr(sys, "argv", [str(SETUP_PATH), command] + (["code"] if command == "--auth-code" else []))

    with pytest.raises(SystemExit) as failure:
        module.main()

    assert failure.value.code == 1
    ensure.assert_called_once_with("google")
    assert "restart Hermes" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in tmp_path.glob("*.json")} == before


@pytest.mark.parametrize("command", ["--install-deps", "--auth-url"])
def test_standalone_without_hermes_reports_setup_not_ambient_installs(command, tmp_path):
    # -I -S excludes both the checkout and installed site packages, just as a
    # copied skill run with an unrelated interpreter has no Hermes PM module.
    (tmp_path / "google_client_secret.json").write_text("{}")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SETUP_PATH), command],
        env={**os.environ, "HERMES_HOME": str(tmp_path), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "Hermes environment" in result.stdout
    assert "hermes setup" in result.stdout
    assert "pip" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def _load_setup_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_pm_still_runs_when_the_google_extra_is_importable(monkeypatch):
    # `sys.path[0]` is the script's own directory and the editable install does
    # not export `pm`, so a correctly installed Hermes reaches this path. The
    # Google libraries are what the command actually needs.
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: True)

    module._ensure_deps()


def test_absent_pm_with_missing_google_extra_still_refuses(monkeypatch, capsys):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: False)

    with pytest.raises(SystemExit) as failure:
        module._ensure_deps()

    assert failure.value.code == 1
    output = capsys.readouterr().out
    assert "Hermes environment" in output
    assert "hermes setup" in output


def test_google_anchor_probe_reports_a_missing_module(monkeypatch):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "_GOOGLE_ANCHORS", ("googleapiclient", "hermes_absent_anchor"))

    assert module._google_deps_importable() is False
