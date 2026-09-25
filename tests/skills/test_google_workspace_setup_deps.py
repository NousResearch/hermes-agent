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
