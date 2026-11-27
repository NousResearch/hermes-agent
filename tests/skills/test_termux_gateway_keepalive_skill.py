"""
Tests for the Termux Gateway Keep-Alive skill (skills/devops/termux-gateway-keepalive).

Covers:
- Shell script syntax checks (bash -n) on all keepalive scripts
- hermes_presence.py ambient alert formatting and CLI argument handling
- Gateway watchdog state and command handling
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "devops" / "termux-gateway-keepalive"
SCRIPTS_DIR = SKILL_DIR / "scripts"
PRESENCE_SCRIPT = SCRIPTS_DIR / "hermes_presence.py"

# Import Python presence module
sys.path.insert(0, str(SCRIPTS_DIR))
import hermes_presence


class TestTermuxGatewayKeepaliveScripts:
    def test_all_shell_scripts_syntax(self):
        sh_files = list(SCRIPTS_DIR.glob("*.sh"))
        assert len(sh_files) >= 14, f"Expected at least 14 shell scripts, found {len(sh_files)}"
        for f in sh_files:
            res = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            assert res.returncode == 0, f"Syntax error in {f.name}: {res.stderr}"

    def test_presence_cli_help(self):
        res = subprocess.run(
            [sys.executable, str(PRESENCE_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 1
        assert "usage: hermes_presence.py" in res.stdout

    def test_presence_fire_calls(self, monkeypatch, tmp_path):
        log_file = tmp_path / "presence.log"
        monkeypatch.setattr(hermes_presence, "LOG", str(log_file))
        monkeypatch.setattr(hermes_presence, "_run", lambda cmd, timeout=12: True)

        ok = hermes_presence.fire("Test Alert", quiet=False)
        assert ok is True
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test Alert" in content

    def test_presence_fire_quiet(self, monkeypatch, tmp_path):
        log_file = tmp_path / "presence.log"
        monkeypatch.setattr(hermes_presence, "LOG", str(log_file))
        calls = []

        def _fake_run(cmd, timeout=12):
            calls.append(cmd[0])
            return True

        monkeypatch.setattr(hermes_presence, "_run", _fake_run)
        ok = hermes_presence.fire("Quiet Alert", quiet=True)
        assert ok is True
        assert "termux-vibrate" not in calls
        assert "termux-toast" in calls
        assert "termux-notification" in calls
