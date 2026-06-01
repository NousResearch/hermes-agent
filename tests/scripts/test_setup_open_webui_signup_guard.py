"""Tests for LAN/signup safety guard in scripts/setup_open_webui.sh (#36121)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "setup_open_webui.sh"

_GUARD = f"""
set -euo pipefail
source {SCRIPT.as_posix()}
assert_signup_safe_for_bind
"""


def _run_guard(*, host: str, signup: str, ack: str = "false") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": env.get("HOME", "/tmp"),
            "OPEN_WEBUI_HOST": host,
            "OPEN_WEBUI_ENABLE_SIGNUP": signup,
            "OPEN_WEBUI_I_UNDERSTAND_LAN_SIGNUP_RACE": ack,
            "OPEN_WEBUI_PORT": "8080",
        }
    )
    return subprocess.run(
        ["bash", "-c", _GUARD],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_loopback_allows_signup_enabled():
    result = _run_guard(host="127.0.0.1", signup="true")
    assert result.returncode == 0


def test_lan_bind_blocks_signup_enabled_without_ack():
    result = _run_guard(host="0.0.0.0", signup="true")
    assert result.returncode == 1
    assert "Refusing to continue" in result.stderr
    assert "36121" in result.stderr


def test_lan_bind_allows_signup_disabled():
    result = _run_guard(host="0.0.0.0", signup="false")
    assert result.returncode == 0


def test_lan_bind_allows_signup_with_explicit_ack():
    result = _run_guard(host="0.0.0.0", signup="true", ack="true")
    assert result.returncode == 0
    assert "WARNING" in result.stdout
