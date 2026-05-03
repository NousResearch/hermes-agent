"""Smoke tests for scripts/diagnose_dashboard_session_token.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "diagnose_dashboard_session_token.py"


def _child_env(hermes_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_DASHBOARD_SESSION_TOKEN", None)
    env.pop("HERMES_DASHBOARD_TOKEN", None)
    return env


def _run_script(*extra_args: str, hermes_home: Path) -> subprocess.CompletedProcess[str]:
    assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra_args],
        cwd=str(_REPO_ROOT),
        env=_child_env(hermes_home),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture
def isolated_hermes_home(tmp_path: Path) -> Path:
    hm = tmp_path / "hm"
    hm.mkdir(parents=True, mode=0o700)
    return hm


def test_diagnose_unset_token_reason(isolated_hermes_home: Path) -> None:
    env_file = isolated_hermes_home / ".env"
    env_file.write_text("# no dashboard session token\n", encoding="utf-8")

    proc = _run_script(hermes_home=isolated_hermes_home)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "reject_reason:" in proc.stdout
    assert "unset" in proc.stdout


def test_diagnose_short_token_random_branch(isolated_hermes_home: Path) -> None:
    env_file = isolated_hermes_home / ".env"
    env_file.write_text("HERMES_DASHBOARD_SESSION_TOKEN=short\n", encoding="utf-8")

    proc = _run_script(hermes_home=isolated_hermes_home)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout
    assert "branch:" in out
    assert "random" in out
    assert "too_short" in out


def test_diagnose_fixed_token_branch(isolated_hermes_home: Path) -> None:
    secret = "z" * 16
    env_file = isolated_hermes_home / ".env"
    env_file.write_text(f"HERMES_DASHBOARD_SESSION_TOKEN={secret}\n", encoding="utf-8")

    proc = _run_script(hermes_home=isolated_hermes_home)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "branch:" in proc.stdout
    assert "fixed" in proc.stdout


def test_diagnose_import_web_server_matches_fixed_secret(isolated_hermes_home: Path) -> None:
    secret = "y" * 16
    env_file = isolated_hermes_home / ".env"
    env_file.write_text(f"HERMES_DASHBOARD_SESSION_TOKEN={secret}\n", encoding="utf-8")

    proc = _run_script("--import-web-server", hermes_home=isolated_hermes_home)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "_SESSION_TOKEN matches stripped fixed secret: True" in proc.stdout


def test_diagnose_workspace_token_hint_when_only_alt_set(isolated_hermes_home: Path) -> None:
    env_file = isolated_hermes_home / ".env"
    env_file.write_text(
        "HERMES_DASHBOARD_TOKEN=abcdefghijklmnop\n",
        encoding="utf-8",
    )

    proc = _run_script(hermes_home=isolated_hermes_home)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "HERMES_DASHBOARD_TOKEN" in proc.stdout
    assert "HERMES_DASHBOARD_SESSION_TOKEN" in proc.stdout
