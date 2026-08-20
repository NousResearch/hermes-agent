"""E2E tests for skills/github/github-auth/scripts/gh-env.sh.

Reported on Discord/GitHub (henning, hng) against the official Docker layout:
the GitHub skills' auth detection reads the Hermes ``.env``. PR #43785 made the
script resolve it via ``${HERMES_HOME:-$HOME/.hermes}/.env`` so it works when
``HERMES_HOME=/opt/data`` and the subprocess ``HOME`` is redirected to
``/opt/data/home``. These tests exercise the *real* bash script (not a mock)
against temp homes to lock in:

  * token found via ``$HERMES_HOME/.env`` even when ``HOME`` has no ``.env``
    (the Docker case),
  * a ``none`` result self-reports the exact path(s) it checked plus
    HERMES_HOME/HOME, so the common Docker mismatch is diagnosable instead of
    silent.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "github"
    / "github-auth"
    / "scripts"
    / "gh-env.sh"
)


def _run(env_overrides: dict, cwd: Path) -> str:
    """Source gh-env.sh in a clean bash and return its combined output.

    A clean ``PATH`` (no ``gh``) keeps the gh-CLI branch from being taken on
    developer machines / CI runners that happen to have ``gh`` installed and
    authenticated, so the env-file resolution under test is what runs.
    """
    bash = shutil.which("bash") or "/bin/bash"
    env = {
        "PATH": "/usr/bin:/bin",
        "GITHUB_TOKEN": "",
    }
    env.update(env_overrides)
    proc = subprocess.run(
        [bash, "-c", f'source "{SCRIPT}"'],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=20,
    )
    return proc.stdout + proc.stderr


@pytest.mark.skipif(not SCRIPT.exists(), reason="gh-env.sh not present")
def test_token_resolved_from_hermes_home_under_docker_layout(tmp_path):
    """Docker layout: HERMES_HOME=/data, HOME=/data/home (no .env there).
    The token lives at $HERMES_HOME/.env and must still be found."""
    hermes_home = tmp_path / "data"
    sub_home = hermes_home / "home"
    sub_home.mkdir(parents=True)
    (hermes_home / ".env").write_text("GITHUB_TOKEN=ghp_dockertoken\n")

    out = _run(
        {"HERMES_HOME": str(hermes_home), "HOME": str(sub_home)},
        cwd=tmp_path,
    )

    assert "GitHub Auth: curl" in out
    # It reports the real resolved path, not a hardcoded ~/.hermes literal.
    assert f"Token source: {hermes_home / '.env'}" in out
    assert "Not authenticated" not in out


@pytest.mark.skipif(not SCRIPT.exists(), reason="gh-env.sh not present")
def test_token_resolved_from_home_fallback_when_hermes_home_unset(tmp_path):
    """Bare install: no HERMES_HOME, token in ~/.hermes/.env still works."""
    home = tmp_path / "home"
    (home / ".hermes").mkdir(parents=True)
    (home / ".hermes" / ".env").write_text("GITHUB_TOKEN=ghp_baretoken\n")

    env = {"HOME": str(home)}
    env.pop("HERMES_HOME", None)
    out = _run(env, cwd=tmp_path)

    assert "GitHub Auth: curl" in out
    assert f"Token source: {home / '.hermes' / '.env'}" in out


@pytest.mark.skipif(not SCRIPT.exists(), reason="gh-env.sh not present")
def test_none_result_reports_checked_paths(tmp_path):
    """No token anywhere → the failure self-reports the exact path checked plus
    HERMES_HOME/HOME, so the Docker `.env` mismatch is obvious (regression for
    the 'still does not work, insists on ~/.hermes/.env' report)."""
    hermes_home = tmp_path / "data"
    sub_home = hermes_home / "home"
    sub_home.mkdir(parents=True)
    # Deliberately no .env anywhere.

    out = _run(
        {"HERMES_HOME": str(hermes_home), "HOME": str(sub_home)},
        cwd=tmp_path,
    )

    assert "GitHub Auth: none" in out
    assert "Not authenticated" in out
    # The exact path it checked is surfaced (resolved via HERMES_HOME), and the
    # HERMES_HOME/HOME values so a user can see the redirect.
    assert f"Checked: {hermes_home / '.env'}" in out
    assert f"HERMES_HOME={hermes_home}" in out
    assert "/opt/data/.env" in out  # the Docker hint
