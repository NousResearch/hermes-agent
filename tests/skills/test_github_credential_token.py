"""Regression tests for Tirith-safe GitHub credential extraction (#22722)."""

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "skills/software-development/github/scripts/git-credential-token.py"


def _extract(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _credential_file(tmp_path: Path, value: str) -> Path:
    credentials = tmp_path / "credentials"
    credentials.write_text(value, encoding="utf-8", newline="")
    return credentials


@pytest.mark.parametrize(
    ("credential", "token"),
    [
        ("https://octocat:password-form-token@github.com\n", "password-form-token"),
        ("https://oauth-token:x-oauth-basic@github.com\n", "oauth-token"),
        ("https://ghp_token_only@github.com\n", "ghp_token_only"),
        ("https://github_pat_token_only@github.com\n", "github_pat_token_only"),
    ],
)
def test_extracts_supported_git_credential_url_forms(tmp_path, credential, token):
    result = _extract(_credential_file(tmp_path, credential))

    assert result.returncode == 0
    assert result.stdout == f"{token}\n"
    assert result.stderr == ""


def test_extracts_password_from_exact_github_https_credential(tmp_path):
    credentials = _credential_file(
        tmp_path,
        "https://ignored:wrong@example.com\n"
        "https://octocat:secret%2Ftoken@github.com\n",
    )

    result = _extract(credentials)

    assert result.returncode == 0
    assert result.stdout == "secret/token\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "credential",
    [
        "https://octocat:stolen@github.com.attacker.example\n",
        "https://octocat@github.com\n",
        "https://%6fctocat@github.com\n",
        "https://octocat:token@github.com%2eattacker.example\n",
        "https://octocat:token%0D%0AX-Injected%3Ayes@github.com\n",
        "https://ghp_token%0Ainjected@github.com\n",
        "https://octocat:token%00suffix@github.com\n",
        "https://octocat:token%09suffix@github.com\n",
        "https://octocat:token%C2%85suffix@github.com\n",
        "https://ghp_token%1Fsuffix@github.com\n",
        "https://ghp_token%C2%9Fsuffix@github.com\n",
        "https://octocat:bad%ZZtoken@github.com\n",
        "https://octocat:token@github.com:bogus\n",
        "http://octocat:token@github.com\n",
    ],
)
def test_rejects_ambiguous_lookalike_or_malformed_credentials(tmp_path, credential):
    result = _extract(_credential_file(tmp_path, credential))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == ""


GH_ENV = REPO_ROOT / "skills/software-development/github/scripts/gh-env.sh"


@pytest.mark.platforms("posix")
def test_gh_env_finds_the_credential_helper_next_to_itself(tmp_path):
    """The .git-credentials fallback must resolve git-credential-token.py next to
    gh-env.sh; it used to point at a non-existent skills/github/github-auth path."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git-credentials").write_text(
        "https://octocat:tok-from-creds@github.com\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    # Stub `uv run python3 <helper>`: drop `run python3`, run the helper with this interpreter.
    uv.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    uv.chmod(0o755)

    proc = subprocess.run(
        ["bash", "-c",
         f'source "{GH_ENV}" >/dev/null 2>&1; echo "M=$GH_AUTH_METHOD"; echo "T=$GITHUB_TOKEN"'],
        capture_output=True, text=True, check=False,
        env={"HOME": str(home), "HERMES_HOME": str(home),
             "PATH": f"{bin_dir}:/usr/bin:/bin", "GITHUB_TOKEN": ""})

    assert "M=curl" in proc.stdout, proc.stdout + proc.stderr
    assert "T=tok-from-creds" in proc.stdout, proc.stdout + proc.stderr
