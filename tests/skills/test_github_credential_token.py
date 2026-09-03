"""Regression tests for Tirith-safe GitHub credential extraction (#22722)."""

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "skills/github/github-auth/scripts/git-credential-token.py"
LEGACY_SED = r"sed 's|https://[^:]*:\([^@]*\)@.*|\1|'"
SHIPPED_TREES = (
    REPO_ROOT / "skills/github",
    REPO_ROOT / "website/docs/user-guide/skills/bundled/github",
    REPO_ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/skills/bundled/github",
)


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


def test_bundled_github_skills_and_docs_do_not_ship_legacy_sed_url_regex():
    offenders = []
    for tree in SHIPPED_TREES:
        for path in tree.rglob("*"):
            if path.suffix in {".md", ".sh", ".py"} and LEGACY_SED in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
