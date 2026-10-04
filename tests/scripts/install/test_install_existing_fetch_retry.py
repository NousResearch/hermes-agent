"""Existing installs retry a transient branch fetch before failing (#98049)."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
INSTALL_PS1 = ROOT / "scripts" / "install.ps1"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=Fixture",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _existing_checkout(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README").write_text("before\n", encoding="utf-8")
    _git(origin, "add", "README")
    _git(origin, "commit", "-qm", "before")

    install = tmp_path / "install"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(install)],
        check=True,
        capture_output=True,
        text=True,
    )
    (origin / "README").write_text("after\n", encoding="utf-8")
    _git(origin, "commit", "-qam", "after")
    return origin, install


@pytest.mark.platforms("posix")
def test_posix_existing_checkout_retries_transient_fetch(tmp_path):
    origin, install = _existing_checkout(tmp_path)
    attempts = tmp_path / "fetch-attempts"
    env = dict(
        os.environ,
        HOME=tmp_path.as_posix(),
        HERMES_HOME=(tmp_path / "home").as_posix(),
        HERMES_INSTALL_DIR=install.as_posix(),
        HERMES_REPO_URL=origin.as_posix(),
    )
    script = f"""source {shlex.quote(INSTALL_SH.as_posix())} --manifest
sleep() {{ :; }}
git() {{
    case " $* " in
        *" fetch origin +refs/heads/main:refs/remotes/origin/main "*)
            printf 'attempt\\n' >> {shlex.quote(attempts.as_posix())}
            [ "$(wc -l < {shlex.quote(attempts.as_posix())})" -ge 3 ] || return 128
            ;;
    esac
    command git "$@"
}}
stage_repository
"""

    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert attempts.read_text(encoding="utf-8-sig").splitlines() == ["attempt"] * 3
    assert _git(install, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")


@pytest.mark.platforms("windows")
def test_windows_existing_checkout_retries_transient_fetch(tmp_path):
    origin, install = _existing_checkout(tmp_path)
    attempts = tmp_path / "fetch-attempts.txt"
    real_git = Path(shutil.which("git") or pytest.fail("git is required"))

    def quote(value: object) -> str:
        return str(value).replace("'", "''")

    script = (
        f". '{quote(INSTALL_PS1)}' -HermesHome '{quote(tmp_path / 'home')}' "
        f"-InstallDir '{quote(install)}'; "
        "function Ensure-Git { return $true }; "
        "function Start-Sleep { param([int]$Seconds) }; "
        "$script:FetchAttempts = 0; "
        "function git { "
        "param([Parameter(ValueFromRemainingArguments=$true)][string[]]$GitArgs); "
        "$joined = $GitArgs -join ' '; "
        "if ($joined -like '* fetch origin +refs/heads/main:refs/remotes/origin/main*') { "
        "$script:FetchAttempts += 1; "
        f"Add-Content -LiteralPath '{quote(attempts)}' -Value 'attempt'; "
        "if ($script:FetchAttempts -lt 3) { cmd /c exit 128; return } }; "
        f"& '{quote(real_git)}' @GitArgs }}; "
        "Stage-Repository"
    )

    result = subprocess.run(
        [
            str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert attempts.read_text(encoding="utf-8-sig").splitlines() == ["attempt"] * 3
    assert _git(install, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")
