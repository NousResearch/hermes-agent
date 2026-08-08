"""Managed installers preserve unique local commits on divergent updates."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
GIT = ["git", "-c", "user.name=Hermes Test", "-c", "user.email=test@example.invalid"]


def _extract_sh_function(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _extract_ps_function(name: str) -> str:
    text = INSTALL_PS1.read_text(encoding="ascii")
    match = re.search(
        rf"^function {re.escape(name)} \{{.*?^\}}", text, re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"{name} not found in install.ps1"
    return match.group(0)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    home = repo.parent / "home"
    home.mkdir(exist_ok=True)
    env = {**os.environ, "HOME": str(home), "HERMES_HOME": str(home / ".hermes")}
    return subprocess.run(
        GIT + list(args),
        cwd=repo,
        env=env,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _init_git_repo(repo: Path) -> None:
    """Create a repo whose rebases do not depend on the host's Git identity."""
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "test@example.invalid")


def _write_commit(repo: Path, path: str, text: str, message: str) -> str:
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _diverged_repo(tmp_path: Path, *, conflict: bool = False) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _write_commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "branch", "remote-main")
    local_sha = _write_commit(
        repo,
        "tracked.txt",
        "local\n" if conflict else "base\nlocal\n",
        "local carried fix",
    )
    _git(repo, "checkout", "remote-main")
    if conflict:
        remote_sha = _write_commit(repo, "tracked.txt", "remote\n", "remote conflict")
    else:
        remote_sha = _write_commit(repo, "remote.txt", "remote\n", "remote change")
    _git(repo, "checkout", "main")
    return repo, local_sha, remote_sha


def _run_sh_recovery(
    repo: Path, *, fail_count: bool = False, count_output: bytes | None = None
) -> subprocess.CompletedProcess[str]:
    git_wrapper = """
git() {
    if [ "$1" = "rev-list" ]; then
        return 128
    fi
    command git "$@"
}
""" if fail_count else ""
    script = f"""
log_info() {{ :; }}
log_warn() {{ :; }}
log_error() {{ :; }}
{git_wrapper}
{_extract_sh_function("recover_diverged_update")}
recover_diverged_update remote-main
"""
    home = repo.parent / "home"
    temp_dir = repo.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "TMPDIR": str(temp_dir),
    }
    if count_output is not None:
        fake_bin = repo.parent / "fake-bin"
        fake_bin.mkdir()
        count_file = fake_bin / "count-output"
        count_file.write_bytes(count_output)
        wrapper = fake_bin / "git"
        wrapper.write_text(
            """#!/bin/sh
saw_rev_list=false
saw_count=false
for arg do
    [ "$arg" = "rev-list" ] && saw_rev_list=true
    [ "$arg" = "--count" ] && saw_count=true
done
if [ "$saw_rev_list" = true ] && [ "$saw_count" = true ]; then
    cat "$HERMES_FAKE_COUNT"
    exit 0
fi
exec "$HERMES_REAL_GIT" "$@"
""",
            encoding="ascii",
        )
        wrapper.chmod(0o755)
        env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "HERMES_FAKE_COUNT": str(count_file),
                "HERMES_REAL_GIT": shutil.which("git") or "git",
            }
        )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_install_sh_rebases_local_commit_onto_remote(tmp_path: Path) -> None:
    repo, _local_sha, remote_sha = _diverged_repo(tmp_path)

    result = _run_sh_recovery(repo)

    assert result.returncode == 0, result.stderr
    assert _git(repo, "merge-base", "--is-ancestor", remote_sha, "HEAD").returncode == 0
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "local carried fix"
    assert not _git(repo, "status", "--porcelain").stdout


def test_install_sh_conflict_aborts_and_restores_original_head(tmp_path: Path) -> None:
    repo, local_sha, _remote_sha = _diverged_repo(tmp_path, conflict=True)

    result = _run_sh_recovery(repo)

    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == local_sha
    assert not _git(repo, "status", "--porcelain").stdout
    assert not (repo / ".git" / "rebase-merge").exists()
    assert not (repo / ".git" / "rebase-apply").exists()


def test_install_sh_no_local_commit_uses_reset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _write_commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "branch", "remote-main")
    _git(repo, "checkout", "remote-main")
    remote_sha = _write_commit(repo, "remote.txt", "remote\n", "remote change")
    _git(repo, "checkout", "main")

    result = _run_sh_recovery(repo)

    assert result.returncode == 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == remote_sha


def test_install_sh_count_failure_fails_closed(tmp_path: Path) -> None:
    repo, local_sha, _remote_sha = _diverged_repo(tmp_path)

    result = _run_sh_recovery(repo, fail_count=True)

    assert result.returncode == 128
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == local_sha
    assert not _git(repo, "status", "--porcelain").stdout
    assert not any((tmp_path / "tmp").iterdir())


@pytest.mark.parametrize(
    "count_output",
    [b"1\n", b"1", b"000000001\n"],
    ids=["terminated", "unterminated", "nine-digit-zero-padded"],
)
def test_install_sh_exact_single_count_record_rebases(
    count_output: bytes, tmp_path: Path
) -> None:
    repo, local_sha, remote_sha = _diverged_repo(tmp_path)

    result = _run_sh_recovery(repo, count_output=count_output)

    assert result.returncode == 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() != local_sha
    assert _git(repo, "merge-base", "--is-ancestor", remote_sha, "HEAD").returncode == 0
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "local carried fix"
    assert not _git(repo, "status", "--porcelain").stdout
    assert not any((tmp_path / "tmp").iterdir())


@pytest.mark.parametrize(
    "count_output",
    [b"0\n", b"0", b"000000000\n"],
    ids=["terminated", "unterminated", "nine-digit-all-zero"],
)
def test_install_sh_exact_zero_count_record_resets(
    count_output: bytes, tmp_path: Path
) -> None:
    repo, _local_sha, remote_sha = _diverged_repo(tmp_path)

    result = _run_sh_recovery(repo, count_output=count_output)

    assert result.returncode == 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == remote_sha
    assert not _git(repo, "status", "--porcelain").stdout
    assert not any((tmp_path / "tmp").iterdir())


@pytest.mark.parametrize(
    "count_output",
    [
        b"+1\n",
        b"-1\n",
        b" 1 \n",
        b"1x\n",
        b"",
        b"1000000000\n",
        b"9" * 36 + b"\n",
        b"0000000001\n",
        b"0000000000\n",
        b"0" * 4301 + b"1\n",
        b"0" * 4301 + b"0\n",
        b"1\n2\n",
        b"1\n\n",
        b"1\n\n\n",
        b"1\n2",
        "\u0661\n".encode(),
    ],
    ids=[
        "positive-sign",
        "negative-sign",
        "padded",
        "suffix",
        "empty",
        "over-bound",
        "oversized",
        "ten-digit-zero-padded-one",
        "ten-digit-all-zero",
        "giant-zero-padded-one",
        "giant-all-zero",
        "multiple-records",
        "second-empty-record",
        "double-trailing-empty-records",
        "partial-second-record",
        "non-ascii-digit",
    ],
)
def test_install_sh_malformed_count_matrix_fails_closed(
    count_output: bytes, tmp_path: Path
) -> None:
    repo, local_sha, _remote_sha = _diverged_repo(tmp_path)

    result = _run_sh_recovery(repo, count_output=count_output)

    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == local_sha
    assert not _git(repo, "status", "--porcelain").stdout
    assert not any((tmp_path / "tmp").iterdir())


def test_install_sh_rebases_only_actual_local_commit_after_force_push(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    base = _write_commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "update-ref", "--create-reflog", "refs/remotes/origin/main", base)
    old_upstream = _write_commit(repo, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(repo, "update-ref", "refs/remotes/origin/main", old_upstream)
    _write_commit(repo, "local.txt", "local\n", "actual local commit")
    _git(repo, "checkout", "-b", "rewritten", base)
    rewritten = _write_commit(repo, "replacement.txt", "replacement\n", "rewritten upstream")
    _git(repo, "update-ref", "refs/remotes/origin/main", rewritten)
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "rewritten")

    script = f"""
log_info() {{ :; }}
log_warn() {{ :; }}
log_error() {{ :; }}
{_extract_sh_function("recover_diverged_update")}
recover_diverged_update refs/remotes/origin/main
"""
    result = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    subjects = _git(repo, "log", "--format=%s", f"{rewritten}..HEAD").stdout.splitlines()
    assert subjects == ["actual local commit"]
    assert _git(repo, "branch", "--show-current").stdout.strip() == "main"
    assert not (repo / "obsolete.txt").exists()


def test_install_sh_merge_only_payload_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    base = _write_commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "update-ref", "--create-reflog", "refs/remotes/origin/main", base)
    _write_commit(repo, "main.txt", "main\n", "main local")
    _git(repo, "checkout", "-b", "side", base)
    _write_commit(repo, "side.txt", "side\n", "side local")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-commit", "side")
    (repo / "payload.txt").write_text("must survive\n", encoding="utf-8")
    _git(repo, "add", "payload.txt")
    _git(repo, "commit", "-m", "merge payload")
    original = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-b", "remote-work", base)
    remote = _write_commit(repo, "remote.txt", "remote\n", "remote change")
    _git(repo, "update-ref", "refs/remotes/origin/main", remote)
    _git(repo, "checkout", "main")

    script = f"""
log_info() {{ :; }}
log_warn() {{ :; }}
log_error() {{ :; }}
{_extract_sh_function("recover_diverged_update")}
recover_diverged_update refs/remotes/origin/main
"""
    result = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True)

    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == original
    assert (repo / "payload.txt").read_text(encoding="utf-8") == "must survive\n"
    assert not _git(repo, "status", "--porcelain").stdout


@pytest.mark.skipif(
    not any(shutil.which(name) for name in ("pwsh", "powershell")),
    reason="PowerShell is not available",
)
@pytest.mark.parametrize(
    ("count_output", "expected"),
    [
        (b"1\n", "rebase"),
        (b"1", "rebase"),
        (b"0\n", "reset"),
        (b"0", "reset"),
        (b"000000001\n", "rebase"),
        (b"000000000\n", "reset"),
        (b"+1\n", "reject"),
        (b"-1\n", "reject"),
        (b" 1 \n", "reject"),
        (b"1x\n", "reject"),
        (b"", "reject"),
        (b"1000000000\n", "reject"),
        (b"9" * 36 + b"\n", "reject"),
        (b"0000000001\n", "reject"),
        (b"0000000000\n", "reject"),
        (b"0" * 4301 + b"1\n", "reject"),
        (b"0" * 4301 + b"0\n", "reject"),
        (b"1\n2\n", "reject"),
    ],
    ids=[
        "one-terminated",
        "one-unterminated",
        "zero-terminated",
        "zero-unterminated",
        "nine-digit-zero-padded-one",
        "nine-digit-all-zero",
        "positive-sign",
        "negative-sign",
        "padded",
        "suffix",
        "empty",
        "over-bound",
        "oversized",
        "ten-digit-zero-padded-one",
        "ten-digit-all-zero",
        "giant-zero-padded-one",
        "giant-all-zero",
        "multiple-records",
    ],
)
def test_install_ps1_count_contract(
    count_output: bytes, expected: str, tmp_path: Path
) -> None:
    """Exercise the real PowerShell helper with an executable git proxy when available."""
    repo, local_sha, remote_sha = _diverged_repo(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    count_file = fake_bin / "count-output"
    count_file.write_bytes(count_output)
    if os.name == "nt":
        (fake_bin / "git.cmd").write_text(
            """@echo off
echo %* | findstr /C:" rev-list " >nul || goto realgit
echo %* | findstr /C:" --count " >nul || goto realgit
type "%HERMES_FAKE_COUNT%"
exit /b 0
:realgit
"%HERMES_REAL_GIT%" %*
""",
            encoding="ascii",
        )
    else:
        wrapper = fake_bin / "git"
        wrapper.write_text(
            """#!/bin/sh
saw_rev_list=false
saw_count=false
for arg do
    [ "$arg" = "rev-list" ] && saw_rev_list=true
    [ "$arg" = "--count" ] && saw_count=true
done
if [ "$saw_rev_list" = true ] && [ "$saw_count" = true ]; then
    cat "$HERMES_FAKE_COUNT"
    exit 0
fi
exec "$HERMES_REAL_GIT" "$@"
""",
            encoding="ascii",
        )
        wrapper.chmod(0o755)

    host = next(
        shutil.which(name) for name in ("pwsh", "powershell") if shutil.which(name)
    )
    assert host is not None
    command = f"""
function Write-Info {{ param([string]$Message) }}
function Write-Err {{ param([string]$Message) }}
{_extract_ps_function("Recover-DivergedUpdate")}
try {{ Recover-DivergedUpdate 'remote-main'; exit 0 }} catch {{ [Console]::Error.WriteLine($_); exit 1 }}
"""
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "HERMES_FAKE_COUNT": str(count_file),
        "HERMES_REAL_GIT": shutil.which("git") or "git",
    }
    result = subprocess.run(
        [host, "-NoProfile", "-Command", command],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if expected == "reject":
        assert result.returncode != 0
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == local_sha
    elif expected == "rebase":
        assert result.returncode == 0, result.stderr
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() != local_sha
        assert _git(repo, "merge-base", "--is-ancestor", remote_sha, "HEAD").returncode == 0
    else:
        assert expected == "reset"
        assert result.returncode == 0, result.stderr
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == remote_sha
    assert not _git(repo, "status", "--porcelain").stdout


@pytest.mark.skipif(
    not any(shutil.which(name) for name in ("pwsh", "powershell")),
    reason="PowerShell is not available",
)
def test_install_ps1_parses() -> None:
    host = next(shutil.which(name) for name in ("pwsh", "powershell") if shutil.which(name))
    result = subprocess.run(
        [host, "-NoProfile", "-Command", f"[scriptblock]::Create((Get-Content -Raw '{INSTALL_PS1}')) | Out-Null"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
