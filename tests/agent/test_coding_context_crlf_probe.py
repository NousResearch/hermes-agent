"""CRLF regression tests for the coding-workspace snapshot probe (#108513).

The probe runs under ``noninteractive_git_env()``, which blanks global/system
config — on Git for Windows that also strips the platform
``core.autocrlf=true`` default, so a clean CRLF checkout reads as "1 modified".
The probe must pin back the checkout's effective ``core.autocrlf``.

All fixtures are real git repos; the "user's global config" is a temp HOME.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent import coding_context as cc

GIT = shutil.which("git")

IDENT = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _run_git(repo: Path, *args: str, home: Path) -> subprocess.CompletedProcess:
    assert GIT is not None, "git not available"
    env = dict(IDENT, HOME=str(home))
    return subprocess.run([GIT, "-C", str(repo), *args], capture_output=True, text=True, env=env)


@pytest.fixture
def crlf_repo(tmp_path, monkeypatch):
    """LF-committed code file with a CRLF worktree copy; global autocrlf=true.

    Returns (repo, home). The user's own ``git status`` is clean; the blanked
    probe env sees "1 modified" (the bug).
    """
    if GIT is None:
        pytest.skip("git not available")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    for var in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_bytes(b"print('hi')\n")
    assert _run_git(repo, "init", "-q", "-b", "main", home=home).returncode == 0
    assert _run_git(repo, "add", "-A", home=home).returncode == 0
    assert _run_git(repo, "commit", "-q", "-m", "init commit", home=home).returncode == 0
    # CRLF worktree over an LF blob — exactly what Git for Windows checks out.
    # The `add` refreshes the index stat the way a real checkout does (without
    # it, status reports stat-dirt even to the user's own git, which is a
    # different, pre-existing git behavior, not this probe's bug).
    (repo / "main.py").write_bytes(b"print('hi')\r\n")
    assert _run_git(repo, "add", "main.py", home=home).returncode == 0
    user_view = _run_git(repo, "status", "--porcelain", home=home)
    assert user_view.stdout.strip() == "", f"fixture must look clean to the user: {user_view.stdout}"
    return repo, home


def test_crlf_checkout_reads_clean(crlf_repo):
    """The blanked probe used to report phantom dirt; the fix restores clean."""
    repo, _ = crlf_repo
    block = cc.build_coding_workspace_block(repo)
    assert "Status: clean" in block


def test_effective_autocrlf_matches_user_config(crlf_repo):
    repo, _ = crlf_repo
    assert cc._effective_autocrlf(repo) == "true"


def test_real_edit_still_reported(crlf_repo):
    """Normalization must not hide a genuine content change."""
    repo, _ = crlf_repo
    (repo / "main.py").write_bytes(b"print('bye')\r\n")
    block = cc.build_coding_workspace_block(repo)
    assert "Status: clean" not in block
    assert "modified" in block


def test_unset_autocrlf_pins_nothing(tmp_path, monkeypatch):
    """No global/system/local value → no ``-c`` pin (POSIX behavior unchanged).

    The blanked system scope is what the probe itself sees; with nothing set
    anywhere the effective value is empty and the probe argv is untouched.
    """
    if GIT is None:
        pytest.skip("git not available")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_bytes(b"print('hi')\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        assert _run_git(repo, *args, home=home).returncode == 0
    assert cc._effective_autocrlf(repo) == ""
    assert "Status: clean" in cc.build_coding_workspace_block(repo)


def test_platform_default_is_restored_not_assumed(tmp_path, monkeypatch):
    """Git for Windows ships ``core.autocrlf=true`` in the *system* config: the
    effective read picks it up (no per-user setting needed) and the probe pins
    exactly that — the platform default the blanking strips."""
    if GIT is None:
        pytest.skip("git not available")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_bytes(b"print('hi')\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        assert _run_git(repo, *args, home=home).returncode == 0
    value = cc._effective_autocrlf(repo)
    assert value in ("", "true")  # "" on POSIX, "true" on Git for Windows
    assert "Status: clean" in cc.build_coding_workspace_block(repo)


def test_explicit_local_false_is_honored(tmp_path, monkeypatch):
    """An explicit repo-local ``false`` is pinned as-is (never overridden)."""
    if GIT is None:
        pytest.skip("git not available")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_bytes(b"print('hi')\n")
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "core.autocrlf", "false"],
        ["add", "-A"],
        ["commit", "-q", "-m", "x"],
    ):
        assert _run_git(repo, *args, home=home).returncode == 0
    assert cc._effective_autocrlf(repo) == "false"
    assert "Status: clean" in cc.build_coding_workspace_block(repo)
