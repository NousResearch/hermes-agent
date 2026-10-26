"""The startup identity probes must bound the call even when a killed git leaves a descendant.

``_run_git`` and the dirty-status probe both used ``subprocess.run(..., timeout=3)`` and treated
that as a bound. On Windows it is not one: ``run()`` kills only the direct child and then does a
*blocking* ``communicate()`` (CPython's own comment, ``Lib/subprocess.py``), so a git descendant
that inherited duplicates of the captured pipe handles keeps EOF from arriving and the probe never
returns. The class is documented at ``hermes_cli/_subprocess_compat.kill_process_tree`` and guarded
there by ``bounded_probe_run`` (#68609, #66037, #87134); #144964 is the version-info site.

The shim below is the smallest faithful stand-in for a two-level launcher: a native launcher process
that starts a descendant which sleeps while inheriting the captured pipes. Without the bound, the
call returns only when that descendant exits.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import version_info
from hermes_cli.version_info import _git_version_info, _run_git

_BLOCK_SECONDS = 12
_BOUND_CEILING_S = 6.0  # timeout=3 + the bounded kill/drain budget; well under _BLOCK_SECONDS


def _find_msys_bash() -> tuple[Path, Path] | None:
    """``bash.exe`` + its msys runtime belonging to the git on PATH, if this git ships one."""
    git = shutil.which("git")
    if git is None:
        return None
    git_dir = Path(git).resolve().parent
    for root in (git_dir, *git_dir.parents[:3]):
        bash = root / "usr" / "bin" / "bash.exe"
        msys = root / "usr" / "bin" / "msys-2.0.dll"
        if bash.is_file() and msys.is_file():
            return bash, msys
    return None


def _write_sh_launcher(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _prepend_to_path(monkeypatch, shim: Path) -> None:
    # subprocess resolves the executable with the *caller's* PATH, so the child env cannot do this.
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ.get("PATH", ""))


def _install_blocking_shim(tmp_path: Path, monkeypatch) -> Path:
    """Two-level ``git`` on PATH whose descendant holds the captured pipes for every argv."""
    shim = tmp_path / "shim"
    shim.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    if sys.platform == "win32":
        fixture = _find_msys_bash()
        if fixture is None:
            pytest.skip("this git install ships no usr/bin/bash.exe to build a launcher shim from")
        bash, msys = fixture
        # A copy of bash.exe is a real native launcher: it execs the cwd script named by argv[1]
        # (``rev-parse``) and that script's ``sleep`` child inherits our stdout/stderr handles.
        shutil.copy2(bash, shim / "git.exe")
        shutil.copy2(msys, shim / "msys-2.0.dll")
        (repo / "rev-parse").write_text(
            f"echo shim-launcher-ran\nsleep {_BLOCK_SECONDS}\n", encoding="utf-8")
    else:
        _write_sh_launcher(
            shim / "git",
            "# a launcher that leaves a descendant holding the pipe write ends\n"
            f"sleep {_BLOCK_SECONDS} &\nwait\n",
        )

    _prepend_to_path(monkeypatch, shim)
    return repo


def _install_fast_shim(tmp_path: Path, monkeypatch) -> Path:
    """Same shim wiring, no descendant: proves the shim on PATH is the git being executed."""
    shim = tmp_path / "shim"
    shim.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    if sys.platform == "win32":
        fixture = _find_msys_bash()
        if fixture is None:
            pytest.skip("this git install ships no usr/bin/bash.exe to build a launcher shim from")
        bash, msys = fixture
        shutil.copy2(bash, shim / "git.exe")
        shutil.copy2(msys, shim / "msys-2.0.dll")
        (repo / "rev-parse").write_text("echo shim-was-used\n", encoding="utf-8")
    else:
        _write_sh_launcher(shim / "git", "echo shim-was-used\n")
    _prepend_to_path(monkeypatch, shim)
    return repo


def _git_repo(path: Path) -> Path:
    path.mkdir()
    git = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-C", str(path)]
    subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
    (path / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run([*git, "add", "a.txt"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "-qm", "a"], check=True, capture_output=True)
    return path


def _descendants_in(cwd: Path) -> list[int]:
    psutil = pytest.importorskip("psutil")
    found = []
    for proc in psutil.process_iter(["pid"]):
        try:
            if Path(proc.cwd()) == cwd:
                found.append(proc.pid)
        except Exception:
            continue
    return found


def _reap(cwd: Path) -> None:
    psutil = pytest.importorskip("psutil")
    for pid in _descendants_in(cwd):
        try:
            psutil.Process(pid).kill()
        except Exception:
            pass


def test_shim_on_path_is_the_git_that_runs(tmp_path, monkeypatch):
    """Control: without this, a silently-ignored shim would make the bound test pass for nothing."""
    repo = _install_fast_shim(tmp_path, monkeypatch)
    assert _run_git(repo, "rev-parse", "HEAD") == "shim-was-used"


def test_run_git_bounds_the_call_when_a_descendant_holds_the_pipes(tmp_path, monkeypatch):
    repo = _install_blocking_shim(tmp_path, monkeypatch)
    started = time.monotonic()
    try:
        value = _run_git(repo, "rev-parse", "HEAD")
        elapsed = time.monotonic() - started
        assert value is None
        assert elapsed < _BOUND_CEILING_S, (
            f"_run_git took {elapsed:.1f}s with timeout=3: the descendant that outlived the kill "
            f"held the captured pipes, so run()'s post-timeout communicate() blocked"
        )
        assert _descendants_in(repo) == [], "the timed-out probe left a descendant process behind"
    finally:
        _reap(repo)


def test_status_probe_goes_through_the_bounded_runner(tmp_path, monkeypatch):
    """The dirty ``git status`` probe is the second unbounded site; pin it to the same runner."""
    repo = _git_repo(tmp_path / "real")
    real = version_info.bounded_probe_run
    seen: list[tuple[list[str], float]] = []

    def spy(argv, *, timeout, **kwargs):
        seen.append((list(argv), timeout))
        return real(argv, timeout=timeout, **kwargs)

    monkeypatch.setattr(version_info, "bounded_probe_run", spy)
    info = _git_version_info(repo)
    status_calls = [call for call in seen if "--no-optional-locks" in call[0]]
    assert status_calls, (
        "the dirty-status probe no longer calls bounded_probe_run — it is unbounded on Windows "
        "again (#144964)"
    )
    assert status_calls[0] == (
        ["git", "--no-optional-locks", "status", "--porcelain", "-uno"], 3)
    assert info.base_version == "unknown"  # no release tag in this fixture repo
    assert info.dirty is False

    # A status probe that times out reads as "not dirty", exactly as it did before the routing.
    monkeypatch.setattr(version_info, "bounded_probe_run",
                        lambda argv, *, timeout, **kw: None if "--no-optional-locks" in argv
                        else real(argv, timeout=timeout, **kw))
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert _git_version_info(repo).dirty is False

    # ... and a successful one still reports a dirty tree.
    monkeypatch.setattr(version_info, "bounded_probe_run", real)
    assert _git_version_info(repo).dirty is True


@pytest.mark.platforms("linux", "macos")
def test_status_probe_bounds_the_call_when_a_descendant_holds_the_pipes(tmp_path, monkeypatch):
    """POSIX twin of the ``_run_git`` bound: a shim that blocks only for the status probe.

    Windows cannot drive this path from a test: the status probe's first argument is a flag, which
    a ``bash.exe`` launcher shim parses as its own option, so that shim cannot block there.
    """
    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("git not installed")
    shim = tmp_path / "shim"
    shim.mkdir()
    _write_sh_launcher(
        shim / "git",
        "case \"$*\" in\n"
        f"  *--no-optional-locks*) sleep {_BLOCK_SECONDS} & wait ;;\n"
        f"  *) exec \"{real_git}\" \"$@\" ;;\n"
        "esac\n",
    )
    _prepend_to_path(monkeypatch, shim)
    repo = _git_repo(tmp_path / "real")

    started = time.monotonic()
    try:
        info = _git_version_info(repo, include_untracked=True)
        elapsed = time.monotonic() - started
        assert info.dirty is False
        assert elapsed < _BOUND_CEILING_S, (
            f"_git_version_info took {elapsed:.1f}s with a 3s status timeout"
        )
        assert _descendants_in(repo) == []
    finally:
        _reap(repo)


def test_run_git_keeps_its_contract_on_the_normal_path(tmp_path):
    """Real git, real repo: the identity fields must be byte-for-byte what they were."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = _git_repo(tmp_path / "real")

    head = _run_git(repo, "rev-parse", "HEAD")
    assert head is not None and len(head) == 40
    assert _run_git(repo, "rev-parse", "--short=7", "HEAD") == head[:7]
    assert _run_git(repo, "rev-parse", "--verify", "nosuchref") is None
    assert _run_git(tmp_path / "missing", "rev-parse", "HEAD") is None

    clean = _git_version_info(repo)
    assert (clean.commit, clean.source, clean.dirty) == (head, "git", False)
    assert clean.branch in {"master", "main"}
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert _git_version_info(repo).dirty is True
