"""install.sh reruns, pins and invocation modes, driven through the real stage functions.

Each test sources the installer with ``--manifest`` (definitions only) and runs
one stage against real git repositories in ``tmp_path``.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")
INSTALL_SH = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _origin(path: Path, content: str = "one") -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "README").write_text(content)
    _git(path, "add", "README")
    _git(path, "commit", "-qm", content)
    return path


def _commit(repo: Path, content: str) -> str:
    (repo / "README").write_text(content)
    _git(repo, "commit", "-qam", content)
    return _git(repo, "rev-parse", "HEAD")


def _run(tmp_path: Path, body: str, *, env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    script = f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\nsleep() {{ :; }}\n{body}\n"
    full_env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
                    HERMES_INSTALL_DIR=(tmp_path / "install").as_posix(), **(env or {}))
    return subprocess.run(["bash", "-c", script], env=full_env, capture_output=True, text=True, timeout=60, **kwargs)


def _stage(tmp_path: Path, origin: Path, *, commit: str = "", extra_env: dict | None = None,
           prelude: str = "") -> subprocess.CompletedProcess:
    env = {"HERMES_REPO_URL": origin.as_posix(), **(extra_env or {})}
    return _run(tmp_path, f"{prelude}\nINSTALL_COMMIT={shlex.quote(commit)}\nstage_repository", env=env)


def test_piped_one_liner_runs_the_installer():
    """`curl ... | bash` has an empty BASH_SOURCE; under `set -u` the entry guard must still run main."""
    result = subprocess.run(["bash", "-s", "--", "--manifest"], input=INSTALL_SH.read_bytes(),
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["protocol_version"] == 1


def test_rerun_parks_local_work_before_updating(tmp_path):
    origin = _origin(tmp_path / "origin")
    assert _stage(tmp_path, origin).returncode == 0
    install = tmp_path / "install"
    (install / "README").write_text("local edit")
    _commit(origin, "two")
    result = _stage(tmp_path, origin)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "README").read_text() == "two"
    assert "local edit" in _git(install, "stash", "show", "-p", "stash@{0}")


def test_rerun_stops_when_local_work_cannot_be_parked(tmp_path):
    origin = _origin(tmp_path / "origin")
    assert _stage(tmp_path, origin).returncode == 0
    install = tmp_path / "install"
    before = _git(install, "rev-parse", "HEAD")
    (install / "README").write_text("local edit")
    _commit(origin, "two")
    failing_stash = 'git() { [ "${3:-}" = stash ] && return 1; command git "$@"; }'
    result = _stage(tmp_path, origin, prelude=failing_stash)
    assert result.returncode != 0
    assert (install / "README").read_text() == "local edit"
    assert _git(install, "rev-parse", "HEAD") == before


def test_commit_pin_must_come_from_the_installed_branch(tmp_path):
    origin = _origin(tmp_path / "origin")
    on_branch = _git(origin, "rev-parse", "HEAD")
    _commit(origin, "two")
    _git(origin, "checkout", "-qb", "side")
    off_branch = _commit(origin, "side")
    _git(origin, "checkout", "-q", "main")
    refused = _stage(tmp_path, origin, commit=off_branch)
    assert refused.returncode != 0
    assert "is not on branch main" in refused.stdout + refused.stderr
    pinned = _stage(tmp_path, origin, commit=on_branch)
    assert pinned.returncode == 0, pinned.stdout + pinned.stderr
    assert _git(tmp_path / "install", "rev-parse", "HEAD") == on_branch


def test_commitless_checkout_is_moved_aside_and_recloned(tmp_path):
    origin = _origin(tmp_path / "origin")
    install = tmp_path / "install"
    install.mkdir()
    _git(install, "init", "-q")
    (install / "keep").write_text("user file")
    result = _stage(tmp_path, origin)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "README").read_text() == "one"
    [broken] = tmp_path.glob("install.broken-*")
    assert (broken / "keep").read_text() == "user file"


def test_unmerged_index_is_cleared_then_stashed(tmp_path):
    origin = _origin(tmp_path / "origin")
    assert _stage(tmp_path, origin).returncode == 0
    install = tmp_path / "install"
    _git(install, "checkout", "-qb", "conflict")
    _commit(install, "theirs")
    _git(install, "checkout", "-q", "main")
    _commit(install, "ours")
    subprocess.run(["git", "-C", str(install), "-c", "user.email=t@t", "-c", "user.name=t", "merge", "conflict"],
                   capture_output=True)
    assert _git(install, "ls-files", "--unmerged")
    _commit(origin, "two")
    result = _stage(tmp_path, origin)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "README").read_text() == "two"
    assert not _git(install, "ls-files", "--unmerged")


def test_rerun_follows_an_explicit_repo_url(tmp_path):
    first = _origin(tmp_path / "first")
    assert _stage(tmp_path, first).returncode == 0
    moved = tmp_path / "moved"
    subprocess.run(["git", "clone", "-q", str(first), str(moved)], check=True)
    tip = _commit(moved, "moved")
    result = _stage(tmp_path, moved)
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == tip
    assert _git(install, "remote", "get-url", "origin") == moved.as_posix()


@pytest.mark.parametrize("reported, accepted", [("0.6.17", False), ("99.0.0", True)])
def test_path_uv_is_used_only_when_at_least_the_pin(tmp_path, reported, accepted):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "uv"
    fake.write_text(f"#!/bin/sh\necho 'uv {reported}'\n")
    fake.chmod(0o755)
    # No pinned target: rejecting the PATH uv must surface as a failure to stage one.
    body = 'uv_bootstrap_target() { return 1; }\nensure_uv\necho "UV_CMD=$UV_CMD"'
    result = _run(tmp_path, body, env={"PATH": f"{bindir}:{os.environ['PATH']}"})
    if accepted:
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"UV_CMD={fake}" in result.stdout
    else:
        assert result.returncode != 0
        assert "older than the pinned" in result.stdout + result.stderr


def test_interactive_stages_skip_without_a_terminal(tmp_path):
    install = tmp_path / "install"
    (install / ".hermes" / "bin").mkdir(parents=True)
    marker = tmp_path / "ran"
    hermes = install / ".hermes" / "bin" / "hermes"
    hermes.write_text(f"#!/bin/sh\ntouch {shlex.quote(marker.as_posix())}\n")
    hermes.chmod(0o755)
    # A new session has no controlling terminal, so opening /dev/tty fails.
    result = _run(tmp_path, "NON_INTERACTIVE=false\nstage_setup\nstage_gateway", start_new_session=True,
                  stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    assert "no terminal" in result.stdout + result.stderr
