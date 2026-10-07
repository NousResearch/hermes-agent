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


def _origin(path: Path, content: str = "one", *, bootstrap: bool = True) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "README").write_text(content)
    # The bootstrap-protocol interfaces the installer ladder invokes after the
    # checkout: every hermes-agent revision this installer can complete has them.
    # bootstrap=False builds a pre-PM stand-in for the #134733 gate.
    if bootstrap:
        (path / "pm").mkdir()
        (path / "pm" / "lock.json").write_text(
            '{"packages": {"python": {"version": "3.14.0"}}}'
        )
        (path / "pm" / "cli.py").write_text("")
        (path / "hermes_cli").mkdir()
        (path / "hermes_cli" / "source_completion.py").write_text("")
        _git(path, "add", "README", "pm", "hermes_cli")
    else:
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


def test_pinned_fresh_clone_never_materializes_the_branch_tip(tmp_path):
    """A pinned install (the desktop bootstrap passes --commit) checks out only the pin.

    Every file the tip changed is fetched and written once for the tip, then again for the pin.
    """
    origin = _origin(tmp_path / "origin")
    pin = _git(origin, "rev-parse", "HEAD")
    _commit(origin, "tip")
    _git(origin, "config", "uploadpack.allowFilter", "true")
    trace = tmp_path / "trace.log"
    tracer = f'git() {{ printf "%s\\n" "$*" >> {shlex.quote(trace.as_posix())}; command git "$@"; }}'
    result = _stage(tmp_path, origin, commit=pin, prelude=tracer)
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == pin
    assert (install / "README").read_text() == "one"
    assert _git(install, "status", "--porcelain") == ""
    [clone] = [line for line in trace.read_text().splitlines() if line.startswith("clone ")]
    assert "--no-checkout" in clone.split()
    unpinned_root = tmp_path / "unpinned"
    unpinned_root.mkdir()
    trace.unlink()
    result = _stage(unpinned_root, origin, prelude=tracer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (unpinned_root / "install" / "README").read_text() == "tip"
    [clone] = [line for line in trace.read_text().splitlines() if line.startswith("clone ")]
    assert "--no-checkout" not in clone.split()


def test_pinned_fresh_clone_publishes_nothing_when_the_pin_fails(tmp_path):
    """A pinned fresh clone pins before it publishes: a refused or failed pin leaves no empty checkout."""
    origin = _origin(tmp_path / "origin")
    pin = _git(origin, "rev-parse", "HEAD")
    _commit(origin, "tip")
    _git(origin, "checkout", "-qb", "side")
    off_branch = _commit(origin, "side")
    _git(origin, "checkout", "-q", "main")
    refused = _stage(tmp_path, origin, commit=off_branch)
    assert refused.returncode != 0
    assert "is not on branch main" in refused.stdout + refused.stderr
    assert not (tmp_path / "install").exists()
    failing_checkout = 'git() { [ "${3:-}" = checkout ] && return 1; command git "$@"; }'
    failed = _stage(tmp_path, origin, commit=pin, prelude=failing_checkout)
    assert failed.returncode != 0
    assert "no checkout published" in failed.stdout + failed.stderr
    assert not (tmp_path / "install").exists()
    assert not list(tmp_path.glob(".hermes-clone-*"))


def test_commit_pin_rejects_revisions_without_the_bootstrap_interfaces(tmp_path):
    """#134733: --commit accepts any SHA on the branch, but this bootstrap then runs
    against the CHECKED-OUT tree (pm/lock.json, `pm.cli`, source_completion.py).
    A pre-PM revision passed the branch check and died mid-ladder during dependency
    prep, leaving a half-prepared install; the gate fails closed at pin time instead."""
    origin = _origin(tmp_path / "origin", "pre-pm", bootstrap=False)
    pre_pm = _git(origin, "rev-parse", "HEAD")
    (origin / "pm").mkdir()
    (origin / "pm" / "lock.json").write_text(
        '{"packages": {"python": {"version": "3.14.0"}}}'
    )
    (origin / "pm" / "cli.py").write_text("")
    (origin / "hermes_cli").mkdir()
    (origin / "hermes_cli" / "source_completion.py").write_text("")
    _git(origin, "add", "pm", "hermes_cli")
    _git(origin, "commit", "-qm", "pm era")
    refused = _stage(tmp_path, origin, commit=pre_pm)
    assert refused.returncode != 0
    combined = refused.stdout + refused.stderr
    assert "predates the PM bootstrap this installer runs" in combined
    assert "pm/lock.json" in combined
    assert pre_pm in combined  # the refusal names the pinned commit
    pinned = _stage(tmp_path, origin, commit=_git(origin, "rev-parse", "HEAD"))
    assert pinned.returncode == 0, pinned.stdout + pinned.stderr


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


def test_path_uv_is_never_used_even_when_newer_than_the_pin(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "uv"
    fake.write_text("#!/bin/sh\necho 'uv 99.0.0'\n")
    fake.chmod(0o755)
    # No pinned target: with the PATH uv ignored, the install must fail rather than adopt it.
    body = 'uv_bootstrap_target() { return 1; }\nensure_uv\necho "UV_CMD=$UV_CMD"'
    result = _run(tmp_path, body, env={"PATH": f"{bindir}:{os.environ['PATH']}"})
    assert result.returncode != 0
    assert f"UV_CMD={fake}" not in result.stdout


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


def test_rerun_marks_partial_clone_packs_when_the_fetch_crashes(tmp_path):
    """git 2.53+ aborts fetches into a partial clone with unmarked packs (#124272); the installer
    rerun is the recovery for installs whose own updater cannot fetch, so it marks them first."""
    origin = _origin(tmp_path / "origin")
    assert _stage(tmp_path, origin).returncode == 0
    install = tmp_path / "install"
    _git(install, "config", "remote.origin.promisor", "true")
    _git(install, "repack", "-adq")
    packs = list((install / ".git" / "objects" / "pack").glob("pack-*.pack"))
    assert packs and not any(p.with_suffix(".promisor").exists() for p in packs)
    _commit(origin, "two")
    # Stand-in for the git 2.53+ index-pack crash: any fetch while an unmarked pack exists dies.
    crashing_fetch = (
        'git() { if [ "${3:-}" = fetch ]; then for p in "$2"/.git/objects/pack/pack-*.pack; do '
        '[ -e "${p%.pack}.promisor" ] || { echo "BUG: should_include_obj" >&2; return 128; }; done; fi; '
        'command git "$@"; }'
    )
    result = _stage(tmp_path, origin, prelude=crashing_fetch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (install / "README").read_text() == "two"
    assert all(p.with_suffix(".promisor").exists() for p in packs)
