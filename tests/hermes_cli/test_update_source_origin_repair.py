"""Regression for #134843: release-swap payloads ship a valid .git with no remotes.

`hermes update --check` and the apply path fetch `origin` unconditionally; a
release-swap/bootstrap installation (``source=git``, ``updateMechanism=self``,
``payload=bootstrap``) whose payload checkout has no ``origin`` died with the raw
``fatal: 'origin' does not appear to be a git repository``.

The updater now resolves the update source before any fetch: a missing origin is
repaired from the canonical repo (an existing ``upstream`` remote's URL when
present, else ``OFFICIAL_REPO_URL``) and the run proceeds. Normal clones and
forks are never touched. When no source can be resolved the run fails with the
installation-contract diagnostic instead of the raw git error, and the
fetch-failure classifier names the same contract for a pre-existing origin that
no longer resolves to a repository.

These tests run the real resolver, the real check/apply entry points, and real
``git remote`` state against local ``file://`` origins — the git behavior is the
thing under test, so mocks would prove nothing. The canonical URL is redirected
to the local origin so nothing touches the network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_git

GIT = ["git"]


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [*GIT, *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _mk_origin(tmp_path: Path) -> tuple[Path, Path]:
    """A seed checkout with one commit and its bare origin (the update source)."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "config", "user.email", "t@example.com")
    _git(seed, "config", "user.name", "t")
    (seed / "f.txt").write_text("one\n", encoding="utf-8")
    _git(seed, "add", "f.txt")
    _git(seed, "commit", "-qm", "c1")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    return seed, origin


def _advance(seed: Path, origin: Path) -> None:
    """One new commit on origin's main (the update the check should report)."""
    (seed / "f.txt").write_text("two\n", encoding="utf-8")
    _git(seed, "commit", "-aqm", "c2")
    _git(seed, "push", "-q", str(origin), "main")


def _mk_release_payload(tmp_path: Path, origin: Path, *, name: str = "payload") -> Path:
    """A release-swap-shaped checkout: full clone, every remote stripped, stamped."""
    payload = tmp_path / name
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(payload))
    _git(payload, "remote", "remove", "origin")
    assert _git(payload, "remote") == ""
    sha = _git(payload, "rev-parse", "HEAD")
    (payload / "install-stamp.json").write_text(
        json.dumps({
            "commit": sha, "branch": "main", "source": "git",
            "updateMechanism": "self", "payload": "bootstrap",
        }),
        encoding="utf-8",
    )
    return payload


def _remote_config(repo: Path) -> str:
    """The repo's full local remote configuration, for byte-identical before/after checks."""
    result = subprocess.run(
        [*GIT, "config", "--local", "--get-regexp", "^remote\\."],
        cwd=str(repo), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _patch_check_env(monkeypatch, payload: Path, canonical: Path) -> None:
    """Point the check at *payload* with the canonical URL redirected to *canonical*."""
    monkeypatch.setattr(update_cmd_git, "OFFICIAL_REPO_URL", f"file://{canonical}")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", payload)
    monkeypatch.setattr("hermes_cli.update_contract.evaluate_update_admission", lambda root: None)
    monkeypatch.setattr("hermes_cli.source_check._github_compare_behind", lambda *a, **k: None)


def _mk_unrepairable_payload(tmp_path: Path, origin: Path) -> Path:
    """A remote-less payload whose .git/config cannot be written (repair must fail)."""
    payload = _mk_release_payload(tmp_path, origin, name="unrepairable")
    config = payload / ".git" / "config"
    config.unlink()
    config.symlink_to(tmp_path / "no-such-dir" / "config")
    return payload


# ---------------------------------------------------------------------------
# resolve_update_origin: the resolution contract
# ---------------------------------------------------------------------------

def test_missing_origin_is_repaired_from_the_canonical_repo(tmp_path, monkeypatch, capsys):
    seed, origin = _mk_origin(tmp_path)
    payload = _mk_release_payload(tmp_path, origin)
    monkeypatch.setattr(update_cmd_git, "OFFICIAL_REPO_URL", f"file://{origin}")

    resolved = update_cmd_git.resolve_update_origin(GIT, payload)

    assert resolved == f"file://{origin}"
    assert _git(payload, "remote", "get-url", "origin") == f"file://{origin}"
    assert "resolved the update source" in capsys.readouterr().out


def test_existing_origin_is_returned_untouched(tmp_path, monkeypatch):
    seed, origin = _mk_origin(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    before = _remote_config(clone)

    resolved = update_cmd_git.resolve_update_origin(GIT, clone)

    assert resolved == f"file://{origin}"
    assert _remote_config(clone) == before


def test_missing_origin_prefers_an_existing_upstream_url(tmp_path):
    seed, origin = _mk_origin(tmp_path)
    payload = _mk_release_payload(tmp_path, origin)
    _git(payload, "remote", "add", "upstream", f"file://{origin}")

    resolved = update_cmd_git.resolve_update_origin(GIT, payload)

    assert resolved == f"file://{origin}"
    assert _git(payload, "remote", "get-url", "origin") == f"file://{origin}"


@pytest.mark.require_symlinks  # the unrepairable payload's .git/config is a dangling symlink
def test_unresolvable_source_returns_none(tmp_path):
    seed, origin = _mk_origin(tmp_path)
    payload = _mk_unrepairable_payload(tmp_path, origin)

    assert update_cmd_git.resolve_update_origin(GIT, payload) is None
    # A directory that is not a git repo at all has no source either.
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    assert update_cmd_git.resolve_update_origin(GIT, not_a_repo) is None


# ---------------------------------------------------------------------------
# `hermes update --check` on a stamped release-swap payload
# ---------------------------------------------------------------------------

def test_check_repairs_the_source_and_reports_the_update(tmp_path, monkeypatch, capsys):
    """The reported failure: a remote-less payload's --check must fetch and report."""
    seed, origin = _mk_origin(tmp_path)
    payload = _mk_release_payload(tmp_path, origin)
    _advance(seed, origin)
    _patch_check_env(monkeypatch, payload, origin)

    update_cmd._cmd_update_check("main")

    out = capsys.readouterr().out
    assert "does not appear to be a git repository" not in out
    assert "resolved the update source" in out
    assert "Update available" in out
    assert _git(payload, "remote", "get-url", "origin") == f"file://{origin}"
    # The verdict compared against the tracking ref the healed remote fetched.
    assert _git(payload, "rev-parse", "origin/main") == _git(origin, "rev-parse", "main")


@pytest.mark.require_symlinks  # the unrepairable payload's .git/config is a dangling symlink
def test_check_without_a_resolvable_source_fails_with_the_contract_diagnostic(tmp_path, monkeypatch, capsys):
    seed, origin = _mk_origin(tmp_path)
    payload = _mk_unrepairable_payload(tmp_path, origin)
    _patch_check_env(monkeypatch, payload, origin)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._cmd_update_check("main")

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "No usable update source" in out
    assert "git remote add origin" in out
    assert "does not appear to be a git repository" not in out


def test_check_never_modifies_normal_clone_or_fork_remotes(tmp_path, monkeypatch, capsys):
    seed, origin = _mk_origin(tmp_path)
    # A normal clone of the official repo.
    normal = tmp_path / "normal"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(normal))
    _patch_check_env(monkeypatch, normal, origin)
    before = _remote_config(normal)
    update_cmd._cmd_update_check("main")
    assert _remote_config(normal) == before
    assert "resolved the update source" not in capsys.readouterr().out

    # A fork clone: origin = the fork, upstream = the official repo.
    fork_remote = tmp_path / "fork.git"
    _git(tmp_path, "clone", "-q", "--bare", str(origin), str(fork_remote))
    fork = tmp_path / "fork"
    _git(tmp_path, "clone", "-q", f"file://{fork_remote}", str(fork))
    _git(fork, "remote", "add", "upstream", f"file://{origin}")
    _patch_check_env(monkeypatch, fork, origin)
    before = _remote_config(fork)
    update_cmd._cmd_update_check("main")
    assert _remote_config(fork) == before
    assert "resolved the update source" not in capsys.readouterr().out


def test_resolved_source_persists_within_a_release_and_across_swaps(tmp_path, monkeypatch, capsys):
    seed, origin = _mk_origin(tmp_path)
    _advance(seed, origin)

    release_a = _mk_release_payload(tmp_path, origin, name="release-a")
    _patch_check_env(monkeypatch, release_a, origin)
    update_cmd._cmd_update_check("main")
    assert "resolved the update source" in capsys.readouterr().out
    config_a = _git(release_a, "config", "--local", "--list")

    # A second run reuses the repaired authority as-is: no re-repair, no config drift.
    update_cmd._cmd_update_check("main")
    assert "resolved the update source" not in capsys.readouterr().out
    assert _git(release_a, "config", "--local", "--list") == config_a

    # A later release swap publishes a fresh remote-less payload; the same
    # resolution applies to it (the authority is re-resolved, never cached away).
    release_b = _mk_release_payload(tmp_path, origin, name="release-b")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", release_b)
    update_cmd._cmd_update_check("main")
    assert "resolved the update source" in capsys.readouterr().out
    assert _git(release_b, "remote", "get-url", "origin") == f"file://{origin}"


# ---------------------------------------------------------------------------
# The apply path resolves the source before its first fetch
# ---------------------------------------------------------------------------

class _ReachedUnshallow(Exception):
    """Sentinel: the apply path got past source resolution to its first fetch step."""


def test_apply_path_resolves_the_source_before_its_first_fetch(tmp_path, monkeypatch, capsys):
    """`_heal_stale_shallow_checkout` fetches too — the origin must exist by then.

    Pre-fix, a shallow remote-less payload died inside that fetch with the raw
    git fatal; the resolution must run first, so this test fails if the call
    moves after the unshallow step (or is dropped from the apply path).
    """
    from hermes_cli import main as hermes_main

    seed, origin = _mk_origin(tmp_path)
    payload = _mk_release_payload(tmp_path, origin)
    _advance(seed, origin)

    monkeypatch.setattr(update_cmd_git, "OFFICIAL_REPO_URL", f"file://{origin}")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", payload)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)

    origin_at_unshallow = []

    def _record_unshallow(repo_root, branch):
        probe = subprocess.run(
            [*GIT, "-C", str(repo_root), "remote", "get-url", "origin"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        origin_at_unshallow.append(probe.returncode == 0)
        raise _ReachedUnshallow()

    monkeypatch.setattr(update_cmd, "_heal_stale_shallow_checkout", _record_unshallow)

    args = SimpleNamespace(branch="main", yes=True, no_gateway_restart=True)
    with pytest.raises(_ReachedUnshallow):
        update_cmd._cmd_update_impl(args, gateway_mode=False)

    out = capsys.readouterr().out
    assert "resolved the update source" in out
    assert origin_at_unshallow == [True]
    assert _git(payload, "remote", "get-url", "origin") == f"file://{origin}"


# ---------------------------------------------------------------------------
# The fetch-failure classifier names the contract for a dead origin
# ---------------------------------------------------------------------------

def test_classifier_names_the_update_source_contract():
    for stderr in (
        "fatal: 'origin' does not appear to be a git repository\n"
        "fatal: Could not read from remote repository.",
        "fatal: '/nonexistent/repo' does not appear to be a git repository",
    ):
        msg = update_cmd._classify_fetch_failure(stderr)
        assert "No usable update source" in msg
        assert "Failed to fetch updates from origin" not in msg
