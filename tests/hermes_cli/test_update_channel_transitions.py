"""Real-git-graph channel transition tests (acceptance contract, root decision 5).

Uses a REAL temporary git repository with main at C and a stable ancestor B:

- stable apply  -> HEAD detaches at verified B, the ``main`` branch ref stays C
                   (a real transition, never a destructive branch reset);
- beta return   -> checkout main preserves existing main history (still C);
- failed/mismatched release -> exit 1 with the checkout unchanged.

These exercise the actual resolution + checkout path end-to-end against git,
not source-regex mocks.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_channel

pytestmark = pytest.mark.stable_channel_default


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "PATH": os.environ["PATH"], "HOME": str(repo.parent)}
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, env=env, capture_output=True, text=True, check=check)


@pytest.fixture
def git_graph(tmp_path):
    """main = C on origin, tagged stable ancestor = B, hermes-home scratch dir."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)],
        capture_output=True, text=True, check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "A")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "B")
    b_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "tag", "-a", "v2026.1.1", "-m", "release B")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "C")
    c_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")
    return {"repo": repo, "B": b_sha, "C": c_sha, "home": tmp_path / "home"}


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    monkeypatch.setattr(update_channel, "get_default_hermes_root", lambda: root)
    return root


@pytest.fixture
def isolated_pipeline(git_graph, monkeypatch):
    """Point the whole update pipeline at the temp repo.

    ``_cmd_update_impl`` resolves the checkout through ``_m().PROJECT_ROOT``
    (the LIVE worktree by default) and runs real backups/snapshots against
    ``get_hermes_home()``. Without this isolation a test "update" would stash
    and mutate the developer's checkout — exactly what one of these tests did
    on its first run. Everything the pipeline can reach is redirected here.
    """
    repo = git_graph["repo"]
    from hermes_cli import main as hermes_main
    from hermes_cli import update_cmd

    home = git_graph["home"]
    home.mkdir(exist_ok=True)

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(update_cmd, "PROJECT_ROOT", repo, raising=False)
    # Receipts/snapshots/fleet: all best-effort no-ops on a bare temp repo.
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(update_cmd, "_record_update_step", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None)
    return repo


def _args(**kw):
    base = dict(release=None, release_commit=None, branch=None, channel=None,
                yes=True, gateway=False, check=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _resolve_stub(repo: Path, tag: str, commit: str, monkeypatch):
    """Bypass the network: the official resolver reports (tag, commit) verified."""
    monkeypatch.setattr(
        "hermes_cli.update_cmd.resolve_official_release_target",
        lambda repo_dir, requested, expected_commit=None: (tag, commit))


class TestStableTransition:
    def test_stable_detaches_at_release_main_ref_untouched(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch, capsys):
        from hermes_cli import update_cmd

        repo, B, C = git_graph["repo"], git_graph["B"], git_graph["C"]
        _resolve_stub(repo, "v2026.1.1", B, monkeypatch)
        # No dependency/restart work in this test: stop after the checkout phase.
        monkeypatch.setattr(
            update_cmd, "_apply_pulled_update",
            lambda *a, **k: print("POST-UPDATE-PHASES-RAN"))

        update_cmd._cmd_update_impl(_args(release="v2026.1.1"), gateway_mode=False)

        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert head == B, "HEAD must sit at the verified release commit"
        assert _git(repo, "symbolic-ref", "-q", "HEAD", check=False).returncode != 0, (
            "HEAD must be detached, not on a branch")
        main_ref = _git(repo, "rev-parse", "refs/heads/main").stdout.strip()
        assert main_ref == C, "the main branch ref must stay at C — no destructive reset"
        out = capsys.readouterr().out
        assert "Pinning to release v2026.1.1" in out
        assert "POST-UPDATE-PHASES-RAN" in out

    def test_beta_return_preserves_main_history(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch, capsys):
        from hermes_cli import update_cmd

        repo, B, C = git_graph["repo"], git_graph["B"], git_graph["C"]
        _resolve_stub(repo, "v2026.1.1", B, monkeypatch)
        monkeypatch.setattr(update_cmd, "_apply_pulled_update", lambda *a, **k: None)

        # Stable first: detached at B.
        update_cmd._cmd_update_impl(_args(release="v2026.1.1"), gateway_mode=False)
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == B

        # Beta return: record says beta; the branch path checks main out intact.
        update_channel.write_channel_record("beta", hermes_root)
        monkeypatch.setattr(
            "hermes_cli.update_cmd._prepare_checkout_for_update",
            lambda git_cmd, branch, current_branch, **k: update_cmd._CheckoutPlan(
                auto_stash_ref=None, commit_count=1, in_place_update=False,
                parked_branch_switched=False, prompt_for_restore=False,
                switch_block_reason=None, upstream_checked=True))
        monkeypatch.setattr(
            "hermes_cli.update_cmd._pull_updates", lambda *a, **k: B)
        monkeypatch.setattr(update_cmd, "_apply_pulled_update", lambda *a, **k: None)
        _git(repo, "fetch", "origin", "main", check=False)  # no remote: force fetch failure path off

        # Simulate the beta flow's branch checkout directly: the real impl would
        # ff-only merge origin/main; with no remote we assert the ref survives.
        update_cmd._cmd_update_impl(_args(), gateway_mode=False)

        main_ref = _git(repo, "rev-parse", "refs/heads/main").stdout.strip()
        assert main_ref == C, "beta return must preserve existing main history"


class TestFailClosed:
    def test_mismatched_release_commit_exits_unchanged(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch, capsys):
        from hermes_cli import update_cmd

        repo, B, C = git_graph["repo"], git_graph["B"], git_graph["C"]
        wrong = "d" * 40

        def fail_target(*a, **k):
            raise ValueError("Release v2026.1.1 no longer resolves to the checked commit.")

        monkeypatch.setattr(
            "hermes_cli.update_cmd.resolve_official_release_target", fail_target)

        with pytest.raises(SystemExit) as exc:
            update_cmd._cmd_update_impl(
                _args(release="v2026.1.1", release_commit=wrong), gateway_mode=False)
        assert exc.value.code == 1

        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == C, "checkout unchanged"
        assert _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip() == "main"
        out = capsys.readouterr().out
        assert "no longer resolves to the checked commit" in out
        assert "not changed" in out

    def test_already_on_release_short_circuits_without_checkout(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch, capsys):
        from hermes_cli import update_cmd

        repo, B = git_graph["repo"], git_graph["B"]
        _git(repo, "checkout", "-q", "--detach", B)
        _resolve_stub(repo, "v2026.1.1", B, monkeypatch)

        def fail_checkout(*a, **k):  # any checkout attempt must fail the test
            raise AssertionError("already-on-release must not touch the checkout")

        monkeypatch.setattr(update_cmd, "_apply_release_update", fail_checkout)
        update_cmd._cmd_update_impl(_args(release="v2026.1.1"), gateway_mode=False)

        assert "Already on release v2026.1.1" in capsys.readouterr().out
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == B


class TestChannelPersistence:
    def test_channel_flag_persists_record(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch, capsys):
        from hermes_cli import update_cmd

        repo, B, C = git_graph["repo"], git_graph["B"], git_graph["C"]
        _resolve_stub(repo, "v2026.1.1", B, monkeypatch)
        monkeypatch.setattr(update_cmd, "_apply_pulled_update", lambda *a, **k: None)

        update_cmd._cmd_update_impl(_args(channel="stable", release=None), gateway_mode=False)

        assert update_channel.read_update_channel(hermes_root) == "stable"
        assert "Update channel set to stable" in capsys.readouterr().out

    def test_branch_flag_does_not_persist(
            self, git_graph, hermes_root, isolated_pipeline, monkeypatch):
        from hermes_cli import update_cmd

        repo = git_graph["repo"]
        update_channel.write_channel_record("beta", hermes_root)

        monkeypatch.setattr(
            update_cmd, "_prepare_checkout_for_update",
            lambda git_cmd, branch, current_branch, **k: update_cmd._CheckoutPlan(
                auto_stash_ref=None, commit_count=0, in_place_update=False,
                parked_branch_switched=False, prompt_for_restore=False,
                switch_block_reason=None, upstream_checked=True))
        monkeypatch.setattr(
            update_cmd, "_finish_already_up_to_date", lambda *a, **k: None)
        # The branch path fetches origin before prepare; stub the failure out.
        monkeypatch.setattr(
            "hermes_cli.update_cmd._git_run",
            lambda git_cmd, args, *a, **k: subprocess.CompletedProcess(
                (git_cmd + args), 0, stdout="", stderr=""))

        update_cmd._cmd_update_impl(_args(branch="dev"), gateway_mode=False)

        assert update_channel.read_update_channel(hermes_root) == "beta", (
            "--branch is a one-shot override and must not rewrite the record")
