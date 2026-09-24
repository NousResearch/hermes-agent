"""Regression tests for the parked-branch guard in ``hermes update``.

Live incident (2026-08-17, Teknium's Linux box): the source checkout was
parked on a stale feature branch (``claude-code-inspired/local-terminal-
memory-limit``, days behind main) left there by earlier tooling. ``hermes
update`` autostashed, refreshed lazy backends, synced skills and printed
"✓ Code updated!" / "✓ Update complete!" — while the checkout stayed on the
stale branch with none of main's new code. Two sessions burned time on
"the fix is missing" confusion that was really this.

The guard (``_assess_parked_branch_switch``):
- clean tree + branch fully merged into origin/<target>  → safe to
  auto-switch back to the target (and STAY there — no switch-back).
- dirty tree, unmerged commits, git failure, or the
  ``updates.auto_switch_parked_branch: false`` opt-out → do NOT touch the
  branch; warn loudly and mark the code update SKIPPED.

These tests run the guard against REAL git repositories (init, commit,
branch, clone) — not mocked subprocess.run — so they exercise the actual
``git status`` / ``git cherry`` semantics the guard depends on.
"""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
from hermes_cli.update_cmd_git import _restore_local_patch_branch


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(
        GIT + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture()
def repo_pair(tmp_path):
    """A real origin repo + local clone, with main two commits ahead of the
    clone's parked state.

    Returns (clone_path,). The clone starts parked on feature branch
    ``old-feature`` cut from the first commit; origin/main has moved on.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    # Park the clone on a feature branch cut at c1.
    _git(clone, "checkout", "-qb", "old-feature")

    # main advances upstream (two commits).
    (origin / "a.txt").write_text("two\n")
    _git(origin, "commit", "-aqm", "c2")
    (origin / "b.txt").write_text("three\n")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "c3")

    _git(clone, "fetch", "-q", "origin", "main")
    return clone


@pytest.fixture(autouse=True)
def _no_config(monkeypatch):
    """Isolate the guard from the machine's real config.yaml."""
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config", lambda: {})


# ---------------------------------------------------------------------------
# _assess_parked_branch_switch against real repos
# ---------------------------------------------------------------------------

def test_clean_fully_merged_branch_is_safe_to_switch(repo_pair):
    """Parked branch == ancestor of origin/main, clean tree → auto-switch."""
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is True
    assert reason == ""


def test_dirty_tree_blocks_auto_switch(repo_pair):
    """Uncommitted changes on the parked branch → do not touch it."""
    (repo_pair / "a.txt").write_text("local edit\n")
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is False
    assert reason == "dirty"


def test_untracked_file_blocks_auto_switch(repo_pair):
    """Untracked files count as dirty too — they'd ride along on checkout."""
    (repo_pair / "scratch.py").write_text("wip\n")
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is False
    assert reason == "dirty"


def test_unmerged_commits_switch_with_kept_notice(repo_pair):
    """Commits on the parked branch not in origin/main: still safe to switch
    (checkout keeps them on the branch) — reason carries the count so the
    caller prints the loud 'kept' notice. Non-interactive callers (desktop
    update button, gateway /update, cron) depend on this: they cannot
    resolve a skip."""
    (repo_pair / "feature.txt").write_text("unmerged work\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")

    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is True
    assert reason == "unmerged:1"


def test_equivalent_cherry_picked_commit_is_still_safe(repo_pair):
    """A commit whose patch already landed upstream (git cherry '-') does
    not block the switch — only genuinely unmerged '+' commits do."""
    # Cherry-pick origin/main's c2 onto the parked branch: patch-identical.
    _git(repo_pair, "cherry-pick", "origin/main~1")
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is True
    assert reason == ""


def test_config_opt_out_blocks_auto_switch(repo_pair, monkeypatch):
    """updates.auto_switch_parked_branch: false disables auto-switch even
    when the branch is clean and merged."""
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"updates": {"auto_switch_parked_branch": False}},
    )
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is False
    assert reason == "disabled"


def test_missing_origin_ref_is_unverifiable(repo_pair):
    """If origin/<target> can't be resolved, the guard refuses to switch."""
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "no-such-branch"
    )
    assert safe is False
    assert reason == "unverifiable"


# ---------------------------------------------------------------------------
# Skip warning content
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Summary branch/HEAD visibility
# ---------------------------------------------------------------------------


def test_branch_head_suffix_empty_on_non_repo(tmp_path):
    assert update_cmd._branch_head_suffix(GIT, tmp_path / "not-a-repo") == ""


def test_print_update_completion_carries_branch_and_sha(
    repo_pair, monkeypatch, capsys
):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    update_cmd._print_update_completion("✓ Update complete!")
    out = capsys.readouterr().out
    short = _git(repo_pair, "rev-parse", "--short", "HEAD").stdout.strip()
    completion = next(line for line in out.splitlines() if "Update complete" in line)
    assert "old-feature" in completion and short in completion


# ---------------------------------------------------------------------------
# Full update flow: parked branch dirty/unmerged → SKIPPED, no false success
# ---------------------------------------------------------------------------

def _patch_update_flow(monkeypatch, repo, run_real_git=True):
    """Point _cmd_update_impl at the real repo and neuter the long tail.

    Matches the monkeypatch surface of test_update_head_moved_gate.py, but
    keeps REAL subprocess.run so the git plumbing runs against the fixture
    repo (the whole point of these regressions).
    """
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main, "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(update_cmd, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_normalize_managed_eol", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None)
    monkeypatch.setattr(main_web_build, "_record_bytecode_fingerprint", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_capture_active_lazy_features", lambda: [])
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: [])


def test_update_skips_and_warns_on_dirty_parked_branch(
    repo_pair, monkeypatch, capsys
):
    """Tonight's incident shape: parked branch + dirty tree. The update must
    NOT print '✓ Code updated!', must warn loudly, and must exit non-zero
    with the branch named in the summary."""
    (repo_pair / "a.txt").write_text("local edit\n")
    _patch_update_flow(monkeypatch, repo_pair)
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(args)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "old-feature" in out
    assert "✓ Code updated!" not in out
    assert "✓ Update complete!" not in out
    # Branch untouched.
    branch = _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch == "old-feature"
    # No autostash was created — the guard fires before any stash.
    stashes = _git(repo_pair, "stash", "list").stdout.strip()
    assert stashes == ""


def test_update_switches_unmerged_parked_branch_with_kept_notice(
    repo_pair, monkeypatch, capsys
):
    """Local-only commits are rebased onto the updated target and reactivated."""
    (repo_pair / "feature.txt").write_text("unmerged work\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    feature_sha = _git(repo_pair, "rev-parse", "old-feature").stdout.strip()
    _patch_update_flow(monkeypatch, repo_pair)

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" not in out
    # The original branch is active with its patch on top of updated main.
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "old-feature"
    )
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = _git(repo_pair, "rev-parse", "origin/main").stdout.strip()
    assert head != feature_sha
    assert _git(repo_pair, "merge-base", "--is-ancestor", remote, head).returncode == 0
    assert (repo_pair / "feature.txt").read_text() == "unmerged work\n"
    snapshots = _git(repo_pair, "branch", "--list", "hermes-update-snapshot/*").stdout
    assert snapshots.strip()
    snapshot = snapshots.strip().removeprefix("*").strip()
    assert _git(repo_pair, "rev-parse", snapshot).stdout.strip() == feature_sha
    assert _git(repo_pair, "status", "--porcelain").stdout.strip() == "?? .update-incomplete"


def test_local_patch_comparison_failure_keeps_original_checkout(repo_pair, monkeypatch, capsys):
    (repo_pair / "feature.txt").write_text("local\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "local patch")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(update_cmd, "_apply_parked_branch_guard", lambda *a, **k: (True, False, "unmerged:1"))
    calls = []
    def failed_comparison(git_cmd, args, *a, **k):
        calls.append(args)
        return SimpleNamespace(returncode=1, stderr="broken comparison", stdout="")
    monkeypatch.setattr(update_cmd, "_git_run", failed_comparison)
    with pytest.raises(SystemExit) as exc:
        update_cmd._prepare_checkout_for_update(
            GIT, "release", "old-feature", is_fork=False, assume_yes=True,
            gateway_mode=False, gw_input_fn=None, switch_branch=False,
            _windows_gateway_resume=None)
    assert exc.value.code == 1
    assert "Could not compare" in capsys.readouterr().out
    assert calls == [["cherry", "origin/release", "old-feature"]]
    assert _git(repo_pair, "rev-parse", "HEAD").stdout.strip() == original
    assert _git(repo_pair, "status", "--porcelain").stdout == ""


def test_local_patch_rebase_conflict_restores_branch_and_snapshot(repo_pair, monkeypatch, capsys):
    (repo_pair / "a.txt").write_text("local conflict\n")
    _git(repo_pair, "add", "a.txt")
    _git(repo_pair, "commit", "-qm", "local conflict")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    _git(repo_pair, "checkout", "main")
    _git(repo_pair, "merge", "--ff-only", "origin/main")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    with pytest.raises(SystemExit) as exc:
        _restore_local_patch_branch(GIT, "main", "old-feature")
    assert exc.value.code == 1
    assert "Rebase conflicted" in capsys.readouterr().out
    assert _git(repo_pair, "rev-parse", "old-feature").stdout.strip() == original
    assert _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    snapshot = _git(repo_pair, "branch", "--list", "hermes-update-snapshot/*").stdout.strip()
    assert _git(repo_pair, "rev-parse", snapshot).stdout.strip() == original
    assert _git(repo_pair, "status", "--porcelain").stdout == ""


def test_update_conflict_recovers_with_incomplete_marker(repo_pair, monkeypatch, capsys):
    (repo_pair / "a.txt").write_text("local conflict\n")
    _git(repo_pair, "add", "a.txt")
    _git(repo_pair, "commit", "-qm", "local conflict")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    _patch_update_flow(monkeypatch, repo_pair)
    with pytest.raises(SystemExit) as exc:
        hermes_main.cmd_update(SimpleNamespace(branch=None, yes=False, force=False, force_venv=False))
    assert exc.value.code == 1
    assert "Updated main is active" in capsys.readouterr().out
    assert _git(repo_pair, "rev-parse", "old-feature").stdout.strip() == original
    assert _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    snapshot = _git(repo_pair, "branch", "--list", "hermes-update-snapshot/*").stdout.strip()
    assert _git(repo_pair, "rev-parse", snapshot).stdout.strip() == original
    assert _git(repo_pair, "status", "--porcelain").stdout.strip() == "?? .update-incomplete"


def test_local_patch_rebases_onto_selected_target(repo_pair, monkeypatch):
    (repo_pair / "feature.txt").write_text("local\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "local patch")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    _git(repo_pair, "checkout", "-b", "release", "origin/main")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    _restore_local_patch_branch(GIT, "release", "old-feature")
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    target = _git(repo_pair, "rev-parse", "release").stdout.strip()
    assert _git(repo_pair, "merge-base", "--is-ancestor", target, head).returncode == 0
    assert _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "old-feature"
    assert (repo_pair / "feature.txt").read_text() == "local\n"
    snapshot = _git(repo_pair, "branch", "--list", "hermes-update-snapshot/*").stdout.strip()
    assert _git(repo_pair, "rev-parse", snapshot).stdout.strip() == original
    assert _git(repo_pair, "status", "--porcelain").stdout == ""


def test_snapshot_failure_does_not_checkout_or_rebase(repo_pair, monkeypatch, capsys):
    (repo_pair / "feature.txt").write_text("local\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "local patch")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    real_run = update_cmd._git_run
    def fail_snapshot(git_cmd, args, *a, **k):
        if args[0] == "branch":
            return SimpleNamespace(returncode=1, stdout="", stderr="cannot write ref")
        return real_run(git_cmd, args, *a, **k)
    monkeypatch.setattr(update_cmd, "_git_run", fail_snapshot)
    with pytest.raises(SystemExit) as exc:
        _restore_local_patch_branch(GIT, "main", "old-feature")
    assert exc.value.code == 1
    assert "Could not save" in capsys.readouterr().out
    assert _git(repo_pair, "rev-parse", "old-feature").stdout.strip() == original
    assert _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "old-feature"
    assert _git(repo_pair, "status", "--porcelain").stdout == ""


def test_no_new_target_commits_still_restores_local_patch_branch(repo_pair, monkeypatch):
    (repo_pair / "feature.txt").write_text("local\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "local patch")
    original = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    _git(repo_pair, "checkout", "main")
    _git(repo_pair, "merge", "--ff-only", "origin/main")
    _git(repo_pair, "checkout", "old-feature")
    _patch_update_flow(monkeypatch, repo_pair)
    import hermes_cli.managed_uv as managed_uv
    class _StopFlow(Exception):
        pass
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda *a, **k: (_ for _ in ()).throw(_StopFlow()))
    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(SimpleNamespace(branch=None, yes=False, force=False, force_venv=False))
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    assert _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "old-feature"
    assert head != original
    assert _git(repo_pair, "merge-base", "--is-ancestor", "main", "old-feature").returncode == 0
    assert _git(repo_pair, "status", "--porcelain").stdout == ""


def test_update_updates_unmerged_branch_in_place_when_configured(
    repo_pair, monkeypatch, capsys
):
    """updates.parked_branch_strategy: update_in_place — a maintained custom
    branch (local patches on top of main) is updated in place from
    origin/<target> instead of switched away from. The running code must
    advance (origin/main's files arrive) AND the local commits must survive,
    with the checkout never moving."""
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}},
    )
    (repo_pair / "feature.txt").write_text("unmerged work\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    _patch_update_flow(monkeypatch, repo_pair)

    # Stop right after the pull/branch logic, before dependency install.
    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" not in out
    # The checkout never moved.
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "old-feature"
    )
    # origin/main's code actually arrived (b.txt lands with c3)...
    assert (repo_pair / "b.txt").exists()
    assert (repo_pair / "a.txt").read_text() == "two\n"
    # ...and the branch's own commit survived it.
    assert (repo_pair / "feature.txt").read_text() == "unmerged work\n"
    assert "feature work" in _git(repo_pair, "log", "--oneline").stdout


def test_switch_branch_flag_overrides_in_place_strategy(
    repo_pair, monkeypatch, capsys
):
    """--switch-branch overrides updates.parked_branch_strategy:
    update_in_place for one run: the unmerged branch is LEFT ALONE and the
    update runs on the target instead.

    A long-lived feature branch does not want an update-driven merge commit
    in its history (#89507 review). The branch tip must be byte-identical
    afterwards, while the checkout ends up on the updated target.
    """
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}},
    )
    (repo_pair / "feature.txt").write_text("unmerged work\n")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    branch_tip_before = _git(
        repo_pair, "rev-parse", "old-feature"
    ).stdout.strip()
    _patch_update_flow(monkeypatch, repo_pair)

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(
        branch=None, yes=False, force=False, force_venv=False,
        switch_branch=True,
    )

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" not in out
    # Checkout moved to the target and picked up its code...
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "main"
    )
    assert (repo_pair / "b.txt").exists()
    # ...and the feature branch was not written to at all.
    assert (
        _git(repo_pair, "rev-parse", "old-feature").stdout.strip()
        == branch_tip_before
    )


def test_update_auto_switches_clean_merged_parked_branch(
    repo_pair, monkeypatch, capsys
):
    """Clean + fully merged parked branch → auto-switch back to main, pull,
    say so, and STAY on main afterwards (sabotage-proven: reverting the
    guard re-parks the checkout and this test fails on the branch assert)."""
    _patch_update_flow(monkeypatch, repo_pair)
    # Stop the flow right after the pull/branch logic: the dependency
    # install phase begins with _abort_dependency_sync_if_self_locked.
    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" not in out
    # The checkout ends up ON main, fast-forwarded to origin/main.
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "main"
    )
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = _git(repo_pair, "rev-parse", "origin/main").stdout.strip()
    assert head == remote


def test_update_up_to_date_path_does_not_repark_merged_branch(tmp_path, monkeypatch):
    """commit_count == 0 path: before this fix, the updater switched BACK to
    the parked feature branch after checking main ("Restore stash and switch
    back to original branch") — silently re-parking the checkout so every
    subsequent update repeated the incident. A clean, fully merged parked
    branch must now END on main."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "checkout", "-qb", "old-feature")
    # No new upstream commits: local main == origin/main == old-feature tip.

    _patch_update_flow(monkeypatch, clone)

    class _StopFlow(Exception):
        pass

    import hermes_cli.managed_uv as managed_uv

    monkeypatch.setattr(
        managed_uv,
        "update_managed_uv",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    # The regression: old code ran `git checkout old-feature` here.
    assert (
        _git(clone, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    )


def test_update_on_main_fast_path_unchanged(repo_pair, monkeypatch, capsys):
    """On the target branch already: no guard prints, normal pull flow."""
    _git(repo_pair, "checkout", "-q", "main")

    _patch_update_flow(monkeypatch, repo_pair)

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "parked on" not in out
    assert "CODE UPDATE SKIPPED" not in out
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "main"
    )
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = _git(repo_pair, "rev-parse", "origin/main").stdout.strip()
    assert head == remote
