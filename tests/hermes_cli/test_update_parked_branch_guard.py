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

def test_skip_warning_names_branch_behind_count_and_commands(repo_pair, capsys):
    update_cmd._print_parked_branch_skip_warning(
        GIT, repo_pair, "old-feature", "main", "dirty"
    )
    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" in out
    assert "old-feature" in out
    assert "2 commit(s) BEHIND" in out
    assert f"git -C {repo_pair} checkout main && hermes update" in out


def test_skip_warning_dirty_reason(repo_pair, capsys):
    update_cmd._print_parked_branch_skip_warning(
        GIT, repo_pair, "old-feature", "main", "dirty"
    )
    out = capsys.readouterr().out
    assert "uncommitted changes" in out


def test_kept_notice_names_branch_count_and_recovery(capsys):
    update_cmd._print_parked_branch_kept_notice("old-feature", "main", "3")
    out = capsys.readouterr().out
    assert "parked on 'old-feature'" in out
    assert "3 commit(s) not merged into origin/main" in out
    assert "safe on 'old-feature'" in out
    assert "git checkout old-feature" in out
    assert "CODE UPDATE SKIPPED" not in out


# ---------------------------------------------------------------------------
# Summary branch/HEAD visibility
# ---------------------------------------------------------------------------

def test_branch_head_label_reflects_real_checkout(repo_pair):
    label = update_cmd._branch_head_label(GIT, repo_pair)
    short = _git(repo_pair, "rev-parse", "--short", "HEAD").stdout.strip()
    assert label == f"old-feature @ {short}"


def test_branch_head_label_detached(repo_pair):
    _git(repo_pair, "checkout", "-q", "--detach")
    label = update_cmd._branch_head_label(GIT, repo_pair)
    assert label is not None
    assert label.startswith("detached @ ")


def test_branch_head_suffix_empty_on_non_repo(tmp_path):
    assert update_cmd._branch_head_suffix(GIT, tmp_path / "not-a-repo") == ""


def test_print_update_completion_carries_branch_and_sha(
    repo_pair, monkeypatch, capsys
):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    update_cmd._print_update_completion("✓ Update complete!")
    out = capsys.readouterr().out
    short = _git(repo_pair, "rev-parse", "--short", "HEAD").stdout.strip()
    assert f"✓ Update complete! [old-feature @ {short}]" in out


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
    assert "CODE UPDATE SKIPPED" in out
    assert "old-feature" in out
    assert "code update SKIPPED" in out
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
    """Default strategy ("switch"): clean tree + unmerged commits → the
    update proceeds (non-interactive callers like the desktop update button
    cannot resolve a skip), prints the loud 'kept' notice, ends on main
    fast-forwarded to origin/main, and the commits stay on the parked
    branch untouched."""
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
    assert "1 commit(s) not merged into origin/main" in out
    assert "safe on 'old-feature'" in out
    assert "CODE UPDATE SKIPPED" not in out
    assert "updating it in place" not in out
    # Ends on main, fast-forwarded.
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "main"
    )
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = _git(repo_pair, "rev-parse", "origin/main").stdout.strip()
    assert head == remote
    # The unmerged commit is still exactly where it was, on the branch.
    assert (
        _git(repo_pair, "rev-parse", "old-feature").stdout.strip()
        == feature_sha
    )


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
    assert "updating it in place" in out
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
    assert "1 commit(s) not merged into origin/main" in out
    assert "updating it in place" not in out
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


def test_unmerged_branch_still_updates_in_place_without_the_flag(
    repo_pair, monkeypatch, capsys
):
    """--switch-branch is opt-in: with the in-place strategy configured and
    no flag, the update stays in place."""
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

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        hermes_main,
        "_abort_dependency_sync_if_self_locked",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(
        branch=None, yes=False, force=False, force_venv=False,
        switch_branch=False,
    )

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "updating it in place" in out
    assert "--switch-branch" not in out
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "old-feature"
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
    assert "parked on 'old-feature'" in out
    assert "fully merged" in out
    assert "switching back to main" in out
    assert "CODE UPDATE SKIPPED" not in out
    # The checkout ends up ON main, fast-forwarded to origin/main.
    assert (
        _git(repo_pair, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        == "main"
    )
    head = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = _git(repo_pair, "rev-parse", "origin/main").stdout.strip()
    assert head == remote


def test_update_up_to_date_path_does_not_repark_merged_branch(
    tmp_path, monkeypatch, capsys
):
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

    out = capsys.readouterr().out
    assert "switched back to main" in out
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


@pytest.fixture
def maintained_checkout(repo_pair, monkeypatch):
    """Exercise checkout/stash/pull directly, without the updater's host lifecycle."""
    import hermes_cli.config as hermes_config

    config = {"updates": {"parked_branch_strategy": "update_in_place"}}
    monkeypatch.setattr(hermes_config, "load_config", lambda: config)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo_pair)
    # This tiny Git fixture has no installed agent to import-check. Git and
    # stash restoration, including restored-file syntax checks, stay real.
    monkeypatch.setattr(update_cmd, "_critical_module_import_failures", lambda *a, **k: {})
    return repo_pair, config


@pytest.mark.parametrize("keep_stash", [False, True], ids=["cli-restore", "desktop-keep"])
def test_maintained_branch_survives_consecutive_dirty_updates(
    maintained_checkout, keep_stash
):
    repo, _ = maintained_checkout
    origin = repo.parent / "origin"
    committed = "committed local feature\n"
    (repo / "feature.txt").write_text(committed)
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-qm", "maintained local feature")
    feature_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    retained = []

    for revision in (1, 2):
        upstream = f"upstream revision {revision}\n"
        (origin / "upstream.txt").write_text(upstream)
        _git(origin, "add", "upstream.txt")
        _git(origin, "commit", "-qm", f"upstream update {revision}")
        _git(repo, "fetch", "-q", "origin", "main")
        tracked = f"{committed}tracked draft {revision}\n"
        untracked = f"untracked draft {revision}\n"
        (repo / "feature.txt").write_text(tracked)
        (repo / "scratch.txt").write_text(untracked)

        plan = update_cmd._prepare_checkout_for_update(
            GIT, "main", "old-feature", is_fork=False, assume_yes=True,
            gateway_mode=False, gw_input_fn=None, switch_branch=False,
            _windows_gateway_resume=None,
        )
        assert plan.in_place_update and not plan.parked_branch_switched
        assert plan.commit_count > 0
        stash = plan.auto_stash_ref
        assert stash == _git(repo, "rev-parse", "refs/stash").stdout.strip()
        assert _git(repo, "show", f"{stash}:feature.txt").stdout == tracked
        assert _git(repo, "show", f"{stash}^3:scratch.txt").stdout == untracked
        assert not _git(repo, "status", "--porcelain").stdout

        pre_pull_sha = update_cmd._pull_updates(
            GIT, "main", stash, prompt_for_restore=plan.prompt_for_restore,
            gw_input_fn=None, discard_local_changes=False, keep_stash=keep_stash,
        )
        post_pull_sha = update_cmd._verify_head_after_pull(
            GIT, "main", pre_pull_sha, in_place_update=plan.in_place_update,
            _windows_gateway_resume=None,
        )
        assert post_pull_sha == _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert post_pull_sha != pre_pull_sha

        assert _git(repo, "branch", "--show-current").stdout.strip() == "old-feature"
        _git(repo, "merge-base", "--is-ancestor", feature_sha, "HEAD")
        _git(repo, "merge-base", "--is-ancestor", "origin/main", "HEAD")
        assert _git(repo, "show", "HEAD:feature.txt").stdout == committed
        assert (repo / "upstream.txt").read_text() == upstream
        assert (repo / "a.txt").read_text() == "two\n"
        assert (repo / "b.txt").read_text() == "three\n"
        if keep_stash:
            retained.insert(0, stash)
            assert (repo / "feature.txt").read_text() == committed
            assert not (repo / "scratch.txt").exists()
            assert not _git(repo, "status", "--porcelain").stdout
        else:
            assert (repo / "feature.txt").read_text() == tracked
            assert (repo / "scratch.txt").read_text() == untracked
        assert _git(repo, "stash", "list", "--format=%H").stdout.splitlines() == retained

    if keep_stash:
        # The second update must not consume or overwrite the first saved edits.
        _git(repo, "stash", "apply", retained[-1])
        assert (repo / "feature.txt").read_text() == f"{committed}tracked draft 1\n"
        assert (repo / "scratch.txt").read_text() == "untracked draft 1\n"


@pytest.mark.parametrize(
    "case",
    ["conflict-cli", "conflict-desktop", "explicit-switch", "opt-out",
     "fully-merged", "cherry-equivalent", "default-strategy", "pre-existing-conflict"],
)
def test_maintained_update_failure_preserves_branch_and_edits(
    maintained_checkout, case, capsys
):
    repo, config = maintained_checkout
    conflict = case.startswith("conflict-")
    pre_existing_conflict = case == "pre-existing-conflict"
    if case == "cherry-equivalent":
        _git(repo, "cherry-pick", "origin/main~1")
        _git(repo, "commit", "--amend", "-qm", "same upstream patch, local metadata")
        cherry = _git(repo, "cherry", "origin/main").stdout.splitlines()
        assert cherry and all(line.startswith("-") for line in cherry)
    elif case != "fully-merged":
        (repo / "feature.txt").write_text("committed feature\n")
        if conflict or pre_existing_conflict:
            (repo / "a.txt").write_text("conflicting committed feature\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "local feature")
    if case == "opt-out":
        config["updates"]["auto_switch_parked_branch"] = False
    elif case == "default-strategy":
        config["updates"].pop("parked_branch_strategy")

    before_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    before_files = {p.name: p.read_bytes() for p in repo.glob("*.txt")}
    dirty = "uncommitted tracked edits\n"
    scratch = "untracked work to recover\n"
    if pre_existing_conflict:
        assert _git(repo, "merge", "--no-edit", "origin/main", check=False).returncode != 0
        assert _git(repo, "ls-files", "--unmerged").stdout
        dirty = (repo / "a.txt").read_text()
    else:
        (repo / "a.txt").write_text(dirty)
    (repo / "scratch.txt").write_text(scratch)
    before_status = _git(repo, "status", "--porcelain").stdout
    before_index = _git(repo, "ls-files", "--stage").stdout
    before_merge_head = _git(repo, "rev-parse", "--verify", "MERGE_HEAD", check=False).stdout

    with pytest.raises(SystemExit) as exc_info:
        plan = update_cmd._prepare_checkout_for_update(
            GIT, "main", "old-feature", is_fork=False, assume_yes=True,
            gateway_mode=False, gw_input_fn=None, switch_branch=case == "explicit-switch",
            _windows_gateway_resume=None,
        )
        assert conflict, "unsafe checkout reached the pull phase"
        assert plan.in_place_update and plan.auto_stash_ref
        update_cmd._pull_updates(
            GIT, "main", plan.auto_stash_ref, prompt_for_restore=plan.prompt_for_restore,
            gw_input_fn=None, discard_local_changes=False, keep_stash=case == "conflict-desktop",
        )

    assert exc_info.value.code == 1
    assert _git(repo, "branch", "--show-current").stdout.strip() == "old-feature"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(repo, "ls-files", "--stage").stdout == before_index
    assert _git(repo, "rev-parse", "--verify", "MERGE_HEAD", check=False).stdout == before_merge_head
    out = capsys.readouterr().out
    if conflict:
        assert "Merge conflict" in out
        assert "Local changes preserved in stash" in out
        assert {p.name: p.read_bytes() for p in repo.glob("*.txt")} == before_files
        assert not _git(repo, "status", "--porcelain").stdout
        assert _git(repo, "rev-parse", "--verify", "MERGE_HEAD", check=False).returncode != 0
        assert _git(repo, "stash", "list", "--format=%H").stdout.splitlines() == [plan.auto_stash_ref]
        _git(repo, "stash", "apply", plan.auto_stash_ref)
    else:
        assert "CODE UPDATE SKIPPED" in out
        assert not _git(repo, "stash", "list").stdout
        if pre_existing_conflict:
            assert "unresolved conflicts" in out
    assert (repo / "a.txt").read_text() == dirty
    assert (repo / "scratch.txt").read_text() == scratch
    assert _git(repo, "status", "--porcelain").stdout == before_status
