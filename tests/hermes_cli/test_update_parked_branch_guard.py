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

import json
import os
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
    (origin / "a.txt").write_text("one\n", encoding="utf-8")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    # Park the clone on a feature branch cut at c1.
    _git(clone, "checkout", "-qb", "old-feature")

    # main advances upstream (two commits).
    (origin / "a.txt").write_text("two\n", encoding="utf-8")
    _git(origin, "commit", "-aqm", "c2")
    (origin / "b.txt").write_text("three\n", encoding="utf-8")
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
    (repo_pair / "a.txt").write_text("local edit\n", encoding="utf-8")
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, repo_pair, "old-feature", "main"
    )
    assert safe is False
    assert reason == "dirty"


def test_untracked_file_blocks_auto_switch(repo_pair):
    """Untracked files count as dirty too — they'd ride along on checkout."""
    (repo_pair / "scratch.py").write_text("wip\n", encoding="utf-8")
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
    (repo_pair / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
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


def _treeless_repo_pair(tmp_path, *, promisor_reachable=False):
    """A treeless clone parked on ``old-feature`` cut from c1, with origin/main
    two commits ahead whose commits are local but whose trees are not — the
    #124767 shape on a real partial clone. By default the promisor remote can
    no longer satisfy a lazy fetch; ``promisor_reachable=True`` keeps it live
    (the #131444 shape, where every lazy fetch succeeds and nothing bounds them).

    Returns ``(clone,)``. Local commits/trees are all present; only the
    upstream-side trees are missing, which is exactly what ``git cherry``'s
    patch-id walk needs and ``rev-list`` does not.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    # file:// (not a plain path) plus allowFilter: a local origin that speaks
    # the partial-clone protocol, like GitHub does for real tree:0 clones.
    _git(origin, "config", "uploadpack.allowFilter", "true")
    (origin / "a.txt").write_text("one\n", encoding="utf-8")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "checkout", "-qb", "old-feature")

    (origin / "a.txt").write_text("two\n", encoding="utf-8")
    _git(origin, "commit", "-aqm", "c2")
    (origin / "b.txt").write_text("three\n", encoding="utf-8")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "c3")

    # c2/c3 arrive as commits only (no trees).
    _git(clone, "fetch", "-q", "--filter=tree:0", "origin", "main")
    if not promisor_reachable:
        _git(clone, "remote", "set-url", "origin", f"file://{tmp_path / 'nowhere'}")
    return clone


def _pack_count(clone) -> int:
    """Packfiles in the clone — every lazy fetch from the promisor adds one (#131444)."""
    return len(list((clone / ".git" / "objects" / "pack").glob("*.pack")))


def test_treeless_clone_verifies_merged_parked_branch_from_commit_graph(tmp_path):
    """Fully merged parked branch on a tree:0 clone with an unreachable promisor
    remote: ``git cherry`` needs origin/main's trees and its lazy fetch fails,
    which made the guard call a clean, fully-merged checkout unverifiable and
    skip the update (#124767). The verdict must come from the commit graph
    alone."""
    clone = _treeless_repo_pair(tmp_path)
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, clone, "old-feature", "main"
    )
    assert safe is True
    assert reason == ""


def test_treeless_cherry_failure_degrades_to_conservative_unmerged(tmp_path):
    """Clean parked branch with local commits whose patch-equivalence cannot be
    established (cherry's lazy fetch fails): a clean checkout must still reach
    the target — degrade to the commit-graph count instead of the old
    "unverifiable" skip (#124767)."""
    clone = _treeless_repo_pair(tmp_path)
    (clone / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
    _git(clone, "add", "feature.txt")
    _git(clone, "commit", "-qm", "feature work")
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, clone, "old-feature", "main"
    )
    assert safe is True
    assert reason == "unmerged:1"


@pytest.mark.parametrize("repo_config", [
    {},
    # Foreground auto-GC folds each new pack away inside the same command, so the pack
    # count alone cannot see a fetch; the Trace2 child count still does.
    {"gc.auto": "1", "gc.autoPackLimit": "1", "gc.autoDetach": "false"},
], ids=["plain", "foreground-gc"])
def test_parked_branch_guard_never_lazy_fetches_from_a_live_promisor(tmp_path, monkeypatch, repo_config):
    """Clean parked branch with a local commit on a tree:0 clone whose promisor
    remote IS reachable: ``git cherry`` would lazy-fetch a tree batch per
    upstream commit, and with nothing bounding that walk one such assessment
    wrote 332 packs / 180 GiB in 7 h on Windows (#131444). The guard must not
    start a single fetch and settles for the commit-graph count."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    clone = _treeless_repo_pair(tmp_path, promisor_reachable=True)
    for key, value in repo_config.items():
        _git(clone, "config", key, value)
    (clone / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
    _git(clone, "add", "feature.txt")
    _git(clone, "commit", "-qm", "feature work")
    packs_before = _pack_count(clone)
    trace = tmp_path / "trace2.json"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    safe, reason = update_cmd._assess_parked_branch_switch(
        GIT, clone, "old-feature", "main"
    )
    monkeypatch.delenv("GIT_TRACE2_EVENT")
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    fetches = [e["argv"] for e in events if e.get("event") == "child_start" and "fetch" in e.get("argv", [])]
    assert (safe, reason) == (True, "unmerged:1")
    assert fetches == []
    assert _pack_count(clone) == packs_before


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


def test_update_refuses_stopped_rebase_without_moving_head(
    repo_pair, monkeypatch, capsys
):
    """A user-owned rebase controls the checkout; update must not switch branches underneath it."""
    (repo_pair / "a.txt").write_text("topic\n", encoding="utf-8")
    _git(repo_pair, "add", "a.txt")
    _git(repo_pair, "commit", "-qm", "topic")
    rebase = _git(repo_pair, "rebase", "--merge", "origin/main", check=False)
    assert rebase.returncode != 0
    assert (repo_pair / ".git" / "rebase-merge").is_dir()
    assert _git(repo_pair, "branch", "--show-current").stdout.strip() == ""
    head_before = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()

    _patch_update_flow(monkeypatch, repo_pair)
    args = SimpleNamespace(branch=None, yes=True, force=False, force_venv=False)

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._cmd_update_impl(args, False)

    assert exc_info.value.code == 1
    assert _git(repo_pair, "rev-parse", "HEAD").stdout.strip() == head_before
    assert (repo_pair / ".git" / "rebase-merge").is_dir()
    assert _git(repo_pair, "stash", "list").stdout.strip() == ""
    out = capsys.readouterr().out
    assert "Git rebase is in progress" in out
    assert str(repo_pair) in out
    assert "git rebase --abort" in out


def test_update_reports_git_am_abort_for_apply_state(
    repo_pair, monkeypatch, capsys
):
    """rebase-apply is shared by rebase --apply and git am; recovery advice must match Git's owner."""
    (repo_pair / "a.txt").write_text("topic\n", encoding="utf-8")
    _git(repo_pair, "add", "a.txt")
    _git(repo_pair, "commit", "-qm", "topic")
    patch = repo_pair.parent / "upstream.patch"
    patch.write_text(
        _git(repo_pair, "format-patch", "-1", "origin/main~1", "--stdout").stdout, encoding="utf-8"
    )
    applied = _git(repo_pair, "am", str(patch), check=False)
    assert applied.returncode != 0
    assert (repo_pair / ".git" / "rebase-apply" / "applying").is_file()
    head_before = _git(repo_pair, "rev-parse", "HEAD").stdout.strip()

    _patch_update_flow(monkeypatch, repo_pair)
    args = SimpleNamespace(branch=None, yes=True, force=False, force_venv=False)

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._cmd_update_impl(args, False)

    assert exc_info.value.code == 1
    assert _git(repo_pair, "rev-parse", "HEAD").stdout.strip() == head_before
    assert (repo_pair / ".git" / "rebase-apply" / "applying").is_file()
    out = capsys.readouterr().out
    assert "Git am is in progress" in out
    assert "git am --abort" in out


def test_update_skips_and_warns_on_dirty_parked_branch(
    repo_pair, monkeypatch, capsys
):
    """Tonight's incident shape: parked branch + dirty tree. The update must
    NOT print '✓ Code updated!', must warn loudly, and must exit non-zero
    with the branch named in the summary."""
    (repo_pair / "a.txt").write_text("local edit\n", encoding="utf-8")
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
    """Default strategy ("switch"): clean tree + unmerged commits → the
    update proceeds (non-interactive callers like the desktop update button
    cannot resolve a skip), prints the loud 'kept' notice, ends on main
    fast-forwarded to origin/main, and the commits stay on the parked
    branch untouched."""
    (repo_pair / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    feature_sha = _git(repo_pair, "rev-parse", "old-feature").stdout.strip()
    _patch_update_flow(monkeypatch, repo_pair)

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        update_cmd,
        "_complete_source_update",
        lambda *a, **k: (_ for _ in ()).throw(_StopFlow()),
    )
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)

    with pytest.raises(_StopFlow):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "CODE UPDATE SKIPPED" not in out
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
    (repo_pair / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    _patch_update_flow(monkeypatch, repo_pair)

    # Stop right after the pull/branch logic, before dependency install.
    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        update_cmd,
        "_complete_source_update",
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
    assert (repo_pair / "a.txt").read_text(encoding="utf-8-sig") == "two\n"
    # ...and the branch's own commit survived it.
    assert (repo_pair / "feature.txt").read_text(encoding="utf-8-sig") == "unmerged work\n"
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
    (repo_pair / "feature.txt").write_text("unmerged work\n", encoding="utf-8")
    _git(repo_pair, "add", "feature.txt")
    _git(repo_pair, "commit", "-qm", "feature work")
    branch_tip_before = _git(
        repo_pair, "rev-parse", "old-feature"
    ).stdout.strip()
    _patch_update_flow(monkeypatch, repo_pair)

    class _StopFlow(Exception):
        pass

    monkeypatch.setattr(
        update_cmd,
        "_complete_source_update",
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
        update_cmd,
        "_complete_source_update",
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
    (origin / "a.txt").write_text("one\n", encoding="utf-8")
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

    monkeypatch.setattr(
        update_cmd,
        "_complete_source_update",
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
        update_cmd,
        "_complete_source_update",
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
