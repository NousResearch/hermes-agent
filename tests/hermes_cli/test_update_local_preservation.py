"""Native local-change preservation lifecycle across successive source updates.

Regression for #128159: source checkouts carrying intentional local
customizations preserve recoverability plus an explicit active/inactive
record across updates, without redefining ``update_in_place`` or silently
re-applying edits.

All tests run against real disposable Git repositories (no mocks for the
git plumbing itself) so they exercise the actual ref/stash semantics the
lifecycle depends on.
"""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_local_preservation as pres
import hermes_cli.config as hermes_config


GIT = ["git"]


def _git(cwd, *args, check=True):
    result = subprocess.run(
        [*GIT, *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr[:500]}")
    return result


@pytest.fixture(autouse=True)
def _isolated_git_and_config(tmp_path, monkeypatch):
    """Hermetic git identity plus isolation from the machine's config.yaml."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "git-config-global"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    monkeypatch.setattr(hermes_config, "load_config", lambda: {})


def _init_repo(path: Path, *, branch="main"):
    _git(path, "init", "-q", "-b", branch)
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")


def _origin_clone_pair(tmp_path, *, parked_branch="old-feature"):
    """Origin + clone parked on *parked_branch*; origin/main two ahead."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "checkout", "-qb", parked_branch)
    (origin / "a.txt").write_text("two\n")
    _git(origin, "commit", "-aqm", "c2")
    (origin / "b.txt").write_text("three\n")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "c3")
    _git(clone, "fetch", "-q", "origin", "main")
    return origin, clone


def _stash_shas(repo):
    out = _git(repo, "stash", "list", "--format=%H").stdout
    return tuple(line.strip() for line in out.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Flag / config plumbing
# ---------------------------------------------------------------------------

def test_parser_accepts_preservation_flags():
    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_update_parser(subparsers, cmd_update=lambda args: None)
    args = parser.parse_args(["update"])
    assert args.preserve_local_changes is False
    assert args.no_preserve_local_changes is False
    assert args.restore_policy is None
    args = parser.parse_args(
        ["update", "--preserve-local-changes", "--restore-policy", "safe"])
    assert args.preserve_local_changes is True
    assert args.restore_policy == "safe"
    args = parser.parse_args(["update", "--no-preserve-local-changes"])
    assert args.no_preserve_local_changes is True


def test_config_defaults_declare_preservation_keys():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    updates = DEFAULT_CONFIG["updates"]
    assert updates["local_change_preservation"] == "off"
    assert updates["local_change_restore_policy"] == "never"


def test_preservation_enabled_precedence(monkeypatch):
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"local_change_preservation": "preserve"}})
    assert pres.preservation_enabled(None) is True
    assert pres.preservation_enabled(
        SimpleNamespace(preserve_local_changes=False,
                        no_preserve_local_changes=True)) is False
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"local_change_preservation": "off"}})
    assert pres.preservation_enabled(None) is False
    assert pres.preservation_enabled(
        SimpleNamespace(preserve_local_changes=True,
                        no_preserve_local_changes=False)) is True
    # Bool config values are honored too.
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"local_change_preservation": True}})
    assert pres.preservation_enabled(None) is True


def test_restore_policy_keep_stash_forces_never(monkeypatch):
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"local_change_restore_policy": "safe"}})
    assert pres.restore_policy(None) == "safe"
    assert pres.restore_policy(
        SimpleNamespace(keep_stash=True, restore_policy="safe")) == "never"
    assert pres.restore_policy(
        SimpleNamespace(keep_stash=False, restore_policy="never")) == "never"
    assert pres.restore_policy(
        SimpleNamespace(keep_stash=False, restore_policy=None)) == "safe"
    monkeypatch.setattr(hermes_config, "load_config", lambda: {})
    assert pres.restore_policy(None) == "never"


# ---------------------------------------------------------------------------
# Preserve: groups, pre-existing stashes, fail-closed
# ---------------------------------------------------------------------------

def test_preserve_captures_all_four_kinds(tmp_path):
    """Committed + staged/unstaged edits to one file + untracked are recoverable."""
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("base\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    # Committed local work.
    (tmp_path / "feat.txt").write_text("feat\n")
    _git(tmp_path, "add", "feat.txt")
    _git(tmp_path, "commit", "-qm", "local commit")
    # Staged + unstaged edits to the same tracked file.
    (tmp_path / "a.txt").write_text("staged\n")
    _git(tmp_path, "add", "a.txt")
    (tmp_path / "a.txt").write_text("staged\nunstaged\n")
    (tmp_path / "new.txt").write_text("untracked\n")

    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="T1")

    assert state.base_ref.startswith(pres.PRESERVATION_REF_PREFIX)
    assert state.groups["committed"].present is True
    assert state.groups["tracked"].present is True
    assert state.groups["untracked"].present is True
    # Tree is clean so the base installation can advance.
    assert _git(tmp_path, "status", "--porcelain").stdout == ""
    # Base ref pins the pre-preservation HEAD.
    assert _git(tmp_path, "rev-parse", state.base_ref).stdout.strip() == state.pre_sha
    # Every present working-tree group has a verifiable new stash entry.
    shas = _stash_shas(tmp_path)
    assert state.groups["tracked"].ref in shas
    assert state.groups["untracked"].ref in shas
    # Marker records the run for interruption recovery.
    marker = pres.read_in_progress_marker(tmp_path)
    assert marker is not None and marker["id"] == state.preservation_id
    # Content round-trips through the stashes.
    tracked_show = _git(tmp_path, "stash", "show", "-p", state.groups["tracked"].ref).stdout
    assert "unstaged" in tracked_show
    pres.clear_in_progress_marker(tmp_path)


def test_preserve_leaves_preexisting_stashes_untouched(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("base\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "a.txt").write_text("precious\n")
    _git(tmp_path, "stash", "push", "-m", "my-precious")
    before = _stash_shas(tmp_path)
    assert len(before) == 1
    precious = _git(tmp_path, "stash", "show", "-p", "stash@{0}").stdout

    (tmp_path / "a.txt").write_text("new edit\n")
    (tmp_path / "extra.txt").write_text("untracked\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="T2")

    after = _stash_shas(tmp_path)
    assert len(after) == 3  # one pre-existing + two new preservation entries
    assert all(sha in after for sha in before)
    assert _git(tmp_path, "stash", "show", "-p", "stash@{2}").stdout == precious
    assert state.groups["tracked"].ref in after
    pres.clear_in_progress_marker(tmp_path)


def test_preserve_clean_tree_writes_no_ref_or_marker(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("base\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")

    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="T3")

    assert state.base_ref == ""
    assert all(not g.present for g in state.groups.values())
    assert pres.read_in_progress_marker(tmp_path) is None
    assert _git(tmp_path, "for-each-ref", pres.PRESERVATION_REF_PREFIX).stdout == ""


def test_preserve_fails_closed_when_head_unresolvable(tmp_path, monkeypatch, capsys):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("x\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    monkeypatch.setattr(pres, "_head_sha", lambda *a, **k: "")
    with pytest.raises(SystemExit) as exc:
        pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="T4")
    assert exc.value.code == 1
    assert "refusing to preserve" in capsys.readouterr().out.lower()
    assert pres.read_in_progress_marker(tmp_path) is None


# ---------------------------------------------------------------------------
# Restore: policies, conflicts, validation
# ---------------------------------------------------------------------------

def _dirty_clone_with_compatible_edit(tmp_path):
    origin, clone = _origin_clone_pair(tmp_path)
    _git(clone, "checkout", "-q", "main")
    (clone / "mod.py").write_text("X = 1\n")
    _git(clone, "add", "mod.py")
    _git(clone, "commit", "-qm", "track mod")
    _git(origin, "fetch", check=False)
    (clone / "mod.py").write_text("X = 2\n")
    (clone / "notes.md").write_text("private\n")
    return origin, clone


def test_safe_policy_restores_compatible_groups_and_drops_stashes(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "mod.py").write_text("X = 2\n")
    (tmp_path / "notes.md").write_text("notes\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="SAFE")
    # Upstream advances an unrelated file.
    (tmp_path / "b.txt").write_text("b\n")
    _git(tmp_path, "add", "b.txt")
    _git(tmp_path, "commit", "-qm", "upstream")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=False)

    assert outcome["tracked"]["status"] == "active"
    assert outcome["untracked"]["status"] == "active"
    assert (tmp_path / "mod.py").read_text() == "X = 2\n"
    assert (tmp_path / "notes.md").read_text() == "notes\n"
    # Successful groups are dropped; nothing extra lingers.
    assert _git(tmp_path, "stash", "list").stdout.strip() == ""
    pres.clear_in_progress_marker(tmp_path)


def test_never_policy_parks_everything_with_handles(tmp_path, capsys):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "mod.py").write_text("X = 2\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="NEVER")
    (tmp_path / "b.txt").write_text("b\n")
    _git(tmp_path, "add", "b.txt")
    _git(tmp_path, "commit", "-qm", "upstream")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="never", keep_stash=False)

    assert outcome["tracked"]["status"] == "inactive"
    assert state.groups["tracked"].ref in outcome["tracked"]["reason"]
    # Tree stays on the clean updated revision; the stash survives.
    assert (tmp_path / "mod.py").read_text() == "X = 1\n"
    assert state.groups["tracked"].ref in _stash_shas(tmp_path)
    head = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    receipt = pres.write_preservation_receipt(
        tmp_path, state, head, outcome, policy="never", keep_stash=False)
    pres.print_preservation_summary(receipt)
    out = capsys.readouterr().out
    assert "inactive customizations" in out
    assert "recover with" in out
    pres.clear_in_progress_marker(tmp_path)


def test_keep_stash_forces_never_even_under_safe(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "mod.py").write_text("X = 2\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="KEEP")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=True)

    assert outcome["tracked"]["status"] == "inactive"
    assert "--keep-stash" in outcome["tracked"]["reason"]
    pres.clear_in_progress_marker(tmp_path)


def test_conflicting_group_stays_inactive_with_handle(tmp_path):
    origin, clone = _origin_clone_pair(tmp_path)
    _git(clone, "checkout", "-q", "main")
    # Tracked edit (committed base first so the change is tracked, not untracked).
    (clone / "a.txt").write_text("local\n")
    state = pres.preserve_local_changes(GIT, clone, "origin/main", stamp="CONF")
    # Upstream writes the same path: the preserved edit can no longer apply.
    (origin / "a.txt").write_text("upstream\n")
    _git(origin, "commit", "-aqm", "upstream a")
    _git(clone, "fetch", "-q", "origin", "main")
    _git(clone, "merge", "--ff-only", "-q", "origin/main")

    outcome = pres.restore_preserved_groups(
        GIT, clone, state, policy="safe", keep_stash=False)

    assert outcome["tracked"]["status"] == "inactive"
    assert "conflict" in outcome["tracked"]["reason"].lower()
    assert state.groups["tracked"].ref in outcome["tracked"]["reason"]
    # Base installation is never rolled back.
    assert (clone / "a.txt").read_text() == "upstream\n"
    assert state.groups["tracked"].ref in _stash_shas(clone)
    pres.clear_in_progress_marker(clone)


def test_syntax_broken_restore_stays_inactive(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "good.py").write_text("X = 1\n")
    _git(tmp_path, "add", "good.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "good.py").write_text("X = (\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="SYN")
    (tmp_path / "other.txt").write_text("o\n")
    _git(tmp_path, "add", "other.txt")
    _git(tmp_path, "commit", "-qm", "upstream")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=False)

    # A clean textual apply must not count as behavioral compatibility.
    assert outcome["tracked"]["status"] == "inactive"
    assert "validation" in outcome["tracked"]["reason"]
    assert (tmp_path / "good.py").read_text() == "X = 1\n"
    assert state.groups["tracked"].ref in _stash_shas(tmp_path)
    pres.clear_in_progress_marker(tmp_path)


def test_import_failure_restore_stays_inactive(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "mod.py").write_text("X = 2\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="IMP")
    monkeypatch.setattr(
        pres, "_validate_restored_tree",
        lambda *a, **k: "agent import hermes_cli.main: boom")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=False)

    assert outcome["tracked"]["status"] == "inactive"
    assert "boom" in outcome["tracked"]["reason"]
    pres.clear_in_progress_marker(tmp_path)


# ---------------------------------------------------------------------------
# Interruption recovery
# ---------------------------------------------------------------------------

def test_pending_with_newer_edits_fails_closed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("base\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "a.txt").write_text("dirty\n")
    pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="PEND")
    # Newer edits land on top of the unfinished marker.
    (tmp_path / "newer.txt").write_text("newer\n")

    with pytest.raises(SystemExit) as exc:
        pres.check_pending_preservation(GIT, tmp_path)
    assert exc.value.code == 1
    # Nothing is lost: the newer edit and the preserved stash both survive.
    assert (tmp_path / "newer.txt").read_text() == "newer\n"
    assert len(_stash_shas(tmp_path)) == 1


def test_pending_after_source_movement_stays_recoverable(tmp_path, capsys):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("base\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "a.txt").write_text("dirty\n")
    pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="MOVE")
    (tmp_path / "b.txt").write_text("upstream\n")
    _git(tmp_path, "add", "b.txt")
    _git(tmp_path, "commit", "-qm", "upstream")

    marker = pres.check_pending_preservation(GIT, tmp_path)

    assert marker is not None and marker["id"].startswith("MOVE-")
    assert "already moved" in capsys.readouterr().out
    pres.clear_in_progress_marker(tmp_path)


# ---------------------------------------------------------------------------
# Successive updates + updater's own change
# ---------------------------------------------------------------------------

def test_two_successive_runs_accumulate_receipts_and_stashes(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")

    (tmp_path / "mod.py").write_text("X = 2\n")
    first = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="R1")
    (tmp_path / "u1.txt").write_text("u1\n")
    _git(tmp_path, "add", "u1.txt")
    _git(tmp_path, "commit", "-qm", "upstream-1")
    outcome1 = pres.restore_preserved_groups(
        GIT, tmp_path, first, policy="never", keep_stash=False)
    head1 = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    receipt1 = pres.write_preservation_receipt(
        tmp_path, first, head1, outcome1, policy="never", keep_stash=False)
    pres.clear_in_progress_marker(tmp_path)
    stashes_after_first = _stash_shas(tmp_path)

    # Second update: the first run's parked stash is pre-existing now.
    (tmp_path / "second.txt").write_text("second\n")
    second = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="R2")
    assert all(sha in _stash_shas(tmp_path) for sha in stashes_after_first)
    assert second.base_ref != first.base_ref
    (tmp_path / "u2.txt").write_text("u2\n")
    _git(tmp_path, "add", "u2.txt")
    _git(tmp_path, "commit", "-qm", "upstream-2")
    outcome2 = pres.restore_preserved_groups(
        GIT, tmp_path, second, policy="never", keep_stash=False)
    head2 = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    pres.write_preservation_receipt(
        tmp_path, second, head2, outcome2, policy="never", keep_stash=False)
    pres.clear_in_progress_marker(tmp_path)

    receipts = sorted(
        p.name for p in (tmp_path / ".git" / "hermes-local-preservation").glob("receipt-*.json"))
    assert receipt1.name in receipts
    assert len(receipts) == 2
    payload = json.loads((tmp_path / ".git" / "hermes-local-preservation" / receipts[1]).read_text())
    assert payload["schema"] == pres.PRESERVATION_SCHEMA
    assert payload["pre_sha"] == second.pre_sha


def test_updater_own_conflicting_group_stays_recoverable(tmp_path):
    """A locally patched updater that conflicts with its own source update is
    saved inactive — and the new tree still ships the lifecycle mechanism."""
    _init_repo(tmp_path)
    updater = tmp_path / "updater.py"
    updater.write_text("VALUE = 1\n")
    _git(tmp_path, "add", "updater.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    updater.write_text("VALUE = 2  # local patch to the updater\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="SELF")
    # Upstream rewrites the same updater file: the local patch conflicts.
    updater.write_text("VALUE = 3  # upstream rewrite\n")
    _git(tmp_path, "add", "updater.py")
    _git(tmp_path, "commit", "-qm", "upstream updater change")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=False)

    assert outcome["tracked"]["status"] == "inactive"
    assert updater.read_text() == "VALUE = 3  # upstream rewrite\n"
    # Recovery handle retains the operator's patch on demand (it conflicts,
    # so recovery means resolving — the content must still be in the stash).
    stash_diff = _git(
        tmp_path, "stash", "show", "-p", state.groups["tracked"].ref).stdout
    assert "local patch" in stash_diff
    assert state.groups["tracked"].ref in _stash_shas(tmp_path)
    pres.clear_in_progress_marker(tmp_path)


# ---------------------------------------------------------------------------
# Checkout integration: guards are relaxed only where promised
# ---------------------------------------------------------------------------

def _prepare(monkeypatch, clone, **kwargs):
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    defaults = dict(
        branch="main", current_branch="old-feature", is_fork=False,
        assume_yes=True, gateway_mode=False, gw_input_fn=None,
        switch_branch=False, target_ref="origin/main",
        _windows_gateway_resume=None, preserve_requested=True)
    defaults.update(kwargs)
    return update_cmd._prepare_checkout_for_update(GIT, **defaults)


def test_dirty_in_place_reaches_preservation_instead_of_exit(tmp_path, monkeypatch):
    origin, clone = _origin_clone_pair(tmp_path)
    (clone / "feat.txt").write_text("unmerged\n")
    _git(clone, "add", "feat.txt")
    _git(clone, "commit", "-qm", "local commit")
    (clone / "a.txt").write_text("dirty\n")
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}})

    # Without the opt-in the historical guard still exits before any stash.
    with pytest.raises(SystemExit) as exc:
        _prepare(monkeypatch, clone, preserve_requested=False)
    assert exc.value.code == 1
    assert _git(clone, "stash", "list").stdout.strip() == ""

    plan = _prepare(monkeypatch, clone, preserve_requested=True)

    assert plan.in_place_update is True
    assert getattr(plan.preservation_state, "base_ref", "") != ""
    assert _git(clone, "status", "--porcelain").stdout == ""
    pres.clear_in_progress_marker(clone)


def test_switch_branch_still_refuses_dirty_tree(tmp_path, monkeypatch):
    _, clone = _origin_clone_pair(tmp_path)
    (clone / "feat.txt").write_text("unmerged\n")
    _git(clone, "add", "feat.txt")
    _git(clone, "commit", "-qm", "local commit")
    (clone / "a.txt").write_text("dirty\n")
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}})

    with pytest.raises(SystemExit) as exc:
        _prepare(monkeypatch, clone, preserve_requested=True, switch_branch=True)
    assert exc.value.code == 1


def test_unverifiable_state_still_fails_closed(tmp_path, monkeypatch, capsys):
    origin, clone = _origin_clone_pair(tmp_path)
    (clone / "feat.txt").write_text("unmerged\n")
    _git(clone, "add", "feat.txt")
    _git(clone, "commit", "-qm", "local commit")
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}})
    # Point at a target the guard cannot verify: preservation must not invent
    # a way forward around it.
    with pytest.raises(SystemExit) as exc:
        _prepare(monkeypatch, clone, preserve_requested=True,
                 target_ref="origin/no-such-branch", branch="no-such-branch")
    assert exc.value.code == 1


def test_up_to_date_path_restores_preserved_tree(tmp_path, monkeypatch):
    """commit_count == 0 with preservation parked: the tree comes back (undo)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "checkout", "-q", "main")
    (clone / "local.txt").write_text("local\n")
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda: {"updates": {"parked_branch_strategy": "update_in_place"}})
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda *a, **k: None)
    state = pres.preserve_local_changes(GIT, clone, "origin/main", stamp="UPTODATE")
    assert (clone / "local.txt").exists() is False
    plan = update_cmd._CheckoutPlan(
        auto_stash_ref=None, commit_count=0, in_place_update=True,
        parked_branch_switched=False, prompt_for_restore=False,
        switch_block_reason=None, upstream_checked=True,
        preservation_state=state)
    update_cmd._finish_already_up_to_date(
        GIT, "main", "main", plan, gw_input_fn=None, completion_request=None)
    # Undo even under the default never policy: no base movement happened.
    assert (clone / "local.txt").read_text() == "local\n"
    assert pres.read_in_progress_marker(clone) is None


def test_settle_writes_receipt_and_desktop_fact(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "mod.py").write_text("X = 1\n")
    _git(tmp_path, "add", "mod.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "mod.py").write_text("X = 2\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="FACT")
    plan = update_cmd._CheckoutPlan(
        auto_stash_ref=None, commit_count=1, in_place_update=True,
        parked_branch_switched=False, prompt_for_restore=False,
        switch_block_reason=None, upstream_checked=True,
        preservation_state=state)
    from hermes_cli import main as hermes_main
    from hermes_cli import update_receipt

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)

    class _Probe:
        def __init__(self):
            self.steps = []
            self.data = {}

    probe = _Probe()
    token = update_receipt._current.set(probe)
    try:
        update_cmd._settle_preservation_after_update(
            GIT, plan, policy="never", keep_stash=False)
    finally:
        update_receipt._current.reset(token)

    receipt_files = list(
        (tmp_path / ".git" / "hermes-local-preservation").glob("receipt-*.json"))
    assert len(receipt_files) == 1
    payload = json.loads(receipt_files[0].read_text())
    assert payload["inactive_groups"] == ["tracked"]
    assert pres.read_in_progress_marker(tmp_path) is None

    from hermes_cli.web_routers.actions import _latest_update_receipt_summary

    summary = _latest_update_receipt_summary()
    # No active receipt outside the probe: the summary helper fails closed.
    assert summary is None or isinstance(summary, dict)


def test_failed_untracked_group_does_not_wipe_successful_tracked_group(tmp_path, monkeypatch):
    """Cross-group isolation: a later group's failure must not wipe an earlier success.

    The exact gap in the suite (#130902): tracked applies cleanly and validates,
    then untracked fails validation (syntax-broken file). The tracked content must
    survive in the tree, its receipt must stay truthfully active, the untracked
    stash must survive with its handle, and only the untracked paths revert.
    """
    _init_repo(tmp_path)
    # Keep HERMES_HOME outside the repo: the global pytest sandbox materializes
    # the home (SOUL.md, logs) inside tmp_path, and preserve would stash those
    # home files into the untracked group — then validation's import probe
    # recreates SOUL.md mid-restore and the untracked apply conflicts with
    # itself. Real layouts never nest the home inside the repo.
    outside_home = tmp_path.parent / f"{tmp_path.name}-home"
    outside_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(outside_home))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(outside_home))
    (tmp_path / "keep.py").write_text("X = 1\n")
    _git(tmp_path, "add", "keep.py")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "keep.py").write_text("X = 2  # MY_LOCAL\n")
    (tmp_path / "broken_local.py").write_text("def broken(:\n")
    state = pres.preserve_local_changes(GIT, tmp_path, "origin/main", stamp="XGRP")
    assert state.groups["tracked"].present and state.groups["untracked"].present
    # Upstream advances an unrelated file.
    (tmp_path / "b.txt").write_text("b\n")
    _git(tmp_path, "add", "b.txt")
    _git(tmp_path, "commit", "-qm", "upstream")

    outcome = pres.restore_preserved_groups(
        GIT, tmp_path, state, policy="safe", keep_stash=False)

    assert outcome["tracked"]["status"] == "active", outcome
    assert outcome["untracked"]["status"] == "inactive", outcome
    assert "validation" in outcome["untracked"]["reason"]
    # The successful group's content survived the later group's failure.
    assert (tmp_path / "keep.py").read_text() == "X = 2  # MY_LOCAL\n"
    # The failed group's file is gone from the tree but its stash survives.
    assert not (tmp_path / "broken_local.py").exists()
    assert state.groups["untracked"].ref in _stash_shas(tmp_path)
    # The successful group's stash was dropped (normal) — and the receipt's
    # active claim is backed by content actually in the tree.
    assert state.groups["tracked"].ref not in _stash_shas(tmp_path)
    receipt = pres.write_preservation_receipt(
        tmp_path, state, "upstream-rev", outcome, policy="safe", keep_stash=False)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["active_groups"] == ["tracked"]
    assert "MY_LOCAL" in (tmp_path / "keep.py").read_text()
    pres.clear_in_progress_marker(tmp_path)
