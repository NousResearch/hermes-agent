"""Pre-flight abort + post-sync import smoke with snapshot rollback for ``hermes update``.

Issue #115639: the git-pull path synced first and asked questions later. A dirty
tree (autostash skipped or failed) was merged over, same-branch local commits were
destroyed by the diverged-checkout ``reset --hard``, and pulled code that *parses*
but does not *import* sailed through the syntax guard and rode the drain /
forced-restart onto a bricked install.

The loop is deliberately small:
* ``_check_pull_preflight`` runs before the ff-only merge — a dirty tree, or
  same-branch local commits ahead of origin that the reset would destroy,
  aborts the update before anything is changed.
* ``_smoke_check_pulled_imports_or_rollback`` runs right after the syntax guard —
  pulled code that fails to import rolls back to the pre-pull SHA.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


PRE_SHA = "1111111111111111111111111111111111111beef"

CLEAN_V2 = (
    "# branch.oid aaa\n"
    "# branch.head main\n"
    "# branch.upstream origin/main\n"
    "# branch.ab +0 -3\n"
)
DIRTY_V2 = CLEAN_V2 + "1 M. N... 100644 100644 100644 abc123 abc123 hermes_cli/config.py\n"
UNTRACKED_V2 = CLEAN_V2 + "? scratch.txt\n"
AHEAD_V2 = (
    "# branch.oid aaa\n"
    "# branch.head main\n"
    "# branch.upstream origin/main\n"
    "# branch.ab +2 -1\n"
)
CUSTOM_BRANCH_V2 = (
    "# branch.oid aaa\n"
    "# branch.head my-feature\n"
    "# branch.upstream origin/my-feature\n"
    "# branch.ab +5 -0\n"
)


def _ns(stdout="", returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _fake_git_run(status_out=CLEAN_V2, status_rc=0, ff_only_rc=0):
    """Fake ``_git_run``: status probe per-test, rev-parse HEAD -> PRE_SHA."""
    recorded = []

    def fake(git_cmd, args, cwd=None, **kwargs):
        recorded.append(list(args))
        joined = " ".join(str(c) for c in args)
        if "status" in joined:
            return _ns(status_out, status_rc)
        if joined.endswith("rev-parse HEAD"):
            return _ns(PRE_SHA + "\n")
        if "--ff-only" in joined:
            return _ns("Updating abc..def\n" if ff_only_rc == 0 else "",
                       ff_only_rc,
                       "" if ff_only_rc == 0 else "fatal: Not possible to fast-forward, aborting.\n")
        if "reset" in joined and "--hard" in joined:
            return _ns("HEAD is now at abc123\n")
        return _ns()

    return fake, recorded


def _pull_kwargs():
    return {"prompt_for_restore": False, "gw_input_fn": None,
            "discard_local_changes": False, "keep_stash": False}


# ---------------------------------------------------------------------------
# _check_pull_preflight
# ---------------------------------------------------------------------------

def test_preflight_passes_on_clean_tree(monkeypatch):
    fake, _ = _fake_git_run()
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, _detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind is None


def test_preflight_reports_dirty_tree(monkeypatch):
    fake, _ = _fake_git_run(status_out=DIRTY_V2)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind == "dirty"
    assert detail


def test_preflight_reports_untracked_files_as_dirty(monkeypatch):
    fake, _ = _fake_git_run(status_out=UNTRACKED_V2)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, _detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind == "dirty"


def test_preflight_reports_ahead_local_commits_on_same_branch(monkeypatch):
    """Local commits on the target branch would be destroyed by the diverged
    reset --hard — abort before the merge instead."""
    fake, _ = _fake_git_run(status_out=AHEAD_V2)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind == "diverged"
    assert detail


def test_preflight_skips_ahead_check_on_custom_branch(monkeypatch):
    """A parked custom branch *expects* commits beyond origin/<branch> — the
    merge path in _reconcile_diverged_checkout owns that shape, not the abort."""
    fake, _ = _fake_git_run(status_out=CUSTOM_BRANCH_V2)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, _detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind is None


def test_preflight_fails_open_when_probe_errors(monkeypatch):
    """A broken probe must never block an update — fail open, like every other
    advisory preflight in this pipeline."""
    fake, _ = _fake_git_run(status_out="", status_rc=128)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, _detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind is None


def test_preflight_fails_open_when_status_unparseable(monkeypatch):
    fake, _ = _fake_git_run(status_out="")
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    kind, _detail = update_cmd._check_pull_preflight(["git"], "main")

    assert kind is None


# ---------------------------------------------------------------------------
# _pull_updates wiring: abort before the merge
# ---------------------------------------------------------------------------

def test_pull_updates_aborts_before_merge_on_dirty_tree(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run(status_out=DIRTY_V2)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(["git"], "main", None, **_pull_kwargs())

    assert exc.value.code == 1
    flat = [" ".join(c) for c in recorded]
    assert any("status" in c for c in flat)
    assert not any("merge" in c for c in flat)
    assert not any("reset" in c for c in flat)
    out = capsys.readouterr().out
    assert "uncommitted changes" in out


def test_pull_updates_aborts_before_reset_on_ahead_tree(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run(status_out=AHEAD_V2, ff_only_rc=128)
    monkeypatch.setattr(update_cmd, "_git_run", fake)

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(["git"], "main", None, **_pull_kwargs())

    assert exc.value.code == 1
    flat = [" ".join(c) for c in recorded]
    # The diverged reset --hard must never run while local commits are at risk.
    assert not any("reset" in c for c in flat)
    out = capsys.readouterr().out
    assert "local commit" in out


def test_pull_updates_still_syncs_clean_tree(monkeypatch, tmp_path):
    """The preflight is transparent on the happy path: merge runs, PRE_HEAD returned."""
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run()
    monkeypatch.setattr(update_cmd, "_git_run", fake)
    monkeypatch.setattr(update_cmd, "_validate_critical_modules_import",
                        lambda root: (True, None, None))

    pre_sha = update_cmd._pull_updates(["git"], "main", None, **_pull_kwargs())

    assert pre_sha == PRE_SHA
    flat = [" ".join(c) for c in recorded]
    assert any("--ff-only" in c for c in flat)


# ---------------------------------------------------------------------------
# Post-sync import smoke with snapshot rollback
# ---------------------------------------------------------------------------

def test_smoke_success_leaves_tree_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run()
    monkeypatch.setattr(update_cmd, "_git_run", fake)
    monkeypatch.setattr(update_cmd, "_validate_critical_modules_import",
                        lambda root: (True, None, None))

    update_cmd._smoke_check_pulled_imports_or_rollback(["git"], PRE_SHA)

    assert not any("reset" in " ".join(c) for c in recorded)


def test_smoke_failure_rolls_back_to_pre_pull_sha(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run()
    monkeypatch.setattr(update_cmd, "_git_run", fake)
    monkeypatch.setattr(update_cmd, "_validate_critical_modules_import",
                        lambda root: (False, "hermes_cli.main", "cannot import name X"))

    with pytest.raises(SystemExit) as exc:
        update_cmd._smoke_check_pulled_imports_or_rollback(["git"], PRE_SHA)

    assert exc.value.code == 1
    resets = [c for c in recorded if "reset" in c and "--hard" in c]
    assert resets and resets[0][-1] == PRE_SHA
    out = capsys.readouterr().out
    assert "fails to import" in out
    assert "Rolling back" in out
    assert "Rollback complete" in out


def test_pull_updates_rolls_back_when_smoke_fails(monkeypatch, tmp_path, capsys):
    """End to end through _pull_updates: merge lands, syntax passes on the empty
    tmp tree, smoke fails -> reset --hard PRE_HEAD, exit 1, nothing proceeds."""
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path)
    fake, recorded = _fake_git_run()
    monkeypatch.setattr(update_cmd, "_git_run", fake)
    monkeypatch.setattr(update_cmd, "_validate_critical_modules_import",
                        lambda root: (False, "run_agent", "No module named run_agent"))

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(["git"], "main", None, **_pull_kwargs())

    assert exc.value.code == 1
    resets = [c for c in recorded if "reset" in c and "--hard" in c]
    assert resets and resets[0][-1] == PRE_SHA
