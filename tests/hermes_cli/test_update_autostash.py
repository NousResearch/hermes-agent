from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main


# ---------------------------------------------------------------------------
# Managed-uv compatibility for tests that patch shutil.which
# ---------------------------------------------------------------------------
# The production code now uses ``ensure_uv()`` / ``update_managed_uv()``
# instead of ``shutil.which("uv")``.  Many tests in this file patch
# ``shutil.which`` to control whether uv is "available" — these autouse
# fixtures make the managed_uv functions delegate to the patched
# ``shutil.which`` so the existing test setup keeps working without
# per-test changes.
@pytest.fixture(autouse=True)
def _patch_managed_uv(request):
    """Make managed_uv helpers follow shutil.which mocking in tests."""
    import shutil

    # resolve_uv delegates to shutil.which("uv") so that test patches
    # on shutil.which flow through naturally.
    def _fake_resolve_uv(**kwargs):
        return shutil.which("uv")

    def _fake_ensure_uv(**kwargs):
        return shutil.which("uv")

    def _fake_update_managed_uv(**kwargs):
        return None  # never actually self-update in tests

    with patch("hermes_cli.managed_uv.resolve_uv", side_effect=_fake_resolve_uv), \
         patch("hermes_cli.managed_uv.ensure_uv", side_effect=_fake_ensure_uv), \
         patch("hermes_cli.managed_uv.update_managed_uv", side_effect=_fake_update_managed_uv):
        yield













# ---------------------------------------------------------------------------
# Update uses .[all] with fallback to .
# ---------------------------------------------------------------------------

def _setup_update_mocks(monkeypatch, tmp_path):
    """Common setup for cmd_update tests."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **kw: True)
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda: (5, 5))
    monkeypatch.setattr(hermes_config, "migrate_config", lambda **kw: {"env_added": [], "config_added": []})
    monkeypatch.setattr(hermes_main, "_upgrade_pip_before_lazy_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda *a, **kw: True)




def test_refresh_active_memory_provider_dependencies_reinstalls_active_provider(monkeypatch):
    """#53272/#70636: update must re-run the active provider's dep install."""
    recorded = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda provider_name, force=False: recorded.append((provider_name, force)),
    )

    hermes_main._refresh_active_memory_provider_dependencies()

    assert recorded == [("mem0", True)]




def test_reload_updated_runtime_modules_restores_new_hermes_constants_symbol(monkeypatch):
    """A pre-pull module object missing a new helper is repaired by reload."""
    import hermes_constants

    monkeypatch.delattr(hermes_constants, "apply_subprocess_home_env", raising=False)
    assert not hasattr(hermes_constants, "apply_subprocess_home_env")

    hermes_main._reload_updated_runtime_modules()

    assert callable(hermes_constants.apply_subprocess_home_env)






# ---------------------------------------------------------------------------
# ff-only fallback to reset --hard on diverged history
# ---------------------------------------------------------------------------

def _make_update_side_effect(
    current_branch="main",
    upstream_commit_count="3",
    local_ahead_count="0",
    ff_only_fails=False,
    reset_fails=False,
    fetch_fails=False,
    fetch_stderr="",
    rebase_succeeds=True,
):
    """Build a subprocess.run side_effect for cmd_update tests.

    ``upstream_commit_count`` answers the pre-#74885 update check
    ``git rev-list HEAD..origin/<branch> --count`` ("how many commits on
    origin is the local branch missing?"). The default of ``"3"`` matches
    the pre-#74885 expectation that origin has new commits, so existing
    tests that drive the "fetch → pull → reset" flow observe the same
    behaviour they did before.

    ``local_ahead_count`` answers the #74885 ``git rev-list
    origin/<branch>..HEAD --count`` check ("how many local commits does
    origin NOT have?"). The default of ``"0"`` matches the pre-#74885
    expectation that the local branch has nothing ahead of origin, so the
    destructive ``reset --hard`` fallback still fires when ff-only fails.
    Tests that want to exercise the new rebase-preferred path set
    ``local_ahead_count="3"`` (or similar) and optionally
    ``rebase_succeeds=False`` to drive the conflict-then-reset branch.
    """
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "fetch" in joined and "origin" in joined:
            if fetch_fails:
                return SimpleNamespace(stdout="", stderr=fetch_stderr, returncode=128)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(stdout=f"{current_branch}\n", stderr="", returncode=0)
        if "checkout" in joined and "main" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-list" in joined and "HEAD..origin" in joined:
            return SimpleNamespace(stdout=f"{upstream_commit_count}\n", stderr="", returncode=0)
        if "rev-list" in joined and "origin" in joined and "..HEAD" in joined:
            return SimpleNamespace(stdout=f"{local_ahead_count}\n", stderr="", returncode=0)
        if "rebase" in joined and "origin" in joined:
            if rebase_succeeds:
                return SimpleNamespace(stdout="Rebase succeeded.\n", stderr="", returncode=0)
            return SimpleNamespace(stdout="", stderr="CONFLICT in foo.py\n", returncode=1)
        if "--ff-only" in joined:
            if ff_only_fails:
                return SimpleNamespace(
                    stdout="",
                    stderr="fatal: Not possible to fast-forward, aborting.\n",
                    returncode=128,
                )
            return SimpleNamespace(stdout="Updating abc..def\n", stderr="", returncode=0)
        if "reset" in joined and "--hard" in joined:
            if reset_fails:
                return SimpleNamespace(stdout="", stderr="error: unable to write\n", returncode=1)
            return SimpleNamespace(stdout="HEAD is now at abc123\n", stderr="", returncode=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect, recorded


# ---------------------------------------------------------------------------
# Non-main branch → auto-checkout main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fetch failure — friendly error messages
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reset --hard failure — don't attempt stash restore
# ---------------------------------------------------------------------------

def test_cmd_update_skips_stash_restore_when_reset_fails(monkeypatch, tmp_path, capsys):
    """When reset --hard fails, stash restore is skipped with a helpful message."""
    _setup_update_mocks(monkeypatch, tmp_path)
    # Re-enable stash so it actually returns a ref
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )

    side_effect, _ = _make_update_side_effect(ff_only_fails=True, reset_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    # Stash restore should NOT have been called
    assert len(restore_calls) == 0

    out = capsys.readouterr().out
    assert "preserved in stash" in out


def test_cmd_update_rebases_when_local_ahead_of_origin(monkeypatch, tmp_path, capsys):
    """#74885: when ff-only fails AND the local branch has commits ahead of
    origin, prefer ``git rebase origin/<branch>`` over the destructive
    ``reset --hard`` so the user's local commits are preserved. Successful
    rebase should fall through to the normal post-pull path."""
    _setup_update_mocks(monkeypatch, tmp_path)
    restore_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )

    side_effect, recorded = _make_update_side_effect(
        local_ahead_count="3",
        ff_only_fails=True,
        rebase_succeeds=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    # No reset --hard should have been attempted (rebase preserved the work).
    assert not any(
        "reset" in " ".join(c) and "--hard" in " ".join(c) for c in recorded
    ), f"reset --hard should not run when rebase succeeds; recorded: {recorded}"
    # Rebase SHOULD have been invoked.
    assert any("rebase" in " ".join(c) for c in recorded), (
        f"rebase should run when local branch is ahead of origin; recorded: {recorded}"
    )
    out = capsys.readouterr().out
    assert "rebasing onto remote to preserve your work" in out


def test_cmd_update_falls_back_to_reset_when_rebase_conflicts(
    monkeypatch, tmp_path, capsys
):
    """#74885: rebase fails (conflict). Abort the rebase and fall back to the
    destructive reset so the install can still recover. The user must still
    get a clear warning that local commits will be lost and ``git reflog``
    can recover them. The fallback reset succeeding is fine — the install
    continues; the warning is the load-bearing change."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        local_ahead_count="2",
        ff_only_fails=True,
        rebase_succeeds=False,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    # No SystemExit expected: rebase-conflict falls back to reset, which
    # succeeds in this fixture, so the install continues. The user-visible
    # behaviour change is the warning + git reflog hint, which we assert below.
    hermes_main.cmd_update(SimpleNamespace())

    # Both rebase AND reset --hard should have been invoked.
    assert any("rebase" in " ".join(c) for c in recorded)
    assert any(
        "reset" in " ".join(c) and "--hard" in " ".join(c) for c in recorded
    )
    # rebase --abort was issued to leave the user's history untouched.
    assert any("--abort" in " ".join(c) for c in recorded)
    out = capsys.readouterr().out
    # Backup ref message must appear so the user knows where to recover
    # their commits outside the reflog window (#74885 reviewer feedback).
    assert "Backed up local branch to refs/hermes-backups/pre-update-" in out
    assert "Your local commits are preserved at" in out


def test_cmd_update_exits_when_rebase_and_reset_both_fail(
    monkeypatch, tmp_path, capsys
):
    """#74885: worst case — rebase fails AND reset fails (e.g. disk full). We
    must still surface the recovery instructions and exit 1 so the user
    doesn't continue thinking the install succeeded."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, recorded = _make_update_side_effect(
        local_ahead_count="2",
        ff_only_fails=True,
        rebase_succeeds=False,
        reset_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    out = capsys.readouterr().out
    # The new backup-ref message plus the original recovery instructions
    # should both surface so the user knows their commits are safe at
    # refs/hermes-backups/pre-update-* and how to retry the reset manually.
    assert "Your commits are safe at refs/hermes-backups/pre-update-" in out
    assert "git fetch origin && git reset --hard origin/main" in out


def test_cmd_update_exits_when_revlis_fails_without_resetting(
    monkeypatch, tmp_path, capsys
):
    """#74885 reviewer feedback: ``git rev-list origin/HEAD..HEAD --count``
    returning non-zero exit (corrupt repo, missing origin ref, etc.) must NOT
    be coerced to "0 local commits ahead" — that path leads straight into the
    destructive ``reset --hard`` and silently wipes user work. Refuse to
    proceed and exit 1 instead."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect_calls = {"rev_list": 0, "reset": 0, "rebase": 0}
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "rev-list" in joined and "origin" in joined and "..HEAD" in joined:
            side_effect_calls["rev_list"] += 1
            # Simulate git failing (e.g. origin ref missing): non-zero exit,
            # empty stdout. This is the dangerous case the reviewer flagged:
            # pre-fix this was coerced to ahead_count=0 and proceeded to
            # `reset --hard origin/<branch>`.
            return SimpleNamespace(
                stdout="",
                stderr="fatal: bad revision 'origin/main'\n",
                returncode=128,
            )
        if "rev-list" in joined and "HEAD..origin" in joined:
            # Earlier "are there updates?" check; just report 3 commits so the
            # flow reaches our patch path (it must arrive there for the rev-list
            # returncode != 0 path to be exercised).
            return SimpleNamespace(stdout="3\n", stderr="", returncode=0)
        if "reset" in joined and "--hard" in joined:
            side_effect_calls["reset"] += 1
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rebase" in joined:
            side_effect_calls["rebase"] += 1
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "--ff-only" in joined:
            return SimpleNamespace(
                stdout="",
                stderr="fatal: Not possible to fast-forward, aborting.\n",
                returncode=128,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    # rev-list was called exactly once; reset and rebase must NOT have run.
    assert side_effect_calls["rev_list"] == 1
    assert side_effect_calls["reset"] == 0
    assert side_effect_calls["rebase"] == 0

    out = capsys.readouterr().out
    assert "refusing to reset without knowing whether local commits exist" in out


def test_cmd_update_exits_when_backup_ref_creation_fails(
    monkeypatch, tmp_path, capsys
):
    """#74885 reviewer feedback: if the durable backup ref cannot be created
    (e.g. permissions on refs/hermes-backups/), we MUST refuse to rebase or
    reset. Without a backup, the rebase+reset dance can destroy local commits
    even with the conflict-abort fallback, because ``git reset --hard`` walks
    past the reflog window."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect_calls = {"backup": 0, "reset": 0, "rebase": 0}
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "rev-list" in joined and "origin" in joined and "..HEAD" in joined:
            return SimpleNamespace(stdout="2\n", stderr="", returncode=0)
        if "rev-list" in joined and "HEAD..origin" in joined:
            return SimpleNamespace(stdout="3\n", stderr="", returncode=0)
        if "update-ref" in joined:
            side_effect_calls["backup"] += 1
            return SimpleNamespace(
                stdout="",
                stderr="error: cannot lock ref\n",
                returncode=1,
            )
        if "reset" in joined and "--hard" in joined:
            side_effect_calls["reset"] += 1
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rebase" in joined:
            side_effect_calls["rebase"] += 1
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "--ff-only" in joined:
            return SimpleNamespace(
                stdout="",
                stderr="fatal: Not possible to fast-forward, aborting.\n",
                returncode=128,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    # Backup ref creation was attempted exactly once; rebase/reset must NOT
    # have run, otherwise we'd be right back at the reviewer's complaint
    # about silent data loss.
    assert side_effect_calls["backup"] == 1
    assert side_effect_calls["rebase"] == 0
    assert side_effect_calls["reset"] == 0

    out = capsys.readouterr().out
    assert "refusing to rebase or reset without a safety net" in out


# ---------------------------------------------------------------------------
# Non-interactive update.non_interactive_local_changes setting
# (chat app / gateway): "discard" throws stashed changes away, "stash"
# (default) restores them. Interactive terminal updates ignore the setting
# and always go through the restore path.
# ---------------------------------------------------------------------------

def _setup_setting_test(monkeypatch, tmp_path, mode):
    """Common wiring: real stash returns a ref, restore + discard are
    recorded, and load_config reports the given non_interactive_local_changes
    mode."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    discard_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda *a, **kw: discard_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **kw: {"updates": {"non_interactive_local_changes": mode}},
    )
    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    return restore_calls, discard_calls, recorded






def test_bootstrap_marker_not_autostashed_by_update(tmp_path):
    """#38529: the Desktop bootstrap marker must be git-ignored so that
    ``hermes update``'s ``git stash push --include-untracked`` does not sweep it
    into an autostash on every run.

    Behavioral + hermetic: build a throwaway repo that adopts the project's real
    ``.gitignore`` (the contract under test), drop the marker, and confirm the
    same stash invocation the updater uses leaves it untouched.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo_gitignore = Path(hermes_main.__file__).resolve().parents[1] / ".gitignore"

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(repo_gitignore.read_text())
    (tmp_path / "tracked.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    marker = tmp_path / ".hermes-bootstrap-complete"
    marker.write_text("")

    # Exact flags used by hermes update (hermes_cli/main.py).
    git("stash", "push", "--include-untracked", "-m", "hermes-update-autostash")

    assert marker.exists(), (
        ".hermes-bootstrap-complete was swept into the update autostash — it must "
        "be listed in .gitignore so `git stash -u` skips it (#38529)."
    )
    # It must not even register as a dirty/untracked change.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert ".hermes-bootstrap-complete" not in status


# ---------------------------------------------------------------------------
# Permission-denied autostash class: undeletable untracked files (root-owned
# packaging/ etc.) must not abort the update when the stash entry was created.
# ---------------------------------------------------------------------------






def test_update_autostash_survives_undeletable_untracked_dir(tmp_path):
    """Behavioral E2E of the whole permission-denied class with real git:
    root-owned-style undeletable untracked dir → stash succeeds, update-style
    reset works, restore round-trips, nothing lost. (#70127 follow-up)"""
    import os
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    if os.name == "nt":
        pytest.skip("POSIX permission semantics")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write bits")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text("v2 local change\n")
    pkg = tmp_path / "packaging" / "homebrew"
    pkg.mkdir(parents=True)
    (pkg / "hermes-agent.rb").write_text("formula\n")
    os.chmod(pkg, 0o555)  # undeletable contents, like a root-owned dir
    try:
        stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
        assert stash_ref

        # The tracked change is stashed; simulate the updater's checkout window.
        assert (tmp_path / "tracked.txt").read_text() == "v1\n"

        restored = hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )
        assert restored is True
        assert (tmp_path / "tracked.txt").read_text() == "v2 local change\n"
        assert (pkg / "hermes-agent.rb").read_text() == "formula\n"
    finally:
        os.chmod(pkg, 0o755)
