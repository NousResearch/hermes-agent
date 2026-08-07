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
    commit_count="3",
    ff_only_fails=False,
    reset_fails=False,
    fetch_fails=False,
    fetch_stderr="",
):
    """Build a subprocess.run side_effect for cmd_update tests."""
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
        if "rev-list" in joined:
            return SimpleNamespace(stdout=f"{commit_count}\n", stderr="", returncode=0)
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


# ---------------------------------------------------------------------------
# reset --hard on diverged history must not silently discard local COMMITS
# (as opposed to uncommitted changes, which the autostash above already
# covers). A committed local fix shows up clean in `git status` and was
# never protected by the stash -- it needs its own preserve-and-reapply path.
# ---------------------------------------------------------------------------

def _make_divergence_side_effect(
    *, local_only_count="0", cherry_pick_fails=False,
    local_only_count_fails=False, update_ref_fails=False,
):
    """subprocess.run side_effect for the ff-only-fails -> reset -> preserve
    -> cherry-pick flow. Distinguishes the two different rev-list calls by
    range direction: "HEAD..origin/<branch>" is the pre-existing "how many
    new commits" check; "origin/<branch>..HEAD" is this fix's "how many
    local-only commits would the reset destroy" check.
    """
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(list(cmd))
        joined = " ".join(str(c) for c in cmd)
        if "fetch" in joined and "origin" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(stdout="main\n", stderr="", returncode=0)
        if "checkout" in joined and "main" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-list" in joined and "HEAD..origin" in joined:
            return SimpleNamespace(stdout="3\n", stderr="", returncode=0)
        if "rev-list" in joined and "origin/main..HEAD" in joined:
            if local_only_count_fails:
                # _count_commits_between returns -1 for this.
                return SimpleNamespace(
                    stdout="", stderr="fatal: bad revision\n", returncode=128,
                )
            return SimpleNamespace(stdout=f"{local_only_count}\n", stderr="", returncode=0)
        if "--ff-only" in joined:
            return SimpleNamespace(
                stdout="", stderr="fatal: Not possible to fast-forward, aborting.\n",
                returncode=128,
            )
        if "update-ref" in joined:
            if update_ref_fails:
                return SimpleNamespace(
                    stdout="", stderr="fatal: cannot lock ref\n", returncode=128,
                )
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "reset" in joined and "--hard" in joined:
            return SimpleNamespace(stdout="HEAD is now at abc123\n", stderr="", returncode=0)
        if "cherry-pick" in joined and "--abort" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "cherry-pick" in joined:
            if cherry_pick_fails:
                return SimpleNamespace(
                    stdout="", stderr="error: could not apply ...\nCONFLICT\n", returncode=1,
                )
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect, recorded


def test_cmd_update_preserves_and_reapplies_local_commits_on_reset(monkeypatch, tmp_path, capsys):
    """Local commits ahead of origin get backed up before the hard reset and
    cleanly reapplied on top afterward -- the fix's happy path."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)

    side_effect, recorded = _make_divergence_side_effect(local_only_count="2")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    monkeypatch.setattr(hermes_main, "_validate_critical_files_syntax", lambda root: (True, None, None))
    monkeypatch.setattr(hermes_main, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_install_python_dependencies_with_optional_fallback", lambda *a, **kw: None)

    hermes_main.cmd_update(SimpleNamespace())

    calls = [" ".join(str(c) for c in cmd) for cmd in recorded]
    update_ref_calls = [c for c in calls if "update-ref" in c]
    cherry_pick_calls = [c for c in calls if "cherry-pick" in c and "--abort" not in c]
    assert len(update_ref_calls) == 1
    assert "refs/hermes-update-backups/main-" in update_ref_calls[0]
    assert len(cherry_pick_calls) == 1
    assert "HEAD..refs/hermes-update-backups/main-" in cherry_pick_calls[0]

    out = capsys.readouterr().out
    assert "Preserving 2 local commit(s)" in out
    assert "Reapplied 2 local commit(s)" in out


def test_cmd_update_cherry_pick_conflict_leaves_backup_ref_and_prints_recovery(monkeypatch, tmp_path, capsys):
    """A conflicting reapply aborts cleanly and points at the backup ref --
    nothing is lost, no conflict markers are left in the tree."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)

    side_effect, recorded = _make_divergence_side_effect(local_only_count="1", cherry_pick_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    monkeypatch.setattr(hermes_main, "_validate_critical_files_syntax", lambda root: (True, None, None))
    monkeypatch.setattr(hermes_main, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_install_python_dependencies_with_optional_fallback", lambda *a, **kw: None)

    hermes_main.cmd_update(SimpleNamespace())

    calls = [" ".join(str(c) for c in cmd) for cmd in recorded]
    assert any("cherry-pick --abort" in c for c in calls)

    out = capsys.readouterr().out
    assert "Could not automatically reapply your 1 local" in out
    assert "refs/hermes-update-backups/main-" in out
    assert "git cherry-pick HEAD..refs/hermes-update-backups/main-" in out


def test_cmd_update_skips_backup_when_no_local_commits_diverged(monkeypatch, tmp_path, capsys):
    """A pure remote-side divergence (e.g. upstream rebase) with zero local
    commits ahead must not create a backup ref -- nothing local to lose."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)

    side_effect, recorded = _make_divergence_side_effect(local_only_count="0")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    monkeypatch.setattr(hermes_main, "_validate_critical_files_syntax", lambda root: (True, None, None))
    monkeypatch.setattr(hermes_main, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_install_python_dependencies_with_optional_fallback", lambda *a, **kw: None)

    hermes_main.cmd_update(SimpleNamespace())

    calls = [" ".join(str(c) for c in cmd) for cmd in recorded]
    assert not any("update-ref" in c for c in calls)
    assert not any("cherry-pick" in c for c in calls)


def test_cmd_update_backs_up_when_local_commit_count_is_unknown(monkeypatch, tmp_path, capsys):
    """If the local-only commit count itself fails, the reset must still be
    preceded by a backup ref. _count_commits_between returns -1 on error, and
    the old `if local_only > 0` guard silently skipped the backup -- letting
    `reset --hard` destroy exactly the commits this code exists to protect."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)

    side_effect, recorded = _make_divergence_side_effect(local_only_count_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    monkeypatch.setattr(hermes_main, "_validate_critical_files_syntax", lambda root: (True, None, None))
    monkeypatch.setattr(hermes_main, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_install_python_dependencies_with_optional_fallback", lambda *a, **kw: None)

    hermes_main.cmd_update(SimpleNamespace())

    calls = [" ".join(str(c) for c in cmd) for cmd in recorded]
    backup_idx = next(i for i, c in enumerate(calls) if "update-ref" in c)
    reset_idx = next(i for i, c in enumerate(calls) if "reset --hard" in c)
    assert backup_idx < reset_idx, "backup ref must be created before the hard reset"

    out = capsys.readouterr().out
    assert "Could not determine how many local commits" in out
    # never render the -1 sentinel at the user
    assert "-1 local commit" not in out


def test_cmd_update_aborts_when_backup_ref_cannot_be_created(monkeypatch, tmp_path, capsys):
    """If `git update-ref` fails, the destructive reset must NOT run. Previously
    the return code was unchecked, so the code claimed 'backed up to <ref>' and
    then reset --hard anyway, losing the commits it said it had saved."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)

    side_effect, recorded = _make_divergence_side_effect(
        local_only_count="2", update_ref_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    monkeypatch.setattr(hermes_main, "_validate_critical_files_syntax", lambda root: (True, None, None))
    monkeypatch.setattr(hermes_main, "_venv_core_imports_healthy", lambda: (True, ""))
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_install_python_dependencies_with_optional_fallback", lambda *a, **kw: None)

    with pytest.raises(SystemExit) as excinfo:
        hermes_main.cmd_update(SimpleNamespace())
    assert excinfo.value.code == 1

    calls = [" ".join(str(c) for c in cmd) for cmd in recorded]
    assert not any("reset --hard" in c for c in calls), (
        "must not reset --hard when the backup ref could not be created"
    )

    out = capsys.readouterr().out
    assert "Could not create backup ref" in out
    assert "still intact" in out
