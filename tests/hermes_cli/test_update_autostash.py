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
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: (None, []))
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
        lambda *a, **kw: ("abc123deadbeef", []),
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
        lambda *a, **kw: ("abc123deadbeef", []),
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
        stash_ref, _ = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
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
# Windows reserved-device-name guard
# ---------------------------------------------------------------------------

def test_is_reserved_device_name_matches_all_reserved_names():
    """Every Windows reserved device name is detected, case-insensitive."""
    from hermes_cli.update_cmd import _is_reserved_device_name

    reserved = [
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ]
    for name in reserved:
        assert _is_reserved_device_name(name) is True
        assert _is_reserved_device_name(name.lower()) is True
        assert _is_reserved_device_name(name + ".txt") is True
        assert _is_reserved_device_name(name + ".TXT") is True

    # Non-matching names should return False
    assert _is_reserved_device_name("README") is False
    assert _is_reserved_device_name("convention") is False
    assert _is_reserved_device_name("com10") is False
    assert _is_reserved_device_name("lpt0") is False
    assert _is_reserved_device_name("") is False


def test_rename_reserved_untracked_renames_only_matching_files(tmp_path, monkeypatch):
    """Only untracked files matching reserved names are renamed."""
    from hermes_cli.update_cmd import _rename_reserved_untracked
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

    # Create files
    (tmp_path / "CON.txt").write_text("con content")
    (tmp_path / "README.md").write_text("readme content")
    (tmp_path / "PRN").write_text("prn content")

    ls_output = "CON.txt\nREADME.md\nPRN\n"
    renames = _rename_reserved_untracked(tmp_path, ls_output)

    # CON.txt and PRN should be renamed
    assert len(renames) == 2
    renamed_names = [r[1].name for r in renames]
    assert ".hermes_reserved_name_CON.txt" in renamed_names
    assert ".hermes_reserved_name_PRN" in renamed_names

    # Original files should no longer exist
    assert not (tmp_path / "CON.txt").exists()
    assert not (tmp_path / "PRN").exists()

    # README.md should be untouched
    assert (tmp_path / "README.md").exists()

    # Renamed files should exist
    assert (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert (tmp_path / ".hermes_reserved_name_PRN").exists()


def test_rename_reserved_untracked_noop_on_non_windows(tmp_path, monkeypatch):
    """On non-Windows platforms, no renaming happens."""
    from hermes_cli.update_cmd import _rename_reserved_untracked
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    (tmp_path / "CON.txt").write_text("con content")
    ls_output = "CON.txt\n"
    renames = _rename_reserved_untracked(tmp_path, ls_output)

    assert renames == []
    assert (tmp_path / "CON.txt").exists()


def test_restore_reserved_renames_restores_original_names(tmp_path, monkeypatch):
    """Renamed files are restored to their original names when passed explicitly."""
    from hermes_cli.update_cmd import _restore_reserved_renames
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

    # Simulate pre-stash renamed files
    (tmp_path / ".hermes_reserved_name_CON.txt").write_text("con content")
    (tmp_path / ".hermes_reserved_name_AUX.md").write_text("aux content")

    renames = [
        (tmp_path / "CON.txt", tmp_path / ".hermes_reserved_name_CON.txt"),
        (tmp_path / "AUX.md", tmp_path / ".hermes_reserved_name_AUX.md"),
    ]
    _restore_reserved_renames(tmp_path, renames)

    assert (tmp_path / "CON.txt").exists()
    assert (tmp_path / "AUX.md").exists()
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert not (tmp_path / ".hermes_reserved_name_AUX.md").exists()


def test_restore_reserved_renames_noop_on_non_windows(tmp_path, monkeypatch):
    """On non-Windows platforms, restore is a no-op."""
    from hermes_cli.update_cmd import _restore_reserved_renames
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    (tmp_path / ".hermes_reserved_name_CON.txt").write_text("con content")
    _restore_reserved_renames(tmp_path)

    # File should still be there (no rename happened)
    assert (tmp_path / ".hermes_reserved_name_CON.txt").exists()


def test_stash_local_changes_renames_reserved_untracked_on_windows(
    tmp_path, monkeypatch, capsys
):
    """E2E: _stash_local_changes_if_needed renames reserved files before stash."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    from hermes_cli import main as hermes_main
    from hermes_cli.update_cmd import _stash_local_changes_if_needed

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

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

    # Create a reserved-name untracked file
    (tmp_path / "CON.txt").write_text("con content")

    stash_ref, _ = _stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref is not None

    # The reserved file was renamed before stash, then the renamed file
    # was stashed (as untracked) and removed from the working tree.
    assert not (tmp_path / "CON.txt").exists()
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()

    out = capsys.readouterr().out
    assert "Temporarily renamed Windows reserved-device-name file" in out


def test_restore_stashed_changes_restores_reserved_names_after_apply(
    tmp_path, monkeypatch, capsys
):
    """E2E: _restore_stashed_changes restores reserved file names after stash apply."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    from hermes_cli import main as hermes_main
    from hermes_cli.update_cmd import (
        _stash_local_changes_if_needed,
        _restore_stashed_changes,
    )

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

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

    # Create a reserved-name untracked file
    (tmp_path / "CON.txt").write_text("con content")

    stash_ref, reserved_renames = _stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref is not None

    # After restore, original name should be back
    restored = _restore_stashed_changes(
        ["git"], tmp_path, stash_ref, prompt_user=False,
        reserved_renames=reserved_renames,
    )
    assert restored is True

    assert (tmp_path / "CON.txt").exists()
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert (tmp_path / "CON.txt").read_text() == "con content"

    out = capsys.readouterr().out
    assert "Restored reserved file name: CON.txt" in out


def test_restore_reserved_renames_via_sidecar_file(tmp_path, monkeypatch):
    """Renamed files are restored from the sidecar map when no explicit list is given."""
    import json
    from hermes_cli.update_cmd import (
        _restore_reserved_renames,
        _RESERVED_RENAME_MAP_FILE,
    )
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

    (tmp_path / ".hermes_reserved_name_CON.txt").write_text("con content")
    (tmp_path / ".hermes_reserved_name_PRN.md").write_text("prn content")

    map_data = {
        "CON.txt": ".hermes_reserved_name_CON.txt",
        "PRN.md": ".hermes_reserved_name_PRN.md",
    }
    (tmp_path / _RESERVED_RENAME_MAP_FILE).write_text(
        json.dumps(map_data), encoding="utf-8"
    )

    _restore_reserved_renames(tmp_path)

    assert (tmp_path / "CON.txt").exists()
    assert (tmp_path / "PRN.md").exists()
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert not (tmp_path / ".hermes_reserved_name_PRN.md").exists()
    assert not (tmp_path / _RESERVED_RENAME_MAP_FILE).exists()


def test_stash_push_failure_restores_reserved_names(
    tmp_path, monkeypatch
):
    """If stash push fails, temporarily-renamed files are restored immediately."""
    import shutil
    import subprocess
    from unittest.mock import MagicMock

    if shutil.which("git") is None:
        pytest.skip("git not available")

    from hermes_cli import main as hermes_main
    from hermes_cli.update_cmd import _stash_local_changes_if_needed

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

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

    # Create a reserved-name untracked file
    (tmp_path / "CON.txt").write_text("con content")

    orig_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        # Intercept only the stash push call to make it fail.
        if (
            isinstance(cmd, list)
            and len(cmd) >= 5
            and cmd[1:5] == ["stash", "push", "--include-untracked", "-m"]
        ):
            mock = MagicMock()
            mock.returncode = 1
            mock.stdout = ""
            mock.stderr = "stash push failed"
            mock.args = cmd
            return mock
        return orig_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _stash_local_changes_if_needed(["git"], tmp_path)

    # The original file should have been restored
    assert (tmp_path / "CON.txt").exists()
    assert (tmp_path / "CON.txt").read_text() == "con content"
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()


def test_manual_stash_apply_recovery_via_sidecar(
    tmp_path, monkeypatch
):
    """Simulate manual `git stash apply` then recover via sidecar map."""
    import shutil
    import subprocess
    import json

    if shutil.which("git") is None:
        pytest.skip("git not available")

    from hermes_cli import main as hermes_main
    from hermes_cli.update_cmd import (
        _stash_local_changes_if_needed,
        _restore_stashed_changes,
        _restore_reserved_renames,
        _RESERVED_RENAME_MAP_FILE,
    )

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

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

    # Create a reserved-name untracked file
    (tmp_path / "CON.txt").write_text("con content")

    stash_ref, _ = _stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref is not None
    assert not (tmp_path / "CON.txt").exists()
    # Sidecar was stashed along with the renamed file; not in working tree.
    assert not (tmp_path / _RESERVED_RENAME_MAP_FILE).exists()

    # Simulate user manually applying the stash (the renamed file and sidecar come back)
    git("stash", "apply", "stash@{0}")
    # After manual apply, the prefixed file and sidecar exist again.
    assert (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert (tmp_path / _RESERVED_RENAME_MAP_FILE).exists()

    # Recovery via sidecar (no explicit list → reads sidecar)
    _restore_reserved_renames(tmp_path)

    assert (tmp_path / "CON.txt").exists()
    assert not (tmp_path / ".hermes_reserved_name_CON.txt").exists()
    assert not (tmp_path / _RESERVED_RENAME_MAP_FILE).exists()
