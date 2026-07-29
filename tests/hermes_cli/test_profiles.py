"""Comprehensive tests for hermes_cli.profiles module.

Tests cover: validation, directory resolution, CRUD operations, active profile
management, export/import, renaming, alias collision checks, profile isolation,
and shell completion generation.
"""

import json
import io
import os
import shutil
import sys
import tarfile
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from hermes_cli import profiles
from hermes_cli.profiles import (
    normalize_profile_name,
    validate_profile_name,
    get_profile_dir,
    create_profile,
    delete_profile,
    list_profiles,
    set_active_profile,
    get_active_profile,
    get_active_profile_name,
    resolve_profile_env,
    check_alias_collision,
    create_wrapper_script,
    remove_wrapper_script,
    validate_alias_name,
    rename_profile,
    export_profile,
    import_profile,
    _get_profiles_root,
    _get_default_hermes_home,
    seed_profile_skills,
    has_bundled_skills_opt_out,
    NO_BUNDLED_SKILLS_MARKER,
    backfill_profile_envs,
    profiles_to_serve,
)
from hermes_cli.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared fixture: redirect Path.home() and HERMES_HOME for profile tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Set up an isolated environment for profile tests.

    * Path.home() -> tmp_path  (so _get_profiles_root() = tmp_path/.hermes/profiles)
    * HERMES_HOME  -> tmp_path/.hermes  (so get_hermes_home() agrees)
    * Creates the bare-minimum ~/.hermes directory.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


# ===================================================================
# TestValidateProfileName
# ===================================================================

class TestNormalizeProfileName:
    """Tests for normalize_profile_name()."""

    def test_title_case_normalized(self):
        assert normalize_profile_name("Jules") == "jules"
        assert normalize_profile_name("  Librarian ") == "librarian"


class TestValidateProfileName:
    """Tests for validate_profile_name()."""

    @pytest.mark.parametrize("name", ["coder", "work-bot", "a1", "my_agent"])
    def test_valid_names_accepted(self, name):
        # Should not raise
        validate_profile_name(name)


    @pytest.mark.parametrize("name", ["UPPER", "has space", ".hidden", "-leading"])
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ValueError):
            validate_profile_name(name)


# ===================================================================
# TestGetProfileDir
# ===================================================================

class TestGetProfileDir:
    """Tests for get_profile_dir()."""

    def test_default_returns_hermes_home(self, profile_env):
        tmp_path = profile_env
        result = get_profile_dir("default")
        assert result == tmp_path / ".hermes"


# ===================================================================
# TestCreateProfile
# ===================================================================

class TestCreateProfile:
    """Tests for create_profile()."""

    def test_creates_directory_with_subdirs(self, profile_env):
        profile_dir = create_profile("coder", no_alias=True)
        assert profile_dir.is_dir()
        for subdir in ["memories", "sessions", "skills", "skins", "logs",
                        "plans", "workspace", "cron"]:
            assert (profile_dir / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_seeds_placeholder_env_file(self, profile_env):
        """Fresh profiles get their own .env (owner-only) so channel/env
        writes are profile-scoped from day one instead of falling through
        to the shell environment / root install."""
        import stat
        profile_dir = create_profile("coder", no_alias=True)
        env_path = profile_dir / ".env"
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        # Placeholder only — no credentials leak in from anywhere.
        assert all(
            line.startswith("#") or not line.strip()
            for line in content.splitlines()
        )
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600

    def test_seeded_env_does_not_clobber_cloned_env(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        (default_home / ".env").write_text("KEY=val")
        profile_dir = create_profile("coder", clone_config=True, no_alias=True)
        assert (profile_dir / ".env").read_text() == "KEY=val"

    def test_duplicate_raises_file_exists(self, profile_env):
        create_profile("coder", no_alias=True)
        with pytest.raises(FileExistsError):
            create_profile("coder", no_alias=True)


    def test_clone_config_copies_files(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        # Create source config files in default profile
        (default_home / "config.yaml").write_text("model: test")
        (default_home / ".env").write_text("KEY=val")
        (default_home / "SOUL.md").write_text("Be helpful.")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        cloned_config = yaml.safe_load((profile_dir / "config.yaml").read_text())
        assert cloned_config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert cloned_config["model"] == "test"
        assert (profile_dir / ".env").read_text().strip() == "KEY=val"
        assert (profile_dir / "SOUL.md").read_text() == "Be helpful."


    def test_clone_all_excludes_history_artifacts(self, profile_env):
        """--clone-all excludes the source's session history, backups, and
        snapshots — a clone is a fresh workspace, and these can reach tens
        of GB.  Applies to ANY source profile, not just default.
        """
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        (default_home / "state.db").write_text("sessions-data")
        (default_home / "state.db-wal").write_text("wal")
        (default_home / "state.db-shm").write_text("shm")
        (default_home / "sessions" / "20260101_old").mkdir(parents=True)
        (default_home / "backups").mkdir(exist_ok=True)
        (default_home / "backups" / "backup.tar.gz").write_text("archive")
        (default_home / "state-snapshots" / "snap1").mkdir(parents=True)
        (default_home / "checkpoints" / "cp1").mkdir(parents=True)
        # Data that should still copy
        (default_home / "config.yaml").write_text("model: gpt-4")
        # Nested dirs with the same names must NOT be excluded (root-only)
        (default_home / "workspace" / "backups").mkdir(parents=True)
        (default_home / "workspace" / "backups" / "user-data.txt").write_text("mine")

        profile_dir = create_profile("fresh", clone_all=True, no_alias=True)

        for history in (
            "state.db", "state.db-wal", "state.db-shm",
            "sessions", "backups", "state-snapshots", "checkpoints",
        ):
            assert not (profile_dir / history).exists(), history
        assert (profile_dir / "config.yaml").read_text() == "model: gpt-4"
        # Root-only: nested same-name dirs survive
        assert (profile_dir / "workspace" / "backups" / "user-data.txt").read_text() == "mine"

    def test_clone_config_missing_files_skipped(self, profile_env):
        """Clone config gracefully skips files that don't exist in source."""
        profile_dir = create_profile("coder", clone_config=True, no_alias=True)
        # No error; optional files just not copied
        assert not (profile_dir / "config.yaml").exists()
        # .env is always seeded (placeholder) so the profile has its own
        # credentials file even when the clone source lacked one.
        assert (profile_dir / ".env").exists()
        # SOUL.md is always seeded with the default even when clone source lacks it
        assert (profile_dir / "SOUL.md").exists()


# ===================================================================
# TestNoSkillsOptOut
# ===================================================================

class TestNoSkillsOptOut:
    """Tests for `hermes profile create --no-skills` and the opt-out marker."""

    def test_no_skills_writes_marker_and_skips_seeding(self, profile_env):
        profile_dir = create_profile("orchestrator", no_alias=True, no_skills=True)

        # Marker file is present
        marker = profile_dir / NO_BUNDLED_SKILLS_MARKER
        assert marker.is_file(), "expected .no-bundled-skills marker in profile root"
        assert "--no-skills" in marker.read_text()

        # has_bundled_skills_opt_out() agrees
        assert has_bundled_skills_opt_out(profile_dir) is True

        # skills/ dir exists (profile bootstrapping still creates the dir) but
        # contains nothing yet because create_profile itself doesn't seed.
        assert (profile_dir / "skills").is_dir()
        assert list((profile_dir / "skills").iterdir()) == []


    def test_seed_profile_skills_respects_marker(self, profile_env):
        """seed_profile_skills() must no-op on opted-out profiles even when
        called directly (e.g. by `hermes update`'s all-profile sync loop)."""
        profile_dir = create_profile("orchestrator", no_alias=True, no_skills=True)

        # Call seed_profile_skills() directly — it should NOT invoke subprocess,
        # NOT modify the skills/ dir, and return a dict with skipped_opt_out=True.
        result = seed_profile_skills(profile_dir, quiet=True)

        assert result is not None
        assert result.get("skipped_opt_out") is True
        assert result.get("copied") == []
        # skills/ stays empty — no subprocess ran
        assert list((profile_dir / "skills").iterdir()) == []

    def test_default_profile_gets_skills_seeded(self, profile_env, monkeypatch):
        """Sanity: without --no-skills, seed_profile_skills() runs the real
        subprocess path. Mock the subprocess so the test is hermetic, and
        just confirm the marker is NOT checked in the non-opt-out case."""
        import subprocess as _sp

        profile_dir = create_profile("coder", no_alias=True)
        # No marker — not opted out
        assert not (profile_dir / NO_BUNDLED_SKILLS_MARKER).exists()
        assert has_bundled_skills_opt_out(profile_dir) is False

        # Mock subprocess.run to avoid actually running skill sync in tests
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            return _sp.CompletedProcess(
                args=args, returncode=0, stdout='{"copied": ["x"]}', stderr=""
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = seed_profile_skills(profile_dir, quiet=True)

        # Subprocess was invoked (the opt-out branch did NOT short-circuit)
        assert len(calls) == 1
        assert result == {"copied": ["x"]}

    def test_delete_marker_re_enables_seeding(self, profile_env, monkeypatch):
        """Deleting .no-bundled-skills opts the profile back in."""
        import subprocess as _sp

        profile_dir = create_profile("orchestrator", no_alias=True, no_skills=True)
        assert has_bundled_skills_opt_out(profile_dir) is True

        # First call: opted out, returns skipped dict without touching subprocess
        called = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (called.append(a), _sp.CompletedProcess(
                args=a, returncode=0, stdout='{"copied": []}', stderr=""
            ))[1],
        )
        r1 = seed_profile_skills(profile_dir, quiet=True)
        assert r1.get("skipped_opt_out") is True
        assert called == []

        # Delete marker → next call runs the real path
        (profile_dir / NO_BUNDLED_SKILLS_MARKER).unlink()
        assert has_bundled_skills_opt_out(profile_dir) is False
        r2 = seed_profile_skills(profile_dir, quiet=True)
        assert r2 == {"copied": []}
        assert len(called) == 1


# ===================================================================
# TestBackfillProfileEnvs
# ===================================================================

class TestBackfillProfileEnvs:
    """Tests for backfill_profile_envs() — the `hermes update` pass that
    gives pre-#44792 profiles (created before .env seeding) their own
    .env, copied from the default install so credentials don't break."""

    def test_copies_default_env_into_envless_profiles(self, profile_env):
        import stat
        tmp_path = profile_env
        (tmp_path / ".hermes" / ".env").write_text("OPENROUTER_API_KEY=root-key\n")
        p1 = create_profile("old1", no_alias=True)
        p2 = create_profile("old2", no_alias=True)
        # Simulate pre-#44792 profiles: no .env
        (p1 / ".env").unlink()
        (p2 / ".env").unlink()

        backfilled = backfill_profile_envs(quiet=True)

        assert sorted(backfilled) == ["old1", "old2"]
        for p in (p1, p2):
            assert (p / ".env").read_text() == "OPENROUTER_API_KEY=root-key\n"
            assert stat.S_IMODE((p / ".env").stat().st_mode) == 0o600


    def test_placeholder_when_default_has_no_env(self, profile_env):
        p = create_profile("noroot", no_alias=True)
        (p / ".env").unlink()

        backfilled = backfill_profile_envs(quiet=True)

        assert backfilled == ["noroot"]
        content = (p / ".env").read_text(encoding="utf-8")
        assert all(
            line.startswith("#") or not line.strip()
            for line in content.splitlines()
        )

    def test_no_profiles_root_is_noop(self, profile_env):
        assert backfill_profile_envs(quiet=True) == []


# ===================================================================
# TestDeleteProfile
# ===================================================================

class TestDeleteProfile:
    """Tests for delete_profile()."""


    def test_rmtree_failure_raises(self, profile_env):
        profile_dir = create_profile("coder", no_alias=True)
        set_active_profile("coder")

        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles.time.sleep"), \
             patch("hermes_cli.profiles.shutil.rmtree", side_effect=PermissionError("locked")):
            with pytest.raises(RuntimeError, match="Could not remove profile directory"):
                delete_profile("coder", yes=True)

        assert profile_dir.is_dir()
        assert get_active_profile() == "default"

    def test_stops_profile_bound_backends_before_removal(self, profile_env):
        """A Desktop-spawned backend (not in gateway.pid) is stopped first."""
        profile_dir = create_profile("coder", no_alias=True)

        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles._profile_bound_backend_pids", return_value=[4242]) as pids, \
             patch("gateway.status.terminate_pid") as terminate, \
             patch("gateway.status._pid_exists", return_value=False):
            delete_profile("coder", yes=True)

        pids.assert_called_once()
        terminate.assert_any_call(4242)
        assert not profile_dir.is_dir()

    def test_rmtree_retries_transient_enotempty_then_succeeds(self, profile_env):
        """A live writer racing rmtree (ENOTEMPTY) is absorbed by a retry."""
        profile_dir = create_profile("coder", no_alias=True)
        real_rmtree = shutil.rmtree
        calls = {"n": 0}

        def flaky_rmtree(path, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(66, "Directory not empty")
            return real_rmtree(path)

        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles._profile_bound_backend_pids", return_value=[]), \
             patch("hermes_cli.profiles.time.sleep"), \
             patch("hermes_cli.profiles.shutil.rmtree", side_effect=flaky_rmtree):
            delete_profile("coder", yes=True)

        assert calls["n"] == 2
        assert not profile_dir.is_dir()

    def test_backend_scan_only_matches_this_profile(self, profile_env, monkeypatch):
        """The backend PID scan binds by --profile selector and skips self."""
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Backend bound to coder → matched.
            FakeProc(101, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
            # Interactive chat for coder → NOT a backend subcommand, skipped.
            FakeProc(102, ["python", "-m", "hermes_cli.main", "--profile", "coder", "chat"]),
            # Backend for a different profile → skipped.
            FakeProc(103, ["python", "-m", "hermes_cli.main", "--profile", "other", "serve"]),
            # This very process → skipped even if it matched.
            FakeProc(self_pid, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
        ]

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        pids = profiles._profile_bound_backend_pids("coder", profile_dir)
        assert pids == [101]


# ===================================================================
# TestListProfiles
# ===================================================================

class TestListProfiles:
    """Tests for list_profiles()."""

    def test_returns_default_when_no_named_profiles(self, profile_env):
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "default" in names

    def test_includes_named_profiles(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "alpha" in names
        assert "beta" in names


# ===================================================================
# TestActiveProfile
# ===================================================================

class TestActiveProfile:
    """Tests for set_active_profile() / get_active_profile()."""


    def test_set_to_default_removes_file(self, profile_env):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        active_path = tmp_path / ".hermes" / "active_profile"
        assert active_path.exists()

        set_active_profile("default")
        assert not active_path.exists()


# ===================================================================
# TestGetActiveProfileName
# ===================================================================

class TestGetActiveProfileName:
    """Tests for get_active_profile_name()."""


    def test_profile_path_returns_profile_name(self, profile_env, monkeypatch):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        assert get_active_profile_name() == "coder"

    def test_custom_path_returns_default(self, profile_env, monkeypatch):
        """A custom HERMES_HOME (Docker, etc.) IS the default root."""
        tmp_path = profile_env
        custom = tmp_path / "some" / "other" / "path"
        custom.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        # With Docker-aware roots, a custom HERMES_HOME is the default —
        # not "custom".  The user is on the default profile of their
        # custom deployment.
        assert get_active_profile_name() == "default"


# ===================================================================
# TestResolveProfileEnv
# ===================================================================

class TestResolveProfileEnv:
    """Tests for resolve_profile_env()."""


# ===================================================================
# TestAliasCollision
# ===================================================================

class TestAliasCollision:
    """Tests for check_alias_collision()."""

    def test_normal_name_returns_none(self, profile_env):
        # Mock 'which' to return not-found
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = check_alias_collision("mybot")
        assert result is None


    def test_uses_where_on_windows(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            check_alias_collision("mybot")
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "where"


    def test_windows_checks_bat_extension(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        wrapper_dir = profile_env / ".local" / "bin"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        bat_path = wrapper_dir / "mybot.bat"
        bat_path.write_text("@echo off\r\nhermes -p mybot %*\r\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(bat_path),
            )
            result = check_alias_collision("mybot")
        assert result is None  # our own wrapper, safe to overwrite

    def test_traversal_alias_rejected_before_path_lookup(self, profile_env):
        """A path-traversal alias is rejected without ever shelling out to which/where."""
        with patch("subprocess.run") as mock_run:
            result = check_alias_collision("../../.bashrc")
        assert result is not None
        assert "invalid alias name" in result.lower()
        mock_run.assert_not_called()


# ===================================================================
# TestWrapperScript
# ===================================================================

class TestWrapperScript:
    """Tests for create_wrapper_script() and remove_wrapper_script()."""

    def test_creates_sh_on_posix(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("hermes_cli.profiles.shutil.which", lambda name: "/opt/hermes/bin/hermes")
        from hermes_cli.profiles import create_wrapper_script
        wrapper = create_wrapper_script("mybot")
        assert wrapper is not None
        assert wrapper.name == "mybot"
        content = wrapper.read_text()
        assert content.startswith("#!/bin/sh")
        assert "exec /opt/hermes/bin/hermes -p mybot" in content


    def test_remove_finds_bat_on_windows(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        from hermes_cli.profiles import create_wrapper_script, remove_wrapper_script
        wrapper = create_wrapper_script("mybot")
        assert wrapper is not None
        assert wrapper.exists()
        removed = remove_wrapper_script("mybot")
        assert removed is True
        assert not wrapper.exists()


    def test_remove_returns_false_when_absent(self, profile_env):
        from hermes_cli.profiles import remove_wrapper_script
        assert remove_wrapper_script("nonexistent") is False

    def test_custom_alias_target_on_posix(self, profile_env, monkeypatch):
        # Custom alias name pointing at a differently-named profile: the file
        # is named after the alias, the -p content references the profile.
        monkeypatch.setattr("sys.platform", "darwin")
        from hermes_cli.profiles import create_wrapper_script
        wrapper = create_wrapper_script("rq", target="redqueen")
        assert wrapper is not None
        assert wrapper.name == "rq"
        content = wrapper.read_text()
        assert content.startswith("#!/bin/sh")
        assert "hermes -p redqueen" in content

    def test_custom_alias_target_on_windows(self, profile_env, monkeypatch):
        # Regression: custom-name aliases must still produce an executable
        # .bat (not a clobbered #!/bin/sh) on Windows.
        monkeypatch.setattr("sys.platform", "win32")
        from hermes_cli.profiles import create_wrapper_script
        wrapper = create_wrapper_script("rq", target="redqueen")
        assert wrapper is not None
        assert wrapper.name == "rq.bat"
        content = wrapper.read_text()
        assert "@echo off" in content
        assert "hermes -p redqueen" in content
        assert "%*" in content
        assert "#!/bin/sh" not in content


# ===================================================================
# TestWrapperScriptSecurity — path-traversal hardening
# ===================================================================

class TestWrapperScriptSecurity:
    """A crafted alias name must not escape the wrapper directory."""

    def test_validate_alias_name_rejects_traversal(self):
        with pytest.raises(ValueError, match="Invalid alias name"):
            validate_alias_name("../../.bashrc")


    def test_validate_alias_name_accepts_safe_identifier(self):
        validate_alias_name("mybot")  # does not raise

    def test_create_wrapper_rejects_traversal(self, profile_env):
        sentinel = profile_env / ".bashrc"
        sentinel.write_text("keep", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid alias name"):
            create_wrapper_script("../../.bashrc", target="coder")
        # The traversal target was not touched.
        assert sentinel.read_text(encoding="utf-8") == "keep"

    def test_create_wrapper_rejects_absolute_path(self, profile_env, tmp_path):
        target = tmp_path / "abs-wrapper"
        with pytest.raises(ValueError, match="Invalid alias name"):
            create_wrapper_script(str(target))
        assert not target.exists()

    def test_remove_wrapper_rejects_traversal(self, profile_env):
        sentinel = profile_env / ".bashrc"
        sentinel.write_text("keep", encoding="utf-8")
        assert remove_wrapper_script("../../.bashrc") is False
        assert sentinel.read_text(encoding="utf-8") == "keep"


# ===================================================================
# TestFindAliasForProfile — display-side reverse lookup
# ===================================================================

class TestFindAliasForProfile:
    """Tests for find_alias_for_profile() and alias display in list/show."""

    def test_profile_named_alias(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        from hermes_cli.profiles import create_wrapper_script, find_alias_for_profile
        create_wrapper_script("steve")
        assert find_alias_for_profile("steve") == "steve"


    def test_ignores_unrelated_files(self, profile_env, monkeypatch):
        # ~/.local/bin commonly holds unrelated binaries; they must not match.
        monkeypatch.setattr("sys.platform", "darwin")
        from hermes_cli.profiles import _get_wrapper_dir, find_alias_for_profile
        wrapper_dir = _get_wrapper_dir()
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        (wrapper_dir / "pip").write_text("#!/bin/sh\nexec python -m pip \"$@\"\n")
        assert find_alias_for_profile("steve") is None


    def test_list_profiles_surfaces_custom_alias(self, profile_env, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        from hermes_cli.profiles import (
            create_profile,
            create_wrapper_script,
            list_profiles,
        )
        create_profile("steve", no_alias=True)
        create_wrapper_script("qiaobusi", target="steve")
        info = next(p for p in list_profiles() if p.name == "steve")
        assert info.alias_name == "qiaobusi"
        assert info.alias_path is not None
        assert info.alias_path.name == "qiaobusi"


# ===================================================================
# TestRenameProfile
# ===================================================================

class TestRenameProfile:
    """Tests for rename_profile()."""

    def test_renames_directory(self, profile_env):
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"
        assert old_dir.is_dir()

        # Mock alias collision to avoid subprocess calls
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            new_dir = rename_profile("oldname", "newname")

        assert not old_dir.is_dir()
        assert new_dir.is_dir()
        assert new_dir == tmp_path / ".hermes" / "profiles" / "newname"

    def test_renames_root_honcho_host_without_changing_ai_peer(self, profile_env):
        tmp_path = profile_env
        create_profile("ssi_health", no_alias=True)
        honcho_path = tmp_path / ".hermes" / "honcho.json"
        honcho_path.write_text(json.dumps({
            "hosts": {
                "hermes.ssi_health": {
                    "recallMode": "hybrid",
                    "writeFrequency": "async",
                    "sessionStrategy": "per-session",
                    "saveMessages": True,
                    "peerName": "user-peer",
                    "aiPeer": "ssi_health",
                    "workspace": "hermes",
                    "enabled": True,
                }
            }
        }))

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            rename_profile("ssi_health", "heimdall")

        cfg = json.loads(honcho_path.read_text())
        assert "hermes.ssi_health" not in cfg["hosts"]
        assert cfg["hosts"]["hermes_heimdall"]["aiPeer"] == "ssi_health"
        assert cfg["hosts"]["hermes_heimdall"]["peerName"] == "user-peer"


# ===================================================================
# TestExportImport
# ===================================================================

class TestExportImport:
    """Tests for export_profile() / import_profile()."""

    def test_export_creates_tar_gz(self, profile_env, tmp_path):
        create_profile("coder", no_alias=True)
        # Put a marker file so we can verify content
        profile_dir = get_profile_dir("coder")
        (profile_dir / "marker.txt").write_text("hello")

        output = tmp_path / "export" / "coder.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = export_profile("coder", str(output))

        assert Path(result).exists()
        assert tarfile.is_tarfile(str(result))

    def test_import_restores_from_archive(self, profile_env, tmp_path):
        # Create and export a profile
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")
        (profile_dir / "marker.txt").write_text("hello")

        archive_path = tmp_path / "export" / "coder.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        export_profile("coder", str(archive_path))

        # Delete the profile, then import it back under a new name
        import shutil
        shutil.rmtree(profile_dir)
        assert not profile_dir.is_dir()

        imported = import_profile(str(archive_path), name="coder")
        assert imported.is_dir()
        assert (imported / "marker.txt").read_text() == "hello"


    def test_import_rejects_traversal_archive_member(self, profile_env, tmp_path):
        archive_path = tmp_path / "export" / "evil.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        escape_path = tmp_path / "escape.txt"

        with tarfile.open(archive_path, "w:gz") as tf:
            info = tarfile.TarInfo("../../escape.txt")
            data = b"pwned"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Unsafe archive member path"):
            import_profile(str(archive_path), name="coder")

        assert not escape_path.exists()
        assert not get_profile_dir("coder").exists()

    def test_import_rejects_absolute_archive_member(self, profile_env, tmp_path):
        archive_path = tmp_path / "export" / "evil-abs.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_target = tmp_path / "abs-escape.txt"

        with tarfile.open(archive_path, "w:gz") as tf:
            info = tarfile.TarInfo(str(absolute_target))
            data = b"pwned"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Unsafe archive member path"):
            import_profile(str(archive_path), name="coder")

        assert not absolute_target.exists()
        assert not get_profile_dir("coder").exists()


    # ---------------------------------------------------------------
    # Default profile export / import
    # ---------------------------------------------------------------


    def test_export_default_includes_profile_data(self, profile_env, tmp_path):
        """Profile data files end up in the archive (credentials excluded)."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("model: test")
        (default_dir / ".env").write_text("KEY=val")
        (default_dir / "SOUL.md").write_text("Be nice.")
        mem_dir = default_dir / "memories"
        mem_dir.mkdir(exist_ok=True)
        (mem_dir / "MEMORY.md").write_text("remember this")

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(output))

        with tarfile.open(str(output), "r:gz") as tf:
            names = tf.getnames()

        assert "default/config.yaml" in names
        assert "default/.env" not in names  # credentials excluded
        assert "default/SOUL.md" in names
        assert "default/memories/MEMORY.md" in names


    def test_export_default_handles_broken_symlinks(self, profile_env, tmp_path):
        """Broken symlinks inside allowed artifacts are preserved, not crashed (#58394).

        ``shutil.copytree``'s default is ``symlinks=False``, which follows
        symlinks and crashes on broken ones. Use ``symlinks=True`` so stale
        symlinks inside *allowed* artifacts (e.g. ``skills/``) survive as
        symlinks; the link and its target are both retained.
        """
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("ok")
        # Place broken symlink *inside* the allowed ``skills/`` tree so the
        # root-level allow-list passes the directory through; the
        # symlinks=True flag must then preserve the link instead of
        # following and crashing.
        broken_dir = default_dir / "skills" / "with-broken-links"
        broken_dir.mkdir(parents=True)
        (broken_dir / "broken_link").symlink_to("/nonexistent/path")
        # Valid symlink for comparison
        (broken_dir / "valid_target.txt").write_text("real data")
        (broken_dir / "valid_link").symlink_to(
            broken_dir / "valid_target.txt"
        )

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = export_profile("default", str(output))

        assert result.exists()
        with tarfile.open(str(result), "r:gz") as tf:
            names = set(tf.getnames())
        # Allowed artifact survived
        assert any(n.endswith("config.yaml") for n in names)
        # Broken symlink inside an allowed dir was preserved as a symlink
        # (without crashing) — tar entry name recorded as the link path.
        assert any(
            "with-broken-links/broken_link" in n for n in names
        ), (
            f"broken_link should survive; tarfile names: {sorted(names)[:30]}"
        )
        # Valid symlink + target also kept
        assert any("valid_link" in n for n in names)
        assert any("valid_target.txt" in n for n in names)


    def test_import_default_export_with_new_name_roundtrip(self, profile_env, tmp_path):
        """Export default → import under a different name → data preserved."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("model: opus")
        mem_dir = default_dir / "memories"
        mem_dir.mkdir(exist_ok=True)
        (mem_dir / "MEMORY.md").write_text("important fact")

        archive = tmp_path / "export" / "default.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(archive))

        imported = import_profile(str(archive), name="backup")
        assert imported.is_dir()
        assert (imported / "config.yaml").read_text() == "model: opus"
        assert (imported / "memories" / "MEMORY.md").read_text() == "important fact"


# ===================================================================
# TestProfileIsolation
# ===================================================================

class TestProfileIsolation:
    """Verify that two profiles have completely separate paths."""

    def test_separate_config_paths(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        alpha_dir = get_profile_dir("alpha")
        beta_dir = get_profile_dir("beta")
        assert alpha_dir / "config.yaml" != beta_dir / "config.yaml"
        assert str(alpha_dir) not in str(beta_dir)


# ===================================================================
# TestGetProfilesRoot / TestGetDefaultHermesHome (internal helpers)
# ===================================================================

class TestInternalHelpers:
    """Tests for _get_profiles_root() and _get_default_hermes_home()."""


    def test_profiles_root_docker_deployment(self, tmp_path, monkeypatch):
        """In Docker (HERMES_HOME outside ~/.hermes), profiles go under HERMES_HOME."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        root = _get_profiles_root()
        assert root == docker_home / "profiles"

    def test_default_hermes_home_docker(self, tmp_path, monkeypatch):
        """In Docker, _get_default_hermes_home() returns HERMES_HOME itself."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        home = _get_default_hermes_home()
        assert home == docker_home

    def test_profiles_root_profile_mode(self, tmp_path, monkeypatch):
        """In profile mode (HERMES_HOME under ~/.hermes), profiles root is still ~/.hermes/profiles."""
        native = tmp_path / ".hermes"
        profile_dir = native / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        root = _get_profiles_root()
        assert root == native / "profiles"

    def test_active_profile_path_docker(self, tmp_path, monkeypatch):
        """In Docker, active_profile file lives under HERMES_HOME."""
        from hermes_cli.profiles import _get_active_profile_path
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        path = _get_active_profile_path()
        assert path == docker_home / "active_profile"

    def test_create_profile_docker(self, tmp_path, monkeypatch):
        """Profile created in Docker lands under HERMES_HOME/profiles/."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        result = create_profile("orchestrator", no_alias=True)
        expected = docker_home / "profiles" / "orchestrator"
        assert result == expected
        assert expected.is_dir()


    def test_active_profile_name_docker_profile(self, tmp_path, monkeypatch):
        """In Docker with a profile active, get_active_profile_name() returns the profile name."""
        docker_home = tmp_path / "opt" / "data"
        profile = docker_home / "profiles" / "orchestrator"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_active_profile_name() == "orchestrator"


# ===================================================================
# Edge cases and additional coverage
# ===================================================================

class TestEdgeCases:
    """Additional edge-case tests."""


    def test_list_profiles_default_info_fields(self, profile_env):
        profiles = list_profiles()
        default = [p for p in profiles if p.name == "default"][0]
        assert default.is_default is True
        assert default.gateway_running is False
        assert default.skill_count == 0

    def test_gateway_running_check_with_pid_file(self, profile_env):
        """Verify _check_gateway_running uses the shared gateway PID validator."""
        from hermes_cli.profiles import _check_gateway_running
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"

        with patch("gateway.status.get_running_pid", return_value=99999) as mock_get_running_pid:
            assert _check_gateway_running(default_home) is True
        mock_get_running_pid.assert_called_once_with(
            default_home / "gateway.pid",
            cleanup_stale=False,
        )


    def test_gateway_running_check_falls_back_to_runtime_state(self, profile_env):
        """A live gateway whose PID-file/lock check fails closed (separate-process
        reader, e.g. the dashboard s6 service in Docker) is still detected via the
        profile's gateway_state.json validated against the live process table.

        Regression: the Profiles view used to show "Gateway stopped" while the
        sidebar (which already has this fallback) showed "Gateway running" for the
        same live gateway. See get_running_pid() short-circuiting on an
        unheld runtime lock before it inspects the PID record.
        """
        import os
        import gateway.status as gw_status
        from hermes_cli.profiles import _check_gateway_running

        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        default_home.mkdir(parents=True, exist_ok=True)

        # Write a realistic gateway_state.json pointing at THIS live process with
        # a gateway-shaped argv, so get_runtime_status_running_pid validates it.
        live_pid = os.getpid()
        (default_home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "pid": live_pid,
                    "kind": "hermes-gateway",
                    "argv": ["hermes", "gateway", "run"],
                    "start_time": gw_status._get_process_start_time(live_pid),
                    "gateway_state": "running",
                    "active_agents": 0,
                }
            ),
            encoding="utf-8",
        )

        # Primary pid-file/lock check returns None (no lock held by this reader),
        # exactly as it does for a separate-process dashboard. The fallback must
        # then read the state file and confirm the gateway is alive by checking
        # the recorded PID's live command line. In the real separate-process
        # scenario that PID belongs to the live gateway, so mock its command
        # line to a bare ``gateway run`` (this is the default/root home, which
        # runs the gateway with no profile flag).
        with patch("gateway.status.get_running_pid", return_value=None), patch(
            "gateway.status._read_process_cmdline",
            return_value="hermes gateway run --replace",
        ):
            assert _check_gateway_running(default_home) is True


    def test_gateway_running_check_rejects_pid_reused_by_other_profile(self, profile_env):
        """Regression (user report): the dashboard showed a NAMED profile's
        gateway green while ``hermes -p <name> gateway status`` showed it
        stopped.

        Per-profile Docker supervision: a named profile (``coder``) left a
        ``gateway_state=running`` record whose PID the OS later recycled onto a
        DIFFERENT live process (here the default profile's gateway).  The
        ``_check_gateway_running`` fallback must scope the live PID to *this*
        profile's command line, so a recycled PID hosting another profile's
        gateway is not reported running for ``coder``.
        """
        from hermes_cli.profiles import _check_gateway_running

        tmp_path = profile_env
        coder_home = tmp_path / ".hermes" / "profiles" / "coder"
        coder_home.mkdir(parents=True, exist_ok=True)
        (coder_home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "pid": 139,
                    "kind": "hermes-gateway",
                    "argv": ["hermes", "gateway", "run"],
                    "gateway_state": "running",
                    "active_agents": 0,
                }
            ),
            encoding="utf-8",
        )

        # PID 139 is alive but is the DEFAULT gateway (bare, no -p coder), not
        # coder's. start_time is absent so the PID-reuse guard cannot catch it;
        # the profile scope must.
        with patch("gateway.status.get_running_pid", return_value=None), patch(
            "gateway.status._pid_exists", return_value=True
        ), patch("gateway.status._get_process_start_time", return_value=None), patch(
            "gateway.status._read_process_cmdline",
            return_value="hermes gateway run --replace",
        ):
            assert _check_gateway_running(coder_home) is False


    def test_profile_name_boundary_single_char(self):
        """Single alphanumeric character is valid."""
        validate_profile_name("a")
        validate_profile_name("1")


    def test_profile_name_underscore_start(self):
        """Name starting with underscore is invalid (must start with [a-z0-9])."""
        with pytest.raises(ValueError):
            validate_profile_name("_abc")

    def test_clone_from_named_profile(self, profile_env):
        """Clone config from a named (non-default) profile."""
        tmp_path = profile_env
        # Create source profile with config
        source_dir = create_profile("source", no_alias=True)
        (source_dir / "config.yaml").write_text("model: cloned")
        (source_dir / ".env").write_text("SECRET=yes")

        target_dir = create_profile(
            "target", clone_from="source", clone_config=True, no_alias=True,
        )
        cloned_config = yaml.safe_load((target_dir / "config.yaml").read_text())
        assert cloned_config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert cloned_config["model"] == "cloned"
        assert (target_dir / ".env").read_text().strip() == "SECRET=yes"

    def test_delete_clears_active_profile(self, profile_env):
        """Deleting the active profile resets active to default."""
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        assert get_active_profile() == "coder"

        with patch("hermes_cli.profiles._cleanup_gateway_service"):
            delete_profile("coder", yes=True)

        assert get_active_profile() == "default"


class TestProfilesToServe:
    """profiles_to_serve(multiplex) — the gateway's profile-enumeration chokepoint."""

    def test_off_returns_only_active_default(self, profile_env):
        serve = profiles_to_serve(multiplex=False)
        assert len(serve) == 1
        name, home = serve[0]
        assert name == "default"
        assert home == _get_default_hermes_home()

    def test_off_returns_only_active_named(self, profile_env, monkeypatch):
        # A named profile's gateway runs with HERMES_HOME pointing at the
        # profile dir; get_active_profile_name() infers the name from there.
        create_profile("coder", no_alias=True)
        monkeypatch.setenv("HERMES_HOME", str(get_profile_dir("coder")))
        serve = profiles_to_serve(multiplex=False)
        assert len(serve) == 1
        assert serve[0][0] == "coder"
        assert serve[0][1] == get_profile_dir("coder")

    def test_on_returns_default_plus_all_named(self, profile_env):
        create_profile("coder", no_alias=True)
        create_profile("writer", no_alias=True)
        serve = dict(profiles_to_serve(multiplex=True))
        assert set(serve) == {"default", "coder", "writer"}
        assert serve["default"] == _get_default_hermes_home()
        assert serve["coder"] == get_profile_dir("coder")

    def test_on_default_always_first(self, profile_env):
        create_profile("coder", no_alias=True)
        serve = profiles_to_serve(multiplex=True)
        assert serve[0][0] == "default"

    def test_on_active_profile_does_not_change_set(self, profile_env):
        """Enumeration is independent of which profile is active."""
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        serve = dict(profiles_to_serve(multiplex=True))
        assert set(serve) == {"default", "coder"}

