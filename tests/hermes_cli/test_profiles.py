"""Comprehensive tests for hermes_cli.profiles module.

Tests cover: validation, directory resolution, CRUD operations, active profile
management, export/import, renaming, alias collision checks, profile isolation,
and shell completion generation.
"""

import json
import io
import os
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.profiles import (
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
    rename_profile,
    export_profile,
    import_profile,
    generate_bash_completion,
    generate_zsh_completion,
    _get_profiles_root,
    _get_default_hermes_home,
)


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

    def test_too_long_rejected(self):
        long_name = "a" * 65
        with pytest.raises(ValueError):
            validate_profile_name(long_name)

    def test_max_length_accepted(self):
        # 64 chars total: 1 leading + 63 remaining = 64, within [0,63] range
        name = "a" * 64
        validate_profile_name(name)

    def test_default_accepted(self):
        # 'default' is a special-case pass-through
        validate_profile_name("default")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            validate_profile_name("")


# ===================================================================
# TestGetProfileDir
# ===================================================================

class TestGetProfileDir:
    """Tests for get_profile_dir()."""

    def test_default_returns_hermes_home(self, profile_env):
        tmp_path = profile_env
        result = get_profile_dir("default")
        assert result == tmp_path / ".hermes"

    def test_named_profile_returns_profiles_subdir(self, profile_env):
        tmp_path = profile_env
        result = get_profile_dir("coder")
        assert result == tmp_path / ".hermes" / "profiles" / "coder"


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

    def test_duplicate_raises_file_exists(self, profile_env):
        create_profile("coder", no_alias=True)
        with pytest.raises(FileExistsError):
            create_profile("coder", no_alias=True)

    def test_default_raises_value_error(self, profile_env):
        with pytest.raises(ValueError, match="default"):
            create_profile("default", no_alias=True)

    def test_invalid_name_raises_value_error(self, profile_env):
        with pytest.raises(ValueError):
            create_profile("INVALID!", no_alias=True)

    def test_clone_config_copies_files(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        # Create source config files in default profile
        (default_home / "config.yaml").write_text("model: test")
        (default_home / ".env").write_text("KEY=val")
        (default_home / "SOUL.md").write_text("Be helpful.")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        assert (profile_dir / "config.yaml").read_text() == "model: test"
        assert (profile_dir / ".env").read_text() == "KEY=val"
        assert (profile_dir / "SOUL.md").read_text() == "Be helpful."

    def test_clone_all_copies_entire_tree(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        # Populate default with some content
        (default_home / "memories").mkdir(exist_ok=True)
        (default_home / "memories" / "note.md").write_text("remember this")
        (default_home / "config.yaml").write_text("model: gpt-4")
        # Runtime files that should be stripped
        (default_home / "gateway.pid").write_text("12345")
        (default_home / "gateway_state.json").write_text("{}")
        (default_home / "processes.json").write_text("[]")

        profile_dir = create_profile("coder", clone_all=True, no_alias=True)

        # Content should be copied
        assert (profile_dir / "memories" / "note.md").read_text() == "remember this"
        assert (profile_dir / "config.yaml").read_text() == "model: gpt-4"
        # Runtime files should be stripped
        assert not (profile_dir / "gateway.pid").exists()
        assert not (profile_dir / "gateway_state.json").exists()
        assert not (profile_dir / "processes.json").exists()

    def test_clone_config_missing_files_skipped(self, profile_env):
        """Clone config gracefully skips files that don't exist in source."""
        profile_dir = create_profile("coder", clone_config=True, no_alias=True)
        # No error; optional files just not copied
        assert not (profile_dir / "config.yaml").exists()
        assert not (profile_dir / ".env").exists()
        assert not (profile_dir / "SOUL.md").exists()


# ===================================================================
# TestDeleteProfile
# ===================================================================

class TestDeleteProfile:
    """Tests for delete_profile()."""

    def test_removes_directory(self, profile_env):
        profile_dir = create_profile("coder", no_alias=True)
        assert profile_dir.is_dir()
        # Mock gateway import to avoid real systemd/launchd interaction
        with patch("hermes_cli.profiles._cleanup_gateway_service"):
            delete_profile("coder", yes=True)
        assert not profile_dir.is_dir()

    def test_default_raises_value_error(self, profile_env):
        with pytest.raises(ValueError, match="default"):
            delete_profile("default", yes=True)

    def test_nonexistent_raises_file_not_found(self, profile_env):
        with pytest.raises(FileNotFoundError):
            delete_profile("nonexistent", yes=True)


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

    def test_sorted_alphabetically(self, profile_env):
        create_profile("zebra", no_alias=True)
        create_profile("alpha", no_alias=True)
        create_profile("middle", no_alias=True)
        profiles = list_profiles()
        named = [p.name for p in profiles if not p.is_default]
        assert named == sorted(named)

    def test_default_is_first(self, profile_env):
        create_profile("alpha", no_alias=True)
        profiles = list_profiles()
        assert profiles[0].name == "default"
        assert profiles[0].is_default is True


# ===================================================================
# TestActiveProfile
# ===================================================================

class TestActiveProfile:
    """Tests for set_active_profile() / get_active_profile()."""

    def test_set_and_get_roundtrip(self, profile_env):
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        assert get_active_profile() == "coder"

    def test_no_file_returns_default(self, profile_env):
        assert get_active_profile() == "default"

    def test_empty_file_returns_default(self, profile_env):
        tmp_path = profile_env
        active_path = tmp_path / ".hermes" / "active_profile"
        active_path.write_text("")
        assert get_active_profile() == "default"

    def test_set_to_default_removes_file(self, profile_env):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        active_path = tmp_path / ".hermes" / "active_profile"
        assert active_path.exists()

        set_active_profile("default")
        assert not active_path.exists()

    def test_set_nonexistent_raises(self, profile_env):
        with pytest.raises(FileNotFoundError):
            set_active_profile("nonexistent")


# ===================================================================
# TestGetActiveProfileName
# ===================================================================

class TestGetActiveProfileName:
    """Tests for get_active_profile_name()."""

    def test_default_hermes_home_returns_default(self, profile_env):
        # HERMES_HOME points to tmp_path/.hermes which is the default
        assert get_active_profile_name() == "default"

    def test_profile_path_returns_profile_name(self, profile_env, monkeypatch):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        assert get_active_profile_name() == "coder"

    def test_custom_path_returns_custom(self, profile_env, monkeypatch):
        tmp_path = profile_env
        custom = tmp_path / "some" / "other" / "path"
        custom.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert get_active_profile_name() == "custom"


# ===================================================================
# TestResolveProfileEnv
# ===================================================================

class TestResolveProfileEnv:
    """Tests for resolve_profile_env()."""

    def test_existing_profile_returns_path(self, profile_env):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        result = resolve_profile_env("coder")
        assert result == str(tmp_path / ".hermes" / "profiles" / "coder")

    def test_default_returns_default_home(self, profile_env):
        tmp_path = profile_env
        result = resolve_profile_env("default")
        assert result == str(tmp_path / ".hermes")

    def test_nonexistent_raises_file_not_found(self, profile_env):
        with pytest.raises(FileNotFoundError):
            resolve_profile_env("nonexistent")

    def test_invalid_name_raises_value_error(self, profile_env):
        with pytest.raises(ValueError):
            resolve_profile_env("INVALID!")


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

    def test_reserved_name_returns_message(self, profile_env):
        result = check_alias_collision("hermes")
        assert result is not None
        assert "reserved" in result.lower()

    def test_subcommand_returns_message(self, profile_env):
        result = check_alias_collision("chat")
        assert result is not None
        assert "subcommand" in result.lower()

    def test_default_is_reserved(self, profile_env):
        result = check_alias_collision("default")
        assert result is not None
        assert "reserved" in result.lower()


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

    def test_default_raises_value_error(self, profile_env):
        with pytest.raises(ValueError, match="default"):
            rename_profile("default", "newname")

    def test_rename_to_default_raises_value_error(self, profile_env):
        create_profile("coder", no_alias=True)
        with pytest.raises(ValueError, match="default"):
            rename_profile("coder", "default")

    def test_nonexistent_raises_file_not_found(self, profile_env):
        with pytest.raises(FileNotFoundError):
            rename_profile("nonexistent", "newname")

    def test_target_exists_raises_file_exists(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        with pytest.raises(FileExistsError):
            rename_profile("alpha", "beta")


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

    def test_import_to_existing_name_raises(self, profile_env, tmp_path):
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        archive_path = tmp_path / "export" / "coder.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        export_profile("coder", str(archive_path))

        # Importing to same existing name should fail
        with pytest.raises(FileExistsError):
            import_profile(str(archive_path), name="coder")

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

    def test_export_nonexistent_raises(self, profile_env, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_profile("nonexistent", str(tmp_path / "out.tar.gz"))

    # ---------------------------------------------------------------
    # Default profile export / import
    # ---------------------------------------------------------------

    def test_export_default_creates_valid_archive(self, profile_env, tmp_path):
        """Exporting the default profile produces a valid tar.gz."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("model: test")

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = export_profile("default", str(output))

        assert Path(result).exists()
        assert tarfile.is_tarfile(str(result))

    def test_export_default_includes_profile_data(self, profile_env, tmp_path):
        """Profile data files end up in the archive."""
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
        assert "default/.env" in names
        assert "default/SOUL.md" in names
        assert "default/memories/MEMORY.md" in names

    def test_export_default_excludes_infrastructure(self, profile_env, tmp_path):
        """Repo checkout, worktrees, profiles, databases are excluded."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("ok")

        # Create dirs/files that should be excluded
        for d in ("hermes-agent", ".worktrees", "profiles", "bin",
                  "image_cache", "logs", "sandboxes", "checkpoints"):
            sub = default_dir / d
            sub.mkdir(exist_ok=True)
            (sub / "marker.txt").write_text("excluded")

        for f in ("state.db", "gateway.pid", "gateway_state.json",
                  "processes.json", "errors.log", ".hermes_history",
                  "active_profile", ".update_check", "auth.lock"):
            (default_dir / f).write_text("excluded")

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(output))

        with tarfile.open(str(output), "r:gz") as tf:
            names = tf.getnames()

        # Config is present
        assert "default/config.yaml" in names

        # Infrastructure excluded
        excluded_prefixes = [
            "default/hermes-agent", "default/.worktrees", "default/profiles",
            "default/bin", "default/image_cache", "default/logs",
            "default/sandboxes", "default/checkpoints",
        ]
        for prefix in excluded_prefixes:
            assert not any(n.startswith(prefix) for n in names), \
                f"Expected {prefix} to be excluded but found it in archive"

        excluded_files = [
            "default/state.db", "default/gateway.pid",
            "default/gateway_state.json", "default/processes.json",
            "default/errors.log", "default/.hermes_history",
            "default/active_profile", "default/.update_check",
            "default/auth.lock",
        ]
        for f in excluded_files:
            assert f not in names, f"Expected {f} to be excluded"

    def test_export_default_excludes_pycache_at_any_depth(self, profile_env, tmp_path):
        """__pycache__ dirs are excluded even inside nested directories."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("ok")
        nested = default_dir / "skills" / "my-skill" / "__pycache__"
        nested.mkdir(parents=True)
        (nested / "cached.pyc").write_text("bytecode")

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(output))

        with tarfile.open(str(output), "r:gz") as tf:
            names = tf.getnames()

        assert not any("__pycache__" in n for n in names)

    def test_import_default_without_name_raises(self, profile_env, tmp_path):
        """Importing a default export without --name gives clear guidance."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("ok")

        archive = tmp_path / "export" / "default.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(archive))

        with pytest.raises(ValueError, match="Cannot import as 'default'"):
            import_profile(str(archive))

    def test_import_default_with_explicit_default_name_raises(self, profile_env, tmp_path):
        """Explicitly importing as 'default' is also rejected."""
        default_dir = get_profile_dir("default")
        (default_dir / "config.yaml").write_text("ok")

        archive = tmp_path / "export" / "default.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(archive))

        with pytest.raises(ValueError, match="Cannot import as 'default'"):
            import_profile(str(archive), name="default")

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

    def test_separate_state_db_paths(self, profile_env):
        alpha_dir = get_profile_dir("alpha")
        beta_dir = get_profile_dir("beta")
        assert alpha_dir / "state.db" != beta_dir / "state.db"

    def test_separate_skills_paths(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        alpha_dir = get_profile_dir("alpha")
        beta_dir = get_profile_dir("beta")
        assert alpha_dir / "skills" != beta_dir / "skills"
        # Verify both exist and are independent dirs
        assert (alpha_dir / "skills").is_dir()
        assert (beta_dir / "skills").is_dir()


# ===================================================================
# TestCompletion
# ===================================================================

class TestCompletion:
    """Tests for bash/zsh completion generators."""

    def test_bash_completion_contains_complete(self):
        script = generate_bash_completion()
        assert len(script) > 0
        assert "complete" in script

    def test_zsh_completion_contains_compdef(self):
        script = generate_zsh_completion()
        assert len(script) > 0
        assert "compdef" in script

    def test_bash_completion_has_hermes_profiles_function(self):
        script = generate_bash_completion()
        assert "_hermes_profiles" in script

    def test_zsh_completion_has_hermes_function(self):
        script = generate_zsh_completion()
        assert "_hermes" in script


# ===================================================================
# TestGetProfilesRoot / TestGetDefaultHermesHome (internal helpers)
# ===================================================================

class TestInternalHelpers:
    """Tests for _get_profiles_root() and _get_default_hermes_home()."""

    def test_profiles_root_under_home(self, profile_env):
        tmp_path = profile_env
        root = _get_profiles_root()
        assert root == tmp_path / ".hermes" / "profiles"

    def test_default_hermes_home(self, profile_env):
        tmp_path = profile_env
        home = _get_default_hermes_home()
        assert home == tmp_path / ".hermes"


# ===================================================================
# Edge cases and additional coverage
# ===================================================================

class TestEdgeCases:
    """Additional edge-case tests."""

    def test_create_profile_returns_correct_path(self, profile_env):
        tmp_path = profile_env
        result = create_profile("mybot", no_alias=True)
        expected = tmp_path / ".hermes" / "profiles" / "mybot"
        assert result == expected

    def test_list_profiles_default_info_fields(self, profile_env):
        profiles = list_profiles()
        default = [p for p in profiles if p.name == "default"][0]
        assert default.is_default is True
        assert default.gateway_running is False
        assert default.skill_count == 0

    def test_gateway_running_check_with_pid_file(self, profile_env):
        """Verify _check_gateway_running reads pid file and probes os.kill."""
        from hermes_cli.profiles import _check_gateway_running
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"

        # No pid file -> not running
        assert _check_gateway_running(default_home) is False

        # Write a PID file with a JSON payload
        pid_file = default_home / "gateway.pid"
        pid_file.write_text(json.dumps({"pid": 99999}))

        # os.kill(99999, 0) should raise ProcessLookupError -> not running
        assert _check_gateway_running(default_home) is False

        # Mock os.kill to simulate a running process
        with patch("os.kill", return_value=None):
            assert _check_gateway_running(default_home) is True

    def test_gateway_running_check_plain_pid(self, profile_env):
        """Pid file containing just a number (legacy format)."""
        from hermes_cli.profiles import _check_gateway_running
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        pid_file = default_home / "gateway.pid"
        pid_file.write_text("99999")

        with patch("os.kill", return_value=None):
            assert _check_gateway_running(default_home) is True

    def test_profile_name_boundary_single_char(self):
        """Single alphanumeric character is valid."""
        validate_profile_name("a")
        validate_profile_name("1")

    def test_profile_name_boundary_all_hyphens(self):
        """Name starting with hyphen is invalid."""
        with pytest.raises(ValueError):
            validate_profile_name("-abc")

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
        assert (target_dir / "config.yaml").read_text() == "model: cloned"
        assert (target_dir / ".env").read_text() == "SECRET=yes"

    def test_delete_clears_active_profile(self, profile_env):
        """Deleting the active profile resets active to default."""
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        assert get_active_profile() == "coder"

        with patch("hermes_cli.profiles._cleanup_gateway_service"):
            delete_profile("coder", yes=True)

        assert get_active_profile() == "default"
