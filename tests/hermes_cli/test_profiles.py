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
    rename_profile,
    export_profile,
    import_profile,
    generate_bash_completion,
    generate_zsh_completion,
    _get_profiles_root,
    _get_default_hermes_home,
    seed_profile_skills,
    has_bundled_skills_opt_out,
    NO_BUNDLED_SKILLS_MARKER,
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

class TestNormalizeProfileName:
    """Tests for normalize_profile_name()."""

    def test_title_case_normalized(self):
        assert normalize_profile_name("Jules") == "jules"
        assert normalize_profile_name("  Librarian ") == "librarian"

    def test_default_case_insensitive(self):
        assert normalize_profile_name("Default") == "default"
        assert normalize_profile_name("DEFAULT") == "default"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_profile_name("")
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_profile_name("   ")


class TestValidateProfileName:
    """Tests for validate_profile_name()."""

    @pytest.mark.parametrize("name", ["coder", "work-bot", "a1", "my_agent"])
    def test_valid_names_accepted(self, name):
        # Should not raise
        validate_profile_name(name)

    def test_uppercase_rejected(self):
        # validate_profile_name is strict — callers normalize first, then validate.
        with pytest.raises(ValueError):
            validate_profile_name("Jules")

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

    @pytest.mark.parametrize("name", ["hermes", "test", "tmp", "root", "sudo"])
    def test_reserved_names_rejected(self, name):
        """Reserved names collide with the Hermes install itself or with
        common system binaries — reject them at validate time so
        create/install/rename all share one gate."""
        with pytest.raises(ValueError, match="reserved"):
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

    def test_named_profile_returns_profiles_subdir(self, profile_env):
        tmp_path = profile_env
        result = get_profile_dir("coder")
        assert result == tmp_path / ".hermes" / "profiles" / "coder"

    def test_named_profile_matching_is_case_insensitive(self, profile_env):
        tmp_path = profile_env
        assert get_profile_dir("Coder") == tmp_path / ".hermes" / "profiles" / "coder"


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

    def test_clone_config_copies_source_skills(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        skill_dir = default_home / "skills" / "custom" / "installed-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: installed-skill\n---\n")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        assert (
            profile_dir
            / "skills"
            / "custom"
            / "installed-skill"
            / "SKILL.md"
        ).read_text() == "---\nname: installed-skill\n---\n"

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

    def test_clone_all_excludes_sibling_profiles_tree(self, profile_env):
        """--clone-all from default ~/.hermes must not copy profiles/* (nested explosion)."""
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        profiles_root = default_home / "profiles"
        profiles_root.mkdir(exist_ok=True)
        (profiles_root / "other").mkdir(parents=True, exist_ok=True)
        (profiles_root / "other" / "marker.txt").write_text("sibling data")

        (default_home / "memories").mkdir(exist_ok=True)
        (default_home / "memories" / "note.md").write_text("remember this")

        profile_dir = create_profile("coder", clone_all=True, no_alias=True)

        assert (profile_dir / "memories" / "note.md").read_text() == "remember this"
        assert not (profile_dir / "profiles").exists()

    def test_clone_config_missing_files_skipped(self, profile_env):
        """Clone config gracefully skips files that don't exist in source."""
        profile_dir = create_profile("coder", clone_config=True, no_alias=True)
        # No error; optional files just not copied
        assert not (profile_dir / "config.yaml").exists()
        assert not (profile_dir / ".env").exists()
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

    def test_no_skills_conflicts_with_clone(self, profile_env):
        with pytest.raises(ValueError, match="mutually exclusive"):
            create_profile(
                "orchestrator",
                no_alias=True,
                no_skills=True,
                clone_config=True,
            )

    def test_no_skills_conflicts_with_clone_all(self, profile_env):
        with pytest.raises(ValueError, match="mutually exclusive"):
            create_profile(
                "orchestrator",
                no_alias=True,
                no_skills=True,
                clone_all=True,
            )

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
        assert cfg["hosts"]["hermes.heimdall"]["aiPeer"] == "ssi_health"
        assert cfg["hosts"]["hermes.heimdall"]["peerName"] == "user-peer"

    def test_pins_ai_peer_when_absent_on_honcho_host_rename(self, profile_env):
        tmp_path = profile_env
        create_profile("ssi_health", no_alias=True)
        honcho_path = tmp_path / ".hermes" / "honcho.json"
        honcho_path.write_text(json.dumps({
            "hosts": {
                "hermes.ssi_health": {"workspace": "hermes", "enabled": True}
            }
        }))

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            rename_profile("ssi_health", "heimdall")

        cfg = json.loads(honcho_path.read_text())
        assert "hermes.ssi_health" not in cfg["hosts"]
        assert cfg["hosts"]["hermes.heimdall"]["aiPeer"] == "ssi_health"
        assert cfg["hosts"]["hermes.heimdall"]["workspace"] == "hermes"

    def test_does_not_overwrite_existing_honcho_host_on_rename(self, profile_env):
        tmp_path = profile_env
        create_profile("ssi_health", no_alias=True)
        honcho_path = tmp_path / ".hermes" / "honcho.json"
        honcho_path.write_text(json.dumps({
            "hosts": {
                "hermes.ssi_health": {"aiPeer": "ssi_health"},
                "hermes.heimdall": {"aiPeer": "heimdall"},
            }
        }))

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            rename_profile("ssi_health", "heimdall")

        cfg = json.loads(honcho_path.read_text())
        assert cfg["hosts"]["hermes.ssi_health"]["aiPeer"] == "ssi_health"
        assert cfg["hosts"]["hermes.heimdall"]["aiPeer"] == "heimdall"

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

    def test_import_with_explicit_name_does_not_mutate_existing_archive_root_profile(
        self, profile_env, tmp_path
    ):
        create_profile("victim", no_alias=True)
        victim_dir = get_profile_dir("victim")
        (victim_dir / "marker.txt").write_text("original")

        archive_path = tmp_path / "export" / "victim.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tf:
            data = b"imported"
            info = tarfile.TarInfo("victim/marker.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        imported = import_profile(str(archive_path), name="renamed")

        assert imported == get_profile_dir("renamed")
        assert (imported / "marker.txt").read_text() == "imported"
        assert (victim_dir / "marker.txt").read_text() == "original"

    def test_import_rejects_archive_with_multiple_top_level_directories(
        self, profile_env, tmp_path
    ):
        archive_path = tmp_path / "export" / "multi-root.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "w:gz") as tf:
            for member_name, data in (
                ("alpha/marker.txt", b"a"),
                ("beta/marker.txt", b"b"),
            ):
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="exactly one top-level directory"):
            import_profile(str(archive_path), name="coder")

        assert not get_profile_dir("coder").exists()

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

    def test_active_profile_name_docker_default(self, tmp_path, monkeypatch):
        """In Docker (no profile active), get_active_profile_name() returns 'default'."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        assert get_active_profile_name() == "default"

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

    def test_gateway_running_check_plain_pid(self, profile_env):
        """Shared PID validator returning None means the profile is not running."""
        from hermes_cli.profiles import _check_gateway_running
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"

        with patch("gateway.status.get_running_pid", return_value=None) as mock_get_running_pid:
            assert _check_gateway_running(default_home) is False
        mock_get_running_pid.assert_called_once_with(
            default_home / "gateway.pid",
            cleanup_stale=False,
        )

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
