"""Tests for the profile management system (hermes_cli/profiles.py)."""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profiles_home(tmp_path):
    """Create a fake ~/.hermes tree and patch Path.home() to use it."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    hermes_default = fake_home / ".hermes"
    hermes_default.mkdir()
    (hermes_default / "config.yaml").write_text("model:\n  model: anthropic/claude-sonnet-4\n  provider: openrouter\n")
    (hermes_default / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    (hermes_default / "memories").mkdir()
    (hermes_default / "sessions").mkdir()
    (hermes_default / "skills").mkdir()

    with patch("hermes_cli.profiles.Path.home", return_value=fake_home):
        # Also clear HERMES_HOME so get_active_profile_name sees the default
        old = os.environ.pop("HERMES_HOME", None)
        yield fake_home
        if old is not None:
            os.environ["HERMES_HOME"] = old
        else:
            os.environ.pop("HERMES_HOME", None)


@pytest.fixture
def profiles_mod():
    """Import the profiles module (deferred to avoid import-time side effects)."""
    import hermes_cli.profiles as mod
    return mod


# ---------------------------------------------------------------------------
# validate_profile_name
# ---------------------------------------------------------------------------

class TestValidateProfileName:
    def test_valid_names(self, profiles_mod):
        for name in ["work", "personal", "bot-1", "my_agent", "a", "x" * 64]:
            profiles_mod.validate_profile_name(name)  # should not raise

    def test_default_is_valid(self, profiles_mod):
        profiles_mod.validate_profile_name("default")

    def test_invalid_names(self, profiles_mod):
        for name in ["", "Work", "has space", "-starts-hyphen", "_starts-under",
                      "has.dot", "x" * 65, "UPPER"]:
            with pytest.raises(ValueError):
                profiles_mod.validate_profile_name(name)


# ---------------------------------------------------------------------------
# get_profile_dir
# ---------------------------------------------------------------------------

class TestGetProfileDir:
    def test_default_returns_hermes_home(self, profiles_home, profiles_mod):
        result = profiles_mod.get_profile_dir("default")
        assert result == profiles_home / ".hermes"

    def test_named_returns_profiles_subdir(self, profiles_home, profiles_mod):
        result = profiles_mod.get_profile_dir("work")
        assert result == profiles_home / ".hermes" / "profiles" / "work"


# ---------------------------------------------------------------------------
# create_profile
# ---------------------------------------------------------------------------

class TestCreateProfile:
    def test_basic_create(self, profiles_home, profiles_mod):
        path = profiles_mod.create_profile("mybot")
        assert path.is_dir()
        assert path == profiles_home / ".hermes" / "profiles" / "mybot"
        # Check bootstrapped directories
        for subdir in ["memories", "sessions", "skills", "skins", "logs",
                        "plans", "workspace", "audio_cache", "image_cache"]:
            assert (path / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_cannot_create_default(self, profiles_home, profiles_mod):
        with pytest.raises(ValueError, match="default"):
            profiles_mod.create_profile("default")

    def test_duplicate_raises(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("dup")
        with pytest.raises(FileExistsError):
            profiles_mod.create_profile("dup")

    def test_invalid_name_raises(self, profiles_home, profiles_mod):
        with pytest.raises(ValueError):
            profiles_mod.create_profile("Bad Name")

    def test_clone_from_default(self, profiles_home, profiles_mod):
        path = profiles_mod.create_profile("cloned", clone_from="default")
        assert (path / "config.yaml").exists()
        assert (path / ".env").exists()
        # Verify content was actually copied
        assert "anthropic/claude-sonnet-4" in (path / "config.yaml").read_text()
        assert "sk-test-123" in (path / ".env").read_text()

    def test_clone_from_nonexistent_raises(self, profiles_home, profiles_mod):
        with pytest.raises(FileNotFoundError):
            profiles_mod.create_profile("bad", clone_from="nonexistent")

    def test_clone_with_data(self, profiles_home, profiles_mod):
        # Put some data in default profile
        default_home = profiles_home / ".hermes"
        (default_home / "memories" / "memory.md").write_text("I remember things")
        (default_home / "skills" / "test-skill").mkdir(parents=True)
        (default_home / "skills" / "test-skill" / "SKILL.md").write_text("---\nname: test\n---\n# Test")

        path = profiles_mod.create_profile("full-clone", clone_from="default", clone_data=True)
        assert (path / "memories" / "memory.md").exists()
        assert (path / "skills" / "test-skill" / "SKILL.md").exists()

    def test_clone_without_data_skips_memories(self, profiles_home, profiles_mod):
        default_home = profiles_home / ".hermes"
        (default_home / "memories" / "memory.md").write_text("secret")

        path = profiles_mod.create_profile("config-only", clone_from="default")
        # memories dir exists (bootstrapped) but should be empty
        assert (path / "memories").is_dir()
        assert not (path / "memories" / "memory.md").exists()

    def test_clone_from_named_profile(self, profiles_home, profiles_mod):
        # Create source profile first
        src = profiles_mod.create_profile("source")
        (src / "config.yaml").write_text("model:\n  model: openai/gpt-4\n")
        (src / ".env").write_text("OPENAI_API_KEY=sk-source\n")

        # Clone from it
        dst = profiles_mod.create_profile("derived", clone_from="source")
        assert "gpt-4" in (dst / "config.yaml").read_text()
        assert "sk-source" in (dst / ".env").read_text()


# ---------------------------------------------------------------------------
# delete_profile
# ---------------------------------------------------------------------------

class TestDeleteProfile:
    def test_delete_existing(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("doomed")
        path = profiles_mod.delete_profile("doomed")
        assert not path.exists()

    def test_cannot_delete_default(self, profiles_home, profiles_mod):
        with pytest.raises(ValueError, match="default"):
            profiles_mod.delete_profile("default")

    def test_delete_nonexistent_raises(self, profiles_home, profiles_mod):
        with pytest.raises(FileNotFoundError):
            profiles_mod.delete_profile("ghost")

    def test_delete_with_running_gateway_raises(self, profiles_home, profiles_mod):
        path = profiles_mod.create_profile("running")
        # Write a fake PID file with our own PID (so os.kill(pid, 0) succeeds)
        pid_data = {"pid": os.getpid(), "kind": "hermes-gateway"}
        (path / "gateway.pid").write_text(json.dumps(pid_data))

        with pytest.raises(RuntimeError, match="running gateway"):
            profiles_mod.delete_profile("running")


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    def test_default_only(self, profiles_home, profiles_mod):
        profiles = profiles_mod.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "default"
        assert profiles[0].is_default
        assert profiles[0].model == "anthropic/claude-sonnet-4"
        assert profiles[0].has_env

    def test_with_named_profiles(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("alpha")
        profiles_mod.create_profile("beta")
        profiles = profiles_mod.list_profiles()
        names = [p.name for p in profiles]
        assert "default" in names
        assert "alpha" in names
        assert "beta" in names
        assert len(profiles) == 3

    def test_profiles_sorted(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("zebra")
        profiles_mod.create_profile("alpha")
        profiles = profiles_mod.list_profiles()
        named = [p.name for p in profiles if not p.is_default]
        assert named == ["alpha", "zebra"]


# ---------------------------------------------------------------------------
# resolve_profile_env
# ---------------------------------------------------------------------------

class TestResolveProfileEnv:
    def test_default_returns_hermes_home(self, profiles_home, profiles_mod):
        result = profiles_mod.resolve_profile_env("default")
        assert result == str(profiles_home / ".hermes")

    def test_existing_named_profile(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("work")
        result = profiles_mod.resolve_profile_env("work")
        assert result == str(profiles_home / ".hermes" / "profiles" / "work")

    def test_nonexistent_raises(self, profiles_home, profiles_mod):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            profiles_mod.resolve_profile_env("missing")

    def test_invalid_name_raises(self, profiles_home, profiles_mod):
        with pytest.raises(ValueError):
            profiles_mod.resolve_profile_env("Bad Name!")


# ---------------------------------------------------------------------------
# get_active_profile_name
# ---------------------------------------------------------------------------

class TestGetActiveProfileName:
    def test_default_when_no_env(self, profiles_home, profiles_mod):
        assert profiles_mod.get_active_profile_name() == "default"

    def test_named_profile_from_env(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("test-profile")
        profile_path = str(profiles_home / ".hermes" / "profiles" / "test-profile")
        with patch.dict(os.environ, {"HERMES_HOME": profile_path}):
            assert profiles_mod.get_active_profile_name() == "test-profile"

    def test_custom_when_unrecognized_path(self, profiles_home, profiles_mod):
        with patch.dict(os.environ, {"HERMES_HOME": "/opt/custom-hermes"}):
            assert profiles_mod.get_active_profile_name() == "custom"


# ---------------------------------------------------------------------------
# profile_exists
# ---------------------------------------------------------------------------

class TestProfileExists:
    def test_default_exists(self, profiles_home, profiles_mod):
        assert profiles_mod.profile_exists("default")

    def test_created_exists(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("new")
        assert profiles_mod.profile_exists("new")

    def test_uncreated_does_not_exist(self, profiles_home, profiles_mod):
        assert not profiles_mod.profile_exists("nope")


# ---------------------------------------------------------------------------
# Profile isolation: verify each profile is a full HERMES_HOME
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    """Verify that setting HERMES_HOME to a profile dir gives full isolation."""

    def test_config_isolation(self, profiles_home, profiles_mod):
        """Two profiles should have independent config.yaml files."""
        p1 = profiles_mod.create_profile("iso1", clone_from="default")
        p2 = profiles_mod.create_profile("iso2", clone_from="default")

        # Modify p1's config
        (p1 / "config.yaml").write_text("model:\n  model: openai/gpt-4\n")
        # p2 should still have the original
        assert "claude" in (p2 / "config.yaml").read_text()
        assert "gpt-4" in (p1 / "config.yaml").read_text()

    def test_env_isolation(self, profiles_home, profiles_mod):
        """Two profiles should have independent .env files."""
        p1 = profiles_mod.create_profile("env1", clone_from="default")
        p2 = profiles_mod.create_profile("env2", clone_from="default")

        (p1 / ".env").write_text("OPENROUTER_API_KEY=sk-work\n")
        (p2 / ".env").write_text("OPENROUTER_API_KEY=sk-personal\n")

        assert "sk-work" in (p1 / ".env").read_text()
        assert "sk-personal" in (p2 / ".env").read_text()

    def test_memory_isolation(self, profiles_home, profiles_mod):
        """Two profiles should have independent memory directories."""
        p1 = profiles_mod.create_profile("mem1")
        p2 = profiles_mod.create_profile("mem2")

        (p1 / "memories" / "memory.md").write_text("Profile 1 memory")
        assert not (p2 / "memories" / "memory.md").exists()

    def test_session_isolation(self, profiles_home, profiles_mod):
        """Two profiles should have independent session directories."""
        p1 = profiles_mod.create_profile("ses1")
        p2 = profiles_mod.create_profile("ses2")

        (p1 / "sessions" / "test.json").write_text("{}")
        assert not (p2 / "sessions" / "test.json").exists()

    def test_skills_isolation(self, profiles_home, profiles_mod):
        """Two profiles should have independent skill directories."""
        p1 = profiles_mod.create_profile("sk1")
        p2 = profiles_mod.create_profile("sk2")

        skill_dir = p1 / "skills" / "custom-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Custom")
        assert not (p2 / "skills" / "custom-skill").exists()


# ---------------------------------------------------------------------------
# Gateway collision prevention
# ---------------------------------------------------------------------------

class TestGatewayIsolation:
    def test_pid_files_are_separate(self, profiles_home, profiles_mod):
        """Each profile should have its own gateway.pid path."""
        p1 = profiles_mod.create_profile("gw1")
        p2 = profiles_mod.create_profile("gw2")

        pid1_path = p1 / "gateway.pid"
        pid2_path = p2 / "gateway.pid"

        # They should be different paths
        assert pid1_path != pid2_path

        # Writing to one doesn't affect the other
        pid1_path.write_text(json.dumps({"pid": 12345}))
        assert not pid2_path.exists()

    def test_systemd_service_names_differ(self, profiles_home, profiles_mod):
        """Different profiles should get different systemd service names."""
        p1 = profiles_mod.create_profile("svc1")
        p2 = profiles_mod.create_profile("svc2")

        from hermes_cli.gateway import get_service_name

        with patch.dict(os.environ, {"HERMES_HOME": str(p1)}):
            name1 = get_service_name()
        with patch.dict(os.environ, {"HERMES_HOME": str(p2)}):
            name2 = get_service_name()

        assert name1 != name2
        # Both should start with hermes-gateway
        assert name1.startswith("hermes-gateway")
        assert name2.startswith("hermes-gateway")


# ---------------------------------------------------------------------------
# _apply_profile_override (from main.py)
# ---------------------------------------------------------------------------

class TestApplyProfileOverride:
    """Test that --profile/-p pre-parsing sets HERMES_HOME correctly."""

    def test_profile_flag_sets_env(self, profiles_home, profiles_mod):
        profiles_mod.create_profile("test-pre")
        expected = str(profiles_home / ".hermes" / "profiles" / "test-pre")

        from hermes_cli.profiles import resolve_profile_env
        result = resolve_profile_env("test-pre")
        assert result == expected

    def test_default_profile_resolves_to_hermes_home(self, profiles_home, profiles_mod):
        from hermes_cli.profiles import resolve_profile_env
        result = resolve_profile_env("default")
        assert result == str(profiles_home / ".hermes")
