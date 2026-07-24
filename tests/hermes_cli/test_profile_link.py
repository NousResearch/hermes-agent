"""Tests for ``hermes profile link`` — git-backed profile symlinks (issue #44179)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Create a minimal profile environment for testing.

    Returns (hermes_home, profile_dir, external_repo).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir()

    # Create a test profile
    profile_dir = profiles_root / "testprofile"
    profile_dir.mkdir()
    (profile_dir / "config.yaml").write_text("model: test\n")
    (profile_dir / "SOUL.md").write_text("# Test SOUL\n")
    (profile_dir / "skills").mkdir()
    (profile_dir / "skills" / "example.skill").write_text("example")

    # Create an external git repo
    external = tmp_path / "my-dotfiles"
    external.mkdir()
    subprocess.run(["git", "init"], cwd=str(external), capture_output=True)
    (external / "SOUL.md").write_text("# External SOUL\n")
    (external / "config.yaml").write_text("model: external\n")
    (external / "skills").mkdir()
    (external / "skills" / "remote.skill").write_text("remote skill")
    (external / ".gitignore").write_text("*.pyc\n")
    subprocess.run(["git", "add", "."], cwd=str(external), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(external), capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
             "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"},
    )

    # Patch hermes home
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Patch get_profile_dir and related functions
    import hermes_cli.profiles as prof
    monkeypatch.setattr(prof, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(prof, "_get_default_hermes_home", lambda: hermes_home)

    return hermes_home, profile_dir, external


class TestLinkProfile:
    """Tests for link_profile()."""

    def test_link_creates_symlinks(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        results = link_profile("testprofile", external, move_existing=True)

        # config.yaml and SOUL.md should be symlinks
        assert (profile_dir / "config.yaml").is_symlink()
        assert (profile_dir / "SOUL.md").is_symlink()
        assert (profile_dir / "skills").is_symlink()

        # Symlinks should point to external repo
        assert (profile_dir / "config.yaml").resolve() == (external / "config.yaml").resolve()
        assert (profile_dir / "SOUL.md").resolve() == (external / "SOUL.md").resolve()

        # Metadata file should exist
        meta_path = profile_dir / ".profile-link.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["external_path"] == str(external.resolve())
        assert "config.yaml" in meta["linked_items"]
        assert "SOUL.md" in meta["linked_items"]

        # Results should mention LINK
        assert any("LINK" in line for line in results)

    def test_link_skips_missing_items(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        # mcp.json doesn't exist in external repo
        results = link_profile("testprofile", external)

        assert any("SKIP" in line and "mcp.json" in line for line in results)

    def test_link_rejects_non_git_repo(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        non_repo = external.parent / "not-a-repo"
        non_repo.mkdir()

        with pytest.raises(ValueError, match="not a git repository"):
            link_profile("testprofile", non_repo)

    def test_link_rejects_missing_profile(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        with pytest.raises(FileNotFoundError, match="does not exist"):
            link_profile("nonexistent", external)

    def test_link_rejects_missing_external_path(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        with pytest.raises(FileNotFoundError, match="does not exist"):
            link_profile("testprofile", Path("/tmp/nonexistent_path_xyz"))

    def test_link_idempotent(self, profile_env):
        """Running link twice should be safe (OK status for existing links)."""
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        link_profile("testprofile", external, move_existing=True)
        results = link_profile("testprofile", external)

        assert any("OK" in line for line in results)

    def test_link_skips_existing_non_symlink(self, profile_env):
        """Without --move, existing regular files should be skipped."""
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        results = link_profile("testprofile", external)

        # config.yaml existed as a regular file, should be skipped
        assert any("SKIP" in line and "config.yaml" in line for line in results)

    def test_link_with_move(self, profile_env):
        """With --move, existing files should be backed up and replaced."""
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        results = link_profile("testprofile", external, move_existing=True)

        # config.yaml should now be a symlink
        assert (profile_dir / "config.yaml").is_symlink()
        assert (profile_dir / "config.yaml").resolve() == (external / "config.yaml").resolve()

        # Original should be backed up
        assert (profile_dir / "config.yaml.bak").exists()

    def test_link_reads_external_content(self, profile_env):
        """After linking, reading the profile file should show external content."""
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile

        link_profile("testprofile", external, move_existing=True)

        # Read through the symlink
        content = (profile_dir / "config.yaml").read_text()
        assert content == "model: external\n"


class TestCheckProfileLinks:
    """Tests for check_profile_links()."""

    def test_check_ok(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile, check_profile_links

        link_profile("testprofile", external, move_existing=True)
        results = check_profile_links("testprofile")

        assert any("OK" in line for line in results)
        assert any(str(external.resolve()) in line for line in results)

    def test_check_detects_broken(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile, check_profile_links

        link_profile("testprofile", external, move_existing=True)

        # Remove the external file to break the symlink
        (external / "config.yaml").unlink()

        results = check_profile_links("testprofile")
        assert any("BROKEN" in line and "config.yaml" in line for line in results)

    def test_check_rejects_unlinked_profile(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import check_profile_links

        with pytest.raises(FileNotFoundError, match="not linked"):
            check_profile_links("testprofile")


class TestUnlinkProfile:
    """Tests for unlink_profile()."""

    def test_unlink_restores_files(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import link_profile, unlink_profile

        link_profile("testprofile", external, move_existing=True)
        results = unlink_profile("testprofile")

        # Should no longer be symlinks
        assert not (profile_dir / "config.yaml").is_symlink()
        assert not (profile_dir / "SOUL.md").is_symlink()

        # Files should contain external content (copied back)
        assert (profile_dir / "config.yaml").read_text() == "model: external\n"

        # Metadata should be removed
        assert not (profile_dir / ".profile-link.json").exists()

        assert any("RESTORE" in line for line in results)

    def test_unlink_rejects_unlinked_profile(self, profile_env):
        hermes_home, profile_dir, external = profile_env
        from hermes_cli.profiles import unlink_profile

        with pytest.raises(FileNotFoundError, match="not linked"):
            unlink_profile("testprofile")


class TestParserIntegration:
    """Test the CLI parser accepts link subcommand."""

    def test_parser_accepts_link(self):
        """Verify the link subcommand is registered."""
        import argparse
        from hermes_cli.subcommands.profile import build_profile_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_profile_parser(subparsers, cmd_profile=lambda x: None)

        # Should parse without error
        args = parser.parse_args(["profile", "link", "myprofile", "/tmp/repo"])
        assert args.profile_action == "link"
        assert args.profile_name == "myprofile"
        assert args.path == "/tmp/repo"

    def test_parser_accepts_link_check(self):
        import argparse
        from hermes_cli.subcommands.profile import build_profile_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_profile_parser(subparsers, cmd_profile=lambda x: None)

        args = parser.parse_args(["profile", "link", "--check", "myprofile"])
        assert args.check is True
        assert args.profile_name == "myprofile"

    def test_parser_accepts_link_unlink(self):
        import argparse
        from hermes_cli.subcommands.profile import build_profile_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_profile_parser(subparsers, cmd_profile=lambda x: None)

        args = parser.parse_args(["profile", "link", "--unlink", "myprofile"])
        assert args.unlink is True
        assert args.profile_name == "myprofile"

    def test_parser_accepts_link_move(self):
        import argparse
        from hermes_cli.subcommands.profile import build_profile_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_profile_parser(subparsers, cmd_profile=lambda x: None)

        args = parser.parse_args(["profile", "link", "--move", "myprofile", "/tmp/repo"])
        assert args.move is True
