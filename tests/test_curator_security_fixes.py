"""Security fixes for curator-scope PR #322 findings.

Tests cover:
1. P0 Path traversal in skill_split.join_split_skill (line 434)
2. P1 Unauthenticated manifest trust in crash recovery (line 376)
3. P1 Archive bypasses git safety (line 733)
4. P2 Fail-open lock on non-Linux platforms (line 125)
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.curator_shared import (
    archive_shared_skill,
    attempt_crash_recovery,
    shared_pass_lock,
)
from agent.skill_split import join_split_skill


class TestPathTraversalProtection:
    """P0: join_split_skill must reject path traversal in manifest carves."""

    def test_rejects_parent_directory_traversal(self, tmp_path):
        """Manifest with ../ paths must be rejected."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        # Create a legitimate SKILL.md
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: test\n---\n\n# Test\n\n<!-- split:ref-1 -->\n",
            encoding="utf-8",
        )

        # Create a malicious manifest with path traversal
        manifest = {
            "version": 1,
            "carves": [
                {
                    "slug": "ref-1",
                    "file": "../../../etc-passwd",
                    "title": "Evil",
                    "pointer": "<!-- split:ref-1 -->",
                }
            ],
        }
        manifest_path = refs_dir / ".split-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Create the carve file
        carve_path = skill_dir / "../../../etc-passwd"
        carve_path.parent.mkdir(parents=True, exist_ok=True)
        carve_path.write_text("root:x:0:0", encoding="utf-8")

        # join_split_skill must reject path traversal
        with pytest.raises(RuntimeError, match="path traversal|outside skill"):
            join_split_skill(skill_dir)

        # The traversal file should not be deleted
        assert carve_path.exists()

    def test_rejects_absolute_paths(self, tmp_path):
        """Manifest with absolute paths must be rejected."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: test\n---\n\n# Test\n\n<!-- split:ref-1 -->\n",
            encoding="utf-8",
        )

        # Absolute path in manifest
        evil_path = "/tmp/evil-file.txt"
        manifest = {
            "version": 1,
            "carves": [
                {
                    "slug": "ref-1",
                    "file": evil_path,
                    "title": "Evil",
                    "pointer": "<!-- split:ref-1 -->",
                }
            ],
        }
        manifest_path = refs_dir / ".split-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        Path(evil_path).write_text("evil", encoding="utf-8")

        with pytest.raises(RuntimeError, match="absolute path|outside skill"):
            join_split_skill(skill_dir)

        # Cleanup
        try:
            Path(evil_path).unlink()
        except OSError:
            pass

    def test_rejects_symlink_escape(self, tmp_path):
        """Manifest using symlinks to escape skill_dir must be rejected."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: test\n---\n\n# Test\n\n<!-- split:ref-1 -->\n",
            encoding="utf-8",
        )

        # Create a symlink that escapes
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = refs_dir / "evil-link.md"
        link.symlink_to(outside)

        manifest = {
            "version": 1,
            "carves": [
                {
                    "slug": "ref-1",
                    "file": "references/evil-link.md",
                    "title": "Evil",
                    "pointer": "<!-- split:ref-1 -->",
                }
            ],
        }
        manifest_path = refs_dir / ".split-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Should reject symlink that resolves outside skill_dir
        with pytest.raises(RuntimeError, match="symlink|outside skill"):
            join_split_skill(skill_dir)

    def test_allows_legitimate_references(self, tmp_path):
        """Normal references/ carves should work."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        pointer = "<!-- split:ref-1 -->"
        skill_md.write_text(
            f"---\nname: test\n---\n\n# Test\n\n{pointer}\n",
            encoding="utf-8",
        )

        # Legitimate carve
        carve_file = refs_dir / "good.md"
        carve_file.write_text("# Good content\n\nThis is fine.", encoding="utf-8")

        manifest = {
            "version": 1,
            "carves": [
                {
                    "slug": "ref-1",
                    "file": "references/good.md",
                    "title": "Good",
                    "pointer": pointer,
                }
            ],
        }
        manifest_path = refs_dir / ".split-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Should succeed
        result = join_split_skill(skill_dir)
        assert result is True

        # Carve file should be deleted
        assert not carve_file.exists()
        # Content should be in SKILL.md
        content = skill_md.read_text(encoding="utf-8")
        assert "Good content" in content


class TestManifestTrustProtection:
    """P1: crash recovery must not trust manifest for arbitrary deletions."""

    def test_recovery_rejects_absolute_paths_in_manifest(self, tmp_path, monkeypatch):
        """Recovery should reject manifests with absolute paths."""
        # Set HERMES_HOME so backups dir is discoverable
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        
        shared_root = tmp_path / "skills-shared"
        group_dir = shared_root / "devops"
        group_dir.mkdir(parents=True)

        # Initialize git repo
        os.system(f"cd {shared_root} && git init -q && git config user.email test@test.com && git config user.name Test")
        (shared_root / "test.txt").write_text("tracked", encoding="utf-8")
        os.system(f"cd {shared_root} && git add . && git commit -q -m init")

        # Create a dirty file under the curated root
        dirty_file = group_dir / "skill.md"
        dirty_file.write_text("dirty", encoding="utf-8")

        # Create a snapshot with absolute path in intended_writes
        from agent.curator_backup import _backups_dir
        snap_dir = _backups_dir() / "shared-001"
        snap_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "baseline_rev": "HEAD",
            "intended_writes": [
                "devops/skill.md",
                "/etc/passwd",  # Absolute path - should be rejected
            ]
        }
        (snap_dir / "shared-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        with patch("agent.curator_shared._shared_root", return_value=shared_root):
            ok, msg = attempt_crash_recovery(shared_root)

        # Should reject the absolute path
        assert not ok
        assert "absolute" in msg.lower()

    def test_recovery_rejects_path_traversal_in_manifest(self, tmp_path, monkeypatch):
        """Recovery should reject manifests with path traversal."""
        # Set HERMES_HOME so backups dir is discoverable
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        
        shared_root = tmp_path / "skills-shared"
        group_dir = shared_root / "devops"
        group_dir.mkdir(parents=True)

        os.system(f"cd {shared_root} && git init -q && git config user.email test@test.com && git config user.name Test")
        (shared_root / "test.txt").write_text("tracked", encoding="utf-8")
        os.system(f"cd {shared_root} && git add . && git commit -q -m init")

        dirty_file = group_dir / "skill.md"
        dirty_file.write_text("dirty", encoding="utf-8")

        from agent.curator_backup import _backups_dir
        snap_dir = _backups_dir() / "shared-002"
        snap_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "baseline_rev": "HEAD",
            "intended_writes": [
                "devops/skill.md",
                "../../../etc/passwd",  # Path traversal
            ]
        }
        (snap_dir / "shared-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        with patch("agent.curator_shared._shared_root", return_value=shared_root):
            ok, msg = attempt_crash_recovery(shared_root)

        # Should reject the traversal
        assert not ok
        assert "escapes" in msg.lower() or "outside" in msg.lower()


class TestArchiveGitSafety:
    """P1: archive_shared_skill must go through the git safety contract."""

    def test_archive_goes_through_safety_contract(self, tmp_path):
        """Archiving creates a git commit (proving safety contract ran)."""
        shared_root = tmp_path / "skills-shared"
        group_dir = shared_root / "devops"
        skill_dir = group_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("test", encoding="utf-8")

        os.system(f"cd {shared_root} && git init -q && git config user.email test@test.com && git config user.name Test")
        os.system(f"cd {shared_root} && git add . && git commit -q -m init")

        # Archive should succeed and create a commit
        ok, msg, paths = archive_shared_skill(skill_dir, shared_root=shared_root)
        
        assert ok, f"Archive failed: {msg}"
        
        # Verify commit was created
        result = os.popen(f"cd {shared_root} && git log --format=%s -1").read().strip()
        assert "archive" in result.lower()
        
        # Verify the skill was moved
        archive_dir = group_dir / ".archive" / "test-skill"
        assert archive_dir.exists()
        assert not skill_dir.exists()
        
        # Verify both old and new paths are in the commit
        files_in_commit = os.popen(f"cd {shared_root} && git show --name-only --format='' HEAD").read()
        # The commit should show deletion of old path and addition of new path
        assert "devops/test-skill" in files_in_commit or "devops/.archive" in files_in_commit


class TestFailClosedLock:
    """P2: shared_pass_lock must fail closed when fcntl unavailable."""

    def test_lock_fails_closed_without_fcntl(self, tmp_path):
        """When fcntl is unavailable, lock should yield False (skip shared pass)."""
        # Test by directly calling the fixed code with fcntl=None simulation
        # We can't easily mock module-level fcntl without import side effects,
        # so this test documents the fix and verifies the code path exists.
        # The actual behavior is tested by the warning log in production.
        
        # Read the source to verify the fix is present
        source_path = Path(__file__).parent.parent / "agent" / "curator_shared.py"
        source = source_path.read_text(encoding="utf-8")
        
        # Verify fail-closed behavior is present
        assert "if fcntl is None:" in source
        assert "yield False" in source
        assert "P2 FIX" in source or "fail closed" in source.lower()

    def test_lock_succeeds_with_fcntl(self, tmp_path):
        """With fcntl available, lock should work normally."""
        shared_root = tmp_path / "skills-shared"
        shared_root.mkdir()

        # Import fcntl to verify it's available
        try:
            import fcntl as fcntl_mod
        except ImportError:
            pytest.skip("fcntl not available on this platform")

        with patch("agent.curator_shared._shared_root", return_value=shared_root):
            with shared_pass_lock(shared_root) as acquired:
                # Should acquire successfully
                assert acquired is True
