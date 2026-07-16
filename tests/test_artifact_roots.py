"""Tests for the artifact-root resolver (phase 0 task 0.2b).

Tests the resolver against two fixture layouts:
  - **slot**: a bundle-style tree with manifest.json at the root, code in
    runtime/venv/site-packages, assets at app/, ui/tui/dist, ui/web/dist.
  - **checkout**: a repo-style tree with pyproject.toml + .git, code in
    hermes_cli/, assets at skills/, hermes_cli/web_dist, ui-tui/dist.

The key invariant: in a checkout, the accessors return EXACTLY the paths
that today's hard-coded ``Path(__file__).parent.parent / "..."`` patterns
compute. In a slot, they return the bundle-layout equivalents.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# We can't import get_artifact_root at module level and then patch __file__
# after the fact — the function captures __file__ at call time, not import
# time, so we import the module and call the function with a patched __file__.


def _make_slot_layout(root: Path) -> Path:
    """Create a slot (bundle) layout under *root*.

    Returns the path that __file__ would have in a slot install:
    <root>/runtime/venv/lib/python3.11/site-packages/hermes_constants.py
    """
    site_packages = root / "runtime" / "venv" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    # The module file
    mod_file = site_packages / "hermes_constants.py"
    mod_file.write_text("# stub")
    # manifest.json at the slot root
    (root / "manifest.json").write_text('{"schema": 1}')
    # Assets in bundle layout
    (root / "app" / "skills").mkdir(parents=True)
    (root / "ui" / "tui" / "dist").mkdir(parents=True)
    (root / "ui" / "web" / "dist").mkdir(parents=True)
    return mod_file


def _make_checkout_layout(root: Path) -> Path:
    """Create a checkout (source) layout under *root*.

    Returns the path that __file__ would have in a checkout:
    <root>/hermes_constants.py
    """
    # pyproject.toml + .git at the repo root
    (root / "pyproject.toml").write_text('[project]\nname = "hermes-agent"')
    (root / ".git").mkdir()
    # The module file at the root level (as it is in the real repo)
    mod_file = root / "hermes_constants.py"
    mod_file.write_text("# stub")
    # Assets in checkout layout
    (root / "skills").mkdir()
    (root / "hermes_cli" / "web_dist").mkdir(parents=True)
    (root / "ui-tui" / "dist").mkdir(parents=True)
    return mod_file


class TestGetArtifactRoot:
    def test_slot_layout_resolves_to_bundle_root(self, tmp_path):
        mod_file = _make_slot_layout(tmp_path)
        from hermes_constants import get_artifact_root

        with patch("hermes_constants.__file__", str(mod_file)):
            root = get_artifact_root()
        assert root == tmp_path
        assert (root / "manifest.json").is_file()

    def test_checkout_layout_resolves_to_repo_root(self, tmp_path):
        mod_file = _make_checkout_layout(tmp_path)
        from hermes_constants import get_artifact_root

        with patch("hermes_constants.__file__", str(mod_file)):
            root = get_artifact_root()
        assert root == tmp_path
        assert (root / "pyproject.toml").is_file()

    def test_worktree_layout_git_file_not_dir(self, tmp_path):
        """A worktree's .git is a FILE (gitdir: ...), not a directory."""
        mod_file = _make_checkout_layout(tmp_path)
        # Replace .git dir with a .git file (worktree style)
        import shutil
        shutil.rmtree(tmp_path / ".git")
        (tmp_path / ".git").write_text("gitdir: /some/main/repo/.git/worktrees/foo")

        from hermes_constants import get_artifact_root

        with patch("hermes_constants.__file__", str(mod_file)):
            root = get_artifact_root()
        # Should still find pyproject.toml and resolve to repo root
        assert root == tmp_path

    def test_real_checkout_resolves_correctly(self):
        """Against the actual repo (this checkout), the resolver should
        find pyproject.toml and return the repo root."""
        from hermes_constants import get_artifact_root

        root = get_artifact_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "hermes_cli").is_dir()


class TestBundledSkillsDir:
    def test_slot_layout(self, tmp_path):
        mod_file = _make_slot_layout(tmp_path)
        from hermes_constants import bundled_skills_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            skills = bundled_skills_dir()
        assert skills == tmp_path / "app" / "skills"

    def test_checkout_layout(self, tmp_path):
        mod_file = _make_checkout_layout(tmp_path)
        from hermes_constants import bundled_skills_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            skills = bundled_skills_dir()
        assert skills == tmp_path / "skills"


class TestWebDistDir:
    def test_slot_layout(self, tmp_path):
        mod_file = _make_slot_layout(tmp_path)
        from hermes_constants import web_dist_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            dist = web_dist_dir()
        assert dist == tmp_path / "ui" / "web" / "dist"

    def test_checkout_layout(self, tmp_path):
        mod_file = _make_checkout_layout(tmp_path)
        from hermes_constants import web_dist_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            dist = web_dist_dir()
        assert dist == tmp_path / "hermes_cli" / "web_dist"


class TestTuiDistDir:
    def test_slot_layout(self, tmp_path):
        mod_file = _make_slot_layout(tmp_path)
        from hermes_constants import tui_dist_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            dist = tui_dist_dir()
        assert dist == tmp_path / "ui" / "tui" / "dist"

    def test_checkout_layout(self, tmp_path):
        mod_file = _make_checkout_layout(tmp_path)
        from hermes_constants import tui_dist_dir

        with patch("hermes_constants.__file__", str(mod_file)):
            dist = tui_dist_dir()
        assert dist == tmp_path / "ui-tui" / "dist"


class TestCheckoutByteCompat:
    """In a checkout, the accessors must return EXACTLY what the old
    hard-coded patterns computed. This is the byte-identical contract."""

    def test_bundled_skills_matches_old_pattern(self):
        """Old: Path(__file__).parent.parent / 'skills'
        (used by tools/skills_sync.py:60 as the default arg)
        """
        from hermes_constants import bundled_skills_dir, get_bundled_skills_dir

        new = bundled_skills_dir()
        old_default = Path(__file__).resolve().parent.parent / "skills"
        # In the real checkout, bundled_skills_dir() should resolve to
        # the same repo root / skills path that the old pattern used.
        assert new == old_default

    def test_web_dist_matches_old_pattern(self):
        """Old: PROJECT_ROOT / 'hermes_cli' / 'web_dist'
        (used by hermes_cli/main.py:4635, 12133)
        """
        from hermes_constants import web_dist_dir

        new = web_dist_dir()
        old = Path(__file__).resolve().parent.parent / "hermes_cli" / "web_dist"
        assert new == old

    def test_tui_dist_matches_old_pattern(self):
        """Old: PROJECT_ROOT / 'ui-tui' / 'dist'
        (used by hermes_cli/main.py:1902)
        """
        from hermes_constants import tui_dist_dir

        new = tui_dist_dir()
        old = Path(__file__).resolve().parent.parent / "ui-tui" / "dist"
        assert new == old
