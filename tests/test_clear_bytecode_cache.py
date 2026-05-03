"""Tests for _clear_bytecode_cache — stale .pyc cleanup after updates.

Covers the fix for issue #6207: cleanup must reach user data directories
(skills, plugins, per-profile dirs) under ~/.hermes/, not just the
hermes-agent source tree (PROJECT_ROOT).
"""

from pathlib import Path

import pytest

from hermes_cli.main import _clear_bytecode_cache


def _make_pycache(base: Path, *parts: str) -> Path:
    """Create a __pycache__ dir (with a dummy .pyc) at base/parts/__pycache__."""
    d = base.joinpath(*parts, "__pycache__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "module.cpython-312.pyc").write_bytes(b"\x00" * 16)
    return d


class TestClearBytecodeCache:
    def test_cleans_pycache_under_root(self, tmp_path):
        cache = _make_pycache(tmp_path, "package")
        count = _clear_bytecode_cache(tmp_path)
        assert count == 1
        assert not cache.exists()

    def test_cleans_nested_pycache(self, tmp_path):
        _make_pycache(tmp_path, "a", "b", "c")
        _make_pycache(tmp_path, "x")
        count = _clear_bytecode_cache(tmp_path)
        assert count == 2

    def test_skips_venv(self, tmp_path):
        _make_pycache(tmp_path, "venv", "lib")
        _make_pycache(tmp_path, "src")
        count = _clear_bytecode_cache(tmp_path)
        # only src/__pycache__ removed; venv is skipped
        assert count == 1

    def test_skips_dotworktrees(self, tmp_path):
        _make_pycache(tmp_path, ".worktrees", "feature")
        count = _clear_bytecode_cache(tmp_path)
        assert count == 0

    def test_extra_skip_excludes_named_dir(self, tmp_path):
        """extra_skip lets callers avoid re-walking already-cleaned subtrees."""
        _make_pycache(tmp_path, "hermes-agent", "tools")
        _make_pycache(tmp_path, "skills", "my_skill")
        count = _clear_bytecode_cache(tmp_path, extra_skip={"hermes-agent"})
        # skills/__pycache__ removed; hermes-agent/__pycache__ skipped
        assert count == 1
        assert not (tmp_path / "skills" / "my_skill" / "__pycache__").exists()
        assert (tmp_path / "hermes-agent" / "tools" / "__pycache__").exists()

    def test_user_data_dirs_cleaned_by_hermes_root_walk(self, tmp_path):
        """Simulates the multi-profile scenario from issue #6207.

        hermes_root (~/.hermes) contains:
          hermes-agent/  — source tree (PROJECT_ROOT)
          skills/        — default-profile skills
          plugins/       — user plugins
          profiles/dev/skills/  — per-profile skills

        The second _clear_bytecode_cache call on hermes_root (with source
        dir skipped) must reach skills, plugins, and per-profile dirs.
        """
        hermes_root = tmp_path / ".hermes"
        project_root = hermes_root / "hermes-agent"

        # Simulate what the first call (PROJECT_ROOT) already cleaned
        # (no __pycache__ here — already gone)

        # User data dirs that should be cleaned by the hermes_root call
        skills_cache = _make_pycache(hermes_root, "skills", "my_skill")
        plugins_cache = _make_pycache(hermes_root, "plugins", "myplugin")
        profile_cache = _make_pycache(
            hermes_root, "profiles", "dev", "skills", "custom_skill"
        )

        # PROJECT_ROOT is a direct child of hermes_root
        assert project_root.parent.resolve() == hermes_root.resolve()

        count = _clear_bytecode_cache(
            hermes_root, extra_skip={project_root.name}
        )
        assert count == 3
        assert not skills_cache.exists()
        assert not plugins_cache.exists()
        assert not profile_cache.exists()
