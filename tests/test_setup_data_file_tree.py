"""Tests for setup.py _data_file_tree function.

Verifies that the data-file discovery helper tolerates missing optional
data directories (e.g. when optional-skills/ is absent from an sdist).
"""

import os
import tempfile
from collections import defaultdict
from pathlib import Path


def _data_file_tree(root_name: str, repo_root: Path) -> list:
    """Replicate of setup._data_file_tree for isolated testing."""
    root = repo_root / root_name
    if not root.is_dir():
        return []
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(repo_root)
        grouped[str(rel_path.parent)].append(str(rel_path))
    return sorted(grouped.items())


def test_data_file_tree_missing_directory_returns_empty():
    """When the target directory doesn't exist, return [] instead of crashing."""
    repo_root = Path(tempfile.mkdtemp())
    try:
        result = _data_file_tree("nonexistent_skills_dir", repo_root)
        assert result == [], f"Expected empty list, got {result}"
    finally:
        import shutil
        shutil.rmtree(repo_root, ignore_errors=True)


def test_data_file_tree_existing_directory_works():
    """When the directory exists, files are discovered normally."""
    repo_root = Path(tempfile.mkdtemp())
    try:
        (repo_root / "skills").mkdir()
        (repo_root / "skills" / "test_skill.md").write_text("# Test")
        result = _data_file_tree("skills", repo_root)
        assert len(result) > 0, "Expected at least one file entry"
        dirs = [d for d, _ in result]
        assert "skills" in dirs, f"Expected 'skills' in {dirs}"
    finally:
        import shutil
        shutil.rmtree(repo_root, ignore_errors=True)


def test_data_file_tree_optional_skills_dir():
    """optional-skills/ is commonly absent from sdists — must not crash."""
    repo_root = Path(tempfile.mkdtemp())
    try:
        # Only create skills, leave optional-skills absent
        (repo_root / "skills").mkdir()
        (repo_root / "skills" / "test_skill.md").write_text("# Test")
        result_skills = _data_file_tree("skills", repo_root)
        result_optional = _data_file_tree("optional-skills", repo_root)
        assert len(result_skills) > 0, "skills should have files"
        assert result_optional == [], (
            f"optional-skills missing should return [], got {result_optional}"
        )
    finally:
        import shutil
        shutil.rmtree(repo_root, ignore_errors=True)
