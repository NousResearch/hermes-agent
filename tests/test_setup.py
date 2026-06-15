"""Tests for setup.py -- _data_file_tree and module-level behavior."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest


def _import_setup():
    """Import the setup module with setup() patched out so module-level
    code doesn't attempt a real setuptools.setup() call."""
    with patch("setuptools.setup") as mock_setup:
        import setup  # noqa: F811
    return setup, mock_setup


class TestDataFileTree:
    """Tests for setup._data_file_tree."""

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """An empty directory (no files at any depth) yields [].

        Covers the case where rglob("*") produces no file paths.
        """
        mod, _ = _import_setup()
        d = tmp_path / "empty"
        d.mkdir(parents=True)
        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("empty")
        assert result == []

    def test_flat_files_in_root(self, tmp_path: Path) -> None:
        """Files directly under the named dir are grouped under that dir.

        Covers the basic grouping path -- rel_path.parent is the dir
        itself and rel_path is a child file.
        """
        mod, _ = _import_setup()
        d = tmp_path / "mydir"
        d.mkdir(parents=True)
        (d / "a.txt").touch()
        (d / "b.txt").touch()

        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("mydir")

        assert result == [
            ("mydir", ["mydir/a.txt", "mydir/b.txt"]),
        ]

    def test_nested_subdirectories(self, tmp_path: Path) -> None:
        """Files in nested subdirs appear under their own parent key.

        Covers the rel_path.parent != "" branch: a file in mydir/sub/
        gets grouped under "mydir/sub", not "mydir".
        """
        mod, _ = _import_setup()
        d = tmp_path / "mydir"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "root.txt").touch()
        (sub / "nested.txt").touch()

        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("mydir")

        assert result == [
            ("mydir", ["mydir/root.txt"]),
            ("mydir/sub", ["mydir/sub/nested.txt"]),
        ]

    def test_skips_non_file_entries(self, tmp_path: Path) -> None:
        """Directories returned by rglob(*) are skipped (continue).

        Covers the ``if not path.is_file(): continue`` branch: only true
        files appear in the output.
        """
        mod, _ = _import_setup()
        d = tmp_path / "mydir"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "file.txt").touch()

        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("mydir")

        # rglob yields: mydir (dir), mydir/file.txt (file), mydir/sub (dir)
        # The two directories are skipped; only file.txt is kept.
        assert result == [
            ("mydir", ["mydir/file.txt"]),
        ]
        # sub/ is absent because it has no files inside it either.

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        """The returned list is sorted alphabetically by parent path.

        Covers the ``sorted(grouped.items())`` return statement: groups
        appear in lexicographic order regardless of the order they were
        populated.
        """
        mod, _ = _import_setup()
        b_dir = tmp_path / "b_skills"
        a_dir = tmp_path / "a_skills"
        a_sub = a_dir / "sub"
        a_dir.mkdir(parents=True)
        a_sub.mkdir(parents=True)
        b_dir.mkdir(parents=True)
        (a_dir / "b.txt").touch()
        (a_sub / "a.txt").touch()
        (b_dir / "z.txt").touch()

        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree(".")

        # a_skills comes before a_skills/sub (which is a prefix) before b_skills
        assert result == [
            ("a_skills", ["a_skills/b.txt"]),
            ("a_skills/sub", ["a_skills/sub/a.txt"]),
            ("b_skills", ["b_skills/z.txt"]),
        ]

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """A directory that does not exist returns [].

        rglob on a nonexistent path returns an empty iterator, so the
        function returns [] with no error.
        """
        mod, _ = _import_setup()
        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("nonexistent")
        assert result == []

    def test_multiple_files_same_group(self, tmp_path: Path) -> None:
        """Multiple files in the same parent appear in the same group.

        Ensures the list append in grouped[key].append(...) works
        correctly when multiple files share a parent.
        """
        mod, _ = _import_setup()
        d = tmp_path / "pkg"
        d.mkdir(parents=True)
        (d / "mod1.py").touch()
        (d / "mod2.py").touch()
        (d / "mod3.py").touch()

        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("pkg")

        assert result == [
            ("pkg", ["pkg/mod1.py", "pkg/mod2.py", "pkg/mod3.py"]),
        ]


class TestModuleLevelBehavior:
    """Tests for the module-level setup() call and its imports."""

    def test_setup_called_once(self) -> None:
        """setup() is called exactly once at module import time."""
        _, mock_setup = _import_setup()
        mock_setup.assert_called_once()

    def test_setup_receives_data_files(self) -> None:
        """setup() is invoked with a data_files keyword argument."""
        _, mock_setup = _import_setup()
        _, kwargs = mock_setup.call_args
        assert "data_files" in kwargs
        assert isinstance(kwargs["data_files"], list)

    def test_repo_root_is_path(self) -> None:
        """REPO_ROOT is a Path instance pointing to the project root."""
        mod, _ = _import_setup()
        assert isinstance(mod.REPO_ROOT, Path)

    def test_data_file_tree_signature(self) -> None:
        """_data_file_tree has the expected function signature."""
        mod, _ = _import_setup()
        sig = inspect.signature(mod._data_file_tree)
        assert "root_name" in sig.parameters
        assert sig.parameters["root_name"].annotation is str

    def test_data_file_tree_returns_correct_type(self, tmp_path: Path) -> None:
        """_data_file_tree returns list[tuple[str, list[str]]]."""
        mod, _ = _import_setup()
        d = tmp_path / "somedir"
        d.mkdir(parents=True)
        (d / "f.py").touch()
        with patch.object(mod, "REPO_ROOT", tmp_path):
            result = mod._data_file_tree("somedir")
        assert isinstance(result, list)
        if result:
            key, files = result[0]
            assert isinstance(key, str)
            assert isinstance(files, list)
            assert isinstance(files[0], str)
