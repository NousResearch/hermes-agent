"""Regression for the #8751 class: parent-walking marker probes must treat an
unreadable ancestor as 'no marker here', never raise.

The merged prompt_builder fix (337113c1b8) closed the ``.git`` walk; the same
crash shape remained in the two other parent-walking marker probes —
checkpoint_manager's project-marker walk and coding_context's ``_marker_root``.
Both walk toward the filesystem root, where a mode-700 parent for another user
makes every probe under it raise PermissionError out of tool/turn construction.

The denied directory is mode 700 for another user: the dir itself AND every
path under it raise on stat. The walk must read those probes as False and
continue upward — a marker ABOVE the denied ancestor is still found.
"""

from pathlib import Path

import pytest


def _deny_dir_and_descendants(monkeypatch, denied: Path):
    """Path.exists() raises PermissionError for *denied* and anything under it."""
    real_exists = Path.exists

    def fake_exists(self, *args, **kwargs):
        try:
            if self == denied or denied in self.parents:
                raise PermissionError(f"denied: {self}")
        except TypeError:
            pass
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", fake_exists)


class TestCheckpointWorkingDirWalk:
    def test_fully_denied_tree_falls_back_to_own_dir(self, tmp_path, monkeypatch):
        from tools.checkpoint_manager import CheckpointManager

        work = tmp_path / "work"
        work.mkdir()
        _deny_dir_and_descendants(monkeypatch, tmp_path)

        mgr = CheckpointManager.__new__(CheckpointManager)
        assert mgr.get_working_dir_for_path(str(work)) == str(work)

    def test_marker_above_denied_ancestor_still_found(self, tmp_path, monkeypatch):
        from tools.checkpoint_manager import CheckpointManager

        project = tmp_path / "project"
        (project / ".git").mkdir(parents=True)
        work = project / "a" / "b"
        work.mkdir(parents=True)
        # 'a' is mode 700 for another user: every probe under it raises. The walk
        # must read those as 'no marker' and CONTINUE upward — .git at project wins.
        _deny_dir_and_descendants(monkeypatch, work.parent)

        mgr = CheckpointManager.__new__(CheckpointManager)
        assert mgr.get_working_dir_for_path(str(work / "f.txt")) == str(project)


class TestCodingContextMarkerRoot:
    def test_fully_denied_tree_reads_as_no_root(self, tmp_path, monkeypatch):
        import agent.coding_context as cc

        work = tmp_path / "work"
        work.mkdir()
        _deny_dir_and_descendants(monkeypatch, tmp_path)

        assert cc._marker_root(work) is None

    def test_marker_above_denied_ancestor_still_found(self, tmp_path, monkeypatch):
        import agent.coding_context as cc

        project = tmp_path / "proj2"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        work = project / "sub" / "deep"
        work.mkdir(parents=True)
        _deny_dir_and_descendants(monkeypatch, project / "sub")

        assert cc._marker_root(work) == project


class TestGitRootShared:
    def test_git_root_skips_denied_ancestor_and_resolves(self, tmp_path, monkeypatch):
        from agent.prompt_builder import _find_git_root

        project = tmp_path / "repo"
        (project / ".git").mkdir(parents=True)
        work = project / "x"
        work.mkdir(parents=True)
        # 'x' itself is unreadable: the walk must read 'no .git' there and continue to repo.
        _deny_dir_and_descendants(monkeypatch, work)

        assert _find_git_root(work) == project
