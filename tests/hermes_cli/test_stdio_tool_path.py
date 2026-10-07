r"""``hermes_cli.stdio._augment_path_with_known_tools``: the Hermes-managed tools it must expose.

The pinned Git-for-Windows lives in the PM store under a version-stamped entry dir, and
install.ps1 puts it on the installer process's PATH only. Nothing persists it, so a hermes started
any other way has to find it the way pm published it (#134600).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")


def _staged_git(tmp_path: Path) -> Path:
    r"""A PM store entry shaped like the real pinned git (``cmd`` holds git.exe, ``usr\bin`` bash)."""
    entry = tmp_path / "tools" / "git-2.53.0+3-win32-x64"
    for leaf in (("cmd",), ("bin",), ("usr", "bin")):
        (entry.joinpath(*leaf)).mkdir(parents=True)
    (entry / "cmd" / "git.exe").write_bytes(b"MZ")
    (entry / "usr" / "bin" / "bash.exe").write_bytes(b"MZ")
    return entry


def _fake_pm(monkeypatch, entry: Path | None) -> None:
    """Stand in for the ``pm`` package so the real lookup runs against a known published entry."""
    installed = None if entry is None else type("_Installed", (), {"path": entry})()
    monkeypatch.setitem(
        sys.modules, "pm",
        type("_PM", (), {"installed_package": staticmethod(
            lambda name, **_kw: installed if name == "git" else None)}))


class TestPinnedGitOnPath:
    def test_pm_pinned_git_dirs_are_prepended(self, tmp_path, monkeypatch):
        r"""A host whose only git is the pinned one: its ``cmd`` / ``bin`` / ``usr\bin`` must reach
        PATH, so ``git`` AND the ``bash.exe`` contract the tree depends on resolve. The prepend
        list named only the pre-PM ``<data root>\git\*`` dirs, none of which exist on such a host,
        and both tools were invisible."""
        from hermes_cli import stdio

        entry = _staged_git(tmp_path)
        _fake_pm(monkeypatch, entry)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-such-appdata"))
        monkeypatch.setenv("PATH", str(tmp_path / "unrelated"))

        stdio._augment_path_with_known_tools()

        entries = os.environ["PATH"].split(os.pathsep)
        assert str(entry / "cmd") in entries, "the pinned git never reached PATH"
        assert str(entry / "usr" / "bin") in entries, "the pinned bash.exe never reached PATH"
        # Prepended, not appended: the pinned git must win over anything already on PATH.
        assert entries.index(str(entry / "cmd")) < entries.index(str(tmp_path / "unrelated"))

    def test_nothing_published_leaves_path_alone(self, tmp_path, monkeypatch):
        """No pinned entry and no legacy dirs: PATH is untouched, never padded with
        directories that do not exist."""
        from hermes_cli import stdio

        _fake_pm(monkeypatch, None)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-such-appdata"))
        monkeypatch.setenv("PATH", str(tmp_path / "unrelated"))

        stdio._augment_path_with_known_tools()

        assert os.environ["PATH"] == str(tmp_path / "unrelated")

    def test_a_broken_pm_does_not_stop_console_setup(self, tmp_path, monkeypatch):
        """The lookup runs before logging exists and has nowhere to report: a pm that raises must
        degrade to the legacy dirs, never propagate out of console configuration."""
        from hermes_cli import stdio

        def _boom(_name, **_kw):
            raise RuntimeError("no lockfile")

        monkeypatch.setitem(sys.modules, "pm", type("_PM", (), {"installed_package": staticmethod(_boom)}))
        legacy = tmp_path / "hermes" / "git" / "cmd"
        legacy.mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", str(tmp_path / "unrelated"))

        stdio._augment_path_with_known_tools()

        assert str(legacy) in os.environ["PATH"].split(os.pathsep)
