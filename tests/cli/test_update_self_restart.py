"""Tests for the Desktop self-restart helpers added by commit b795d7fc8 / 332c2f8c4.

These cover the public surface that's safe to call in isolation:
  * _DESKTOP_EXE_CANDIDATES is a stable, non-empty tuple
  * _find_desktop_exe_path() returns a real file when Hermes.exe is installed
  * _find_desktop_main_pids() returns only Hermes.exe PIDs that own a
    visible top-level window (so renderer / GPU / utility helpers are
    excluded)
  * _relaunch_desktop_after_update() no-ops on None / empty / missing
  * --no-auto-restart-desktop is registered as a CLI flag

The kill-and-relaunch flow is *not* exercised here (it would actually
kill the running Desktop). The end-to-end behavior was verified manually
against the live install: see the commit message on 332c2f8c4.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Import shim — main.py is a 15k-line god-file that has heavy top-level
# imports. Importing the module we need is fine; calling the entry points
# is not. We import the specific functions under test only.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def update_mod():
    """Lazily import hermes_cli.main so test collection is fast and side-effect free."""
    from hermes_cli import main  # noqa: F401

    return main


# ---------------------------------------------------------------------------
# _DESKTOP_EXE_CANDIDATES
# ---------------------------------------------------------------------------
class TestDesktopExeCandidates:
    def test_is_tuple(self, update_mod):
        assert isinstance(update_mod._DESKTOP_EXE_CANDIDATES, tuple)

    def test_non_empty(self, update_mod):
        assert len(update_mod._DESKTOP_EXE_CANDIDATES) > 0

    def test_all_end_in_exe(self, update_mod):
        for name in update_mod._DESKTOP_EXE_CANDIDATES:
            assert name.lower().endswith(".exe"), f"{name!r} should end in .exe"

    def test_includes_hermes(self, update_mod):
        assert "Hermes.exe" in update_mod._DESKTOP_EXE_CANDIDATES


# ---------------------------------------------------------------------------
# _find_desktop_exe_path
# ---------------------------------------------------------------------------
class TestFindDesktopExePath:
    def test_finds_real_install(self, update_mod, tmp_path, monkeypatch):
        """When the desktop is installed at the canonical path, return it."""
        # Create a fake Hermes.exe so the function has something to find.
        fake_hermes = (
            tmp_path
            / "apps"
            / "desktop"
            / "release"
            / "win-unpacked"
            / "Hermes.exe"
        )
        fake_hermes.parent.mkdir(parents=True)
        fake_hermes.write_bytes(b"MZ")  # PE magic — enough to be a "file"

        # Redirect PROJECT_ROOT to our fake tree.
        monkeypatch.setattr(update_mod, "PROJECT_ROOT", tmp_path)
        # Make os.environ lookups for LOCALAPPDATA / ProgramFiles miss so
        # only our fake path can be discovered.
        for k in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            monkeypatch.delenv(k, raising=False)

        result = update_mod._find_desktop_exe_path()
        assert result is not None
        assert Path(result).resolve() == fake_hermes.resolve()

    def test_returns_none_when_no_install(self, update_mod, tmp_path, monkeypatch):
        """No candidate exists → returns None, never raises."""
        monkeypatch.setattr(update_mod, "PROJECT_ROOT", tmp_path)
        for k in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            monkeypatch.delenv(k, raising=False)
        # Make psutil query raise too so the last-resort path returns None
        with mock.patch("psutil.process_iter", side_effect=OSError("nope")):
            assert update_mod._find_desktop_exe_path() is None


# ---------------------------------------------------------------------------
# _relaunch_desktop_after_update — no-op safety paths
# ---------------------------------------------------------------------------
class TestRelaunchDesktopAfterUpdate:
    def test_none_token_is_noop(self, update_mod, capsys):
        update_mod._relaunch_desktop_after_update(None)
        # Should print nothing.
        captured = capsys.readouterr()
        assert "Restarting" not in captured.out
        assert captured.err == ""

    def test_empty_token_is_noop(self, update_mod, capsys):
        update_mod._relaunch_desktop_after_update({})
        captured = capsys.readouterr()
        assert "Restarting" not in captured.out

    def test_missing_exe_field_is_noop(self, update_mod, capsys):
        update_mod._relaunch_desktop_after_update({"exe": ""})
        captured = capsys.readouterr()
        assert "Restarting" not in captured.out

    def test_nonexistent_exe_is_noop(self, update_mod, capsys):
        update_mod._relaunch_desktop_after_update({"exe": r"C:\nope\fake.exe"})
        captured = capsys.readouterr()
        assert "Restarting" not in captured.out

    def test_real_exe_does_not_crash(self, update_mod, tmp_path, monkeypatch, capsys):
        """With a real existing exe, Popen is invoked and a success line is printed.

        We stub subprocess.Popen so the test does not actually relaunch the
        desktop during the test run.
        """
        fake_hermes = tmp_path / "Hermes.exe"
        fake_hermes.write_bytes(b"MZ")
        called = {}

        def fake_popen(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return mock.MagicMock()

        monkeypatch.setattr(update_mod.subprocess, "Popen", fake_popen)

        update_mod._relaunch_desktop_after_update({"exe": str(fake_hermes)})
        captured = capsys.readouterr()

        assert "Restarting" in captured.out
        assert "Desktop relaunched" in captured.out or "Relaunch" in captured.out
        assert called, "Popen was not invoked"


# ---------------------------------------------------------------------------
# _find_desktop_main_pids — only main-window PIDs
# ---------------------------------------------------------------------------
class TestFindDesktopMainPids:
    def test_empty_when_psutil_missing(self, update_mod):
        """Force the inner `import psutil` to fail and verify graceful empty return."""
        import sys as _sys

        original = _sys.modules.get("psutil")
        _sys.modules["psutil"] = None  # any attribute access raises AttributeError
        try:
            assert update_mod._find_desktop_main_pids() == []
        finally:
            if original is not None:
                _sys.modules["psutil"] = original
            else:
                _sys.modules.pop("psutil", None)

    def test_filters_by_window(self, update_mod):
        """Only PIDs whose process owns a visible top-level window are returned."""
        # A proc that pretends to be a Hermes.exe main (has window)
        main_proc = mock.MagicMock()
        main_proc.info = {"pid": 100, "name": "Hermes.exe"}
        # A proc that pretends to be a helper (no window)
        helper_proc = mock.MagicMock()
        helper_proc.info = {"pid": 200, "name": "Hermes.exe"}
        # A proc that pretends to be unrelated
        chrome_proc = mock.MagicMock()
        chrome_proc.info = {"pid": 300, "name": "chrome.exe"}

        fake_procs = [main_proc, helper_proc, chrome_proc]

        # Import psutil fresh inside the function — patch via sys.modules.
        import sys as _sys

        original_psutil = _sys.modules.get("psutil")
        fake_psutil = mock.MagicMock()
        fake_psutil.process_iter.return_value = iter(fake_procs)
        _sys.modules["psutil"] = fake_psutil
        try:
            with mock.patch.object(
                update_mod,
                "_process_has_main_window",
                side_effect=lambda p: p.info["pid"] == 100,
            ):
                result = update_mod._find_desktop_main_pids()
        finally:
            if original_psutil is not None:
                _sys.modules["psutil"] = original_psutil
            else:
                _sys.modules.pop("psutil", None)

        assert result == [100]


# ---------------------------------------------------------------------------
# _process_has_main_window — POSIX fast path
# ---------------------------------------------------------------------------
class TestProcessHasMainWindow:
    def test_posix_returns_true(self, update_mod, monkeypatch):
        """Off Windows, always True so the caller can fall back to its own filter."""
        monkeypatch.setattr(update_mod.sys, "platform", "linux")
        fake_proc = mock.MagicMock()
        fake_proc.pid = 1234
        assert update_mod._process_has_main_window(fake_proc) is True


# ---------------------------------------------------------------------------
# CLI flag registration
# ---------------------------------------------------------------------------
class TestUpdateCliFlags:
    def test_no_auto_restart_desktop_registered(self):
        from hermes_cli.subcommands.update import build_update_parser

        # Build a minimal fake subparsers that exposes add_argument + set_defaults.
        class _FakeParser:
            def __init__(self):
                self.kwargs = {}
                self.defaults = {}

            def add_argument(self, *args, **kwargs):
                self.kwargs[args[0]] = kwargs

            def set_defaults(self, **kwargs):
                self.defaults.update(kwargs)

        class _FakeSubparsers:
            def add_parser(self, name, **kwargs):
                self.parser = _FakeParser()
                self.parser._name = name
                return self.parser

        sub = _FakeSubparsers()
        build_update_parser(sub, cmd_update=lambda args: None)
        assert "--no-auto-restart-desktop" in sub.parser.kwargs
        assert sub.parser.kwargs["--no-auto-restart-desktop"]["default"] is False
        assert sub.parser.kwargs["--no-auto-restart-desktop"]["action"] == "store_true"
        # And the handler is wired.
        assert sub.parser.defaults.get("func") is not None
