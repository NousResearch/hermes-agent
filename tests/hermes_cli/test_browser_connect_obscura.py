"""Tests for Obscura browser support in hermes_cli.browser_connect.

Obscura is a lightweight Rust headless browser that exposes a Chrome DevTools
Protocol server via `obscura serve --port <N>`, unlike Chromium-family browsers
which use `--remote-debugging-port`. These tests guard the `_browser_debug_argv`
dispatch helper and the obscura entry in `_LINUX_BROWSER_GROUPS`.
"""

from __future__ import annotations

import shutil

import pytest

from hermes_cli.browser_connect import (
    _browser_debug_argv,
    get_chrome_debug_candidates,
)


class TestBrowserDebugArgv:
    """_browser_debug_argv dispatches per-binary."""

    def test_obscura_returns_serve_argv(self):
        """Obscura uses `obscura serve --port <N>` instead of --remote-debugging-port."""
        argv = _browser_debug_argv("/usr/local/bin/obscura", 9222)
        assert argv == ["/usr/local/bin/obscura", "serve", "--port", "9222"]

    def test_obscura_case_insensitive(self):
        """Binary name matching is case-insensitive."""
        argv = _browser_debug_argv("/x/OBsCuRa", 9222)
        assert argv == ["/x/OBsCuRa", "serve", "--port", "9222"]

    def test_chrome_returns_remote_debugging_argv(self):
        """Chromium-family browsers use --remote-debugging-port."""
        argv = _browser_debug_argv("/usr/bin/google-chrome", 9222)
        assert argv[0] == "/usr/bin/google-chrome"
        assert any("--remote-debugging-port=9222" in arg for arg in argv)
        assert any("--user-data-dir=" in arg for arg in argv)
        assert "--no-first-run" in argv
        assert "--no-default-browser-check" in argv

    def test_chromium_returns_remote_debugging_argv(self):
        """Chromium also uses --remote-debugging-port."""
        argv = _browser_debug_argv("/usr/bin/chromium", 9222)
        assert argv[0] == "/usr/bin/chromium"
        assert "--remote-debugging-port=9222" in argv

    def test_brave_returns_remote_debugging_argv(self):
        """Brave also uses --remote-debugging-port."""
        argv = _browser_debug_argv("/usr/bin/brave-browser", 9222)
        assert argv[0] == "/usr/bin/brave-browser"
        assert "--remote-debugging-port=9222" in argv


class TestObscuraInLinuxCandidates:
    """get_chrome_debug_candidates includes obscura when available."""

    def test_obscura_included_when_which_resolves(self, monkeypatch):
        """When shutil.which('obscura') finds it, obscura is in candidates."""
        def fake_which(name):
            if name == "obscura":
                return "/home/user/.local/bin/obscura"
            return None

        monkeypatch.setattr(shutil, "which", fake_which)
        # Mock os.path.isfile to return True for the which-resolved path
        import os
        original_isfile = os.path.isfile
        def fake_isfile(path):
            if path == "/home/user/.local/bin/obscura":
                return True
            return original_isfile(path)
        monkeypatch.setattr(os.path, "isfile", fake_isfile)

        candidates = get_chrome_debug_candidates("Linux")
        assert "/home/user/.local/bin/obscura" in candidates

    def test_obscura_not_included_when_which_fails(self, monkeypatch):
        """When shutil.which('obscura') returns None, obscura is not in candidates."""
        def fake_which(name):
            return None

        monkeypatch.setattr(shutil, "which", fake_which)
        # Mock os.path.isfile to return False for all obscura paths
        import os
        original_isfile = os.path.isfile
        def fake_isfile(path):
            if "obscura" in path:
                return False
            return original_isfile(path)
        monkeypatch.setattr(os.path, "isfile", fake_isfile)

        candidates = get_chrome_debug_candidates("Linux")
        assert not any("obscura" in c for c in candidates)

    def test_obscura_static_paths_included_when_present(self, monkeypatch):
        """Static paths like /usr/local/bin/obscura are included when they exist."""
        def fake_which(name):
            return None

        monkeypatch.setattr(shutil, "which", fake_which)
        # Mock os.path.isfile to return True for /usr/local/bin/obscura
        import os
        original_isfile = os.path.isfile
        def fake_isfile(path):
            if path == "/usr/local/bin/obscura":
                return True
            return original_isfile(path)
        monkeypatch.setattr(os.path, "isfile", fake_isfile)

        candidates = get_chrome_debug_candidates("Linux")
        assert "/usr/local/bin/obscura" in candidates
