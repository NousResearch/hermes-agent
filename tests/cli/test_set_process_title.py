"""Tests for _set_process_title — verifies graceful handling when ctypes
is unavailable (NousResearch/hermes-agent#42074)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSetProcessTitleCtypesUnavailable:
    """When the _ctypes C extension is missing, _set_process_title must
    return without raising — the function is purely cosmetic."""

    def test_returns_gracefully_when_ctypes_missing(self):
        """Simulate a Python installation without _ctypes (e.g. pyenv-built
        Python compiled before libffi-dev was installed)."""
        import hermes_cli.main as mod

        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "ctypes":
                raise ImportError("No module named '_ctypes'")
            return real_import(name, *args, **kwargs)

        # setproctitle not installed → falls through to ctypes path
        with patch.dict("sys.modules", {"setproctitle": None}):
            with patch("builtins.__import__", side_effect=_fake_import):
                # Must not raise
                mod._set_process_title()

    def test_works_normally_when_ctypes_available(self):
        """When ctypes works, the function should complete without error."""
        import hermes_cli.main as mod

        # Just verify it doesn't crash — we can't assert the process title
        # changed because test runners may not have prctl available.
        with patch.dict("sys.modules", {"setproctitle": None}):
            mod._set_process_title()

    def test_uses_setproctitle_when_available(self):
        """When setproctitle is installed, it should be preferred over ctypes."""
        import hermes_cli.main as mod

        mock_spt = type("MockSPT", (), {"setproctitle": staticmethod(lambda name: None)})()
        with patch.dict("sys.modules", {"setproctitle": mock_spt}):
            with patch("hermes_cli.main.setproctitle", mock_spt, create=True):
                # Should return early via setproctitle without touching ctypes
                mod._set_process_title()
