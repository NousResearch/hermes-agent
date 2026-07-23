"""Regression tests for computer_use permissions status JSON parsing (#65217).

After _json_out was hardened to return None on malformed JSON, _mac_permissions
must set an explicit error instead of silently leaving out["error"] as None.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


def _make_binary() -> str:
    """Return a dummy binary path."""
    return "/usr/local/bin/cua-driver"


class TestMacPermissionsMalformedJSON:
    """_mac_permissions must emit a diagnostic when stdout is not valid JSON."""

    def test_malformed_json_sets_error(self, monkeypatch):
        """When cua-driver returns non-JSON stdout, out['error'] is populated."""
        from tools.computer_use.permissions import _mac_permissions

        captured: dict = {"error": None}

        def fake_run(*args, **kwargs):
            m = MagicMock()
            m.stdout = "this is not json\n"
            m.returncode = 0
            return m

        with patch("tools.computer_use.permissions.subprocess.run", fake_run):
            _mac_permissions(_make_binary(), captured)

        assert captured.get("error") is not None
        assert "malformed JSON" in captured["error"]

    def test_empty_stdout_sets_error(self, monkeypatch):
        """Empty stdout returns None from _json_out which _mac_permissions
        treats as a parse failure — it should set an error."""
        from tools.computer_use.permissions import _mac_permissions

        captured: dict = {"error": None}

        def fake_run(*args, **kwargs):
            m = MagicMock()
            m.stdout = ""
            m.returncode = 0
            return m

        with patch("tools.computer_use.permissions.subprocess.run", fake_run):
            _mac_permissions(_make_binary(), captured)

        assert captured.get("error") is not None

    def test_valid_json_updates_bools(self, monkeypatch):
        """Valid JSON response updates the expected boolean keys."""
        import json
        from tools.computer_use.permissions import _BOOLS, _mac_permissions

        captured: dict = {k: None for k in _BOOLS}
        captured["error"] = None

        def fake_run(*args, **kwargs):
            m = MagicMock()
            m.stdout = json.dumps({"accessibility": True, "screen_recording": False})
            m.returncode = 0
            return m

        with patch("tools.computer_use.permissions.subprocess.run", fake_run):
            _mac_permissions(_make_binary(), captured)

        assert captured["accessibility"] is True
        assert captured["screen_recording"] is False
        assert captured.get("error") is None
