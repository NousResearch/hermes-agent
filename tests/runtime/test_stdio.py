"""Tests for runtime-owned Windows stdio configuration."""

from __future__ import annotations

import io

import pytest

from runtime import stdio


class TestConfigureWindowsStdio:
    """runtime.stdio.configure_windows_stdio wiring."""

    @pytest.fixture(autouse=True)
    def _reset_configured(self, monkeypatch):
        monkeypatch.setattr(stdio, "_CONFIGURED", False)

    def test_no_op_on_posix(self, monkeypatch):
        monkeypatch.setattr(stdio, "is_windows", lambda: False)

        result = stdio.configure_windows_stdio()

        assert result is False

    def test_reconfigure_stream_handles_missing_method(self):
        """StringIO-like objects without .reconfigure() must not blow up."""
        stdio._reconfigure_stream(io.StringIO())
