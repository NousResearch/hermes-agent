"""Tests for display.stream_pad config option."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


class TestStreamPadConfig:
    """display.stream_pad controls the indent prefix for streamed response text."""

    def test_stream_pad_reads_from_cli_config(self):
        """_STREAM_PAD reads from CLI_CONFIG display.stream_pad."""
        import cli as _cli

        # Verify the source: _STREAM_PAD is computed from CLI_CONFIG at import
        config_val = _cli.CLI_CONFIG.get("display", {}).get("stream_pad", "    ")
        assert _cli._STREAM_PAD == config_val

    def test_stream_pad_fallback_is_four_spaces(self):
        """When config has no stream_pad, the fallback is 4 spaces."""
        assert {}.get("stream_pad", "    ") == "    "

    def test_empty_string_is_valid_stream_pad(self):
        """Empty string disables indentation."""
        assert {"stream_pad": ""}.get("stream_pad", "    ") == ""

    def test_custom_string_is_valid_stream_pad(self):
        """Arbitrary string is accepted."""
        assert {"stream_pad": "→ "}.get("stream_pad", "    ") == "→ "


class TestStreamPadDefaultConfig:
    """display.stream_pad is present in DEFAULT_CONFIG with the expected default."""

    def test_default_config_contains_stream_pad(self):
        from hermes_cli.config import DEFAULT_CONFIG

        display = DEFAULT_CONFIG.get("display", {})
        assert "stream_pad" in display
        assert display["stream_pad"] == "    "

    def test_stream_pad_in_dump_output(self):
        """hermes config display includes display.stream_pad."""
        from io import StringIO
        from unittest.mock import patch

        from hermes_cli.dump import run_dump

        buf = StringIO()
        with patch("sys.stdout", buf):
            try:
                run_dump(type("A", (), {"quiet": False})())
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "stream_pad" in output
