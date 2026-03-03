"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(**kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    from cli import HermesCLI
    with patch("cli.get_tool_definitions", return_value=[]):
        return HermesCLI(**kwargs)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_integer(self):
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns > 0

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25

    def test_none_max_turns_gets_default(self):
        cli = _make_cli(max_turns=None)
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns > 0

    def test_env_var_max_turns(self, monkeypatch):
        """Env var is used when config file doesn't set max_turns."""
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "42")
        import cli as cli_module
        original_agent = cli_module.CLI_CONFIG["agent"].get("max_turns")
        original_root = cli_module.CLI_CONFIG.get("max_turns")
        cli_module.CLI_CONFIG["agent"]["max_turns"] = None
        cli_module.CLI_CONFIG.pop("max_turns", None)
        try:
            cli_obj = _make_cli()
            assert cli_obj.max_turns == 42
        finally:
            if original_agent is not None:
                cli_module.CLI_CONFIG["agent"]["max_turns"] = original_agent
            if original_root is not None:
                cli_module.CLI_CONFIG["max_turns"] = original_root

    def test_max_turns_never_none_for_agent(self):
        """The value passed to AIAgent must never be None (causes TypeError in run_conversation)."""
        cli = _make_cli()
        assert cli.max_turns is not None


class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in ("off", "new", "all", "verbose")


class TestProviderResolution:
    def test_api_key_is_string_or_none(self):
        cli = _make_cli()
        assert cli.api_key is None or isinstance(cli.api_key, str)

    def test_base_url_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.base_url, str)
        assert cli.base_url.startswith("http")

    def test_model_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.model, str)
        assert len(cli.model) > 0
