"""Tests for `hermes memory setup [provider]` routing.

The `memory setup` subcommand accepts an optional positional ``provider`` so a
fresh install can configure a specific provider directly (e.g.
``hermes memory setup honcho``) without the interactive picker — which matters
because the per-provider ``hermes <provider>`` subcommand is only registered
once that provider is active.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import memory_setup


class TestMemorySetupProviderRouting:
    def test_setup_with_provider_arg_skips_picker(self):
        """`memory setup honcho` routes straight to cmd_setup_provider."""
        args = SimpleNamespace(memory_command="setup", provider="honcho")
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        direct.assert_called_once_with("honcho")
        picker.assert_not_called()

    def test_setup_without_provider_runs_picker(self):
        """`memory setup` (no provider) runs the interactive picker."""
        args = SimpleNamespace(memory_command="setup", provider=None)
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        picker.assert_called_once_with(args)
        direct.assert_not_called()

    def test_setup_with_missing_provider_attr_runs_picker(self):
        """A SimpleNamespace lacking `provider` must not crash — fall back to picker."""
        args = SimpleNamespace(memory_command="setup")
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        picker.assert_called_once_with(args)
        direct.assert_not_called()

    def test_unknown_provider_reports_and_returns_early(self, capsys):
        """An unknown provider name surfaces a helpful message and returns
        before any config load/save (the not-found guard precedes those imports)."""
        memory_setup.cmd_setup_provider("notaprovider")
        out = capsys.readouterr().out
        assert "not found" in out
        assert "hermes memory setup" in out
