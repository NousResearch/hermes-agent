"""Tests for the /models slash command."""

from __future__ import annotations

import pytest

from hermes_cli.commands import COMMAND_REGISTRY, resolve_command


class TestModelsCommandRegistry:
    """Verify /models is registered correctly."""

    def test_models_command_exists(self):
        """The /models command should be registered."""
        cmd = resolve_command("models")
        assert cmd is not None
        assert cmd.name == "models"

    def test_models_command_category(self):
        """The /models command should be in Configuration category."""
        cmd = resolve_command("models")
        assert cmd.category == "Configuration"

    def test_models_accepts_provider_arg(self):
        """The /models command should accept a [provider] argument."""
        cmd = resolve_command("models")
        assert "[provider]" in cmd.args_hint

    def test_models_not_cli_only(self):
        """The /models command should work in both CLI and gateway."""
        cmd = resolve_command("models")
        assert cmd.cli_only is False
        assert cmd.gateway_only is False


class TestModelsCommandInRegistry:
    """Verify /models appears in the full registry."""

    def test_models_in_registry(self):
        """The /models command should be in COMMAND_REGISTRY."""
        names = [cmd.name for cmd in COMMAND_REGISTRY]
        assert "models" in names

    def test_models_near_provider(self):
        """The /models command should be near /provider in the registry."""
        names = [cmd.name for cmd in COMMAND_REGISTRY]
        provider_idx = names.index("provider")
        models_idx = names.index("models")
        # Should be within a few entries of each other
        assert abs(models_idx - provider_idx) <= 3
