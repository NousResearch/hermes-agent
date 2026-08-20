"""Tests for display.memory_notifications handling in the classic CLI.

The messaging gateway (gateway/run.py) and the TUI/desktop backend
(tui_gateway/server.py) honour ``display.memory_notifications``
(off | on | verbose) when surfacing the background self-improvement
review summary. The classic CLI ignored the setting and always behaved
as "on" (AIAgent's hardcoded default), so a user-configured ``off`` was
silently ignored. These tests lock in the parity fix.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.cli.test_cli_init import _make_cli


def _cli_with_display(display_overrides):
    display = {"compact": False, "tool_progress": "all"}
    display.update(display_overrides)
    return _make_cli(config_overrides={"display": display})


class TestMemoryNotificationsConfig:
    def test_default_is_on(self):
        cli = _make_cli()
        assert cli.memory_notifications == "on"

    def test_off_is_honored(self):
        cli = _cli_with_display({"memory_notifications": "off"})
        assert cli.memory_notifications == "off"

    def test_verbose_is_honored(self):
        cli = _cli_with_display({"memory_notifications": "verbose"})
        assert cli.memory_notifications == "verbose"

    def test_yaml_bool_false_normalized_to_off(self):
        # YAML 1.1 parses bare `off` as boolean False.
        cli = _cli_with_display({"memory_notifications": False})
        assert cli.memory_notifications == "off"

    def test_yaml_bool_true_normalized_to_on(self):
        cli = _cli_with_display({"memory_notifications": True})
        assert cli.memory_notifications == "on"

    def test_mixed_case_normalized(self):
        cli = _cli_with_display({"memory_notifications": "OFF"})
        assert cli.memory_notifications == "off"

    def test_per_platform_cli_override_wins(self):
        cli = _cli_with_display(
            {
                "memory_notifications": "on",
                "platforms": {"cli": {"memory_notifications": "off"}},
            }
        )
        assert cli.memory_notifications == "off"

    def test_other_platform_override_ignored(self):
        cli = _cli_with_display(
            {
                "memory_notifications": "off",
                "platforms": {"weixin": {"memory_notifications": "on"}},
            }
        )
        assert cli.memory_notifications == "off"
