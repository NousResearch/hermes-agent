"""Regression: Discord UI view classes must be defined after lazy-install.

When discord.py is NOT installed at module load time, the
``if DISCORD_AVAILABLE:`` guard at the bottom of gateway/platforms/discord.py
evaluates to False and is skipped — leaving ExecApprovalView and its four
siblings undefined in the module globals.

check_discord_requirements() must call _define_discord_view_classes() after
a successful lazy install so that all view classes are available the moment
DISCORD_AVAILABLE flips to True.  Without this, the first button interaction
(exec approval, slash confirm, etc.) raises NameError even though
DISCORD_AVAILABLE=True.

Fixes: lazy-install path NameError for ExecApprovalView, SlashConfirmView,
UpdatePromptView, ModelPickerView, ClarifyChoiceView.
"""
import importlib
from unittest.mock import patch


_VIEW_NAMES = [
    "ExecApprovalView",
    "SlashConfirmView",
    "UpdatePromptView",
    "ModelPickerView",
    "ClarifyChoiceView",
]


class TestDefineDiscordViewClasses:
    """_define_discord_view_classes() registers all UI view classes in module globals."""

    def test_registers_all_five_view_classes(self, monkeypatch):
        """Calling _define_discord_view_classes() must (re)define all 5 view classes."""
        dp = importlib.import_module("plugins.platforms.discord.adapter")

        # Remove the classes to simulate the state where the module was loaded
        # with DISCORD_AVAILABLE=False (the lazy-install scenario).
        for name in _VIEW_NAMES:
            monkeypatch.delattr(dp, name)

        # Pre-condition: classes are gone
        for name in _VIEW_NAMES:
            assert not hasattr(dp, name), f"{name} should be absent before the call"

        dp._define_discord_view_classes()

        for name in _VIEW_NAMES:
            assert hasattr(dp, name), f"{name} must be defined after _define_discord_view_classes()"
            assert isinstance(getattr(dp, name), type), f"{name} must be a class"

