"""Tests for protected HermesCLI TUI extension hooks.

Verifies that wrapper CLIs can extend the TUI via:
  - _get_extra_tui_widgets()
  - _register_extra_tui_keybindings()
  - _build_tui_layout_children()
without overriding run().
"""

from __future__ import annotations

def _make_cli():
    """Bare HermesCLI (no __init__): the layout hooks need no init state."""
    from cli import HermesCLI

    return HermesCLI.__new__(HermesCLI)

class TestExtensionHookSubclass:
    def test_extra_widgets_inserted_before_status_bar(self):
        cli = _make_cli()
        # Monkey-patch to simulate subclass override
        cli._get_extra_tui_widgets = lambda: ["radio-menu", "mini-player"]

        children = cli._build_tui_layout_children(
            sudo_widget="sudo",
            secret_widget="secret",
            approval_widget="approval",
            clarify_widget="clarify",
            spinner_widget="spinner",
            spacer="spacer",
            status_bar="status",
            input_rule_top="top-rule",
            image_bar="image-bar",
            input_area="input-area",
            input_rule_bot="bottom-rule",
            voice_status_bar="voice-status",
            completions_menu="completions-menu",
        )
        # Extra widgets should appear between spacer and status bar
        spacer_idx = children.index("spacer")
        status_idx = children.index("status")
        assert children[spacer_idx + 1] == "radio-menu"
        assert children[spacer_idx + 2] == "mini-player"
        assert children[spacer_idx + 3] == "status"
        assert status_idx == spacer_idx + 3
