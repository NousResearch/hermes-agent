"""Regression tests for TUILayoutMixin (extracted verbatim from cli.py).

Wave 1 godfile extraction, shard s5 cluster c2 -> hermes_cli/tui_layout_mixin.py.
These tests exercise the moved methods directly on the bare mixin (no cli.py
import, no prompt_toolkit stubs beyond the real ``Window``) so they pin the
extracted code itself. The existing tests/cli/test_cli_extension_hooks.py
continues to cover the same methods via the HermesCLI MRO.
"""

from __future__ import annotations

from hermes_cli.tui_layout_mixin import TUILayoutMixin


class TestTUILayoutMixinDefaults:
    def test_extra_tui_widgets_default_empty(self):
        mixin = TUILayoutMixin()
        assert mixin._get_extra_tui_widgets() == []

    def test_register_extra_tui_keybindings_default_noop(self):
        mixin = TUILayoutMixin()
        # Body is docstring-only: returns None and registers nothing.
        result = mixin._register_extra_tui_keybindings(None, input_area=None)
        assert result is None

    def test_build_tui_layout_children_returns_all_widgets_in_order(self):
        mixin = TUILayoutMixin()
        children = mixin._build_tui_layout_children(
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
        # First element is the zero-height Window, rest are the named widgets.
        assert children[1:] == [
            "sudo", "secret", "approval", "clarify", "spinner",
            "spacer", "status", "top-rule", "image-bar", "input-area",
            "bottom-rule", "voice-status", "completions-menu",
        ]
        assert children[0].__class__.__name__ == "Window"

    def test_build_tui_layout_children_omits_none_widgets(self):
        mixin = TUILayoutMixin()
        children = mixin._build_tui_layout_children(
            sudo_widget=None,
            secret_widget="secret",
            approval_widget=None,
            clarify_widget="clarify",
            spacer="spacer",
            status_bar="status",
            input_rule_top="top-rule",
            image_bar="image-bar",
            input_area="input-area",
            input_rule_bot="bottom-rule",
            voice_status_bar="voice-status",
            completions_menu="completions-menu",
        )
        assert "sudo" not in children
        assert "approval" not in children
        assert children[0].__class__.__name__ == "Window"


class TestTUILayoutMixinWidgetInjection:
    def test_extra_widgets_inserted_between_spacer_and_status_bar(self):
        mixin = TUILayoutMixin()
        mixin._get_extra_tui_widgets = lambda: ["radio-menu", "mini-player"]

        children = mixin._build_tui_layout_children(
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
        spacer_idx = children.index("spacer")
        status_idx = children.index("status")
        assert children[spacer_idx + 1] == "radio-menu"
        assert children[spacer_idx + 2] == "mini-player"
        assert children[spacer_idx + 3] == "status"
        assert status_idx == spacer_idx + 3

    def test_pet_and_stash_widgets_are_getattr_guarded(self):
        """_pet_widget / _stash_panel_widget are read via getattr so a bare
        mixin (or a HermesCLI before run() sets them) yields None and the
        widgets are omitted from the layout."""
        mixin = TUILayoutMixin()
        children = mixin._build_tui_layout_children(
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
        assert "pet-widget" not in children
        assert "stash-widget" not in children

        mixin._pet_widget = "pet-widget"
        mixin._stash_panel_widget = "stash-widget"
        children = mixin._build_tui_layout_children(
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
        assert children[children.index("spacer") + 1] == "pet-widget"
        assert children[children.index("pet-widget") + 1] == "stash-widget"
        assert children[children.index("stash-widget") + 1] == "status"
