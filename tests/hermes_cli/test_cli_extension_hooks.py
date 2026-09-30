"""Tests for protected HermesCLI TUI extension hooks.

Verifies that wrapper CLIs can extend the TUI via:
  - _get_extra_tui_widgets()
  - _register_extra_tui_keybindings()
  - _build_tui_layout_children()
without overriding run().
"""

from __future__ import annotations


def _make_cli():
    """Bare TUI mixin: the layout hooks need no initialized CLI state."""
    from hermes_cli.cli_tui_mixin import CLITuiMixin

    return CLITuiMixin()


def _layout_kwargs():
    return {
        "sudo_widget": "sudo",
        "secret_widget": "secret",
        "approval_widget": "approval",
        "clarify_widget": "clarify",
        "spinner_widget": "spinner",
        "spacer": "spacer",
        "status_bar": "status",
        "input_rule_top": "top-rule",
        "image_bar": "image-bar",
        "input_area": "input-area",
        "input_rule_bot": "bottom-rule",
        "voice_status_bar": "voice-status",
        "completions_menu": "completions-menu",
    }


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


class TestPluginDockHook:
    def test_claimed_container_is_inserted_immediately_after_spacer(self, monkeypatch):
        from agent.shell_hooks import _parse_hooks_block
        from hermes_cli import lifecycle
        from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS
        from prompt_toolkit.layout import Window

        dock = Window()
        later_dock = Window()

        class DockProvider:
            def __pt_container__(self):
                return dock

        calls = []

        def invoke_hook(name, **kwargs):
            calls.append((name, kwargs))
            return [DockProvider(), later_dock]

        monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
        children = _make_cli()._build_tui_layout_children(**_layout_kwargs())

        assert calls and len(calls) == 1
        assert calls[0][0] == "render_cli_dock"
        assert calls[0][1]["platform"] == "cli"
        assert callable(calls[0][1]["invalidate"])
        assert "render_cli_dock" in VALID_HOOKS
        assert "render_cli_dock" in SHELL_UNSUPPORTED_HOOKS
        assert _parse_hooks_block(
            {"render_cli_dock": [{"command": "/tmp/unsupported.sh"}]}
        ) == []
        spacer_idx = children.index("spacer")
        assert children[spacer_idx + 1] is dock
        assert children[spacer_idx + 2] is later_dock
        assert children[spacer_idx + 3] == "status"

    def test_malformed_returns_are_ignored(self, monkeypatch):
        from hermes_cli import lifecycle

        class BrokenProvider:
            def __pt_container__(self):
                raise RuntimeError("broken container")

        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda *_args, **_kwargs: ["text", object(), BrokenProvider()],
        )
        children = _make_cli()._build_tui_layout_children(**_layout_kwargs())

        assert children[children.index("spacer") + 1] == "status"

    def test_plugin_exception_does_not_hide_later_dock(self, monkeypatch, tmp_path):
        from hermes_cli import lifecycle, plugins
        from prompt_toolkit.layout import Window

        dock = Window()
        manager = plugins.PluginManager(scope_key=str(tmp_path))

        def broken_plugin(**_kwargs):
            raise RuntimeError("plugin failed")

        manager._hooks["render_cli_dock"] = [broken_plugin, lambda **_kwargs: dock]
        monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0)
        monkeypatch.setattr(lifecycle, "invoke_hook", manager.invoke_hook)

        children = _make_cli()._build_tui_layout_children(**_layout_kwargs())

        assert children[children.index("spacer") + 1] is dock

    def test_no_return_preserves_legacy_layout_identity(self, monkeypatch):
        from hermes_cli import cli_tui_mixin, lifecycle

        cli = _make_cli()
        zero_height_window = object()
        kwargs = {name: object() for name in _layout_kwargs()}
        monkeypatch.setattr(cli_tui_mixin, "Window", lambda **_kwargs: zero_height_window)

        cli._get_extra_tui_widgets = lambda: []
        legacy_children = cli._build_tui_layout_children(**kwargs)
        del cli._get_extra_tui_widgets

        monkeypatch.setattr(lifecycle, "invoke_hook", lambda *_args, **_kwargs: [])
        plugin_children = cli._build_tui_layout_children(**kwargs)

        assert len(plugin_children) == len(legacy_children)
        assert all(current is legacy for current, legacy in zip(plugin_children, legacy_children))
