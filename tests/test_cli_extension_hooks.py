"""Tests for protected HermesCLI TUI extension hooks.

Verifies that wrapper CLIs can extend the TUI via:
  - _get_extra_tui_widgets()
  - _register_extra_tui_keybindings()
  - _build_tui_layout_children()
without overriding run().
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

from prompt_toolkit.key_binding import KeyBindings


def _make_cli(**kwargs):
    """Create a HermesCLI with prompt_toolkit stubs (same pattern as test_cli_init)."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI(**kwargs)


class TestExtensionHookDefaults:
    def test_extra_tui_widgets_default_empty(self):
        cli = _make_cli()
        assert cli._get_extra_tui_widgets() == []

    def test_register_extra_tui_keybindings_default_noop(self):
        cli = _make_cli()
        kb = KeyBindings()
        result = cli._register_extra_tui_keybindings(kb, input_area=None)
        assert result is None
        assert kb.bindings == []

    def test_build_tui_layout_children_returns_all_widgets_in_order(self):
        cli = _make_cli()
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
            workspace_footer="workspace-footer",
            voice_status_bar="voice-status",
            completions_menu="completions-menu",
        )
        # First element is Window(height=0), rest are the named widgets
        assert children[1:] == [
            "sudo", "secret", "approval", "clarify", "spinner",
            "spacer", "status", "top-rule", "image-bar", "input-area",
            "bottom-rule", "workspace-footer", "voice-status", "completions-menu",
        ]


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
            workspace_footer="workspace-footer",
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

    def test_extra_keybindings_can_add_bindings(self):
        cli = _make_cli()
        kb = KeyBindings()

        def _custom_hook(kb, *, input_area):
            @kb.add("f2")
            def _toggle(event):
                return None

        cli._register_extra_tui_keybindings = _custom_hook
        cli._register_extra_tui_keybindings(kb, input_area=None)
        assert len(kb.bindings) == 1


class TestWorkspaceFooter:
    def test_workspace_footer_shows_github_repo_and_branch(self, monkeypatch):
        cli = _make_cli()

        monkeypatch.setattr("os.getcwd", lambda: "/Users/test/work/hermes-agent")

        def _fake_run(cmd, capture_output, text, timeout, cwd):
            if cmd == ["git", "rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="true\n", stderr="")
            if cmd == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="feat/session-continuity\n", stderr="")
            if cmd == ["git", "config", "--get", "remote.origin.url"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="git@github.com:0xNyk/hermes-agent.git\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", _fake_run)
        cli._workspace_footer_cache = None
        cli._workspace_footer_last_refresh = 0.0

        fragments = cli._get_workspace_footer_fragments()
        rendered = "".join(text for _, text in fragments)

        assert "/Users/test/work/hermes-agent" in rendered
        assert "0xNyk/hermes-agent" in rendered
        assert "feat/session-continuity" in rendered
