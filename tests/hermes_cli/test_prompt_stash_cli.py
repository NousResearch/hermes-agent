"""Tests for the Ctrl+S prompt stash wiring inside HermesCLI.

The state machine itself is covered by tests/hermes_cli/test_prompt_stash.py. These
tests verify the cli.py side:
  - HermesCLI.__init__ creates a PromptStash
  - the layout hook makes room for the stash browse panel
  - _render_stash_panel renders bounded, display-width-correct rows
  - the status-bar indicator appears / disappears with stash contents

Follows the prompt_toolkit-stub construction pattern from
tests/hermes_cli/test_cli_extension_hooks.py.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

def _make_cli(**kwargs):
    """Create a HermesCLI with prompt_toolkit stubbed out."""
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

@pytest.fixture(scope="module")
def cli():
    return _make_cli()

class TestStashStateInit:

    def test_stash_is_per_instance_not_shared(self):
        """Two CLIs must not share one stash — drafts would leak across sessions."""
        a = _make_cli()
        b = _make_cli()
        a._prompt_stash.stash("only in a")
        assert len(a._prompt_stash) == 1
        assert len(b._prompt_stash) == 0

class TestLayoutSlot:
    def test_layout_includes_stash_panel_when_present(self, cli):
        cli._stash_panel_widget = "stash-panel"
        try:
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
            assert "stash-panel" in children
            # Panel sits directly above the status bar.
            assert children.index("stash-panel") < children.index("status")
        finally:
            cli._stash_panel_widget = None

    def test_layout_omits_stash_panel_when_absent(self, cli):
        cli._stash_panel_widget = None
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
        assert None not in children

class TestRenderStashPanel:
    """Contributor's panel renderer, now measured in display cells."""

    @staticmethod
    def _rows(cli, count=3, width=100):
        stash = type(cli._prompt_stash)()
        for i in range(count):
            stash.stash(f"draft number {i}")
        return cli._render_stash_panel(stash.panel_rows(), 0, width)

    @pytest.mark.parametrize("width", [16, 20, 40, 120])
    def test_no_line_exceeds_terminal_width(self, cli, width):
        """Rows must never bleed past the terminal — the bug the PR's three
        follow-up commits kept failing to fix by tweaking len()."""
        from prompt_toolkit.utils import get_cwidth

        text = "".join(t for _, t in self._rows(cli, count=3, width=width))
        for line in text.split("\n"):
            if line:
                assert get_cwidth(line) <= max(width, 12), (
                    f"line {line!r} is {get_cwidth(line)} cells, width={width}"
                )

    def test_multiline_draft_renders_as_one_row(self, cli):
        stash = type(cli._prompt_stash)()
        stash.stash("first line\nsecond line\nthird line")
        frags = cli._render_stash_panel(stash.panel_rows(), 0, 100)
        text = "".join(t for _, t in frags)
        # header + 1 entry row + footer = 3 rendered lines
        assert len([ln for ln in text.split("\n") if ln]) == 3

    def test_wide_glyph_preview_does_not_overflow(self, cli):
        """CJK previews are 2 cells per char — must still fit the box."""
        from prompt_toolkit.utils import get_cwidth

        stash = type(cli._prompt_stash)()
        stash.stash("中文" * 80)
        frags = cli._render_stash_panel(stash.panel_rows(), 0, 60)
        for line in "".join(t for _, t in frags).split("\n"):
            if line:
                assert get_cwidth(line) <= 60

    def test_cursor_row_is_styled_differently(self, cli):
        stash = type(cli._prompt_stash)()
        stash.stash("a")
        stash.stash("b")
        rows = stash.panel_rows()
        assert cli._render_stash_panel(rows, 0, 100) != cli._render_stash_panel(rows, 1, 100)

class TestStatusBarIndicator:

    def test_indicator_appears_after_stashing(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        cli._prompt_stash.stash("a parked draft")
        try:
            text = "".join(t for _, t in cli._get_status_bar_fragments())
            assert "📌 1" in text
        finally:
            cli._prompt_stash.clear()

    def test_indicator_clears_after_restore(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        cli._prompt_stash.stash("a")
        cli._prompt_stash.pop()
        text = "".join(t for _, t in cli._get_status_bar_fragments())
        assert "📌" not in text
