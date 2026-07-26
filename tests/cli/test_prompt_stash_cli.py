"""Tests for the Ctrl+S prompt stash wiring inside HermesCLI.

The state machine itself is covered by tests/cli/test_prompt_stash.py. These
tests verify the cli.py side:
  - HermesCLI.__init__ creates a PromptStash
  - the layout hook makes room for the stash browse panel
  - _render_stash_panel renders bounded, display-width-correct rows
  - the status-bar indicator appears / disappears with stash contents

Follows the prompt_toolkit-stub construction pattern from
tests/cli/test_cli_extension_hooks.py.
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
    def test_cli_has_prompt_stash(self, cli):
        assert hasattr(cli, "_prompt_stash")

    def test_stash_starts_empty(self, cli):
        # Duck-typed rather than isinstance: _make_cli reloads the `cli`
        # module, which re-imports hermes_cli.prompt_toolkit stubs and can
        # yield a distinct-but-equivalent PromptStash class object.
        stash = cli._prompt_stash
        assert type(stash).__name__ == "PromptStash"
        assert len(stash) == 0
        assert stash.panel_open is False
        assert stash.indicator() == ""
        assert stash.placeholder_hint() == ""

    def test_stash_is_per_instance_not_shared(self):
        """Two CLIs must not share one stash — drafts would leak across sessions."""
        a = _make_cli()
        b = _make_cli()
        a._prompt_stash.stash("only in a")
        assert len(a._prompt_stash) == 1
        assert len(b._prompt_stash) == 0


class TestKeybindingRegistration:
    """The Ctrl+S binding must actually be registered on the TUI KeyBindings."""

    def test_ctrl_s_source_binding_exists(self):
        """cli.py registers a c-s handler (regression guard for PR #4771).

        The original PR's head lost its keybinding during a rebase, shipping
        the panel renderer with no way to reach it. Assert on the source so
        that cannot silently regress again.
        """
        from pathlib import Path

        source = Path(
            importlib.import_module("cli").__file__ or ""
        ).read_text(encoding="utf-8")
        assert "@kb.add('c-s'" in source
        assert "def handle_prompt_stash(" in source

    def test_panel_navigation_bindings_exist(self):
        from pathlib import Path

        source = Path(
            importlib.import_module("cli").__file__ or ""
        ).read_text(encoding="utf-8")
        for handler in (
            "def handle_stash_panel_up(",
            "def handle_stash_panel_down(",
            "def handle_stash_panel_restore(",
            "def handle_stash_panel_delete(",
            "def handle_stash_panel_close(",
        ):
            assert handler in source, f"missing panel handler: {handler}"

    def test_extension_hook_still_a_noop(self, cli):
        """The stash binding lives in run(), not in the wrapper extension hook."""
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()
        assert cli._register_extra_tui_keybindings(kb, input_area=None) is None
        assert kb.bindings == []


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

    def test_returns_fragments(self, cli):
        frags = self._rows(cli)
        assert frags
        assert all(isinstance(f, tuple) and len(f) == 2 for f in frags)

    def test_header_and_footer_present(self, cli):
        text = "".join(t for _, t in self._rows(cli))
        assert "📌 Stash" in text
        assert "Ctrl+S" in text
        assert "Enter=restore" in text
        assert "D=delete" in text

    def test_row_per_entry(self, cli):
        text = "".join(t for _, t in self._rows(cli, count=3))
        for i in range(3):
            assert f"[{i + 1}]" in text

    def test_singular_plural_item_label(self, cli):
        one = "".join(t for _, t in self._rows(cli, count=1))
        assert "(1 item)" in one
        two = "".join(t for _, t in self._rows(cli, count=2))
        assert "(2 items)" in two

    @pytest.mark.parametrize("width", [16, 20, 30, 40, 60, 80, 120, 400])
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
        styles = {s for s, _ in cli._render_stash_panel(stash.panel_rows(), 1, 100)}
        assert "class:subagent-selected" in styles

    def test_empty_list_still_renders_frame(self, cli):
        frags = cli._render_stash_panel([], 0, 80)
        text = "".join(t for _, t in frags)
        assert "(0 items)" in text


class TestStatusBarIndicator:
    def test_no_indicator_when_stash_empty(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        text = "".join(t for _, t in cli._get_status_bar_fragments())
        assert "📌" not in text

    def test_indicator_appears_after_stashing(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        cli._prompt_stash.stash("a parked draft")
        try:
            text = "".join(t for _, t in cli._get_status_bar_fragments())
            assert "📌 1" in text
        finally:
            cli._prompt_stash.clear()

    def test_indicator_count_tracks_stash_size(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        cli._prompt_stash.stash("a")
        cli._prompt_stash.stash("b")
        try:
            text = "".join(t for _, t in cli._get_status_bar_fragments())
            assert "📌 2" in text
        finally:
            cli._prompt_stash.clear()

    def test_indicator_clears_after_restore(self, cli):
        cli._prompt_stash.clear()
        cli._status_bar_visible = True
        cli._prompt_stash.stash("a")
        cli._prompt_stash.pop()
        text = "".join(t for _, t in cli._get_status_bar_fragments())
        assert "📌" not in text


class TestFmtStashAge:
    def test_age_buckets(self, cli):
        import time

        now = time.monotonic()
        assert cli._fmt_stash_age(now) == "just now"
        assert cli._fmt_stash_age(now - 30).endswith("s ago")
        assert "min ago" in cli._fmt_stash_age(now - 300)
        assert cli._fmt_stash_age(now - 7200).endswith("h ago")
