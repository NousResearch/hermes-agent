"""Tests for bracketed-paste timeout safety valve (#16263).

Verifies the production helper in cli.py monkey-patches prompt_toolkit's
Vt100Parser.feed() so the parser auto-escapes from bracketed-paste mode when
the ESC[201~ end mark is never received.
"""
import ast
import importlib
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

from prompt_toolkit.keys import Keys


ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "cli.py"


def _load_production_patch_helper():
    """Load cli._apply_bracketed_paste_timeout_patch without importing cli.

    Importing cli.py pulls optional runtime deps that aren't required for this
    parser-level regression.  AST-loading the exact helper keeps the test tied
    to production code while avoiding unrelated import side effects.  If the
    production helper is removed, this test fails.
    """
    source = CLI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_bracketed_paste_timeout_patch"
        ),
        None,
    )
    assert helper_node is not None, (
        "cli.py must define _apply_bracketed_paste_timeout_patch()"
    )
    helper_source = ast.get_source_segment(source, helper_node)
    namespace = {"time": time, "logger": logging.getLogger("test.cli")}
    exec(helper_source, namespace)
    return namespace["_apply_bracketed_paste_timeout_patch"]


def _reset_and_apply_production_patch():
    """Reload prompt_toolkit's parser and apply Hermes' production patch."""
    import prompt_toolkit.input.vt100_parser as vt100_mod

    vt100_mod = importlib.reload(vt100_mod)
    # importlib.reload() preserves module dict entries that the reloaded source
    # does not redefine, so clear Hermes' sentinel before re-applying.
    if hasattr(vt100_mod, "_hermes_bp_timeout_patched"):
        delattr(vt100_mod, "_hermes_bp_timeout_patched")
    _load_production_patch_helper()()
    assert getattr(vt100_mod, "_hermes_bp_timeout_patched", False)
    return vt100_mod


class TestBracketedPasteTimeout:
    """Verify the Vt100Parser monkey-patch prevents frozen bracketed-paste."""

    def _make_parser(self):
        """Create a Vt100Parser after applying the production patch."""
        vt100_mod = _reset_and_apply_production_patch()
        callback = MagicMock()
        parser = vt100_mod.Vt100Parser(callback)
        return parser, callback

    def test_normal_bracketed_paste_works(self):
        """A complete bracketed-paste sequence should work normally."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~hello world\x1b[201~")
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert call_args.data == "hello world"

    def test_incomplete_paste_times_out(self):
        """If ESC[201~ is never received, parser should recover after timeout."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~some pasted text")
        assert parser._in_bracketed_paste

        parser._hermes_bp_start = time.monotonic() - 3.0
        parser.feed("more data")

        assert not parser._in_bracketed_paste
        assert callback.called

    def test_timeout_preserves_buffered_content(self):
        """Auto-escape should flush buffered content, not lose it."""
        parser, callback = self._make_parser()
        content = "line1\nline2\nline3"
        parser.feed(f"\x1b[200~{content}")
        parser._hermes_bp_start = time.monotonic() - 3.0
        parser.feed("")

        paste_events = [
            c[0][0]
            for c in callback.call_args_list
            if hasattr(c[0][0], "key") and c[0][0].key == Keys.BracketedPaste
        ]
        assert len(paste_events) >= 1
        assert content in paste_events[0].data

    def test_normal_keys_after_timeout_recovery(self):
        """After timeout recovery, normal key processing should resume."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~stuck")
        parser._hermes_bp_start = time.monotonic() - 3.0
        parser.feed("")

        assert not parser._in_bracketed_paste
        callback.reset_mock()
        parser.feed("a")
        assert not parser._in_bracketed_paste

    def test_no_timeout_when_end_mark_arrives_quickly(self):
        """No timeout should fire if end mark arrives within the window."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~quick paste\x1b[201~")
        assert not parser._in_bracketed_paste
        callback.assert_called_once()

    def test_subsequent_data_after_incomplete_paste(self):
        """Data arriving after a stuck paste should be processable."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~content")
        parser._hermes_bp_start = time.monotonic() - 5.0
        parser.feed("x")

        assert not parser._in_bracketed_paste
        assert callback.call_count >= 1

    def test_torn_end_mark_recovers(self):
        """If end mark arrives split across feeds within timeout, it still works."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~some content\x1b[20")
        assert parser._in_bracketed_paste

        parser.feed("1~")
        assert not parser._in_bracketed_paste
        callback.assert_called_once()
        assert callback.call_args[0][0].data == "some content"

    def test_no_timeout_under_threshold(self):
        """Bracketed-paste mode should not timeout within the 2s window."""
        parser, callback = self._make_parser()
        parser.feed("\x1b[200~waiting")
        parser._hermes_bp_start = time.monotonic() - 0.5
        parser.feed("more waiting")

        assert parser._in_bracketed_paste
        assert not callback.called
