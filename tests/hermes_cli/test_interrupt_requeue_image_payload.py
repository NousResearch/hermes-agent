"""Regression for #110737: CLI silently drops interrupt messages carrying an image attachment.

Background
----------
``cli_tui_mixin._tui_on_enter`` bundles an Enter submission as a
``(text, images)`` tuple when images are attached, and ``_tui_enter_while_busy``
puts that tuple on ``_interrupt_queue`` in interrupt mode. ``_chat_render_turn``
then did ``"\\n".join(all_parts)`` over the raw payloads, raising ``TypeError``
on the tuple — swallowed by the outer ``chat()`` handler, so the message was
lost and no next turn ever started.

The re-queue block must unpack tuple payloads and re-attach their images to the
combined turn instead of joining them as strings.
"""

from __future__ import annotations

import importlib
import queue
import sys
from unittest.mock import MagicMock, patch


def _make_cli():
    """Build a HermesCLI instance with prompt_toolkit stubbed out.

    Mirrors the helper in ``test_cli_interrupt_drain_regression.py``.
    """
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
            return _cli_mod.HermesCLI()


def _stub_render_deps(cli):
    """Neutralize the display/audio side effects of ``_chat_render_turn``."""
    cli._chat_print_reasoning_box = lambda turn: None
    cli._chat_print_response_panel = lambda turn, response: None
    cli._emit_focus_recovery_line = lambda: None
    cli._ring_bell = lambda **kwargs: None
    cli._voice_tts = None


def _interrupted_turn(interrupt_payload):
    turn = MagicMock()
    turn.result = {"interrupted": True, "interrupt_message": interrupt_payload}
    turn.use_streaming_tts = False
    return turn


class TestInterruptRequeueImagePayload:
    """The interrupt re-queue block preserves (text, images) tuple payloads."""

    def test_tuple_payload_requeued_with_images_attached(self):
        cli = _make_cli()
        _stub_render_deps(cli)
        images = ["shot1.png", "shot2.png"]

        cli._chat_render_turn(
            _interrupted_turn(("look at this", images)), MagicMock(), None
        )

        payload = cli._pending_input.get_nowait()
        assert isinstance(payload, tuple)
        assert payload == ("look at this", images)

    def test_multiple_tuple_payloads_join_text_and_merge_images(self):
        cli = _make_cli()
        _stub_render_deps(cli)
        cli._interrupt_queue.put(("second", ["b.png"]))

        cli._chat_render_turn(
            _interrupted_turn(("first", ["a.png"])), MagicMock(), None
        )

        payload = cli._pending_input.get_nowait()
        assert payload == ("first\nsecond", ["a.png", "b.png"])

    def test_plain_text_payload_stays_a_plain_string(self):
        cli = _make_cli()
        _stub_render_deps(cli)

        cli._chat_render_turn(_interrupted_turn("plain text"), MagicMock(), None)

        payload = cli._pending_input.get_nowait()
        assert payload == "plain text"
        assert isinstance(payload, str)
