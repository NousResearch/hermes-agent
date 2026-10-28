"""Assistant response prose must inherit the terminal's default foreground.

Pinning the body to the skin's ``banner_text`` (raw or light-mode remapped) makes it unreadable
when the terminal theme differs from what Hermes detected at startup, e.g. a light/dark switch
in Ghostty or cmux while Hermes is running: near-white text on a light background, or the
remapped near-black on a dark one. Only the chrome (border, title) keeps the skin colors.
"""

from types import SimpleNamespace

import pytest
from rich.panel import Panel

import cli as cli_mod
from hermes_cli.skin_engine import get_active_skin, set_active_skin

_FG_ESCAPE = "\x1b[38;"


@pytest.fixture(autouse=True)
def _light_mode_default_skin(monkeypatch):
    # Light-mode remap active: the old code painted the body #1A1A1A here, the raw skin #FFF8DC
    # in dark mode. Either way the body carried an explicit foreground.
    monkeypatch.setattr(cli_mod, "_LIGHT_MODE_CACHE", True)
    set_active_skin("default")
    yield
    set_active_skin("default")


class _RecordingConsole:
    def __init__(self, *args, **kwargs):
        self.printed = []

    def print(self, *objects, **kwargs):
        self.printed.extend(objects)


def _panels(printed):
    return [obj for obj in printed if isinstance(obj, Panel)]


def _assert_body_unstyled(panel: Panel):
    assert str(panel.style) == "none", f"response body pinned to {panel.style!r}"
    # The chrome still follows the skin.
    expected = cli_mod._maybe_remap_for_light_mode(get_active_skin().get_color("response_border", "#CD7F32"))
    assert str(panel.border_style) == expected


def _stream_cli():
    cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    cli.show_reasoning = False
    cli.show_timestamps = False
    cli.final_response_markdown = "strip"
    cli._reasoning_box_opened = False
    cli._stream_box_opened = False
    cli._stream_buf = ""
    cli._stream_table_buf = []
    cli._in_stream_table = False
    cli._close_reasoning_box = lambda: None
    cli._scrollback_box_width = lambda *a: 80
    return cli


def test_streamed_response_lines_carry_no_foreground_escape(monkeypatch):
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)
    cli = _stream_cli()

    cli._emit_stream_text("hello\nworld\n")

    body = [line for line in printed if "hello" in line or "world" in line]
    assert body == [f"{cli_mod._STREAM_PAD}hello", f"{cli_mod._STREAM_PAD}world"]
    assert not any(_FG_ESCAPE in line for line in body)


def test_flushed_partial_line_carries_no_foreground_escape(monkeypatch):
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)
    cli = _stream_cli()
    cli._release_held_status_lines = lambda: None

    cli._emit_stream_text("no trailing newline")
    cli._flush_stream()

    assert f"{cli_mod._STREAM_PAD}no trailing newline" in printed


def test_final_response_panel_leaves_body_foreground_unset(monkeypatch):
    console = _RecordingConsole()
    monkeypatch.setattr(cli_mod, "ChatConsole", lambda *a, **kw: console)
    cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    cli.final_response_markdown = "strip"
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._last_turn_interrupted = False
    cli._streamed_text_this_turn = ""
    cli._scrollback_box_width = lambda *a: 80
    turn = SimpleNamespace(result={}, use_streaming_tts=False, box_opened=False)

    cli._chat_print_response_panel(turn, "hello")

    (panel,) = _panels(console.printed)
    _assert_body_unstyled(panel)


def test_background_result_panel_leaves_body_foreground_unset():
    from hermes_cli.cli_commands_mixin import _print_side_result_panel

    console = _RecordingConsole()
    cli = SimpleNamespace(_app=None, final_response_markdown="strip", _scrollback_box_width=lambda *a: 80)

    _print_side_result_panel(
        cli, header_lines=["  Background task #1 complete"], body="hello", title_suffix="(bg #1)",
        empty_note="  (no response)", console=console)

    (panel,) = _panels(console.printed)
    _assert_body_unstyled(panel)
