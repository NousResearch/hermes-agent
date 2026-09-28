"""The timestamp setting controls live transcript labels and stored-turn history.

The `/timestamps` command persists the flag; live renderers honor the configured
format, while `/history` only timestamps turns with a stored unix `timestamp`.
"""

import io
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    def __init__(self):
        self.show_timestamps = False


def _seed(tmp_path, monkeypatch, value=False):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text(
        f"display:\n  timestamps: {str(value).lower()}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import cli

    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


def test_timestamps_on_sets_and_persists(tmp_path, monkeypatch):
    hh = _seed(tmp_path, monkeypatch)
    s = _Stub()
    s._handle_timestamps_command("/timestamps on")
    assert s.show_timestamps is True
    assert yaml.safe_load((hh / "config.yaml").read_text(encoding="utf-8-sig"))["display"]["timestamps"] is True


def _render_history(history, show_ts):
    from cli import HermesCLI

    h = HermesCLI.__new__(HermesCLI)
    h.show_timestamps = show_ts
    h.conversation_history = history
    h._show_recent_sessions = lambda reason="history", limit=10: True
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        h.show_history()
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_history_shows_timestamp_for_stored_turns():
    ts = time.time()
    hist = [
        {"role": "user", "content": "hello", "timestamp": ts},
        {"role": "assistant", "content": "hi", "timestamp": ts + 60},
        {"role": "user", "content": "live turn, no ts"},
    ]
    out = _render_history(hist, show_ts=True)
    hhmm = datetime.fromtimestamp(ts).strftime("%H:%M")
    assert f"[You #1]  [{hhmm}]" in out
    assert "[Hermes #2]  [" in out
    # a turn with no stored timestamp must NOT get a fabricated time
    assert "[You #3]\n" in out


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("timestamp_format", [None, "%H:%M:%S"])
@pytest.mark.parametrize("surface", [
    "user_single", "user_multiline", "reasoning_stream", "reasoning_preview",
    "reasoning_final", "assistant_stream", "assistant_final", "tool_preparing", "tool_completed",
])
def test_live_transcript_honors_timestamp_setting(tmp_path, monkeypatch, enabled, timestamp_format, surface):
    """The actual render entry points must honor the loaded setting, including non-stream paths."""
    hh = _seed(tmp_path, monkeypatch, enabled)
    if timestamp_format is not None:
        (hh / "config.yaml").write_text(
            f"display:\n  timestamps: {str(enabled).lower()}\n  timestamp_format: '{timestamp_format}'\n",
            encoding="utf-8")
    import cli
    from rich.console import Console

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 27, 12, 34, 56)

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "CLI_CONFIG", cli.load_cli_config())
    instance = cli.HermesCLI.__new__(cli.HermesCLI)
    instance._init_display_options(verbose=False, compact=False)
    instance._reset_stream_state()
    instance._last_turn_interrupted = False
    instance.show_reasoning = True
    instance._reasoning_shown_this_turn = False
    instance._pending_tool_info = {}
    instance._last_scrollback_tool = None
    instance._invalidate = lambda: None
    instance._turn_summary_record = lambda *args: None
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)
    monkeypatch.setattr(cli, "ChatConsole", lambda: console)
    monkeypatch.setattr(cli, "_cprint", lambda text: output.write(text + "\n"))
    turn = SimpleNamespace(result={"last_reasoning": "Considering the image."}, use_streaming_tts=False)
    renders = {
        "user_single": lambda: instance._print_user_message_preview("Inspect this image."),
        "user_multiline": lambda: instance._print_user_message_preview("Inspect this image.\nCheck the labels."),
        "reasoning_stream": lambda: instance._stream_reasoning_delta("Considering the image.\n"),
        "reasoning_preview": lambda: instance._emit_reasoning_preview("Considering the image."),
        "reasoning_final": lambda: instance._chat_print_reasoning_box(turn),
        "assistant_stream": lambda: instance._emit_stream_text("Image inspected.\n"),
        "assistant_final": lambda: instance._chat_print_response_panel(turn, "Image inspected."),
        "tool_preparing": lambda: instance._on_tool_gen_start("vision_analyze"),
        "tool_completed": lambda: instance._on_tool_progress(
            "tool.completed", "vision_analyze", duration=1.0, result="Image inspected."),
    }
    renders[surface]()
    rendered = output.getvalue()
    assert rendered.strip(), surface
    stamp = "12:34:56" if timestamp_format else "12:34"
    assert (stamp in rendered) is enabled, (surface, rendered)
    if timestamp_format is None:
        assert "12:34:56" not in rendered
