"""Tests for the CLI `/timestamps` toggle and timestamps in `/history`.

`display.timestamps` already drove the live `[HH:MM]` label suffix on
submitted/streamed messages but had no runtime toggle and `/history`
ignored it. These assert the new `/timestamps` command flips and persists
the flag and that `/history` renders `[HH:MM]` only for turns that carry a
stored unix `timestamp` (never fabricating one for live unsaved turns).
"""

import io
import sys
import time
from datetime import datetime

import yaml

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    def __init__(self):
        self.show_timestamps = False


def _seed(tmp_path, monkeypatch, value=False):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text(f"display:\n  timestamps: {str(value).lower()}\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import cli

    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


def test_timestamps_on_sets_and_persists(tmp_path, monkeypatch):
    hh = _seed(tmp_path, monkeypatch)
    s = _Stub()
    s._handle_timestamps_command("/timestamps on")
    assert s.show_timestamps is True
    assert yaml.safe_load((hh / "config.yaml").read_text())["display"]["timestamps"] is True


def test_timestamps_bare_toggles(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    s = _Stub()
    s.show_timestamps = True
    s._handle_timestamps_command("/timestamps")
    assert s.show_timestamps is False


def test_timestamps_status_is_noop(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    s = _Stub()
    s.show_timestamps = True
    s._handle_timestamps_command("/timestamps status")
    assert s.show_timestamps is True


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


def test_history_hides_timestamps_when_off():
    ts = time.time()
    hist = [{"role": "user", "content": "hello", "timestamp": ts}]
    out = _render_history(hist, show_ts=False)
    # label present, no [HH:MM] suffix
    first_label_line = out.split("[You #1]")[1].split("\n")[0]
    assert "[" not in first_label_line
