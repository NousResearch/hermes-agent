"""display.terminal_title / display.terminal_progress drive the tab title and its progress bar."""

import threading

import pytest

from cli import HermesCLI
from hermes_cli import terminal_activity


def _cli(**attrs):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None  # no prompt_toolkit app: writes go to write_tty
    # Resolved flags are cached per CLI instance; set both so no test reads config.yaml.
    cli._terminal_title_cfg_enabled = True
    cli._terminal_progress_cfg_enabled = True
    for key, value in attrs.items():
        setattr(cli, key, value)
    return cli


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(terminal_activity, "_EMITTED", False)
    monkeypatch.setattr(terminal_activity, "_tty_available", lambda cli: True)
    monkeypatch.setattr(terminal_activity.os, "getcwd", lambda: "/tmp/hermes-work")
    written = []
    monkeypatch.setattr("hermes_cli.terminal_notify.write_tty", written.append)
    yield written
    terminal_activity.stop_spinner()


def test_idle_and_busy_titles_come_from_the_session_then_the_cwd():
    assert terminal_activity.compose_title(session_title="Fix Ghostty notifications",
                                           cwd="/tmp/hermes-work") == "Hermes · Fix Ghostty notifications"
    # No session title yet (fresh session, auto-title still running) → cwd basename labels it.
    assert terminal_activity.compose_title(session_title="", cwd="/tmp/hermes-work") == "Hermes · hermes-work"
    # A running turn prefixes the spinner glyph, which alternates per frame and cycles back.
    frames = [terminal_activity.compose_title(session_title="t", cwd="/tmp/x", busy=True, frame=i)[0]
              for i in range(len(terminal_activity.SPINNER_FRAMES) + 1)]
    assert frames[0] == terminal_activity.SPINNER_FRAMES[0]
    assert frames[-1] == frames[0] and len(set(frames)) == len(terminal_activity.SPINNER_FRAMES)


def test_control_characters_cannot_escape_the_osc():
    seq = terminal_activity.title_sequence("Hermes · evil\x07\x1b]0;injected\x1b\\")
    # One terminator only, and the smuggled sequence is gone from the payload.
    assert seq.count("\x07") == 1 and seq.endswith("\x07")
    assert "\x1b]0;injected" not in seq[5:]


def test_turn_sets_title_and_indeterminate_bar_then_clears_both(_isolate):
    written = _isolate
    busy_title = "\x1b]0;\u25d0 Hermes \u00b7 hermes-work\x07"
    idle_title = "\x1b]0;Hermes \u00b7 hermes-work\x07"
    cli = _cli(session_id="s1")

    assert cli._set_terminal_activity(True) is True
    assert terminal_activity._SPINNER.active is True  # the tab keeps spinning while the turn runs
    assert cli._set_terminal_activity(False) is True
    assert terminal_activity._SPINNER.active is False  # ...and stops the moment the turn ends
    # Busy title + indeterminate bar first, idle title + bar removed last; every repaint in
    # between is a busy frame of the same label.
    bar_on, bar_off = "\x1b]9;4;3;0\x07", "\x1b]9;4;0\x07"
    assert written[:2] == [busy_title, bar_on]
    assert written[-2:] == [idle_title, bar_off]
    spin_seq = "\x1b]0;\u25d1 Hermes \u00b7 hermes-work\x07"
    assert all(seq in (busy_title, bar_on, spin_seq, idle_title, bar_off) for seq in written), written


def test_each_display_flag_can_be_switched_off_independently(_isolate):
    written = _isolate
    # display.terminal_progress=false → title only, no OSC 9;4 at any point.
    title_only = _cli(session_id="s1")
    assert title_only._set_terminal_activity(True, config={"display": {"terminal_progress": False}}) is True
    assert title_only._set_terminal_activity(False, config={"display": {"terminal_progress": False}}) is True
    assert written and all("9;4" not in seq for seq in written), written

    # display.terminal_title=false → progress only.
    written.clear()
    bar_only = _cli(session_id="s1")
    assert bar_only._set_terminal_activity(True, config={"display": {"terminal_title": False}}) is True
    assert bar_only._set_terminal_activity(False, config={"display": {"terminal_title": False}}) is True
    assert written == ["\x1b]9;4;3;0\x07", "\x1b]9;4;0\x07"], written

    # Both off → nothing at all, and no spinner left running.
    written.clear()
    off = _cli(session_id="s1")
    both_off = {"display": {"terminal_title": False, "terminal_progress": False}}
    assert off._set_terminal_activity(True, config=both_off) is False
    assert written == [] and terminal_activity._SPINNER.active is False


def test_busy_turn_keeps_repainting_the_tab(monkeypatch, _isolate):
    """The tab animates for the whole turn (Claude Code does the same). Event-based, so the
    assertion never depends on scheduling luck."""
    frames = []
    spun = threading.Event()
    real_write = terminal_activity._write

    def _record(cli, seq):
        real_write(cli, seq)
        frames.append(seq)
        if len(frames) > 2:  # the first two are the busy title + the progress bar
            spun.set()

    monkeypatch.setattr(terminal_activity, "_write", _record)
    monkeypatch.setattr(terminal_activity, "SPINNER_INTERVAL", 0.01)
    cli = _cli(session_id="s1")
    try:
        assert cli._set_terminal_activity(True) is True
        assert spun.wait(2.0), frames  # a repaint landed after the first frames
        assert frames[2] != frames[0]  # ...and it shows the other glyph
    finally:
        cli._set_terminal_activity(False)
    assert terminal_activity._SPINNER.active is False


def test_piped_run_never_writes_indicators(monkeypatch, _isolate):
    """`hermes -z ... > out`, cron, and a closed stdout stay clean: no OSC in the output."""
    written = _isolate
    monkeypatch.setattr(terminal_activity, "_tty_available", lambda cli: False)

    class _NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr(terminal_activity.sys, "stdout", _NotATty())
    assert _cli(session_id="s1")._set_terminal_activity(False) is False
    assert written == []


def test_live_app_gets_indicators_on_its_loop_and_exit_clears_them(monkeypatch, _isolate):
    """A live app must be written through its own loop (never a second tty writer), and the exit
    reset must undo exactly what we emitted."""

    class _Output:
        raw = []

        def write_raw(self, data):
            self.raw.append(data)

        def flush(self):
            pass

    class _Loop:
        queued = []

        def call_soon_threadsafe(self, fn):
            self.queued.append(fn)

    class _App:
        _is_running = True
        loop = _Loop()
        output = _Output()

    monkeypatch.setattr("hermes_cli.terminal_notify.write_tty",
                        lambda seq: pytest.fail(f"stray tty write: {seq!r}"))
    cli = _cli(session_id="s1", _app=_App())
    assert cli._set_terminal_activity(False) is True
    assert _Output.raw == []  # queued for the loop, nothing written from this thread
    while _Loop.queued:
        _Loop.queued.pop(0)()
    assert _Output.raw == ["\x1b]0;Hermes \u00b7 hermes-work\x07", "\x1b]9;4;0\x07"]

    exit_writes = []
    monkeypatch.setattr("hermes_cli.terminal_notify.write_tty", exit_writes.append)
    terminal_activity.reset_on_exit()
    assert exit_writes == ["\x1b]0;\x07\x1b]9;4;0\x07"]
    # Idempotent: a second exit pass emits nothing.
    terminal_activity.reset_on_exit()
    assert exit_writes == ["\x1b]0;\x07\x1b]9;4;0\x07"]
