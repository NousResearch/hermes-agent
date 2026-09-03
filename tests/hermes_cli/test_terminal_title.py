from __future__ import annotations

from types import SimpleNamespace

from cli import HermesCLI
from hermes_cli.terminal_title import (
    compose_terminal_title,
    sanitize_terminal_title,
    write_terminal_title,
)
import hermes_cli.terminal_title as terminal_title


class _Terminal:
    def __init__(self, *, isatty: bool = True) -> None:
        self._isatty = isatty
        self.writes: list[str] = []
        self.flushed = False

    def isatty(self) -> bool:
        return self._isatty

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        self.flushed = True


class _PromptToolkitOutput:
    """Minimal prompt_toolkit Output stand-in backed by a real terminal."""

    def __init__(self, stdout: _Terminal) -> None:
        self.stdout = stdout
        self.raw_writes: list[str] = []
        self.flushed = False

    def write_raw(self, value: str) -> None:
        self.raw_writes.append(value)

    def flush(self) -> None:
        self.flushed = True


class _ScheduledLoop:
    """Capture callbacks so tests can control their event-loop ordering."""

    def __init__(self) -> None:
        self.callbacks = []

    def call_soon_threadsafe(self, callback) -> None:
        self.callbacks.append(callback)


def test_compose_terminal_title_uses_skin_symbol_session_and_busy_marker():
    assert compose_terminal_title(" ⚔ Ares ", "Release prep") == "⚔ Release prep"
    assert compose_terminal_title(" ⚔ Ares ", "Release prep", busy=True) == "⚔ Release prep ⏳"
    assert compose_terminal_title(" ⚔ Ares ") == "⚔"


def test_sanitize_terminal_title_removes_terminal_control_sequences_and_bounds_length():
    title = sanitize_terminal_title("Build\x1b]2;injected\a\nready")

    assert title == "Build]2;injectedready"
    assert len(sanitize_terminal_title("x" * 201)) == 200


def test_write_terminal_title_emits_icon_and_window_sequences(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = _Terminal()

    assert write_terminal_title("⚕ Planning", terminal)
    assert terminal.writes == ["\033]1;⚕ Planning\a\033]2;⚕ Planning\a"]
    assert terminal.flushed


def test_write_terminal_title_uses_prompt_toolkit_raw_output(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = _Terminal()
    output = _PromptToolkitOutput(terminal)

    assert write_terminal_title("⚕ Planning", output)
    assert output.raw_writes == ["\033]1;⚕ Planning\a\033]2;⚕ Planning\a"]
    assert output.flushed
    assert terminal.writes == []


def test_write_terminal_title_updates_windows_console_and_osc_terminals(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(terminal_title.sys, "platform", "win32")
    titles: list[str] = []
    monkeypatch.setattr(
        terminal_title,
        "_set_windows_console_title",
        lambda title: titles.append(title) or True,
    )
    terminal = _Terminal()

    assert write_terminal_title("⚕ Planning", terminal)
    assert titles == ["⚕ Planning"]
    assert terminal.writes == ["\033]1;⚕ Planning\a\033]2;⚕ Planning\a"]
    assert terminal.flushed


def test_write_terminal_title_skips_dumb_and_noninteractive_output(monkeypatch):
    terminal = _Terminal()
    monkeypatch.setenv("TERM", "dumb")
    assert not write_terminal_title("Hermes", terminal)

    monkeypatch.setenv("TERM", "xterm-256color")
    assert not write_terminal_title("Hermes", _Terminal(isatty=False))


def test_cli_terminal_title_writes_on_the_prompt_toolkit_event_loop(monkeypatch):
    loop = _ScheduledLoop()
    cli = HermesCLI.__new__(HermesCLI)
    cli._terminal_title_enabled = True
    cli._agent_running = False
    cli._app = SimpleNamespace(loop=loop, output=object())
    cli._current_session_title = lambda: "Current session"
    cli.session_id = "session-a"
    writes = []
    monkeypatch.setattr(
        terminal_title,
        "write_terminal_title",
        lambda title, output: writes.append((title, output)),
    )

    cli._update_terminal_title()

    assert writes == []
    assert len(loop.callbacks) == 1
    loop.callbacks.pop()()
    assert writes == [("⚕ Current session", cli._app.output)]


def test_cli_auto_title_skips_a_session_that_changed_before_the_scheduled_write(monkeypatch):
    loop = _ScheduledLoop()
    cli = HermesCLI.__new__(HermesCLI)
    cli._terminal_title_enabled = True
    cli._agent_running = False
    cli._app = SimpleNamespace(loop=loop, output=object())
    cli._current_session_title = lambda: "Current session"
    cli.session_id = "session-a"
    writes = []
    monkeypatch.setattr(
        terminal_title,
        "write_terminal_title",
        lambda title, output: writes.append((title, output)),
    )

    cli._update_terminal_title(
        session_title="Generated A",
        expected_session_id="session-a",
    )
    cli.session_id = "session-b"
    loop.callbacks.pop()()

    assert writes == []


def test_cli_auto_title_uses_busy_state_when_the_scheduled_write_runs(monkeypatch):
    loop = _ScheduledLoop()
    cli = HermesCLI.__new__(HermesCLI)
    cli._terminal_title_enabled = True
    cli._agent_running = False
    cli._app = SimpleNamespace(loop=loop, output=object())
    cli._current_session_title = lambda: "Current session"
    cli.session_id = "session-a"
    writes = []
    monkeypatch.setattr(
        terminal_title,
        "write_terminal_title",
        lambda title, output: writes.append((title, output)),
    )

    cli._update_terminal_title(
        session_title="Generated A",
        expected_session_id="session-a",
    )
    cli._agent_running = True
    loop.callbacks.pop()()

    assert writes == [("⚕ Generated A ⏳", cli._app.output)]
