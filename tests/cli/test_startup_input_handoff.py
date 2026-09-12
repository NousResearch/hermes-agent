"""Startup must preserve Enter/newline identity and return the inherited TTY mode."""

import os
import sys
from contextlib import closing

import pytest


@pytest.fixture
def terminal(monkeypatch):
    import termios

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_input
    from prompt_toolkit.output import DummyOutput

    master, slave = os.openpty()
    inherited = termios.tcgetattr(slave)
    try:
        with os.fdopen(os.dup(slave), "r", encoding="utf-8") as stream:
            monkeypatch.setattr(sys, "stdin", stream)
            with closing(create_input(stdin=stream)) as terminal_input:
                with create_app_session(input=terminal_input, output=DummyOutput()):
                    yield master, slave, terminal_input, inherited
    finally:
        os.close(master)
        os.close(slave)


class HandoffObserved(Exception):
    pass


@pytest.mark.linux_only
@pytest.mark.parametrize("reply", ["n", "interrupt", "eof"])
@pytest.mark.parametrize("ending", [b"\r", b"\n"])
def test_startup_preserves_queued_key_identity_and_restores_tty(terminal, monkeypatch, reply, ending):
    import select
    import termios

    from prompt_toolkit.keys import Keys

    from cli import HermesCLI

    master, slave, terminal_input, inherited = terminal
    shell = HermesCLI.__new__(HermesCLI)
    shell._claim_active_session = lambda _kind: True
    observed = []

    def answer(_prompt):
        if reply == "interrupt":
            raise KeyboardInterrupt
        if reply == "eof":
            raise EOFError
        return reply

    def startup():
        assert shell._offer_first_run_setup() is False
        # Type ahead at the real onboarding-to-REPL boundary, before the
        # Application has entered its own raw-mode context.
        os.write(master, b"draft" + ending)
        # Wait for the line discipline to receive the write before changing
        # modes; write completion alone does not order those kernel operations.
        assert select.select([slave], [], [], 2)[0] == [slave]

    def observe_handoff():
        with terminal_input.raw_mode():
            observed.extend(key.key for key in terminal_input.read_keys())
        # A failure during startup must restore the mode just like a normal exit.
        raise HandoffObserved

    monkeypatch.setattr("builtins.input", answer)
    shell._tui_print_startup = startup
    shell._tui_init_run_state = observe_handoff
    with pytest.raises(HandoffObserved):
        shell.run()

    assert termios.tcgetattr(slave) == inherited
    expected = Keys.ControlM if ending == b"\r" else Keys.ControlJ
    assert observed == [*"draft", expected]


@pytest.mark.linux_only
@pytest.mark.parametrize("outcome", ["configured", "cancelled", "failed"])
def test_first_run_dialog_borrows_cooked_mode_without_releasing_raw_owner(terminal, monkeypatch, outcome):
    import termios

    from cli import HermesCLI

    _master, slave, terminal_input, inherited = terminal
    shell = HermesCLI.__new__(HermesCLI)
    shell.requested_provider = "unchanged"
    shell.model = "unchanged"
    shell._runtime_credentials_ready = lambda: True
    dialog_modes = []

    def answer(_prompt):
        dialog_modes.append(termios.tcgetattr(slave)[3])
        return "y"

    def picker():
        dialog_modes.append(termios.tcgetattr(slave)[3])
        if outcome == "cancelled":
            raise KeyboardInterrupt
        if outcome == "failed":
            raise RuntimeError("provider setup failed")

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", picker)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    with terminal_input.raw_mode():
        raw = termios.tcgetattr(slave)
        result = shell._offer_first_run_setup()
        assert termios.tcgetattr(slave) == raw

    assert result is (outcome == "configured")
    assert len(dialog_modes) == 2
    assert all(mode & termios.ICANON and mode & termios.ECHO for mode in dialog_modes)
    assert termios.tcgetattr(slave) == inherited
