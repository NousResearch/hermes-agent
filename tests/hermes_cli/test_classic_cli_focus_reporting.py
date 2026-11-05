"""Regression coverage for #129029: classic CLI focus reporting lifecycle."""

from types import SimpleNamespace
from unittest.mock import patch


def _cli():
    import cli as cli_mod
    return cli_mod


class _FakeOutput:
    def __init__(self):
        self.raw = []
        self.flushed = 0

    def write_raw(self, value):
        self.raw.append(value)

    def flush(self):
        self.flushed += 1


def test_focus_reporting_enable_is_symmetric_with_cleanup_reset():
    cli_mod = _cli()
    output = _FakeOutput()

    with (
        patch.object(cli_mod, "_FOCUS_REPORTING_ENABLE_SEQ", "focus-enable"),
        patch.object(cli_mod, "_FOCUS_REPORTING_DISABLE_SEQ", "focus-disable"),
    ):
        assert cli_mod._enable_focus_reporting(output) is True
        assert cli_mod._disable_focus_reporting(output) is True

    assert output.raw == ["focus-enable", "focus-disable"]
    assert cli_mod._FOCUS_REPORTING_ENABLE_SEQ == "\x1b[?1004h"
    assert cli_mod._FOCUS_REPORTING_DISABLE_SEQ == "\x1b[?1004l"
    assert cli_mod._FOCUS_REPORTING_DISABLE_SEQ in cli_mod._TERMINAL_INPUT_MODE_RESET_SEQ
    assert output.flushed == 2


def test_focus_reporting_is_not_enabled_on_native_windows_input():
    """Native prompt_toolkit Win32 input does not parse CSI focus reports."""
    cli_mod = _cli()
    output = _FakeOutput()

    with patch("hermes_cli.cli_terminal_input.sys.platform", "win32"):
        assert cli_mod._enable_focus_reporting(output) is False
        assert cli_mod._disable_focus_reporting(output) is False

    assert output.raw == []
    assert output.flushed == 0


def test_input_mode_recovery_reenables_focus_reporting():
    cli_mod = _cli()
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    output = _FakeOutput()
    shell._app = SimpleNamespace(output=output)
    shell._last_input_mode_recovery = None
    shell._input_mode_recovery_notice_shown = True
    shell.config = {}

    with (
        patch("hermes_cli.cli_terminal_mixin._write_terminal_sequence") as reset_modes,
        patch.object(cli_mod, "_enable_focus_reporting") as enable_focus,
        patch.object(cli_mod, "_cli_multiline_shortcuts_enabled", return_value=False),
    ):
        shell._recover_terminal_input_modes(reason="test")

    reset_modes.assert_called_once_with(shell._app, cli_mod._TERMINAL_INPUT_MODE_RESET_SEQ)
    enable_focus.assert_called_once_with(output)


class _FakeTask:
    def __init__(self):
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback


def test_external_editor_suspends_focus_reports_until_terminal_returns():
    cli_mod = _cli()
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    output = _FakeOutput()
    task = _FakeTask()
    buffer = SimpleNamespace(open_in_editor=lambda **_kw: task)
    shell._app = SimpleNamespace(output=output, current_buffer=buffer, is_running=True)
    shell._command_running = False
    shell._sudo_state = shell._secret_state = shell._approval_state = None
    shell._slash_confirm_state = shell._clarify_state = shell._connection_state = None
    shell._inline_pastes = lambda _buffer: None
    events = []
    shell._submit_editor_buffer = lambda _buffer: events.append("submit")

    with (
        patch.object(cli_mod, "_disable_focus_reporting", side_effect=lambda _out: events.append("disable")),
        patch.object(cli_mod, "_enable_focus_reporting", side_effect=lambda _out: events.append("enable")),
    ):
        assert shell._open_external_editor() is True
        assert events == ["disable"]
        assert task.callback is not None
        task.callback(task)

    assert events == ["disable", "enable", "submit"]


def test_ctrl_z_suspends_focus_reports_while_shell_owns_terminal():
    cli_mod = _cli()
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    output = _FakeOutput()
    event = SimpleNamespace(app=SimpleNamespace(output=output, invalidate=lambda: None))
    events = []
    skin = SimpleNamespace(get_branding=lambda *_args: "Hermes Agent")

    with (
        patch("hermes_cli.cli_tui_mixin.sys.platform", "darwin"),
        patch("hermes_cli.skin_engine.get_active_skin", return_value=skin),
        patch("prompt_toolkit.application.run_in_terminal", side_effect=lambda fn: fn()),
        patch("hermes_cli.cli_tui_mixin.os.write", side_effect=lambda *_a: events.append("write")),
        patch("hermes_cli.cli_tui_mixin.os.kill", side_effect=lambda *_a: events.append("stop")),
        patch.object(cli_mod, "_disable_focus_reporting", side_effect=lambda _out: events.append("disable")),
        patch.object(cli_mod, "_enable_focus_reporting", side_effect=lambda _out: events.append("enable")),
    ):
        shell._tui_handle_ctrl_z(event)

    assert events == ["disable", "write", "stop", "enable"]


def test_external_editor_does_not_rearm_focus_after_cli_stops():
    cli_mod = _cli()
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    output = _FakeOutput()
    task = _FakeTask()
    buffer = SimpleNamespace(open_in_editor=lambda **_kw: task)
    app = SimpleNamespace(output=output, current_buffer=buffer, is_running=True)
    shell._app = app
    shell._command_running = False
    shell._sudo_state = shell._secret_state = shell._approval_state = None
    shell._slash_confirm_state = shell._clarify_state = shell._connection_state = None
    shell._inline_pastes = lambda _buffer: None
    events = []
    shell._submit_editor_buffer = lambda _buffer: events.append("submit")

    with (
        patch.object(cli_mod, "_disable_focus_reporting", side_effect=lambda _out: events.append("disable")),
        patch.object(cli_mod, "_enable_focus_reporting", side_effect=lambda _out: events.append("enable")),
    ):
        assert shell._open_external_editor() is True
        app.is_running = False
        task.callback(task)

    assert events == ["disable", "submit"]


def test_raw_text_prompt_suspends_focus_reports_while_input_owns_terminal():
    cli_mod = _cli()
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    output = _FakeOutput()
    app = SimpleNamespace(output=output, is_running=True, invalidate=lambda: None)
    shell._app = app
    shell._status_bar_visible = True
    events = []

    # Use the real main-thread branch; run_in_terminal is collapsed synchronously
    # so the test proves only the terminal-ownership lifecycle, not PT scheduling.
    with (
        patch("prompt_toolkit.application.run_in_terminal", side_effect=lambda fn: fn()),
        patch("builtins.input", return_value="answer"),
        patch.object(cli_mod, "_disable_focus_reporting", side_effect=lambda _out: events.append("disable")),
        patch.object(cli_mod, "_enable_focus_reporting", side_effect=lambda _out: events.append("enable")),
    ):
        result = shell._prompt_text_input("Choice: ")

    assert result == "answer"
    assert events == ["disable", "enable"]
