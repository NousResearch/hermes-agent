import ctypes
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cli as cli_module
from cli import HermesCLI


class _TtyStringIO(io.StringIO):
    def isatty(self):
        return True


class _NonTtyStringIO(io.StringIO):
    def isatty(self):
        return False


class _EscapingStdoutProxy(io.StringIO):
    def isatty(self):
        return True

    def write(self, data):
        return super().write(data.replace("\033", "?"))


def _make_cli(stream=None):
    cli = HermesCLI.__new__(HermesCLI)
    cli._tab_activity_indicator = True
    cli._tab_title_idle = "Hermes"
    cli._tab_title_busy = "{spinner} {title}"
    cli._tab_title_done = "{title}"
    cli._tab_title_spinner_frames = ["a", "b"]
    cli._tab_title_frame_idx = 0
    cli._tab_title_last = None
    cli._tab_title_next_update = 0.0
    cli._tab_title_stream = stream or _TtyStringIO()
    cli._pending_title = None
    cli._session_db = SimpleNamespace(get_session_title=lambda session_id: "Project")
    cli.session_id = "session-123456"
    return cli


@pytest.mark.platforms("posix")
def test_tab_title_done_writes_osc_title_sequence():
    cli = _make_cli()

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == "\033]0;Project\a"


@pytest.mark.platforms("posix")
def test_tab_title_busy_advances_configured_spinner_frames():
    cli = _make_cli()
    cli._pending_title = "Project"

    cli._set_tab_title_state("busy", force=True)
    cli._set_tab_title_state("busy", force=False)

    assert cli._tab_title_stream.getvalue() == "\033]0;a Project\a\033]0;b Project\a"


def test_tab_title_indicator_is_opt_in():
    cli = _make_cli()
    cli._tab_activity_indicator = False

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == ""


def test_tab_title_indicator_skips_non_tty_streams():
    stream = _NonTtyStringIO()
    cli = _make_cli(stream=stream)

    cli._set_tab_title_state("done", force=True)

    assert stream.getvalue() == ""


@pytest.mark.platforms("posix")
def test_tab_title_writer_bypasses_patch_stdout_proxy(monkeypatch):
    patched_stdout = _EscapingStdoutProxy()
    raw_stdout = _TtyStringIO()
    cli = _make_cli()
    delattr(cli, "_tab_title_stream")
    monkeypatch.setattr(cli_module.sys, "stdout", patched_stdout)
    monkeypatch.setattr(cli_module.sys, "__stdout__", raw_stdout)

    cli._set_tab_title_state("done", force=True)

    assert patched_stdout.getvalue() == ""
    assert raw_stdout.getvalue() == "\033]0;Project\a"


@pytest.mark.platforms("windows")
def test_tab_title_uses_windows_console_title_api(monkeypatch):
    stream = _NonTtyStringIO()
    cli = _make_cli(stream=stream)
    cli._tab_title_done = "done \033bad\nnext"
    set_console_title = Mock(return_value=1)
    windll = SimpleNamespace(
        kernel32=SimpleNamespace(SetConsoleTitleW=set_console_title),
    )
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    cli._set_tab_title_state("done", force=True)

    set_console_title.assert_called_once_with("done bad next")
    assert stream.getvalue() == ""
    assert cli._tab_title_last == "done bad next"


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("terminal_env", [
    {"TERM_PROGRAM": "vscode"},
    {"MSYSTEM": "MINGW64"},
    {"WT_SESSION": "test-session"},
    {"TERM": "xterm-256color"},
])
def test_tab_title_dual_writes_on_windows_tty(monkeypatch, terminal_env):
    stream = _TtyStringIO()
    cli = _make_cli(stream=stream)
    set_console_title = Mock(return_value=1)
    windll = SimpleNamespace(
        kernel32=SimpleNamespace(SetConsoleTitleW=set_console_title),
    )
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    for name in ("MSYSTEM", "TERM_PROGRAM", "WT_SESSION", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in terminal_env.items():
        monkeypatch.setenv(name, value)

    cli._set_tab_title_state("done", force=True)

    set_console_title.assert_called_once_with("Project")
    assert stream.getvalue() == "\033]0;Project\a"
    assert cli._tab_title_last == "Project"


@pytest.mark.platforms("windows")
def test_tab_title_avoids_osc_on_legacy_windows_console(monkeypatch):
    stream = _TtyStringIO()
    cli = _make_cli(stream=stream)
    set_console_title = Mock(return_value=1)
    windll = SimpleNamespace(
        kernel32=SimpleNamespace(SetConsoleTitleW=set_console_title),
    )
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    for name in ("MSYSTEM", "TERM_PROGRAM", "WT_SESSION", "TERM"):
        monkeypatch.delenv(name, raising=False)

    cli._set_tab_title_state("done", force=True)

    set_console_title.assert_called_once_with("Project")
    assert stream.getvalue() == ""
    assert cli._tab_title_last == "Project"


@pytest.mark.platforms("posix")
def test_tab_title_sanitizes_control_characters():
    cli = _make_cli()
    cli._tab_title_done = "done \033bad\a\nnext\tmore\x7fhidden\x9f"

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == "\033]0;done bad next morehidden\a"


@pytest.mark.platforms("posix")
def test_tab_title_truncates_long_titles():
    cli = _make_cli()
    cli._tab_title_done = "x" * 300

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == f"\033]0;{'x' * 255}\a"


def test_configure_tab_activity_indicator_handles_invalid_spinner_frames():
    cli = HermesCLI.__new__(HermesCLI)
    cli._configure_tab_activity_indicator({"tab_activity_indicator": True, "tab_title_spinner_frames": 123})

    assert cli._tab_activity_indicator is True
    assert cli._tab_title_spinner_frames == ["*"]


@pytest.mark.platforms("windows")
def test_tab_title_windows_native_failure_falls_back_to_osc(monkeypatch):
    cli = _make_cli()
    native_title = Mock(side_effect=OSError("No console attached"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(SetConsoleTitleW=native_title)))
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    cli._set_tab_title_state("done", force=True)
    cli._set_tab_title_state("done")

    assert cli._tab_title_stream.getvalue() == "\033]0;Project\a"
    native_title.assert_called_once_with("Project")


@pytest.mark.platforms("windows")
def test_tab_title_failed_outputs_retry_without_leaking_osc_to_redirected_stdout(monkeypatch):
    stream = _NonTtyStringIO()
    cli = _make_cli(stream)
    native_title = Mock(side_effect=[0, 1])
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(SetConsoleTitleW=native_title)))
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    cli._set_tab_title_state("done", force=True)
    assert cli._tab_title_last is None
    cli._set_tab_title_state("done")

    assert native_title.call_count == 2
    assert cli._tab_title_last == "Project"
    assert stream.getvalue() == ""


@pytest.mark.platforms("posix")
def test_tab_title_writes_sanitized_title_to_real_pty():
    import os
    import pty

    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "w", encoding="utf-8") as stream:
            cli = _make_cli(stream)
            cli._tab_title_done = "done \033bad\nnext"
            cli._set_tab_title_state("done", force=True)
            assert os.read(master, 1024) == b"\033]0;done bad next\a"
    finally:
        os.close(master)


@pytest.mark.platforms("posix")
def test_tab_title_config_reaches_cli_from_yaml(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"display": {
        "tab_activity_indicator": True,
        "tab_title_done": "Finished {title}",
    }})
    monkeypatch.setattr(cli_module, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    loaded = cli_module.load_cli_config()
    monkeypatch.setattr(cli_module, "CLI_CONFIG", loaded)
    cli = HermesCLI.__new__(HermesCLI)
    cli._init_display_options(verbose=False, compact=False)
    cli._tab_title_stream = _TtyStringIO()
    cli._pending_title = "Project"

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == "\033]0;Finished Project\a"
