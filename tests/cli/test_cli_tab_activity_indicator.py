import io
from types import SimpleNamespace

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


def test_tab_title_done_writes_osc_title_sequence():
    cli = _make_cli()

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == "\033]0;Project\a"


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


def test_tab_title_sanitizes_control_characters():
    cli = _make_cli()
    cli._tab_title_done = "done \033bad\a\nnext\tmore\x7fhidden\x9f"

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == "\033]0;done bad next morehidden\a"


def test_tab_title_truncates_long_titles():
    cli = _make_cli()
    cli._tab_title_done = "x" * 300

    cli._set_tab_title_state("done", force=True)

    assert cli._tab_title_stream.getvalue() == f"\033]0;{'x' * 255}\a"


def test_configure_tab_activity_indicator_handles_invalid_spinner_frames(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    monkeypatch.setattr(
        cli_module,
        "CLI_CONFIG",
        {"display": {"tab_activity_indicator": True, "tab_title_spinner_frames": 123}},
    )

    cli._configure_tab_activity_indicator()

    assert cli._tab_activity_indicator is True
    assert cli._tab_title_spinner_frames == ["*"]
