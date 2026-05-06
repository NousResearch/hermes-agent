"""Tests for CLI external-editor support."""

from unittest.mock import patch

from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text=""):
        self.calls = []
        self.text = text
        self.cursor_position = len(text)

    def open_in_editor(self, validate_and_handle=False):
        self.calls.append(validate_and_handle)


class _FakeApp:
    def __init__(self):
        self.current_buffer = _FakeBuffer()


def _make_cli(with_app=True):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._app = _FakeApp() if with_app else None
    cli_obj._command_running = False
    cli_obj._command_status = ""
    cli_obj._command_display = ""
    cli_obj._sudo_state = None
    cli_obj._secret_state = None
    cli_obj._approval_state = None
    cli_obj._clarify_state = None
    cli_obj._skip_paste_collapse = False
    return cli_obj

def test_open_external_editor_uses_prompt_toolkit_buffer_editor():
    cli_obj = _make_cli()

    assert cli_obj._open_external_editor() is True
    assert cli_obj._app.current_buffer.calls == [False]


def test_open_external_editor_rejects_when_no_tui():
    cli_obj = _make_cli(with_app=False)

    with patch("cli._cprint") as mock_cprint:
        assert cli_obj._open_external_editor() is False

    assert mock_cprint.called
    assert "interactive cli" in str(mock_cprint.call_args).lower()


def test_open_external_editor_rejects_modal_prompts():
    cli_obj = _make_cli()
    cli_obj._approval_state = {"selected": 0}

    with patch("cli._cprint") as mock_cprint:
        assert cli_obj._open_external_editor() is False

    assert mock_cprint.called
    assert "active prompt" in str(mock_cprint.call_args).lower()

def test_open_external_editor_uses_explicit_buffer_when_provided():
    cli_obj = _make_cli()
    external_buffer = _FakeBuffer()

    assert cli_obj._open_external_editor(buffer=external_buffer) is True
    assert external_buffer.calls == [False]
    assert cli_obj._app.current_buffer.calls == []


def test_expand_paste_references_replaces_placeholder_with_file_contents(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("line one\nline two", encoding="utf-8")

    text = f"before [Pasted text #1: 2 lines → {paste_file}] after"
    expanded = cli_obj._expand_paste_references(text)

    assert expanded == "before line one\nline two after"


def test_open_external_editor_expands_paste_placeholders_before_open(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("alpha\nbeta", encoding="utf-8")
    buffer = _FakeBuffer(text=f"[Pasted text #1: 2 lines → {paste_file}]")

    assert cli_obj._open_external_editor(buffer=buffer) is True
    assert buffer.text == "alpha\nbeta"
    assert buffer.cursor_position == len("alpha\nbeta")
    assert buffer.calls == [False]


def test_open_external_editor_sets_skip_collapse_flag_during_expansion(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("a\nb\nc\nd\ne\nf", encoding="utf-8")
    buffer = _FakeBuffer(text=f"[Pasted text #1: 6 lines \u2192 {paste_file}]")

    # After expansion the flag should have been set (to prevent re-collapse)
    assert cli_obj._open_external_editor(buffer=buffer) is True
    # Flag is consumed by _on_text_changed, but since no handler is attached
    # in tests it stays True until the handler resets it.
    assert cli_obj._skip_paste_collapse is True
