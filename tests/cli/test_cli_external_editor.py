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
    """Single-level paste reference expansion (original behavior)."""
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("line one\nline two", encoding="utf-8")

    text = f"before [Pasted text #1: 2 lines \u2192 {paste_file}] after"
    expanded = cli_obj._expand_paste_references(text)

    assert expanded == "before line one\nline two after"


def test_expand_paste_references_expands_nested_references_recursively(tmp_path):
    """Nested paste files (chained pastes) should all expand into full content."""
    cli_obj = _make_cli()

    # Simulate chained pastes: paste_2.txt contains a reference to paste_1.txt
    paste_1 = tmp_path / "paste_1.txt"
    paste_1.write_text("Hello from paste_1\nLine 2 of paste_1", encoding="utf-8")

    paste_2 = tmp_path / "paste_2.txt"
    paste_2.write_text(
        f"[Pasted text #1: 2 lines \u2192 {paste_1}]\nLine from paste_2",
        encoding="utf-8",
    )

    # paste_3 contains a reference to paste_2, which contains a reference to paste_1
    paste_3 = tmp_path / "paste_3.txt"
    paste_3.write_text(
        f"[Pasted text #2: 2 lines \u2192 {paste_2}]\nLine from paste_3",
        encoding="utf-8",
    )

    # The top-level input references paste_3
    text = f"before [Pasted text #3: 2 lines \u2192 {paste_3}] after"
    expanded = cli_obj._expand_paste_references(text)

    expected = (
        "before Hello from paste_1\n"
        "Line 2 of paste_1\n"
        "Line from paste_2\n"
        "Line from paste_3 after"
    )
    assert expanded == expected, f"Got: {expanded!r}"


def test_expand_paste_references_guards_against_circular_references(tmp_path):
    """Circular references should not cause infinite loops."""
    cli_obj = _make_cli()

    paste_a = tmp_path / "paste_a.txt"
    paste_b = tmp_path / "paste_b.txt"

    # A references B, B references A (circular)
    paste_a.write_text(
        f"[Pasted text #1: 1 lines \u2192 {paste_b}]\nExtra from A",
        encoding="utf-8",
    )
    paste_b.write_text(
        f"[Pasted text #2: 1 lines \u2192 {paste_a}]\nExtra from B",
        encoding="utf-8",
    )

    text = f"start [Pasted text #1: 2 lines \u2192 {paste_a}] end"
    expanded = cli_obj._expand_paste_references(text)

    # Should not hang; should return something containing the expanded content
    assert "Extra from A" in expanded or "Extra from B" in expanded
    assert "start" in expanded
    assert "end" in expanded


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


def test_inline_pastes_stores_full_content(tmp_path):
    """History should recall the actual pasted text, not the placeholder."""
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("line one\nline two", encoding="utf-8")
    buffer = _FakeBuffer(text=f"[Pasted text #1: 2 lines \u2192 {paste_file}]")

    cli_obj._inline_pastes(buffer)

    assert buffer.text == "line one\nline two"
    assert buffer.cursor_position == len("line one\nline two")
    # Skip flag set so the resulting text-change doesn't re-collapse.
    assert cli_obj._skip_paste_collapse is True


def test_inline_pastes_leaves_plain_text_untouched():
    """No placeholder → buffer text and collapse flag are unchanged."""
    cli_obj = _make_cli()
    buffer = _FakeBuffer(text="just a normal message")

    cli_obj._inline_pastes(buffer)

    assert buffer.text == "just a normal message"
    assert cli_obj._skip_paste_collapse is False


def test_inline_pastes_missing_file_keeps_placeholder(tmp_path):
    """A recalled reference whose file is gone stays as the placeholder."""
    cli_obj = _make_cli()
    placeholder = f"[Pasted text #1: 2 lines \u2192 {tmp_path / 'gone.txt'}]"
    buffer = _FakeBuffer(text=placeholder)

    cli_obj._inline_pastes(buffer)

    assert buffer.text == placeholder
    assert cli_obj._skip_paste_collapse is False
