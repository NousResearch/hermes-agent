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


def test_expand_paste_references_recursively_expands_nested_placeholders(tmp_path):
    cli_obj = _make_cli()
    paste_1 = tmp_path / "paste_1.txt"
    paste_2 = tmp_path / "paste_2.txt"
    paste_3 = tmp_path / "paste_3.txt"

    paste_1.write_text("alpha", encoding="utf-8")
    paste_2.write_text(f"before [Pasted text #1: 1 lines → {paste_1}] after", encoding="utf-8")
    paste_3.write_text(f"outer [Pasted text #2: 1 lines → {paste_2}] done", encoding="utf-8")

    text = f"start [Pasted text #3: 1 lines → {paste_3}] end"

    assert cli_obj._expand_paste_references(text) == "start outer before alpha after done end"


def test_expand_paste_references_expands_multiple_placeholders(tmp_path):
    cli_obj = _make_cli()
    paste_1 = tmp_path / "paste_1.txt"
    paste_2 = tmp_path / "paste_2.txt"
    paste_1.write_text("alpha", encoding="utf-8")
    paste_2.write_text("beta", encoding="utf-8")

    text = f"A [Pasted text #1: 1 lines → {paste_1}] B [Pasted text #2: 1 lines → {paste_2}] C"

    assert cli_obj._expand_paste_references(text) == "A alpha B beta C"


def test_expand_paste_references_allows_duplicate_independent_placeholders(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("alpha", encoding="utf-8")

    text = f"A [Pasted text #1: 1 lines → {paste_file}] B [Pasted text #2: 1 lines → {paste_file}] C"

    assert cli_obj._expand_paste_references(text) == "A alpha B alpha C"


def test_expand_paste_references_result_reports_missing_file(tmp_path):
    cli_obj = _make_cli()
    missing = tmp_path / "missing.txt"
    text = f"before [Pasted text #1: 1 lines → {missing}] after"

    result = cli_obj._expand_paste_references_result(text)

    assert result.text == text
    assert result.unresolved_refs == [f"[Pasted text #1: 1 lines → {missing}]"]
    assert result.errors


def test_expand_paste_references_result_reports_cycle(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text(f"loop [Pasted text #1: 1 lines → {paste_file}]", encoding="utf-8")

    result = cli_obj._expand_paste_references_result(f"start [Pasted text #1: 1 lines → {paste_file}]")

    assert result.unresolved_refs
    assert str(paste_file.resolve()) in result.cycle_paths


def test_expand_paste_references_result_reports_depth_limit(tmp_path):
    cli_obj = _make_cli()
    paste_1 = tmp_path / "paste_1.txt"
    paste_2 = tmp_path / "paste_2.txt"
    paste_1.write_text(f"[Pasted text #2: 1 lines → {paste_2}]", encoding="utf-8")
    paste_2.write_text("final", encoding="utf-8")

    result = cli_obj._expand_paste_references_result(
        f"[Pasted text #1: 1 lines → {paste_1}]",
        max_depth=1,
    )

    assert result.unresolved_refs
    assert result.depth_exceeded is True


def test_expand_user_input_pastes_or_report_blocks_unresolved_refs(tmp_path):
    cli_obj = _make_cli()
    missing = tmp_path / "missing.txt"
    text = f"[Pasted text #1: 1 lines → {missing}]"

    with patch("cli._cprint") as mock_cprint:
        ok, returned = cli_obj._expand_user_input_pastes_or_report(text)

    assert ok is False
    assert returned == text
    rendered = "\n".join(str(call.args[0]) for call in mock_cprint.call_args_list if call.args)
    assert "Paste expansion failed" in rendered
    assert "message was not submitted" in rendered
    assert "Unresolved paste references" in rendered


def test_expand_user_input_pastes_or_report_returns_expanded_text(tmp_path):
    cli_obj = _make_cli()
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text("alpha", encoding="utf-8")
    text = f"before [Pasted text #1: 1 lines → {paste_file}] after"

    ok, returned = cli_obj._expand_user_input_pastes_or_report(text)

    assert ok is True
    assert returned == "before alpha after"


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
