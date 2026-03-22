from datetime import datetime

from hermes_cli.paste_collapse import (
    expand_paste_references,
    materialize_paste_for_insertion,
    should_collapse_pasted_text,
    write_pasted_text_reference,
)


class FakeBuffer:
    def __init__(self, text: str, cursor_position: int):
        self.text = text
        self.cursor_position = cursor_position

    def insert_text(self, value: str):
        self.text = self.text[: self.cursor_position] + value + self.text[self.cursor_position :]
        self.cursor_position += len(value)


def test_should_not_collapse_small_single_word_paste():
    assert should_collapse_pasted_text("hello") is False


def test_should_not_collapse_short_multiline_paste_under_threshold():
    assert should_collapse_pasted_text("a\nb\nc\nd") is False


def test_should_collapse_large_multiline_paste_at_threshold():
    pasted = "1\n2\n3\n4\n5\n6"
    assert should_collapse_pasted_text(pasted) is True


def test_write_reference_creates_file_and_returns_placeholder(tmp_path):
    ref = write_pasted_text_reference(
        "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta",
        paste_dir=tmp_path,
        counter=1,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    assert ref.startswith("[Pasted text #1: 6 lines → ")
    created = next(tmp_path.iterdir())
    assert created.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta"


def test_expand_paste_references_expands_exact_reference(tmp_path):
    ref = write_pasted_text_reference(
        "big\nchunk\nof\ntext\nfor\nagent",
        paste_dir=tmp_path,
        counter=2,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    assert expand_paste_references(ref) == "big\nchunk\nof\ntext\nfor\nagent"


def test_expand_paste_references_expands_inline_reference(tmp_path):
    ref = write_pasted_text_reference(
        "embedded\npaste\ncontent\nline4\nline5\nline6",
        paste_dir=tmp_path,
        counter=3,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    text = f"Intro before\n{ref}\nOutro after"
    expanded = expand_paste_references(text)
    assert "embedded\npaste\ncontent" in expanded
    assert "Intro before" in expanded
    assert "Outro after" in expanded


def test_expand_paste_references_leaves_missing_file_reference_literal():
    text = "before [Pasted text #9: 6 lines → /no/such/file.txt] after"
    assert expand_paste_references(text) == text


def test_small_paste_into_large_existing_draft_does_not_collapse(tmp_path):
    existing = "line1\nline2\nline3\nline4\nline5\nline6"
    paste_dir = tmp_path / "pastes"
    inserted, collapsed = materialize_paste_for_insertion(
        "word",
        current_buffer_text=existing,
        paste_dir=paste_dir,
        counter=1,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    assert collapsed is False
    assert inserted == "word"
    assert paste_dir.exists() is False


def test_large_paste_is_inserted_as_reference_without_replacing_surrounding_text(tmp_path):
    before = "intro\nmore intro\n"
    after = "\noutro"
    buf = FakeBuffer(before + after, cursor_position=len(before))
    inserted, collapsed = materialize_paste_for_insertion(
        "a\nb\nc\nd\ne\nf",
        current_buffer_text=buf.text,
        paste_dir=tmp_path,
        counter=1,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    assert collapsed is True
    buf.insert_text(inserted)
    assert buf.text.startswith(before)
    assert buf.text.endswith(after)
    assert "[Pasted text #1:" in buf.text


def test_large_paste_is_not_collapsed_when_current_buffer_is_slash_command(tmp_path):
    pasted = "a\nb\nc\nd\ne\nf"
    inserted, collapsed = materialize_paste_for_insertion(
        pasted,
        current_buffer_text="/plan ",
        paste_dir=tmp_path,
        counter=1,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    assert collapsed is False
    assert inserted == pasted


def test_inline_paste_reference_expands_inside_larger_message(tmp_path):
    ref = write_pasted_text_reference(
        "pasted\nchunk\nline3\nline4\nline5\nline6",
        paste_dir=tmp_path,
        counter=1,
        now=datetime(2026, 3, 22, 12, 34, 56),
    )
    raw = f"Please summarize this:\n{ref}\nThanks"
    expanded = expand_paste_references(raw)
    assert "Please summarize this:" in expanded
    assert "pasted\nchunk\nline3" in expanded
    assert "Thanks" in expanded
