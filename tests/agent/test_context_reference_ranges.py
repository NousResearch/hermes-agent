"""Ranged attachments follow the documented one-indexed, inclusive file contract."""

from pathlib import Path

import pytest

from agent.context_references import preprocess_context_references


@pytest.mark.parametrize("line_range", ["0", "0-2", "3-1", "3-0"])
def test_invalid_file_ranges_expand_the_same_text_as_unranged_reference(tmp_path: Path, line_range: str):
    text = "First ordinary line.\nSecond ordinary line.\nLast ordinary line.\n"
    (tmp_path / "notes space.txt").write_text(text, encoding="utf-8")

    plain = preprocess_context_references("Read @file:`notes space.txt`", cwd=tmp_path, context_length=100_000)
    ranged = preprocess_context_references(
        f"Read @file:`notes space.txt`:{line_range}", cwd=tmp_path, context_length=100_000,
    )

    assert plain.expanded and text in plain.message
    assert ranged.expanded and not ranged.blocked and not ranged.warnings
    assert text in ranged.message


@pytest.mark.parametrize("line_range, expected", [("2", "second\n"), ("2-2", "second\n"), ("1-2", "first\nsecond\n")])
def test_valid_file_ranges_keep_their_inclusive_slice(tmp_path: Path, line_range: str, expected: str):
    (tmp_path / "notes.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")

    result = preprocess_context_references(
        f"Read @file:notes.txt:{line_range}", cwd=tmp_path, context_length=100_000,
    )

    assert result.expanded and not result.blocked and not result.warnings
    assert f"\n{expected}\n```" in result.message
    assert "third" not in result.message
