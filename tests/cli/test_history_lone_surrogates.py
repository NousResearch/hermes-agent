"""Tests for the history-file guard against lone surrogates.

Behaviour contracts only: the history file must always be valid UTF-8 and a failed
write must never propagate (history is a convenience, not a reason to crash the CLI).
"""

from pathlib import Path

from hermes_cli.safe_history import SafeFileHistory


def _entries(path: Path) -> list[str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")  # raises if the guard ever writes invalid UTF-8
    return [line[1:] for line in text.splitlines() if line.startswith("+")]


def test_surrogate_text_is_stored_as_valid_utf8(tmp_path):
    hist = SafeFileHistory(str(tmp_path / "history"))
    hist.store_string("a\udcffb")  # must not raise
    entries = _entries(tmp_path / "history")
    assert entries == ["a\ufffdb"]
    assert not any("\ud800" <= ch <= "\udfff" for e in entries for ch in e)


def test_clean_text_is_stored_unchanged(tmp_path):
    hist = SafeFileHistory(str(tmp_path / "history"))
    hist.store_string("git status")
    assert _entries(tmp_path / "history") == ["git status"]


def test_unwritable_history_does_not_raise(tmp_path):
    # A directory cannot be opened for append; the guard must swallow the OSError.
    target = tmp_path / "history"
    target.mkdir()
    SafeFileHistory(str(target)).store_string("anything")  # must not raise


def test_bracketed_paste_wrappers_are_normalised_before_storing(tmp_path):
    hist = SafeFileHistory(str(tmp_path / "history"))
    hist.store_string("hello[~[[e")
    entries = _entries(tmp_path / "history")
    assert entries and "\ufffd" not in entries[0]
