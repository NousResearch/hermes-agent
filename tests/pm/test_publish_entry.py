"""Publication behavior when a live Windows process holds an existing entry."""

from pathlib import Path

import pytest

from pm import install
from pm.store import Store


class _VerifiedPackage:
    name = "tool"

    def verify(self, entry, target):
        return None if (entry / "bin/tool").is_file() else "tool is missing"


def _write_entry(path: Path, contents: bytes = b"tool") -> None:
    (path / "bin").mkdir(parents=True)
    (path / "bin/tool").write_bytes(contents)


def _hold_entry_rename(monkeypatch, entry, previous) -> None:
    real_rename = Path.rename

    def held_entry_rename(self, destination):
        if self == entry and destination == previous:
            raise PermissionError("live entry is held")
        return real_rename(self, destination)

    monkeypatch.setattr(install, "retry_held", lambda operation: operation())
    monkeypatch.setattr(Path, "rename", held_entry_rename)


def test_publish_keeps_a_byte_identical_held_entry(tmp_path, monkeypatch):
    """Regression for #139088: an unchanged live entry need not be republished."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged)
    _hold_entry_rename(monkeypatch, entry, previous)

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
        pass

    assert (entry / "bin/tool").read_bytes() == b"tool"
    assert not previous.exists()


def test_publish_raises_when_a_held_entry_differs_from_staged_bytes(tmp_path, monkeypatch):
    """A held entry remains an error unless it is exactly the staged replacement."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged, b"replacement")
    _hold_entry_rename(monkeypatch, entry, previous)

    with pytest.raises(PermissionError, match="live entry is held"):
        with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
            pass
