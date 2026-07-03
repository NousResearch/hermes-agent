from pathlib import Path

import pytest

from orchard.registry import Registry


def test_add_lookup_remove(tmp_path: Path):
    reg = Registry(tmp_path / "r.db")
    emp = reg.add("alice", "Alice A", "mm-123")
    assert reg.get("alice").mm_user_id == "mm-123"
    assert reg.by_mm_user("mm-123").id == "alice"
    assert [e.id for e in reg.all()] == ["alice"]
    assert reg.remove("alice") is True
    assert reg.get("alice") is None


def test_rejects_bad_id(tmp_path: Path):
    reg = Registry(tmp_path / "r.db")
    with pytest.raises(ValueError):
        reg.add("Alice Uppercase", "x", "mm-1")


def test_rejects_duplicate_mm_user(tmp_path: Path):
    reg = Registry(tmp_path / "r.db")
    reg.add("alice", "Alice", "mm-1")
    with pytest.raises(ValueError):
        reg.add("bob", "Bob", "mm-1")  # same mm_user_id
