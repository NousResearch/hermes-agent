"""Regression test for MemoryStore invalid target rejection."""
import tempfile
from pathlib import Path
import sys

HERMES_ROOT = Path(__file__).resolve().parent.parent
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from tools.memory_tool import MemoryStore


def test_invalid_target_add():
    store = MemoryStore()
    result = store.add("invalid_target", "hello")
    assert result["success"] is False
    assert "invalid_target" in result["error"]
    assert "memory" in result["error"]


def test_invalid_target_replace():
    store = MemoryStore()
    result = store.replace("bogus", "old", "new")
    assert result["success"] is False
    assert "bogus" in result["error"]


def test_invalid_target_remove():
    store = MemoryStore()
    result = store.remove("not_a_target", "old")
    assert result["success"] is False
    assert "not_a_target" in result["error"]


def test_valid_target_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore()
        store._memory_dir = Path(tmp)
        result = store.add("memory", "hello world")
        assert result["success"] is True


if __name__ == "__main__":
    test_invalid_target_add()
    test_invalid_target_replace()
    test_invalid_target_remove()
    test_valid_target_still_works()
    print("ALL TESTS PASSED")
