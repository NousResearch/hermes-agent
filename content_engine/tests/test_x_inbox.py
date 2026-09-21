"""Focused tests for the X experience inbox capture store."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

import x_inbox as xi


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "x_inbox_test.db"
    monkeypatch.setattr(xi, "DB_PATH", db)
    yield db


def test_add_and_retrieve():
    cap_id = xi.add_capture("hit a weird Codex rate-limit today", source="user")
    caps = xi.list_captures()
    assert len(caps) == 1
    assert caps[0]["id"] == cap_id
    assert caps[0]["text"] == "hit a weird Codex rate-limit today"
    assert caps[0]["used"] == 0


def test_empty_capture_rejected():
    with pytest.raises(ValueError):
        xi.add_capture("   ")


def test_duplicate_text_is_noop_same_id():
    first = xi.add_capture("  Agents are exposing bad engineering.  ")
    second = xi.add_capture("agents are EXPOSING bad engineering.")
    assert first == second
    assert len(xi.list_captures()) == 1


def test_mark_used_and_filter():
    a = xi.add_capture("capture one")
    b = xi.add_capture("capture two")
    xi.mark_used([a])
    unused = xi.list_captures(used=False)
    used = xi.list_captures(used=True)
    assert [c["id"] for c in unused] == [b]
    assert [c["id"] for c in used] == [a]


def test_counts():
    xi.add_capture("one")
    b = xi.add_capture("two")
    xi.mark_used([b])
    assert xi.counts() == {"total": 2, "used": 1, "unused": 1}
