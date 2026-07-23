"""Behavior contracts for the persistent-memory documentation."""

from pathlib import Path


DOC = (
    Path(__file__).resolve().parents[2]
    / "website/docs/user-guide/features/memory.md"
).read_text()


def test_docs_do_not_recommend_completed_work_as_memory():
    lower = DOC.lower()
    assert "completed task diary entries" not in lower
    assert "**completed work:**" not in lower


def test_docs_do_not_route_procedural_corrections_to_memory():
    lower = DOC.lower()
    assert "don't use `sudo`" not in lower
    assert "procedures and workarounds" in lower


def test_docs_define_a_five_destination_information_router():
    lower = DOC.lower()
    for destination in ("user.md", "memory.md", "skills", "project state", "session search"):
        assert destination in lower


def test_docs_keep_memory_below_capacity_headroom():
    lower = DOC.lower()
    assert "70%" in lower
    assert "headroom" in lower
