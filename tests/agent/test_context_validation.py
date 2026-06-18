"""Never-lose-important-context validation scenarios."""

from pathlib import Path

import pytest

from agent.context_validation import (
    LocalNoteIndex,
    build_context_validation_report,
)
from tools.memory_tool import MemoryStore


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    return tmp_path


def _new_store() -> MemoryStore:
    store = MemoryStore(memory_char_limit=1200, user_char_limit=1200)
    store.load_from_disk()
    return store


def test_preference_survives_compaction_and_restart_via_durable_memory(memory_dir):
    store = _new_store()
    assert store.add(
        "user",
        "User prefers concise morning briefs with revenue blockers first.",
    )["success"]

    # Simulate compaction/app restart: the old working conversation is gone, and
    # a fresh MemoryStore reloads only durable curated memory from disk.
    restarted = _new_store()
    report = build_context_validation_report(
        durable_memory_entries=restarted.user_entries,
        working_memory_entries=(),
        durable_expectations={"brief_style": ("concise", "morning briefs")},
        discarded_expectations={"smalltalk": ("weather banter",)},
    )

    assert report.ok
    assert "brief_style" in report.durable_memory
    assert report.working_memory == {}
    assert report.discarded_context["smalltalk"] == ("weather banter",)
    rendered = report.to_markdown()
    assert "Durable memory" in rendered
    assert "Working memory" in rendered
    assert "Discarded context" in rendered


def test_commitment_survives_garbage_collection_until_resolved(memory_dir):
    store = _new_store()
    assert store.add(
        "memory",
        "Active commitment: submit the Vanta application packet after Chris confirms final consent.",
    )["success"]

    # Filler is treated as garbage-collectable and must not leak into retained
    # surfaces, while the active commitment remains durable until resolved.
    restarted = _new_store()
    report = build_context_validation_report(
        durable_memory_entries=restarted.memory_entries,
        durable_expectations={"vanta_commitment": ("Active commitment", "Vanta", "final consent")},
        discarded_expectations={"filler": ("funny aside about lunch",)},
    )

    assert report.ok
    assert "vanta_commitment" in report.durable_memory
    assert "filler" in report.discarded_context


def test_conflicting_memory_requires_clarification_instead_of_overwrite():
    report = build_context_validation_report(
        conflict_candidates={
            "timezone": (
                "User timezone is America/New_York.",
                "User timezone is America/Los_Angeles.",
            )
        }
    )

    assert not report.ok
    assert report.requires_clarification
    assert report.unresolved_conflicts[0].key == "timezone"
    assert "requires clarification" in report.to_markdown()


def test_durable_note_index_recall_is_separate_from_curated_memory(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "wiki" / "concepts" / "vanta-application.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Vanta application packet\n\n"
        "Remote U.S. Senior Software Engineer, Developer Experience.\n"
        "Packet includes Ashby form gates and resume-choice blockers.\n",
        encoding="utf-8",
    )

    report = build_context_validation_report(
        durable_memory_entries=(),
        note_index=LocalNoteIndex.from_path(Path(vault)),
        note_expectations={"vanta_packet_note": ("Vanta", "application packet", "Remote U.S.")},
    )

    assert report.ok
    hits = report.durable_notes["vanta_packet_note"]
    assert hits[0].surface == "durable_notes"
    assert hits[0].source == "wiki/concepts/vanta-application.md"
    assert report.durable_memory == {}
