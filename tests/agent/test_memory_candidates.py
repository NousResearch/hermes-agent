"""Tests for governed memory candidate queue."""

from __future__ import annotations

from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _with_home(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_stage_memory_candidate_does_not_mutate_memory_until_promoted(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Existing fact", encoding="utf-8")

    def run():
        from agent.memory_candidates import promote_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(
            target="memory",
            content="New verified fact",
            source={"session_id": "s1"},
            rationale="unit test",
        )
        before = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
        promote_memory_candidate(candidate.candidate_id)
        after = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
        return candidate, before, after

    candidate, before, after = _with_home(home, run)
    assert before == "Existing fact"
    assert candidate.status == "staged"
    assert after == "Existing fact\n§\nNew verified fact"


def test_memory_candidate_promotion_is_idempotent(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Existing fact", encoding="utf-8")

    def run():
        from agent.memory_candidates import load_memory_candidate, promote_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(target="memory", content="New verified fact")
        promote_memory_candidate(candidate.candidate_id)
        promote_memory_candidate(candidate.candidate_id)
        return load_memory_candidate(candidate.candidate_id).status, (mem_dir / "MEMORY.md").read_text(encoding="utf-8")

    assert _with_home(home, run) == ("promoted", "Existing fact\n§\nNew verified fact")


def test_duplicate_promotion_records_noop_and_rollback_preserves_existing_entry(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Existing fact", encoding="utf-8")

    def run():
        from agent.memory_candidates import load_memory_candidate, promote_memory_candidate, rollback_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(target="memory", content="Existing fact")
        promote_memory_candidate(candidate.candidate_id)
        loaded = load_memory_candidate(candidate.candidate_id)
        rollback_memory_candidate(candidate.candidate_id)
        return loaded.status, loaded.payload.get("promote_action"), (mem_dir / "MEMORY.md").read_text(encoding="utf-8")

    assert _with_home(home, run) == ("promoted", "noop_duplicate", "Existing fact")


def test_stage_memory_candidate_rejects_separator_in_content(tmp_path):
    home = tmp_path / ".hermes"

    def run():
        from agent.memory_candidates import stage_memory_candidate

        try:
            stage_memory_candidate(target="memory", content="one\n§\ntwo")
        except ValueError as exc:
            return str(exc)
        return "no error"

    assert _with_home(home, run) == "content cannot contain memory entry separator"


def test_reject_memory_candidate_marks_status_and_prevents_promotion(tmp_path):
    home = tmp_path / ".hermes"

    def run():
        from agent.memory_candidates import load_memory_candidate, promote_memory_candidate, reject_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(target="user", content="Bad preference")
        reject_memory_candidate(candidate.candidate_id, reason="untrusted")
        loaded = load_memory_candidate(candidate.candidate_id)
        try:
            promote_memory_candidate(candidate.candidate_id)
        except ValueError as exc:
            promoted = str(exc)
        else:
            promoted = "promoted"
        return loaded.status, promoted

    assert _with_home(home, run) == ("rejected", "candidate is not staged")


def test_memory_candidate_rollback_removes_exact_last_promoted_entry(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Existing fact", encoding="utf-8")

    def run():
        from agent.memory_candidates import load_memory_candidate, promote_memory_candidate, rollback_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(target="memory", content="New verified fact")
        promote_memory_candidate(candidate.candidate_id)
        rollback_memory_candidate(candidate.candidate_id)
        return load_memory_candidate(candidate.candidate_id).status, (mem_dir / "MEMORY.md").read_text(encoding="utf-8")

    assert _with_home(home, run) == ("rolled_back", "Existing fact")


def test_memory_candidate_rollback_preserves_existing_bytes(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    original = "  Existing fact with spacing  \n\n§\nSecond fact\n"
    (mem_dir / "MEMORY.md").write_text(original, encoding="utf-8")

    def run():
        from agent.memory_candidates import promote_memory_candidate, rollback_memory_candidate, stage_memory_candidate

        candidate = stage_memory_candidate(target="memory", content="New verified fact")
        promote_memory_candidate(candidate.candidate_id)
        rollback_memory_candidate(candidate.candidate_id)
        return (mem_dir / "MEMORY.md").read_text(encoding="utf-8")

    assert _with_home(home, run) == original


def test_memory_wiki_find_read_grep_retrieve_helpers(tmp_path):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Project uses pytest\n§\nSpark host runs local LLM", encoding="utf-8")

    def run():
        from agent.memory_candidates import memory_wiki_find, memory_wiki_grep, memory_wiki_read, memory_wiki_retrieve

        found = memory_wiki_find("pytest")
        read = memory_wiki_read(found[0]["id"])
        grep = memory_wiki_grep("Spark")
        retrieved = memory_wiki_retrieve("local LLM", max_chars=500)
        return found[0]["title"], read["text"], grep[0]["title"], retrieved["entries"][0]["title"]

    assert _with_home(home, run) == (
        "Project uses pytest",
        "Project uses pytest",
        "Spark host runs local LLM",
        "Spark host runs local LLM",
    )
