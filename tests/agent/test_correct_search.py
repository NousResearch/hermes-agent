"""Behaviour tests for the host-side search behind ``/correct``.

A wrong fact is often stored in several places at once. The agent cannot find them itself
(there is no full-text search over skills in the model tools, and a remote terminal backend
cannot see the skills directory), so the host searches literal substrings across the memory
files and the skills the agent created. These tests cover what it finds, what it refuses to
look at, and that the ids it hands back are usable.
"""

from pathlib import Path
from typing import Optional

import pytest

from agent.learning_mutations import MAX_SEARCH_HITS, search_learned_text

WRONG = "the deploy port is 8080"


def memories_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "memories"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_memory(filename: str, entries: list[str]) -> Path:
    from tools.memory_tool_store import ENTRY_DELIMITER

    path = memories_dir() / filename
    path.write_text(ENTRY_DELIMITER.join(entries) + "\n", encoding="utf-8")
    return path


def skills_dir() -> Path:
    from agent.skill_utils import get_skills_dir

    path = get_skills_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_skill(name: str, lines: list[str], *, references: Optional[dict[str, str]] = None) -> Path:
    skill_dir = skills_dir() / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Fixture skill for the /correct search.\n"
        "version: 0.1.0\n"
        "author: Hermes\n"
        "license: MIT\n"
        "metadata:\n"
        "  hermes:\n"
        "    tags: [Fixture]\n"
        "---\n\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8")
    for rel, content in (references or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_dir


class TestWhatItFinds:
    def test_finds_a_memory_entry_with_its_line_and_snippet(self):
        write_memory("MEMORY.md", ["User prefers tabs.", WRONG, "Deploys run on Tuesdays."])

        found = search_learned_text([WRONG])

        assert found["terms"] == [WRONG]
        memory_hits = [hit for hit in found["hits"] if hit["kind"] == "memory"]
        assert len(memory_hits) == 1
        hit = memory_hits[0]
        assert hit["label"].startswith("MEMORY.md entry 2")
        assert hit["line"] > 0
        assert WRONG in hit["snippet"]
        assert hit["memory_id"].startswith("memory:memory:")

    def test_finds_a_hit_in_the_user_profile_file(self):
        write_memory("USER.md", [WRONG])

        hits = [hit for hit in search_learned_text([WRONG])["hits"] if hit["kind"] == "memory"]

        assert [hit["memory_id"].split(":")[1] for hit in hits] == ["profile"]

    def test_the_memory_id_it_returns_resolves(self):
        write_memory("MEMORY.md", ["Something else.", WRONG])

        from agent.learning_mutations import node_detail

        hit = next(h for h in search_learned_text([WRONG])["hits"] if h["kind"] == "memory")
        detail = node_detail(hit["memory_id"])

        assert detail["ok"] is True and WRONG in detail["content"]

    def test_finds_skill_lines_in_skill_md_and_in_references(self):
        write_skill("deploy-guide",
                    ["# Deploy", "", "1. Set the port.", f"2. Then check {WRONG}."],
                    references={"references/network.md": f"Firewall: {WRONG}.\n"})

        skill_hits = [hit for hit in search_learned_text([WRONG])["hits"] if hit["kind"] == "skill"]

        assert {(hit["skill"], hit["relative_path"]) for hit in skill_hits} == {
            ("deploy-guide", "SKILL.md"), ("deploy-guide", "references/network.md")}
        assert all(hit["line"] > 0 and WRONG in hit["snippet"] for hit in skill_hits)

    def test_search_is_literal_and_case_insensitive(self):
        write_memory("MEMORY.md", ["The Deploy Port is 8080."])

        assert search_learned_text(["deploy port"])["hits"]
        assert search_learned_text(["deploy por"])["hits"]
        assert not search_learned_text(["deploy prt"])["hits"]

    def test_several_terms_each_match(self):
        write_memory("MEMORY.md", ["Staging port is 8080.", "Production host is 10.0.0.4."])

        found = search_learned_text(["production host", "staging port"])

        assert {hit["term"] for hit in found["hits"]} == {"production host", "staging port"}


class TestWhatItRefuses:
    def test_bundled_hub_and_protected_skills_are_not_searched(self, monkeypatch):
        from tools import skill_usage

        for name in ("bundled-skill", "hub-skill", "protected-skill", "learned-skill"):
            write_skill(name, [f"# {name}", WRONG])

        bundled = {"bundled-skill"}
        hub = {"hub-skill"}
        protected = {"protected-skill"}
        monkeypatch.setattr(skill_usage, "is_agent_created",
                            lambda name: name not in bundled and name not in hub)
        monkeypatch.setattr(skill_usage, "is_protected_builtin", lambda name: name in protected)

        skills = {hit["skill"] for hit in search_learned_text([WRONG])["hits"] if hit["kind"] == "skill"}

        assert skills == {"learned-skill"}

    def test_empty_input_searches_for_nothing(self):
        write_memory("MEMORY.md", [WRONG])

        assert search_learned_text([])["hits"] == []
        assert search_learned_text(["", "   "])["hits"] == []
        assert search_learned_text(None)["hits"] == []

    def test_duplicate_and_blank_terms_are_collapsed(self):
        write_memory("MEMORY.md", [WRONG])

        assert search_learned_text(["deploy port", "DEPLOY PORT", "  "])["terms"] == ["deploy port"]

    def test_the_hit_cap_is_respected_and_reported(self):
        write_memory("MEMORY.md", [f"{WRONG} ({index})" for index in range(1, 4)])
        write_skill("deploy-guide", [WRONG, WRONG, WRONG])

        found = search_learned_text([WRONG], max_hits=2)

        assert len(found["hits"]) == 2
        assert found["truncated"] is True
        assert MAX_SEARCH_HITS >= 2

    def test_a_missing_memory_directory_and_missing_skills_dir_are_not_errors(self):
        assert search_learned_text(["anything at all"])["hits"] == []


class TestExistingWriteGates:
    """The gateway already limits stored writes through the write-approval gates. /correct
    applies its edits through the same tools, so it inherits them — nothing new is wired.
    """

    def test_memory_edits_honour_the_existing_write_approval_gate(self):
        from hermes_constants import get_hermes_home
        from tools.memory_tool import _memory_tool, load_on_disk_store

        write_memory("MEMORY.md", [WRONG])
        (get_hermes_home() / "config.yaml").write_text(
            "memory:\n  write_approval: true\n", encoding="utf-8")

        outcome, payload = _memory_tool("patch", "memory", "the deploy port is 9090", WRONG,
                                        None, None, load_on_disk_store())

        # Staged is also reported as a non-applied outcome — the payload is what says why.
        assert outcome != "ok", payload
        assert '"staged": true' in payload, payload
        assert "write_approval" in payload
        # Staged means not applied: a group-chat user proposing an edit cannot silently
        # change the owner's memory while the gate is on.
        assert WRONG in (memories_dir() / "MEMORY.md").read_text(encoding="utf-8")


class TestEndToEndWithTheToolPaths:
    def test_search_then_edit_records_a_ledger_entry_and_rollback_restores(self):
        """The full correction journey: seed, find, edit through the tools' own code paths,
        then prove the skill edit is undoable from the ledger."""
        from tools.memory_tool import _memory_tool, load_on_disk_store
        from tools.skill_manager_tool import _skill_manage_from
        from tools import skill_ledger

        write_memory("MEMORY.md", ["Unrelated note.", WRONG])
        first = write_skill("deploy-guide", ["# Deploy", WRONG])
        second = write_skill("ops-notes", ["# Ops", f"Note: {WRONG}."])
        original_first = (first / "SKILL.md").read_text(encoding="utf-8")

        found = search_learned_text([WRONG])
        skill_names = {hit["skill"] for hit in found["hits"] if hit["kind"] == "skill"}
        assert skill_names == {"deploy-guide", "ops-notes"}
        assert any(hit["kind"] == "memory" for hit in found["hits"])

        for skill_dir in (first, second):
            fixed = (skill_dir / "SKILL.md").read_text(encoding="utf-8").replace(
                WRONG, "the deploy port is 9090")
            # The skill_manage tool handler, action=edit — the path the prompt tells the
            # agent to use, and the one that records the ledger entry.
            payload = _skill_manage_from({"action": "edit", "name": skill_dir.name, "content": fixed})
            assert '"success": true' in payload or '"success": true' in payload.replace(" ", ""), payload

        outcome, payload = _memory_tool("patch", "memory", "the deploy port is 9090", WRONG,
                                        None, None, load_on_disk_store())
        assert outcome != "rejected", payload

        entries = skill_ledger.list_entries(skill="deploy-guide")
        assert entries, "no ledger entry for the skill edit — it could not be undone"
        entry_id = entries[0]["id"]

        assert WRONG not in (first / "SKILL.md").read_text(encoding="utf-8")
        ok, message = skill_ledger.rollback_entry(entry_id)
        assert ok is True, message
        assert (first / "SKILL.md").read_text(encoding="utf-8") == original_first
        # The memory edit correctly has no undo — the prompt says so before anything is applied.
        assert "the deploy port is 9090" in (memories_dir() / "MEMORY.md").read_text(encoding="utf-8")
