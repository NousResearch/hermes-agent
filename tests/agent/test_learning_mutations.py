"""Behavior contracts for journey node edit/delete (agent.learning_mutations).

Exercises the real on-disk resolution (skills dir + MEMORY.md/USER.md chunking)
against a temp HERMES_HOME, never mocks — the id→file mapping is the whole point.
"""

from __future__ import annotations

import pytest

from agent import learning_mutations as lm
from hermes_constants import get_hermes_home

_SKILL = """---
name: my-skill
description: A test skill.
---

# My Skill

Body.
"""


@pytest.fixture
def home():
    base = get_hermes_home()
    (base / "memories").mkdir(parents=True, exist_ok=True)
    (base / "memories" / "MEMORY.md").write_text("alpha note\nline two\n§\nbeta note", encoding="utf-8")
    (base / "memories" / "USER.md").write_text("user profile note", encoding="utf-8")
    skill = base / "skills" / "my-skill"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    return base


def test_parse_node_kind():
    assert lm.parse_node_kind("memory:memory:0") == "memory"
    assert lm.parse_node_kind("memory:profile:3") == "memory"
    assert lm.parse_node_kind("debugging-hermes") == "skill"








def test_edit_memory_replaces_chunk(home):
    assert lm.edit_node("memory:profile:2", "rewritten profile")["ok"]
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8").strip() == "rewritten profile"








def test_skill_detail_returns_skill_md(home):
    d = lm.node_detail("my-skill")
    assert d["ok"] and d["kind"] == "skill"
    assert "name: my-skill" in d["content"]




def test_delete_pinned_skill_refused(home):
    from tools import skill_usage

    skill_usage.set_pinned("my-skill", True)
    res = lm.delete_node("my-skill")
    assert not res["ok"]
    assert "pinned" in res["message"]
    assert (home / "skills" / "my-skill").exists()






def test_memory_writes_match_memory_tool_format(home):
    """A journey mutation must leave the file byte-identical to what the memory
    tool itself writes — same §-join, no trailing-newline drift — so the two
    surfaces never fight over format and indices stay aligned."""
    from tools.memory_tool import ENTRY_DELIMITER, MemoryStore

    assert lm.edit_node("memory:memory:0", "alpha rewritten")["ok"]
    path = home / "memories" / "MEMORY.md"
    entries = MemoryStore._read_file(path)

    assert entries == ["alpha rewritten", "beta note"]
    assert path.read_text(encoding="utf-8") == ENTRY_DELIMITER.join(entries)


# ── Concurrency / drift: Journey shares MEMORY.md with the live agent ────────


def _memory_entries(home):
    from tools.memory_tool import MemoryStore

    return MemoryStore._read_file(home / "memories" / "MEMORY.md")


def test_delete_keeps_a_memory_the_agent_stored_meanwhile(home, monkeypatch):
    """A Journey delete must not undo an agent write that landed after the graph was drawn.

    The mutation used to read the file, drop one entry and write the whole file back with no
    lock: anything stored in between was silently gone, with no ``.bak`` to restore from.
    """
    real_locate = lm._locate_memory

    def _locate_then_agent_writes(node_id):
        located = real_locate(node_id)  # the graph the user clicked, read before the agent's write
        from tools.memory_tool import load_on_disk_store

        assert load_on_disk_store().add("memory", "gamma — learned mid-turn")["success"]
        return located

    monkeypatch.setattr(lm, "_locate_memory", _locate_then_agent_writes)

    assert lm.delete_node("memory:memory:0")["ok"]

    entries = _memory_entries(home)
    assert "gamma — learned mid-turn" in entries  # survived the rewrite
    assert entries == ["beta note", "gamma — learned mid-turn"]  # and the right entry went


def test_edit_targets_its_own_entry_after_the_list_shifted(home, monkeypatch):
    """Identify the entry by text, not by the index from a stale view."""
    real_locate = lm._locate_memory

    def _locate_then_prepend(node_id):
        located = real_locate(node_id)
        path = home / "memories" / "MEMORY.md"
        from tools.memory_tool import ENTRY_DELIMITER

        path.write_text(ENTRY_DELIMITER.join(["zeta first", *_memory_entries(home)]), encoding="utf-8")
        return located

    monkeypatch.setattr(lm, "_locate_memory", _locate_then_prepend)

    assert lm.edit_node("memory:memory:1", "beta rewritten")["ok"]

    assert _memory_entries(home) == ["zeta first", "alpha note\nline two", "beta rewritten"]


def test_hand_edited_file_is_snapshotted_and_refused_not_rewritten(home):
    """The drift guard (#26045) covers this surface too: a file that would not round-trip is
    backed up and left alone instead of being silently reformatted."""
    path = home / "memories" / "MEMORY.md"
    path.write_text("alpha note\nline two\n§\n\n§\nbeta note\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    result = lm.delete_node("memory:memory:0")

    assert result["ok"] is False
    assert "snapshot" in result["message"].lower() or "drift" in result["message"].lower()
    assert path.read_text(encoding="utf-8") == before  # untouched
    assert list(path.parent.glob("MEMORY.md.bak.*")), "no snapshot was written"


def test_a_vanished_entry_refuses_instead_of_deleting_a_neighbour(home, monkeypatch):
    real_locate = lm._locate_memory

    def _locate_then_remove_target(node_id):
        located = real_locate(node_id)
        (home / "memories" / "MEMORY.md").write_text("beta note", encoding="utf-8")
        return located

    monkeypatch.setattr(lm, "_locate_memory", _locate_then_remove_target)

    result = lm.delete_node("memory:memory:0")

    assert result["ok"] is False and "stale" in result["message"]
    assert _memory_entries(home) == ["beta note"]  # the survivor is untouched


# ── Duplicate entries: a Journey card is an occurrence, not a text ───────────


def _write_memory_file(home, *entries):
    from tools.memory_tool import ENTRY_DELIMITER

    (home / "memories" / "MEMORY.md").write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def test_editing_the_second_of_two_identical_cards_leaves_the_first_alone(home):
    """Journey renders one card per chunk, so identical entries are two cards. Editing the second
    must change that one — not the first, and not collapse the pair into one card."""
    _write_memory_file(home, "same memory", "same memory", "other")

    assert lm.edit_node("memory:memory:1", "edited second")["ok"]

    assert _memory_entries(home) == ["same memory", "edited second", "other"]


def test_deleting_one_of_two_identical_cards_keeps_the_other(home):
    _write_memory_file(home, "same memory", "same memory")

    assert lm.delete_node("memory:memory:0")["ok"]

    assert _memory_entries(home) == ["same memory"]


def test_a_mutation_never_collapses_duplicates_it_did_not_touch(home):
    """The locked reload must not renumber the list being rewritten: an edit elsewhere would
    otherwise persist the collapse and drop a card the user never selected."""
    _write_memory_file(home, "keep", "dup", "dup")

    assert lm.edit_node("memory:memory:0", "keep edited")["ok"]

    assert _memory_entries(home) == ["keep edited", "dup", "dup"]


def test_a_shifted_duplicate_still_leaves_both_cards_standing(home, monkeypatch):
    """Identical copies are interchangeable, so editing either is the same to the reader — what
    must never happen is one of them disappearing."""
    _write_memory_file(home, "alpha", "dup", "dup")
    real_locate = lm._locate_memory

    def _locate_then_shift(node_id):
        located = real_locate(node_id)
        _write_memory_file(home, "dup", "dup")  # "alpha" is gone: the list renumbered
        return located

    monkeypatch.setattr(lm, "_locate_memory", _locate_then_shift)

    assert lm.edit_node("memory:memory:1", "rewritten")["ok"]

    assert sorted(_memory_entries(home)) == ["dup", "rewritten"]


def test_a_vanished_duplicate_pair_refuses_rather_than_picking_one(home, monkeypatch):
    """The clicked text is gone from a list that still holds other cards: nothing to target."""
    _write_memory_file(home, "alpha", "beta")
    real_locate = lm._locate_memory

    def _locate_then_replace(node_id):
        located = real_locate(node_id)
        _write_memory_file(home, "gamma", "delta")
        return located

    monkeypatch.setattr(lm, "_locate_memory", _locate_then_replace)

    result = lm.edit_node("memory:memory:1", "rewritten")

    assert result["ok"] is False and "stale" in result["message"]
    assert _memory_entries(home) == ["gamma", "delta"]
