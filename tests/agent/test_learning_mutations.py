"""Behavior contracts for journey node edit/delete (agent.learning_mutations).

Exercises the real on-disk resolution (skills dir + MEMORY.md/USER.md chunking)
against a temp HERMES_HOME, never mocks — the id→file mapping is the whole point.
"""

from __future__ import annotations

import threading

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


# ── Locking / drift (issue #119668) ─────────────────────────────────────────
# A Journey mutation shares MEMORY.md with the live agent's memory tool, so it must
# take the same lock, re-read under it and honour the drift guard — otherwise a
# memory the agent stored in the meantime is rewritten away from a stale snapshot.


def _race_memory_tool_add(monkeypatch, content: str) -> threading.Thread:
    """Start a lock-respecting ``memory_tool`` writer that appends *content* the
    moment the Journey mutation resolves its node id (the read half of its
    read-modify-write), then give it a moment to land. Under a correct lock the
    writer blocks until the mutation has written; without one it interleaves and
    the mutation's write clobbers it."""
    from agent import learning_graph
    from tools.memory_tool import MemoryStore

    located, landed = threading.Event(), threading.Event()
    real_cards = learning_graph._memory_cards

    def _cards_then_let_writer_in():
        cards = real_cards()
        located.set()
        landed.wait(timeout=0.5)
        return cards

    def _writer():
        located.wait(timeout=5)
        MemoryStore().add("memory", content)
        landed.set()

    monkeypatch.setattr(learning_graph, "_memory_cards", _cards_then_let_writer_in)
    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    return thread


def test_delete_memory_keeps_concurrent_memory_tool_add(home, monkeypatch):
    from tools.memory_tool import MemoryStore

    writer = _race_memory_tool_add(monkeypatch, "gamma note")
    assert lm.delete_node("memory:memory:0")["ok"]
    writer.join(timeout=5)

    assert MemoryStore._read_file(home / "memories" / "MEMORY.md") == ["beta note", "gamma note"]


def test_edit_memory_keeps_concurrent_memory_tool_add(home, monkeypatch):
    from tools.memory_tool import MemoryStore

    writer = _race_memory_tool_add(monkeypatch, "gamma note")
    assert lm.edit_node("memory:memory:0", "alpha rewritten")["ok"]
    writer.join(timeout=5)

    assert MemoryStore._read_file(home / "memories" / "MEMORY.md") == ["alpha rewritten", "beta note", "gamma note"]


@pytest.mark.parametrize("mutate", [lambda: lm.delete_node("memory:memory:0"),
                                    lambda: lm.edit_node("memory:memory:0", "alpha rewritten")],
                         ids=["delete", "edit"])
def test_memory_drift_is_refused_with_backup_like_memory_tool(home, mutate):
    """Hand-edited content that wouldn't round-trip through the § parser is what
    the memory tool's drift guard exists for: snapshot to .bak, refuse, leave the
    file untouched. A Journey mutation must not reformat it silently."""
    from tools.memory_tool_store import _drift_error

    path = home / "memories" / "MEMORY.md"
    raw = "alpha note\n§\n\n§\nbeta note\n"
    path.write_text(raw, encoding="utf-8")

    res = mutate()

    assert not res["ok"]
    assert path.read_text(encoding="utf-8") == raw
    (backup,) = home.glob("memories/MEMORY.md.bak.*")
    assert backup.read_text(encoding="utf-8") == raw
    assert res["message"] == _drift_error(path, str(backup))["error"]


def test_unreadable_memory_file_is_refused_unchanged(home):
    path = home / "memories" / "MEMORY.md"
    path.write_bytes(b"alpha note\n\xc3\x28\n\xc2\xa7\nbeta")

    res = lm.delete_node("memory:memory:0")

    assert not res["ok"] and "could not be read" in res["message"]
    assert path.read_bytes() == b"alpha note\n\xc3\x28\n\xc2\xa7\nbeta"
