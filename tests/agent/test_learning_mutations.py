"""Behavior contracts for journey node edit/delete (agent.learning_mutations).

Exercises the real on-disk resolution (skills dir + MEMORY.md/USER.md chunking)
against a temp HERMES_HOME, never mocks — the id→file mapping is the whole point.
"""

from __future__ import annotations

import os

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


def _lock_is_held(path) -> bool:
    """True when some other holder owns the memory file's lock right now.

    Mirrors ``MemoryStore._file_lock``'s own platform handling (fcntl on Unix, msvcrt on
    Windows) with a non-blocking acquisition, so a test can tell "this write is serialized
    behind the cycle" apart from "this write went straight through" without sleeping.
    """
    from tools import memory_tool as mt

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if mt.fcntl is not None:
            try:
                mt.fcntl.flock(fd, mt.fcntl.LOCK_EX | mt.fcntl.LOCK_NB)
            except OSError:
                return True
            mt.fcntl.flock(fd, mt.fcntl.LOCK_UN)
            return False
        if mt.msvcrt is not None:
            try:
                mt.msvcrt.locking(fd, mt.msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            mt.msvcrt.locking(fd, mt.msvcrt.LK_UNLCK, 1)
            return False
        return False
    finally:
        os.close(fd)


def test_journey_mutation_is_atomic_against_a_concurrent_memory_store_write(home, monkeypatch):
    """A memory-tool write that commits during a journey edit must survive that edit.

    The journey path reads the entry list and then writes it straight back, so unless the
    whole read-modify-write cycle holds ``MemoryStore._file_lock`` a concurrent
    ``MemoryStore.add`` lands in the gap and the stale snapshot silently overwrites it.

    The competitor is released from inside the journey's own read — the exact window — and
    reports which of two mutually exclusive states it reached: it found the lock held (the
    cycle is serialized) or it got straight through and committed mid-cycle. Both outcomes
    are event-signalled, so the ordering is forced rather than raced.
    """
    import threading

    from tools.memory_tool import MemoryStore, load_on_disk_store

    memory_file = home / "memories" / "MEMORY.md"
    store = load_on_disk_store()
    gamma = "gamma concurrent note"

    armed = threading.Event()
    lock_was_held = threading.Event()
    wrote_mid_cycle = threading.Event()

    def competitor():
        armed.set()
        if _lock_is_held(memory_file):
            lock_was_held.set()
        else:
            store.add("memory", gamma)
            wrote_mid_cycle.set()

    real_read_file = MemoryStore._read_file

    def read_then_compete(path):
        chunks = real_read_file(path)
        if path == memory_file and not armed.is_set():
            threading.Thread(target=competitor, daemon=True).start()
            assert armed.wait(5), "competitor thread never started"
            assert lock_was_held.wait(5) or wrote_mid_cycle.wait(5), (
                "competitor neither found the lock held nor completed its write"
            )
        return chunks

    monkeypatch.setattr(MemoryStore, "_read_file", staticmethod(read_then_compete))

    assert lm.edit_node("memory:memory:0", "alpha rewritten")["ok"]

    if lock_was_held.is_set():
        # Serialized: the competing write had to wait for the cycle to finish, so apply it
        # now. It reloads under the lock first, so the journey edit must survive it.
        assert store.add("memory", gamma)["success"]

    assert MemoryStore._read_file(memory_file) == ["alpha rewritten", "beta note", gamma]
