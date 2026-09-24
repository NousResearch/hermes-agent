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


def test_edit_memory_refuses_an_entry_over_the_memory_tools_limit(home):
    """Same char cap as the memory tool's replace: an over-limit entry reads as external drift to
    every later mutation, so the tool's own remove/replace would refuse (with a .bak) until the
    file is fixed by hand."""
    from tools.memory_tool import MemoryStore

    result = lm.edit_node("memory:profile:2", "x" * 5000)

    assert not result["ok"] and "Shorten" in result["message"]
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8") == "user profile note"
    assert MemoryStore().remove("user", "user profile note")["success"]

    # The cap gates what an edit adds, never a delete: pruning is the way out of a file that is
    # already over its total (lowered memory_char_limit, well-formed hand edits).
    from tools.memory_tool import ENTRY_DELIMITER

    path = home / "memories" / "USER.md"
    path.write_text(ENTRY_DELIMITER.join(f"entry {i} " + "y" * 500 for i in range(4)), encoding="utf-8")
    assert lm.delete_node("memory:profile:2")["ok"]
    assert len(MemoryStore._read_file(path)) == 3








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

    # A Notepad BOM must not make the first card's id permanently "stale": the graph and the
    # store must parse the file the same way.
    path.write_text("alpha note\n§\nbeta note", encoding="utf-8-sig")
    from agent.learning_graph import build_learning_graph

    first = next(n for n in build_learning_graph()["nodes"] if n["kind"] == "memory")
    assert first["label"] == "alpha note"  # the BOM is not part of the card's title
    assert lm.edit_node(first["id"], "alpha sans bom")["ok"]
    assert MemoryStore._read_file(path) == ["alpha sans bom", "beta note"]


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


# ── Node identity: the card the user clicked, not the index it sat at ────────


def _memory_entries(home) -> list[str]:
    from tools.memory_tool import MemoryStore

    return MemoryStore._read_file(home / "memories" / "MEMORY.md")


def _write_memory_file(home, *entries):
    from tools.memory_tool import ENTRY_DELIMITER

    (home / "memories" / "MEMORY.md").write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def _journey_id(home, label: str) -> str:
    """The node id Journey renders for the card whose text starts with *label*."""
    from agent.learning_graph import build_learning_graph

    nodes = [n for n in build_learning_graph()["nodes"] if n["kind"] == "memory"]
    index = next(i for i, entry in enumerate(_memory_entries(home)) if entry.startswith(label))
    return nodes[index]["id"]


def test_the_id_journey_renders_names_the_cards_text(home):
    from agent.learning_graph import memory_fingerprint

    node_id = _journey_id(home, "beta")

    assert node_id.split(":")[:3] == ["memory", "memory", "1"]
    assert node_id.split(":")[3] == memory_fingerprint("beta note")


def test_edit_targets_the_clicked_card_when_the_list_shifted_before_submit(home):
    """The window a displayed-index identity cannot see: another writer prepends an entry AFTER
    the graph is drawn and BEFORE the edit is submitted, so by the time the mutation reads the
    file that index names somebody else's card."""
    node_id = _journey_id(home, "beta")  # what Journey showed the user
    _write_memory_file(home, "zeta first", "alpha note\nline two", "beta note")

    assert lm.edit_node(node_id, "beta rewritten")["ok"]

    assert _memory_entries(home) == ["zeta first", "alpha note\nline two", "beta rewritten"]


def test_delete_targets_the_clicked_card_when_the_list_shifted_before_submit(home):
    node_id = _journey_id(home, "beta")
    _write_memory_file(home, "zeta first", "alpha note\nline two", "beta note")

    assert lm.delete_node(node_id)["ok"]

    assert _memory_entries(home) == ["zeta first", "alpha note\nline two"]


def test_detail_prefills_the_clicked_card_after_a_shift(home):
    node_id = _journey_id(home, "beta")
    _write_memory_file(home, "zeta first", "alpha note\nline two", "beta note")

    assert lm.node_detail(node_id)["content"] == "beta note"


def test_a_clicked_card_that_is_gone_refuses_even_though_its_index_still_exists(home):
    node_id = _journey_id(home, "beta")
    _write_memory_file(home, "alpha note\nline two", "gamma replaced beta")

    result = lm.edit_node(node_id, "beta rewritten")

    assert result["ok"] is False and "stale" in result["message"]
    assert _memory_entries(home) == ["alpha note\nline two", "gamma replaced beta"]


def test_an_id_from_an_older_graph_still_resolves_by_position(home):
    """Pre-fingerprint ids (a cached graph, an older shell) keep working."""
    assert lm.edit_node("memory:memory:1", "beta rewritten")["ok"]

    assert _memory_entries(home) == ["alpha note\nline two", "beta rewritten"]


def test_a_profile_card_resolves_through_its_fingerprinted_id(home):
    """USER.md cards sit after the MEMORY.md ones in the id's global index."""
    from agent.learning_graph import build_learning_graph

    profile = next(n for n in build_learning_graph()["nodes"] if n.get("memorySource") == "profile")

    assert lm.edit_node(profile["id"], "rewritten profile")["ok"]

    assert (home / "memories" / "USER.md").read_text(encoding="utf-8").strip() == "rewritten profile"


def test_a_memory_longer_than_the_rendered_card_still_matches(home):
    """The card renders a truncated body; the fingerprint must digest the WHOLE entry or a long
    memory never matches itself."""
    from agent.learning_graph import build_learning_graph

    long_entry = "long memory " + ("x" * 2000)
    _write_memory_file(home, long_entry, "beta note")
    node = [n for n in build_learning_graph()["nodes"] if n["kind"] == "memory"][0]

    assert lm.node_detail(node["id"])["content"] == long_entry
    assert lm.edit_node(node["id"], "trimmed")["ok"]
    assert _memory_entries(home) == ["trimmed", "beta note"]
