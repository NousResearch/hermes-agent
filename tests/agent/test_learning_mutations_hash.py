"""Phase-1 audit fix: starmap (learning-graph) node ids carry a content hash so
a concurrent write between graph render and click can't silently mutate the
wrong memory entry.
"""

import pytest

from agent import learning_mutations as lm
from agent.learning_graph import memory_chunk_hash


@pytest.fixture()
def memdir(tmp_path, monkeypatch):
    """Point both the graph and the mutation module at a tmp memories dir."""
    mem = tmp_path / "memories"
    mem.mkdir()
    monkeypatch.setattr("agent.learning_mutations._memories_dir", lambda: mem)
    monkeypatch.setattr("agent.learning_graph.get_hermes_home", lambda: tmp_path)
    return mem


def _write(mem, entries):
    (mem / "MEMORY.md").write_text("\n§\n".join(entries), encoding="utf-8")


def _mem_id(source, index, text):
    return f"memory:{source}:{index}:{memory_chunk_hash(text)}"


class TestStaleHashGuard:
    def test_delete_with_matching_hash_succeeds(self, memdir):
        _write(memdir, ["alpha entry", "bravo entry", "charlie entry"])
        node_id = _mem_id("memory", 1, "bravo entry")
        result = lm.delete_node(node_id)
        assert result["ok"] is True
        remaining = (memdir / "MEMORY.md").read_text(encoding="utf-8")
        assert "bravo entry" not in remaining
        assert "alpha entry" in remaining and "charlie entry" in remaining

    def test_delete_rejected_when_entry_shifted(self, memdir):
        # Graph rendered [alpha, bravo, charlie]; user clicks bravo (index 1).
        _write(memdir, ["alpha entry", "bravo entry", "charlie entry"])
        node_id = _mem_id("memory", 1, "bravo entry")
        # A concurrent write removes alpha -> index 1 is now charlie.
        _write(memdir, ["bravo entry", "charlie entry"])
        result = lm.delete_node(node_id)
        assert result["ok"] is False
        assert "stale" in result["message"].lower()
        # Nothing was deleted — charlie (now at index 1) survives.
        remaining = (memdir / "MEMORY.md").read_text(encoding="utf-8")
        assert "charlie entry" in remaining and "bravo entry" in remaining

    def test_edit_rejected_when_entry_shifted(self, memdir):
        _write(memdir, ["alpha entry", "bravo entry", "charlie entry"])
        node_id = _mem_id("memory", 1, "bravo entry")
        _write(memdir, ["bravo entry", "charlie entry"])
        result = lm.edit_node(node_id, "REWRITTEN")
        assert result["ok"] is False
        assert "stale" in result["message"].lower()
        assert "REWRITTEN" not in (memdir / "MEMORY.md").read_text(encoding="utf-8")

    def test_legacy_id_without_hash_still_works(self, memdir):
        # Older clients emit position-only ids; those must keep working
        # (position-only addressing, no hash check).
        _write(memdir, ["alpha entry", "bravo entry", "charlie entry"])
        result = lm.delete_node("memory:memory:1")
        assert result["ok"] is True
        assert "bravo entry" not in (memdir / "MEMORY.md").read_text(encoding="utf-8")

    def test_node_detail_rejects_stale_hash(self, memdir):
        _write(memdir, ["alpha entry", "bravo entry"])
        node_id = _mem_id("memory", 0, "alpha entry")
        _write(memdir, ["DIFFERENT now", "bravo entry"])
        detail = lm.node_detail(node_id)
        assert detail["ok"] is False
        assert "stale" in detail["message"].lower()
