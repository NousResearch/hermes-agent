from pathlib import Path

import tools.memory_tool as memory_tool


def test_memory_store_auto_compacts_oversized_legacy_file(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", tmp_path)

    # Legacy file: huge single entry with no delimiter.
    big_entry = "A" * 5000
    (tmp_path / "MEMORY.md").write_text(big_entry, encoding="utf-8")

    store = memory_tool.MemoryStore(memory_char_limit=2200, user_char_limit=1375)
    store.load_from_disk()

    assert store._char_count("memory") <= 2200
    disk_content = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert len(disk_content) <= 2200

    # After compaction, normal writes should work again.
    result = store.add("memory", "recent observation")
    assert result["success"] is True


def test_add_eviction_makes_room_instead_of_error(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", tmp_path)
    store = memory_tool.MemoryStore(memory_char_limit=120, user_char_limit=100)
    store.memory_entries = ["old-a", "old-b", "old-c"]
    store.save_to_disk("memory")

    # Make the list very full first.
    store.memory_entries = [
        "x" * 50,
        "y" * 50,
    ]
    store.save_to_disk("memory")

    res = store.add("memory", "z" * 40)
    assert res["success"] is True
    assert "Compacted memory" in res.get("message", "")
    assert store.memory_entries[-1] == "z" * 40


def test_remove_and_replace_no_match_are_noop_success(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", tmp_path)
    store = memory_tool.MemoryStore(memory_char_limit=2200, user_char_limit=1375)
    store.memory_entries = ["alpha note", "beta note"]
    store.save_to_disk("memory")

    rep = store.replace("memory", "missing key", "new value")
    rem = store.remove("memory", "missing key")
    assert rep["success"] is True
    assert rem["success"] is True
    assert store.memory_entries == ["alpha note", "beta note"]
