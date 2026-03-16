"""Tests for hermes memory / hermes knowledge commands (#1156)."""
import pytest
from unittest.mock import patch, MagicMock
import datetime
from tools.memory_tool import MemoryStore, _make_meta, _strip_meta


# ---------------------------------------------------------------------------
# Unit tests: provenance helpers
# ---------------------------------------------------------------------------

def test_make_meta_format():
    meta = _make_meta("session-abc")
    assert meta.startswith("[saved:")
    assert "|session:session-abc]" in meta
    assert meta.endswith("\n")


def test_make_meta_default_session():
    meta = _make_meta()
    assert "|session:cli]" in meta


def test_strip_meta_with_meta():
    meta = _make_meta("sess-xyz")
    entry = meta + "some content here"
    content, ts, sid = _strip_meta(entry)
    assert content == "some content here"
    assert ts is not None
    assert sid == "sess-xyz"


def test_strip_meta_without_meta():
    entry = "legacy entry without metadata"
    content, ts, sid = _strip_meta(entry)
    assert content == entry
    assert ts is None
    assert sid is None


# ---------------------------------------------------------------------------
# Unit tests: MemoryStore with provenance
# ---------------------------------------------------------------------------

def test_add_stores_provenance(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "test entry", session_id="sess-123")
        entries = store.list_entries("memory")
        assert len(entries) == 1
        assert entries[0]["content"] == "test entry"
        assert entries[0]["session_id"] == "sess-123"
        assert entries[0]["saved_at"] is not None


def test_add_without_session_id(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "no session entry")
        entries = store.list_entries("memory")
        assert entries[0]["session_id"] == "cli"


def test_remove_matches_content_not_meta(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "removable entry", session_id="sess-abc")
        result = store.remove("memory", "removable entry")
        assert result["success"] is True
        assert store.list_entries("memory") == []


def test_remove_does_not_match_session_id(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "keep this", session_id="sess-xyz")
        result = store.remove("memory", "sess-xyz")
        assert result["success"] is False


def test_clear_all(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "entry 1")
        store.add("memory", "entry 2")
        store.clear_all("memory")
        assert store.list_entries("memory") == []


def test_forget_session(tmp_path):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        store.add("memory", "from session A", session_id="sess-A")
        store.add("memory", "from session B", session_id="sess-B")
        store.add("user", "user note A", session_id="sess-A")
        result = store.forget_session("sess-A")
        assert result["removed"] == 2
        mem = store.list_entries("memory")
        usr = store.list_entries("user")
        assert len(mem) == 1
        assert mem[0]["content"] == "from session B"
        assert len(usr) == 0


def test_list_entries_legacy_no_meta(tmp_path):
    """Backward compat: entries without meta prefix still listed."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore()
        # Write legacy entry directly
        (tmp_path / "MEMORY.md").write_text("legacy entry")
        store.load_from_disk()
        entries = store.list_entries("memory")
        assert len(entries) == 1
        assert entries[0]["content"] == "legacy entry"
        assert entries[0]["saved_at"] is None
        assert entries[0]["session_id"] is None


# ---------------------------------------------------------------------------
# Integration: CLI commands
# ---------------------------------------------------------------------------

def test_cmd_memory_list(tmp_path, capsys):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        from hermes_cli.knowledge import cmd_memory
        store = MemoryStore()
        store.add("memory", "hello world", session_id="s1")
        args = MagicMock()
        args.memory_action = "list"
        cmd_memory(args)
        out = capsys.readouterr().out
        assert "hello world" in out
        assert "s1" in out


def test_cmd_memory_clear(tmp_path, capsys):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        from hermes_cli.knowledge import cmd_memory
        store = MemoryStore()
        store.add("memory", "to be cleared")
        args = MagicMock()
        args.memory_action = "clear"
        cmd_memory(args)
        store2 = MemoryStore()
        store2.load_from_disk()
        assert store2.list_entries("memory") == []


def test_cmd_memory_delete(tmp_path, capsys):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        from hermes_cli.knowledge import cmd_memory
        store = MemoryStore()
        store.add("memory", "delete me please")
        args = MagicMock()
        args.memory_action = "delete"
        args.text = "delete me"
        args.target = "all"
        cmd_memory(args)
        store2 = MemoryStore()
        store2.load_from_disk()
        assert store2.list_entries("memory") == []


def test_cmd_memory_forget_session(tmp_path, capsys):
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        from hermes_cli.knowledge import cmd_memory
        store = MemoryStore()
        store.add("memory", "session entry", session_id="forget-me")
        store.add("memory", "keep entry", session_id="keep-me")
        args = MagicMock()
        args.memory_action = "forget-session"
        args.session_id = "forget-me"
        cmd_memory(args)
        out = capsys.readouterr().out
        assert "1" in out


# ---------------------------------------------------------------------------
# TTL tests
# ---------------------------------------------------------------------------

def test_ttl_expires_old_entries(tmp_path):
    """Entries older than ttl_days should be removed on load."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(ttl_days=7)
        # Write an entry with old timestamp directly
        old_ts = (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat(timespec="seconds")
        (tmp_path / "MEMORY.md").write_text(f"[saved:{old_ts}|session:old]\nold entry")
        store.load_from_disk()
        assert store.list_entries("memory") == []


def test_ttl_keeps_recent_entries(tmp_path):
    """Entries newer than ttl_days should be kept."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(ttl_days=7)
        store.add("memory", "recent entry")
        store2 = MemoryStore(ttl_days=7)
        store2.load_from_disk()
        assert len(store2.list_entries("memory")) == 1


def test_ttl_none_keeps_all_entries(tmp_path):
    """When ttl_days is None, no entries are expired."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(ttl_days=None)
        old_ts = (datetime.datetime.now() - datetime.timedelta(days=999)).isoformat(timespec="seconds")
        (tmp_path / "MEMORY.md").write_text(f"[saved:{old_ts}|session:old]\nold entry")
        store.load_from_disk()
        assert len(store.list_entries("memory")) == 1


def test_ttl_keeps_legacy_entries_without_timestamp(tmp_path):
    """Legacy entries without timestamp are kept regardless of TTL."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(ttl_days=1)
        (tmp_path / "MEMORY.md").write_text("legacy entry no timestamp")
        store.load_from_disk()
        assert len(store.list_entries("memory")) == 1


# ---------------------------------------------------------------------------
# Approval mode tests
# ---------------------------------------------------------------------------

def test_approval_mode_blocks_add(tmp_path):
    """When approval_mode=True, add() should return pending_approval."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(approval_mode=True)
        result = store.add("memory", "needs approval")
        assert result["success"] is False
        assert result.get("pending_approval") is True
        assert "approved=True" in result["message"]
        assert store.list_entries("memory") == []


def test_approval_mode_allows_approved_add(tmp_path):
    """When approval_mode=True and approved=True, add() should succeed."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(approval_mode=True)
        result = store.add("memory", "approved entry", approved=True)
        assert result["success"] is True
        assert len(store.list_entries("memory")) == 1


def test_approval_mode_false_adds_normally(tmp_path):
    """When approval_mode=False (default), add() works without approval."""
    with patch("tools.memory_tool.MEMORY_DIR", tmp_path):
        store = MemoryStore(approval_mode=False)
        result = store.add("memory", "normal entry")
        assert result["success"] is True
