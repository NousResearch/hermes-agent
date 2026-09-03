"""Tests for federation Phase 7 — collaboration (memory sync + distributed search)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.federation.federation_collaboration import (
    FederationDistributedSearch,
    FederationMemorySync,
    MemoryEntry,
    SearchResult,
)


# ========================================================================
# MemoryEntry tests
# ========================================================================

class TestMemoryEntry:
    def test_to_dict_roundtrip(self):
        entry = MemoryEntry(
            node_id="test-001",
            content="User prefers concise responses",
            target="memory",
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.node_id == "test-001"
        assert restored.content == "User prefers concise responses"
        assert restored.target == "memory"

    def test_version_default(self):
        entry = MemoryEntry(node_id="x", content="y")
        assert entry.version == 1


# ========================================================================
# FederationMemorySync tests
# ========================================================================

class TestFederationMemorySync:
    def _make_sync(self, tmp_path: Path):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=True)
        adapter.get_peer_count = MagicMock(return_value=2)
        sync = FederationMemorySync(
            device_id="dev-a",
            adapter=adapter,
            hermes_home=tmp_path,
        )
        return sync

    def test_load_empty(self, tmp_path):
        sync = self._make_sync(tmp_path)
        sync._load_local_memories()
        assert sync.entry_count == 0

    def test_load_existing_memories(self, tmp_path):
        mem_dir = tmp_path / "memories"
        mem_dir.mkdir(parents=True)
        (mem_dir / "memory.md").write_text(
            "## pref-001\nUser prefers concise responses\n"
            "## pref-002\nProject uses pytest\n"
        )
        sync = self._make_sync(tmp_path)
        sync._load_local_memories()
        assert sync.entry_count == 2
        assert "pref-001" in sync._local_memories
        assert "pref-002" in sync._local_memories

    @pytest.mark.asyncio
    async def test_on_local_memory_change_broadcasts(self, tmp_path):
        sync = self._make_sync(tmp_path)
        await sync.on_local_memory_change("new-001", "New content", "memory")
        sync.adapter.send.assert_called_once()
        call_args = sync.adapter.send.call_args
        msg = call_args[0][0]
        assert msg.payload["action"] == "update"
        assert msg.payload["entry"]["node_id"] == "new-001"

    def test_apply_remote_entry_creates_file(self, tmp_path):
        sync = self._make_sync(tmp_path)
        entry = MemoryEntry(
            node_id="remote-001",
            content="Remote memory content",
            target="memory",
        )
        sync._apply_remote_entry(entry)
        mem_file = tmp_path / "memories" / "memory.md"
        assert mem_file.exists()
        assert "## remote-001" in mem_file.read_text()

    def test_apply_remote_entry_appends(self, tmp_path):
        sync = self._make_sync(tmp_path)
        mem_dir = tmp_path / "memories"
        mem_dir.mkdir(parents=True)
        (mem_dir / "memory.md").write_text("## existing\nExisting content\n")

        entry = MemoryEntry(
            node_id="remote-002",
            content="Remote content",
            target="memory",
        )
        sync._apply_remote_entry(entry)

        content = (mem_dir / "memory.md").read_text()
        assert "## existing" in content
        assert "## remote-002" in content


# ========================================================================
# FederationDistributedSearch tests
# ========================================================================

class TestFederationDistributedSearch:
    def _make_search(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=True)
        adapter.get_peer_count = MagicMock(return_value=2)
        search = FederationDistributedSearch(
            device_id="dev-a",
            adapter=adapter,
            request_timeout=0.1,  # Fast for tests
        )
        return search

    def test_init(self):
        search = self._make_search()
        assert search.device_id == "dev-a"
        assert search.request_timeout == 0.1

    @pytest.mark.asyncio
    async def test_search_no_peers(self, tmp_path):
        search = self._make_search()
        search.adapter.get_peer_count = MagicMock(return_value=0)
        results = await search.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_collect_search_result(self):
        search = self._make_search()
        search._pending_queries["q-001"] = {}

        msg = MagicMock()
        msg.payload = {
            "query_id": "q-001",
            "device_id": "dev-b",
            "results": [
                {
                    "device_id": "dev-b",
                    "session_id": 123,
                    "session_title": "Test session",
                    "snippet": "Found something",
                    "score": 1.0,
                    "timestamp": 1234567890.0,
                }
            ],
        }
        await search.handle_search_result(msg)
        assert "dev-b" in search._pending_queries["q-001"]
        assert len(search._pending_queries["q-001"]["dev-b"]) == 1

    def test_search_result_dataclass(self):
        result = SearchResult(
            device_id="dev-a",
            session_id=42,
            session_title="My session",
            snippet="A snippet of text",
            score=0.95,
        )
        assert result.device_id == "dev-a"
        assert result.session_id == 42
        assert result.score == 0.95
