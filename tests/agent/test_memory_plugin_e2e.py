"""End-to-end test: a SQLite-backed memory plugin exercising the full interface.

This proves a real plugin can register as a MemoryProvider and get wired
into the agent loop via MemoryManager. Uses SQLite + FTS5 (stdlib, no
external deps, no API keys).
"""

import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager
from agent.builtin_memory_provider import BuiltinMemoryProvider


# ---------------------------------------------------------------------------
# SQLite FTS5 memory provider — a real, minimal plugin implementation
# ---------------------------------------------------------------------------


class SQLiteMemoryProvider(MemoryProvider):
    """Minimal SQLite + FTS5 memory provider for testing.

    Demonstrates the full MemoryProvider interface with a real backend.
    No external dependencies — just stdlib sqlite3.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = None

    @property
    def name(self) -> str:
        return "sqlite_memory"

    def is_available(self) -> bool:
        return True  # SQLite is always available

    def initialize(self, session_id: str, **kwargs) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories
            USING fts5(content, context, session_id)
        """)
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._conn:
            return ""
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count == 0:
            return ""
        return (
            f"# SQLite Memory Plugin\n"
            f"Active. {count} memories stored.\n"
            f"Use sqlite_recall to search, sqlite_retain to store."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._conn or not query:
            return ""
        # FTS5 search
        try:
            rows = self._conn.execute(
                "SELECT content FROM memories WHERE memories MATCH ? LIMIT 5",
                (query,)
            ).fetchall()
            if not rows:
                return ""
            results = [row[0] for row in rows]
            return "## SQLite Memory\n" + "\n".join(f"- {r}" for r in results)
        except sqlite3.OperationalError:
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._conn:
            return
        combined = f"User: {user_content}\nAssistant: {assistant_content}"
        self._conn.execute(
            "INSERT INTO memories (content, context, session_id) VALUES (?, ?, ?)",
            (combined, "conversation", self._session_id),
        )
        self._conn.commit()

    def get_tool_schemas(self):
        return [
            {
                "name": "sqlite_retain",
                "description": "Store a fact to SQLite memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember"},
                        "context": {"type": "string", "description": "Category/context"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "sqlite_recall",
                "description": "Search SQLite memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "sqlite_retain":
            content = args.get("content", "")
            context = args.get("context", "explicit")
            if not content:
                return json.dumps({"error": "content is required"})
            self._conn.execute(
                "INSERT INTO memories (content, context, session_id) VALUES (?, ?, ?)",
                (content, context, self._session_id),
            )
            self._conn.commit()
            return json.dumps({"result": "Stored."})

        elif tool_name == "sqlite_recall":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "query is required"})
            try:
                rows = self._conn.execute(
                    "SELECT content, context FROM memories WHERE memories MATCH ? LIMIT 10",
                    (query,)
                ).fetchall()
                results = [{"content": r[0], "context": r[1]} for r in rows]
                return json.dumps({"results": results})
            except sqlite3.OperationalError:
                return json.dumps({"results": []})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def on_memory_write(self, action, target, content):
        """Mirror built-in memory writes to SQLite."""
        if action == "add" and self._conn:
            self._conn.execute(
                "INSERT INTO memories (content, context, session_id) VALUES (?, ?, ?)",
                (content, f"builtin_{target}", self._session_id),
            )
            self._conn.commit()

    def shutdown(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestSQLiteMemoryPlugin:
    """Full lifecycle test with the SQLite provider."""

    def test_full_lifecycle(self):
        """Exercise init → store → recall → sync → prefetch → shutdown."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        sqlite_mem = SQLiteMemoryProvider()

        mgr.add_provider(builtin)
        mgr.add_provider(sqlite_mem)

        # Initialize
        mgr.initialize_all(session_id="test-session-1", platform="cli")
        assert sqlite_mem._conn is not None

        # System prompt — empty at first
        prompt = mgr.build_system_prompt()
        assert "SQLite Memory Plugin" not in prompt

        # Store via tool call
        result = json.loads(mgr.handle_tool_call(
            "sqlite_retain", {"content": "User prefers dark mode", "context": "preference"}
        ))
        assert result["result"] == "Stored."

        # System prompt now shows count
        prompt = mgr.build_system_prompt()
        assert "1 memories stored" in prompt

        # Recall via tool call
        result = json.loads(mgr.handle_tool_call(
            "sqlite_recall", {"query": "dark mode"}
        ))
        assert len(result["results"]) == 1
        assert "dark mode" in result["results"][0]["content"]

        # Sync a turn (auto-stores conversation)
        mgr.sync_all("What's my theme?", "You prefer dark mode.")
        count = sqlite_mem._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count == 2  # 1 explicit + 1 synced

        # Prefetch for next turn
        prefetched = mgr.prefetch_all("dark mode")
        assert "dark mode" in prefetched

        # Memory bridge — mirroring builtin writes
        mgr.on_memory_write("add", "user", "Timezone: US Pacific")
        count = sqlite_mem._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count == 3

        # Shutdown
        mgr.shutdown_all()
        assert sqlite_mem._conn is None

    def test_tool_routing_with_builtin(self):
        """Verify builtin + plugin tools coexist without conflict."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        sqlite_mem = SQLiteMemoryProvider()
        mgr.add_provider(builtin)
        mgr.add_provider(sqlite_mem)
        mgr.initialize_all(session_id="test-2")

        # Builtin has no tools
        assert len(builtin.get_tool_schemas()) == 0
        # SQLite has 2 tools
        schemas = mgr.get_all_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"sqlite_retain", "sqlite_recall"}

        # Routing works
        assert mgr.has_tool("sqlite_retain")
        assert mgr.has_tool("sqlite_recall")
        assert not mgr.has_tool("memory")  # builtin doesn't register this

    def test_second_external_plugin_rejected(self):
        """Only one external memory provider is allowed at a time."""
        mgr = MemoryManager()
        p1 = SQLiteMemoryProvider()
        p2 = SQLiteMemoryProvider()
        # Hack name for p2
        p2._name_override = "sqlite_memory_2"
        original_name = p2.__class__.name
        type(p2).name = property(lambda self: getattr(self, '_name_override', 'sqlite_memory'))

        mgr.add_provider(p1)
        mgr.add_provider(p2)  # should be rejected

        # Only p1 was accepted
        assert len(mgr.providers) == 1
        assert mgr.provider_names == ["sqlite_memory"]

        # Restore class
        type(p2).name = original_name
        mgr.shutdown_all()

    def test_provider_failure_isolation(self):
        """Failing external provider doesn't break builtin."""
        from agent.builtin_memory_provider import BuiltinMemoryProvider

        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()  # name="builtin", always accepted
        ext = SQLiteMemoryProvider()

        mgr.add_provider(builtin)
        mgr.add_provider(ext)
        mgr.initialize_all(session_id="test-4")

        # Break external provider's connection
        ext._conn.close()
        ext._conn = None

        # Sync — external fails silently, builtin (no-op sync) succeeds
        mgr.sync_all("user", "assistant")  # should not raise

        mgr.shutdown_all()

    def test_plugin_registration_flow(self):
        """Simulate the full plugin load → agent init path."""
        # Simulate what AIAgent.__init__ does via plugins/memory/ discovery
        provider = SQLiteMemoryProvider()

        mem_mgr = MemoryManager()
        mem_mgr.add_provider(BuiltinMemoryProvider())
        if provider.is_available():
            mem_mgr.add_provider(provider)
        mem_mgr.initialize_all(session_id="agent-session")

        assert len(mem_mgr.providers) == 2
        assert mem_mgr.provider_names == ["builtin", "sqlite_memory"]
        assert provider._conn is not None  # initialized = connection established

        mem_mgr.shutdown_all()
