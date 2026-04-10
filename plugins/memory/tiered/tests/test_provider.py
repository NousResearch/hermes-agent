"""Tests for TieredMemoryProvider (__init__.py).

Tests the integration layer: tool routing, input validation, JSON output,
error handling, prefetch formatting, sync_turn threading, and on_memory_write.

Requires PYTHONPATH to include both the repo root and Hermes source:
  PYTHONPATH=/Users/wills_mac_mini/Code/hermes-poc:/Users/wills_mac_mini/.hermes/hermes-agent
"""

import json
import unittest
from unittest.mock import MagicMock, patch

# Import at module level — requires agent.memory_provider on PYTHONPATH
from plugins.memory.tiered import (
    TieredMemoryProvider,
    MEMORY_SEARCH_SCHEMA,
    MEMORY_OBSERVE_SCHEMA,
    GRANOLA_LIST_SCHEMA,
    GRANOLA_GET_SCHEMA,
    GRANOLA_QUERY_SCHEMA,
)


def _make_provider():
    """Create a provider with a mocked DB (no real SQLite needed)."""
    provider = TieredMemoryProvider.__new__(TieredMemoryProvider)
    provider._db = MagicMock()
    provider._session_id = "test"
    provider._sync_thread = None
    return provider


class TestHandleToolCall(unittest.TestCase):
    """Test handle_tool_call routing, validation, and error handling."""

    def test_memory_search_returns_valid_json(self):
        provider = _make_provider()
        with patch("plugins.memory.tiered.hybrid_search", return_value=[
            {"id": "abc", "title": "Test", "content": "hello", "score": 0.5, "search_mode": "fts_only"}
        ]):
            result = json.loads(provider.handle_tool_call("memory_search", {"query": "test"}))
        self.assertIn("results", result)
        self.assertIn("total", result)
        self.assertEqual(result["total"], 1)

    def test_memory_observe_returns_valid_json(self):
        provider = _make_provider()
        provider._db.insert_entry.return_value = "entry-123"
        result = json.loads(provider.handle_tool_call("memory_observe", {
            "title": "Test Title",
            "content": "Test content",
            "category": "project",
        }))
        self.assertEqual(result["id"], "entry-123")
        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["category"], "project")

    def test_unknown_tool_raises(self):
        provider = _make_provider()
        with self.assertRaises(NotImplementedError):
            provider.handle_tool_call("unknown_tool", {})

    def test_search_query_truncated_to_1000(self):
        provider = _make_provider()
        long_query = "x" * 2000
        with patch("plugins.memory.tiered.hybrid_search", return_value=[]) as mock_search:
            provider.handle_tool_call("memory_search", {"query": long_query})
            called_query = mock_search.call_args[0][1]
            self.assertEqual(len(called_query), 1000)

    def test_observe_title_truncated_to_200(self):
        provider = _make_provider()
        provider._db.insert_entry.return_value = "id-1"
        long_title = "t" * 500
        provider.handle_tool_call("memory_observe", {
            "title": long_title,
            "content": "short content",
        })
        called_title = provider._db.insert_entry.call_args[0][0]
        self.assertEqual(len(called_title), 200)

    def test_observe_content_truncated_to_10000(self):
        provider = _make_provider()
        provider._db.insert_entry.return_value = "id-1"
        long_content = "c" * 20000
        provider.handle_tool_call("memory_observe", {
            "title": "Title",
            "content": long_content,
        })
        called_content = provider._db.insert_entry.call_args[0][1]
        self.assertEqual(len(called_content), 10000)

    def test_search_error_returns_empty_results(self):
        provider = _make_provider()
        with patch("plugins.memory.tiered.hybrid_search", side_effect=Exception("DB error")):
            result = json.loads(provider.handle_tool_call("memory_search", {"query": "test"}))
        self.assertEqual(result["results"], [])
        self.assertEqual(result["total"], 0)
        self.assertIn("error", result)

    def test_observe_scrub_failure_returns_rejected(self):
        provider = _make_provider()
        provider._db.insert_entry.side_effect = RuntimeError("agent.redact unavailable")
        result = json.loads(provider.handle_tool_call("memory_observe", {
            "title": "Test", "content": "Test content",
        }))
        self.assertEqual(result["status"], "rejected")
        self.assertIn("error", result)

    def test_observe_general_error_returns_failed(self):
        provider = _make_provider()
        provider._db.insert_entry.side_effect = Exception("DB locked")
        result = json.loads(provider.handle_tool_call("memory_observe", {
            "title": "Test", "content": "Test content",
        }))
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)


class TestHandleGranolaToolCalls(unittest.TestCase):
    """Test handle_tool_call routing for the 3 Granola tools."""

    @patch("plugins.memory.tiered.list_granola_meetings")
    def test_handle_list_granola_meetings(self, mock_list):
        mock_list.return_value = {"content": json.dumps({"meetings": [{"id": "m1", "title": "Standup"}]})}
        provider = _make_provider()
        result = provider.handle_tool_call("list_granola_meetings", {
            "time_range": "this_week",
        })
        parsed = json.loads(result)
        self.assertIn("meetings", parsed)
        mock_list.assert_called_once_with(
            time_range="this_week",
            custom_start=None,
            custom_end=None,
        )

    @patch("plugins.memory.tiered.list_granola_meetings")
    def test_handle_list_granola_meetings_defaults(self, mock_list):
        mock_list.return_value = {"content": json.dumps({"meetings": []})}
        provider = _make_provider()
        provider.handle_tool_call("list_granola_meetings", {})
        mock_list.assert_called_once_with(
            time_range="last_30_days",
            custom_start=None,
            custom_end=None,
        )

    @patch("plugins.memory.tiered.list_granola_meetings")
    def test_handle_list_granola_meetings_custom_range(self, mock_list):
        mock_list.return_value = {"content": json.dumps({"meetings": []})}
        provider = _make_provider()
        provider.handle_tool_call("list_granola_meetings", {
            "time_range": "custom",
            "custom_start": "2026-01-01T00:00:00Z",
            "custom_end": "2026-01-31T23:59:59Z",
        })
        mock_list.assert_called_once_with(
            time_range="custom",
            custom_start="2026-01-01T00:00:00Z",
            custom_end="2026-01-31T23:59:59Z",
        )

    @patch("plugins.memory.tiered.get_granola_meeting")
    def test_handle_get_granola_meeting(self, mock_get):
        mock_get.return_value = {"content": json.dumps({"id": "m1", "notes": "Important stuff"})}
        provider = _make_provider()
        result = provider.handle_tool_call("get_granola_meeting", {
            "id": "m1",
            "include_transcript": True,
        })
        parsed = json.loads(result)
        self.assertEqual(parsed["id"], "m1")
        mock_get.assert_called_once_with(
            meeting_id="m1",
            include_transcript=True,
        )

    @patch("plugins.memory.tiered.get_granola_meeting")
    def test_handle_get_granola_meeting_defaults(self, mock_get):
        mock_get.return_value = {"content": json.dumps({"id": "m2"})}
        provider = _make_provider()
        provider.handle_tool_call("get_granola_meeting", {"id": "m2"})
        mock_get.assert_called_once_with(
            meeting_id="m2",
            include_transcript=False,
        )

    @patch("plugins.memory.tiered.query_granola_meetings")
    def test_handle_query_granola_meetings(self, mock_query):
        mock_query.return_value = {"content": json.dumps({"results": [{"id": "m1", "snippet": "budget"}]})}
        provider = _make_provider()
        result = provider.handle_tool_call("query_granola_meetings", {"query": "budget discussion"})
        parsed = json.loads(result)
        self.assertIn("results", parsed)
        mock_query.assert_called_once_with(query="budget discussion")

    @patch("plugins.memory.tiered.list_granola_meetings")
    def test_handle_granola_list_error(self, mock_list):
        mock_list.side_effect = Exception("API timeout")
        provider = _make_provider()
        result = json.loads(provider.handle_tool_call("list_granola_meetings", {}))
        self.assertIn("error", result)
        self.assertIn("details", result)

    @patch("plugins.memory.tiered.get_granola_meeting")
    def test_handle_granola_get_error(self, mock_get):
        mock_get.side_effect = ValueError("Meeting ID is required")
        provider = _make_provider()
        result = json.loads(provider.handle_tool_call("get_granola_meeting", {"id": ""}))
        self.assertIn("error", result)

    @patch("plugins.memory.tiered.query_granola_meetings")
    def test_handle_granola_query_error(self, mock_query):
        mock_query.side_effect = ValueError("Query must be at least 2 characters")
        provider = _make_provider()
        result = json.loads(provider.handle_tool_call("query_granola_meetings", {"query": "x"}))
        self.assertIn("error", result)


class TestGetToolSchemas(unittest.TestCase):

    def test_returns_five_schemas(self):
        provider = _make_provider()
        schemas = provider.get_tool_schemas()
        self.assertEqual(len(schemas), 5)
        names = {s["name"] for s in schemas}
        self.assertEqual(names, {
            "memory_search",
            "memory_observe",
            "list_granola_meetings",
            "get_granola_meeting",
            "query_granola_meetings",
        })

    def test_granola_list_schema_has_time_range_enum(self):
        self.assertIn("time_range", GRANOLA_LIST_SCHEMA["parameters"]["properties"])
        tr = GRANOLA_LIST_SCHEMA["parameters"]["properties"]["time_range"]
        self.assertEqual(set(tr["enum"]), {"this_week", "last_week", "last_30_days", "custom"})

    def test_granola_get_schema_requires_id(self):
        self.assertIn("id", GRANOLA_GET_SCHEMA["parameters"]["required"])

    def test_granola_query_schema_requires_query(self):
        self.assertIn("query", GRANOLA_QUERY_SCHEMA["parameters"]["required"])


class TestIsAvailable(unittest.TestCase):

    def test_available_when_sqlite_vec_importable(self):
        provider = _make_provider()
        # sqlite-vec is installed in this venv, so should return True
        self.assertTrue(provider.is_available())


class TestPrefetch(unittest.TestCase):

    def test_prefetch_empty_returns_empty_string(self):
        provider = _make_provider()
        with patch("plugins.memory.tiered.hybrid_search", return_value=[]):
            result = provider.prefetch("test query")
        self.assertEqual(result, "")

    def test_prefetch_formats_results(self):
        provider = _make_provider()
        with patch("plugins.memory.tiered.hybrid_search", return_value=[
            {"id": "1", "title": "Meeting notes", "content": "Discussed Q4 goals", "score": 0.85, "search_mode": "hybrid"}
        ]):
            result = provider.prefetch("test query")
        self.assertIn("[Recalled from memory]", result)
        self.assertIn("Meeting notes", result)
        self.assertIn("0.8500", result)

    def test_prefetch_error_returns_empty(self):
        provider = _make_provider()
        with patch("plugins.memory.tiered.hybrid_search", side_effect=Exception("DB error")):
            result = provider.prefetch("test query")
        self.assertEqual(result, "")


class TestSyncTurn(unittest.TestCase):

    def test_sync_turn_calls_insert(self):
        provider = _make_provider()
        provider.sync_turn("hello", "hi there")
        provider._sync_thread.join(timeout=5.0)
        provider._db.insert_entry.assert_called_once()
        call_kwargs = provider._db.insert_entry.call_args[1]
        self.assertEqual(call_kwargs["category"], "chat")

    def test_sync_turn_joins_previous_thread(self):
        provider = _make_provider()
        provider.sync_turn("msg1", "resp1")
        provider._sync_thread.join(timeout=5.0)
        provider.sync_turn("msg2", "resp2")
        provider._sync_thread.join(timeout=5.0)
        self.assertEqual(provider._db.insert_entry.call_count, 2)

    def test_sync_turn_swallows_runtime_error(self):
        provider = _make_provider()
        provider._db.insert_entry.side_effect = RuntimeError("scrub failed")
        provider.sync_turn("hello", "hi")
        provider._sync_thread.join(timeout=5.0)
        # Should not raise — error is caught and logged


class TestOnMemoryWrite(unittest.TestCase):

    def test_add_action_stores_content(self):
        provider = _make_provider()
        provider._db.insert_entry.return_value = "id-1"
        provider.on_memory_write("add", "memory", "Important fact to remember")
        provider._db.insert_entry.assert_called_once()
        call_args = provider._db.insert_entry.call_args
        self.assertEqual(call_args[0][1], "Important fact to remember")
        self.assertEqual(call_args[1]["category"], "project")

    def test_non_add_action_does_not_store(self):
        provider = _make_provider()
        provider.on_memory_write("remove", "memory", "something")
        provider._db.insert_entry.assert_not_called()

    def test_empty_content_does_not_store(self):
        provider = _make_provider()
        provider.on_memory_write("add", "memory", "")
        provider._db.insert_entry.assert_not_called()

    def test_scrub_error_handled_gracefully(self):
        provider = _make_provider()
        provider._db.insert_entry.side_effect = RuntimeError("scrub failed")
        provider.on_memory_write("add", "memory", "test content")
        # Should not raise


class TestShutdown(unittest.TestCase):

    def test_shutdown_closes_db(self):
        provider = _make_provider()
        provider.shutdown()
        provider._db.close.assert_called_once()

    def test_shutdown_without_db_does_not_error(self):
        provider = TieredMemoryProvider.__new__(TieredMemoryProvider)
        provider.shutdown()  # No _db attribute — should not raise


class TestCronRegistration(unittest.TestCase):
    """Test _ensure_cron_registered registers both consolidation and comprehensive-update jobs."""

    @patch("cron.jobs.create_job")
    @patch("cron.jobs.list_jobs", return_value=[])
    def test_cron_registers_comprehensive_update(self, mock_list, mock_create):
        """Verify create_job is called with comprehensive-update params."""
        provider = _make_provider()
        provider._ensure_cron_registered()
        # Find the comprehensive-update call
        cu_calls = [
            c for c in mock_create.call_args_list
            if c.kwargs.get("name") == "Comprehensive Update"
        ]
        self.assertEqual(len(cu_calls), 1, "Expected exactly one Comprehensive Update create_job call")
        call = cu_calls[0]
        self.assertEqual(call.kwargs.get("schedule"), "0 11 * * 1-5")
        self.assertEqual(call.kwargs.get("skill"), "comprehensive-update")
        self.assertEqual(call.kwargs.get("deliver"), "telegram")

    @patch("cron.jobs.create_job")
    @patch("cron.jobs.list_jobs", return_value=[
        {"name": "Comprehensive Update", "schedule": "0 11 * * 1-5"},
    ])
    def test_cron_comprehensive_update_idempotent(self, mock_list, mock_create):
        """If Comprehensive Update already exists, create_job should NOT be called for it."""
        provider = _make_provider()
        provider._ensure_cron_registered()
        cu_calls = [
            c for c in mock_create.call_args_list
            if c.kwargs.get("name") == "Comprehensive Update"
        ]
        self.assertEqual(len(cu_calls), 0, "Should not re-register existing Comprehensive Update job")

    @patch("cron.jobs.create_job")
    @patch("cron.jobs.list_jobs", return_value=[])
    def test_cron_registers_both_jobs(self, mock_list, mock_create):
        """Verify both Tiered Memory Consolidation and Comprehensive Update are registered."""
        provider = _make_provider()
        provider._ensure_cron_registered()
        registered_names = {
            c.kwargs.get("name") for c in mock_create.call_args_list
        }
        self.assertIn("Tiered Memory Consolidation", registered_names)
        self.assertIn("Comprehensive Update", registered_names)
        self.assertEqual(mock_create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
