"""Tests for active_context tool — session-scoped task tracking."""

import json
import time
import unittest

from tools.active_context_tool import (
    ActiveContextStore,
    active_context_tool,
)


class TestActiveContextStore(unittest.TestCase):
    """Unit tests for the in-memory ActiveContextStore."""

    def test_set_and_get(self):
        store = ActiveContextStore()
        result = json.loads(store.set("debugging Memory OS update script"))
        self.assertEqual(result["status"], "set")
        self.assertEqual(result["task"], "debugging Memory OS update script")

        result = json.loads(store.get())
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["task"], "debugging Memory OS update script")

    def test_get_empty(self):
        store = ActiveContextStore()
        result = json.loads(store.get())
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["task"])

    def test_clear(self):
        store = ActiveContextStore()
        store.set("some task")
        result = json.loads(store.clear())
        self.assertEqual(result["status"], "cleared")
        self.assertEqual(result["task"], "some task")

        result = json.loads(store.get())
        self.assertEqual(result["status"], "empty")

    def test_clear_empty(self):
        store = ActiveContextStore()
        result = json.loads(store.clear())
        self.assertEqual(result["status"], "empty")

    def test_overwrite(self):
        store = ActiveContextStore()
        store.set("task A")
        store.set("task B")
        result = json.loads(store.get())
        self.assertEqual(result["task"], "task B")

    def test_set_empty_string_clears(self):
        store = ActiveContextStore()
        store.set("task A")
        result = json.loads(store.set(""))
        self.assertEqual(result["status"], "cleared")
        result = json.loads(store.get())
        self.assertEqual(result["status"], "empty")

    def test_truncation(self):
        store = ActiveContextStore()
        long_task = "x" * 1000
        store.set(long_task)
        result = json.loads(store.get())
        self.assertEqual(len(result["task"]), 500)

    def test_ttl_expiry(self):
        store = ActiveContextStore(ttl_seconds=1)
        store.set("expiring task")
        result = json.loads(store.get())
        self.assertEqual(result["status"], "active")

        time.sleep(1.1)
        result = json.loads(store.get())
        self.assertEqual(result["status"], "expired")

    def test_format_for_prompt_active(self):
        store = ActiveContextStore()
        store.set("fixing browser bug")
        prompt = store.format_for_prompt()
        self.assertIn("fixing browser bug", prompt)
        self.assertIn("[Active context:", prompt)

    def test_format_for_prompt_empty(self):
        store = ActiveContextStore()
        prompt = store.format_for_prompt()
        self.assertIsNone(prompt)

    def test_format_for_prompt_expired(self):
        store = ActiveContextStore(ttl_seconds=1)
        store.set("expiring task")
        time.sleep(1.1)
        prompt = store.format_for_prompt()
        self.assertIsNone(prompt)

    def test_reset(self):
        store = ActiveContextStore()
        store.set("task A")
        store.reset()
        result = json.loads(store.get())
        self.assertEqual(result["status"], "empty")

    def test_set_strips_whitespace(self):
        store = ActiveContextStore()
        store.set("  hello world  ")
        result = json.loads(store.get())
        self.assertEqual(result["task"], "hello world")


class TestActiveContextTool(unittest.TestCase):
    """Tests for the tool handler function."""

    def test_set_action(self):
        store = ActiveContextStore()
        result = json.loads(
            active_context_tool(action="set", task="my task", store=store)
        )
        self.assertEqual(result["status"], "set")
        self.assertEqual(result["task"], "my task")

    def test_get_action(self):
        store = ActiveContextStore()
        store.set("existing task")
        result = json.loads(
            active_context_tool(action="get", store=store)
        )
        self.assertEqual(result["status"], "active")

    def test_clear_action(self):
        store = ActiveContextStore()
        store.set("task")
        result = json.loads(
            active_context_tool(action="clear", store=store)
        )
        self.assertEqual(result["status"], "cleared")

    def test_set_without_task_returns_error(self):
        store = ActiveContextStore()
        result = json.loads(
            active_context_tool(action="set", task=None, store=store)
        )
        self.assertIn("error", result)

    def test_invalid_action(self):
        store = ActiveContextStore()
        result = json.loads(
            active_context_tool(action="invalid", store=store)
        )
        self.assertIn("error", result)

    def test_no_store(self):
        result = json.loads(
            active_context_tool(action="get", store=None)
        )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
