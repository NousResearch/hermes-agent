"""Tests for the ``memory`` inline executor's argument forwarding (agent/inline_tool_executors.py).

The tool schema advertises ``new_text`` as an alias for ``content``
(tools/memory_tool.py implements it at the ``if content is None and new_text``
seam), so the executor's arg_specs must forward it — otherwise the allowlist in
``_call_tool`` silently drops the argument and ``replace`` fails with
"content is required" even though the caller supplied the value.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools.memory_tool import MemoryStore


def _agent(store):
    return SimpleNamespace(_memory_store=store, _memory_manager=None)


def _ctx():
    return InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")


class TestMemoryExecutorForwardsNewText:
    def test_replace_via_new_text_alias_succeeds(self):
        """End-to-end root-cause lock: replace with only ``new_text`` must work."""
        store = MemoryStore(memory_char_limit=500, user_char_limit=300)
        INLINE_TOOL_EXECUTORS["memory"](
            _agent(store),
            {"action": "add", "target": "memory", "content": "Household has two cars: a hatchback and an estate."},
            _ctx(),
        )
        result = INLINE_TOOL_EXECUTORS["memory"](
            _agent(store),
            {
                "action": "replace",
                "target": "memory",
                "old_text": "Household has two cars: a hatchback and an estate.",
                "new_text": "Household has two cars: a hatchback and a saloon.",
            },
            _ctx(),
        )
        payload = json.loads(result)
        assert payload["success"] is True, payload
        assert "a saloon" in store.memory_entries[0]
        assert not any("an estate" in entry for entry in store.memory_entries)

    def test_new_text_is_forwarded_to_memory_tool(self):
        """The executor's arg_specs must pass ``new_text`` through to the tool."""
        with patch("tools.memory_tool.memory_tool") as tool:
            tool.return_value = "{}"
            INLINE_TOOL_EXECUTORS["memory"](
                _agent(store=object()),
                {"action": "replace", "target": "memory", "old_text": "A", "new_text": "B"},
                _ctx(),
            )
        assert tool.call_args.kwargs["new_text"] == "B"

    def test_content_wins_when_both_set(self):
        """Documented precedence: when both are forwarded, ``content`` wins."""
        with patch("tools.memory_tool.memory_tool") as tool:
            tool.return_value = "{}"
            INLINE_TOOL_EXECUTORS["memory"](
                _agent(store=object()),
                {"action": "replace", "target": "memory", "old_text": "A", "content": "C", "new_text": "B"},
                _ctx(),
            )
        kwargs = tool.call_args.kwargs
        assert kwargs["content"] == "C"
        assert kwargs["new_text"] == "B"
