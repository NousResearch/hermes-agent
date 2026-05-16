# Test for memory_write_events fix: ensure only write actions are counted
from __future__ import annotations

import json

from plugin_api import analyze_messages


def test_memory_write_events_counts_only_writes():
    """Verify memory_write_events counts only add/replace actions and mnemosyne_remember."""
    messages = []

    # memory add (write)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_add",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": "add", "content": "fact1"})}
        }]
    })

    # memory replace (write)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_replace",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": "replace", "content": "fact2", "id": "1"})}
        }]
    })

    # memory list (read)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_list",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": "list"})}
        }]
    })

    # memory search (read)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_search",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": "search", "query": "test"})}
        }]
    })

    # mnemosyne_remember (write)
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_mnem",
            "type": "function",
            "function": {"name": "mnemosyne_remember", "arguments": json.dumps({"fact": "fact3"})}
        }]
    })

    result = analyze_messages("test_sess", "Test Session", messages)
    assert result["memory_write_events"] == 3, f"Expected 3 memory writes, got {result['memory_write_events']}"
    # memory_events should count all memory-related tool calls (5 total)
    assert result["memory_events"] == 5, f"Expected 5 memory events, got {result['memory_events']}"
