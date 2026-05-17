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


def test_memory_write_events_dict_arguments():
    """Verify memory tool calls with dict arguments (not JSON string) are handled."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_dict_add",
            "type": "function",
            "function": {"name": "memory", "arguments": {"action": "add", "content": "dict_fact"}}
        }]
    }]

    result = analyze_messages("test_sess", "Dict Args", messages)
    assert result["memory_write_events"] == 1, f"Expected 1 write for dict args, got {result['memory_write_events']}"


def test_memory_write_events_missing_action():
    """Verify memory tool calls with missing action field are not counted as writes."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_no_action",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"content": "fact_no_action"})}
        }]
    }]

    result = analyze_messages("test_sess", "Missing Action", messages)
    assert result["memory_write_events"] == 0, f"Expected 0 writes for missing action, got {result['memory_write_events']}"


def test_memory_write_events_malformed_json():
    """Verify memory tool calls with malformed JSON arguments don't crash."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_bad_json",
            "type": "function",
            "function": {"name": "memory", "arguments": "{invalid json!!!"}
        }]
    }]

    result = analyze_messages("test_sess", "Malformed JSON", messages)
    assert result["memory_write_events"] == 0, f"Expected 0 writes for malformed JSON, got {result['memory_write_events']}"


def test_memory_write_events_uppercase_action():
    """Verify memory tool calls with uppercase action values are handled correctly."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_upper",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": "ADD", "content": "upper_fact"})}
        }]
    }]

    result = analyze_messages("test_sess", "Uppercase Action", messages)
    assert result["memory_write_events"] == 1, f"Expected 1 write for uppercase ADD, got {result['memory_write_events']}"


def test_memory_write_events_none_function():
    """Verify memory tool calls with None function field don't crash."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_none_fn",
            "type": "function",
            "function": None,
            "name": "memory",
            "arguments": json.dumps({"action": "add", "content": "fact_none_fn"})
        }]
    }]

    result = analyze_messages("test_sess", "None Function", messages)
    # The tool name is extracted from call["name"], and arguments from call["arguments"]
    assert result["memory_write_events"] == 1, f"Expected 1 write for None function, got {result['memory_write_events']}"


def test_memory_write_events_non_string_action():
    """Verify memory tool calls with non-string action values (e.g. int) don't crash."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_int_action",
            "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"action": 42})}
        }]
    }]

    result = analyze_messages("test_sess", "Int Action", messages)
    assert result["memory_write_events"] == 0, f"Expected 0 writes for int action, got {result['memory_write_events']}"


def test_memory_write_events_empty_arguments():
    """Verify memory tool calls with empty string arguments don't crash."""
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_empty_args",
            "type": "function",
            "function": {"name": "memory", "arguments": ""}
        }]
    }]

    result = analyze_messages("test_sess", "Empty Args", messages)
    assert result["memory_write_events"] == 0, f"Expected 0 writes for empty args, got {result['memory_write_events']}"
