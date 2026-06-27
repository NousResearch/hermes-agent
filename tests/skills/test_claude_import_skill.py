"""
Tests for the claude-import skill — parse script and memory unit extraction.

Tests load the script as a module (following the pattern from
``test_openclaw_migration.py`` and ``test_telephony_skill.py``)
and exercise its functions with synthetic Claude export data.

No live filesystem or network — all test data is in-memory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterator

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "migration"
    / "claude-import"
    / "scripts"
    / "claude_parse.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("claude_parse", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers — synthetic Claude export data
# ---------------------------------------------------------------------------

def make_conversation(
    uuid: str = "550e8400-e29b-41d4-a716-446655440000",
    name: str = "Test Conversation",
    summary: str = "A discussion about testing.",
    created: str = "2025-08-15T14:30:00Z",
    updated: str = "2025-08-15T15:30:00Z",
    messages: list | None = None,
) -> dict:
    """Build a synthetic Claude conversation object."""
    if messages is None:
        messages = [
            {
                "uuid": "msg-001",
                "sender": "human",
                "text": "What architecture should we use for this project?",
                "content": [],
                "created_at": "2025-08-15T14:30:01Z",
                "updated_at": "2025-08-15T14:30:01Z",
                "attachments": [],
                "files": [],
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
            },
            {
                "uuid": "msg-002",
                "sender": "assistant",
                "text": "For your use case, I'd recommend a modular monolith. "
                        "It gives you the deployment simplicity of a monolith "
                        "with the organizational benefits of separate modules. "
                        "You can split into microservices later if needed.",
                "content": [],
                "created_at": "2025-08-15T14:30:10Z",
                "updated_at": "2025-08-15T14:30:10Z",
                "attachments": [],
                "files": [],
                "parent_message_uuid": "msg-001",
            },
        ]
    return {
        "uuid": uuid,
        "name": name,
        "summary": summary,
        "created_at": created,
        "updated_at": updated,
        "account": {"uuid": "acct-001"},
        "chat_messages": messages,
    }


def write_synthetic_export(tmp_path: Path, conversations: list[dict]) -> Path:
    """Write a synthetic conversations.json to a temp directory."""
    export_dir = tmp_path / "claude-export"
    export_dir.mkdir()
    conv_path = export_dir / "conversations.json"
    conv_path.write_text(json.dumps(conversations), encoding="utf-8")
    return export_dir


# ---------------------------------------------------------------------------
# stream_json_array
# ---------------------------------------------------------------------------

def test_stream_json_array_yields_all_items(tmp_path):
    mod = load_module()
    data = [
        {"id": 1, "name": "first"},
        {"id": 2, "name": "second"},
        {"id": 3, "name": "third"},
    ]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = list(mod.stream_json_array(str(path)))
    assert len(result) == 3
    assert result[0]["id"] == 1
    assert result[2]["id"] == 3


def test_stream_json_array_empty_array(tmp_path):
    mod = load_module()
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    result = list(mod.stream_json_array(str(path)))
    assert result == []


def test_stream_json_array_single_item(tmp_path):
    mod = load_module()
    path = tmp_path / "single.json"
    path.write_text('[{"key": "value"}]', encoding="utf-8")

    result = list(mod.stream_json_array(str(path)))
    assert len(result) == 1
    assert result[0]["key"] == "value"


# ---------------------------------------------------------------------------
# _flatten_content
# ---------------------------------------------------------------------------

def test_flatten_content_string_passthrough():
    mod = load_module()
    assert mod._flatten_content("hello") == "hello"


def test_flatten_content_list_of_blocks():
    mod = load_module()
    blocks = [
        {"type": "text", "text": "Part one."},
        {"type": "text", "text": "Part two."},
    ]
    assert mod._flatten_content(blocks) == "Part one.\nPart two."


def test_flatten_content_list_with_empty_text():
    mod = load_module()
    blocks = [
        {"type": "text", "text": ""},
        {"type": "text", "text": "Only this."},
    ]
    assert mod._flatten_content(blocks) == "Only this."


def test_flatten_content_non_dict_items():
    mod = load_module()
    blocks = ["plain", {"type": "text", "text": "structured"}]
    assert mod._flatten_content(blocks) == "plain\nstructured"


# ---------------------------------------------------------------------------
# _extract_key_exchanges
# ---------------------------------------------------------------------------

def test_extract_key_exchanges_returns_qa_pair():
    mod = load_module()
    conv = make_conversation()
    text = mod._extract_key_exchanges(conv["chat_messages"])
    assert "Q:" in text
    assert "A:" in text
    assert "modular monolith" in text


def test_extract_key_exchanges_only_one_exchange():
    mod = load_module()
    messages = [
        {
            "uuid": "m1", "sender": "human",
            "text": "Question one?", "content": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
        {
            "uuid": "m2", "sender": "assistant",
            "text": "Answer one.", "content": [],
            "created_at": "2025-01-01T00:00:01Z",
            "updated_at": "2025-01-01T00:00:01Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "m1",
        },
        {
            "uuid": "m3", "sender": "human",
            "text": "Follow-up?", "content": [],
            "created_at": "2025-01-01T00:00:02Z",
            "updated_at": "2025-01-01T00:00:02Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "m2",
        },
        {
            "uuid": "m4", "sender": "assistant",
            "text": "Follow-up answer.", "content": [],
            "created_at": "2025-01-01T00:00:03Z",
            "updated_at": "2025-01-01T00:00:03Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "m3",
        },
    ]
    text = mod._extract_key_exchanges(messages)
    assert "Follow-up" not in text
    assert "Answer one" in text


def test_extract_key_exchanges_empty_messages():
    mod = load_module()
    assert mod._extract_key_exchanges([]) == ""


def test_extract_key_exchanges_no_assistant():
    mod = load_module()
    messages = [
        {
            "uuid": "m1", "sender": "human",
            "text": "Hello?", "content": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
    ]
    text = mod._extract_key_exchanges(messages)
    assert "Hello" in text
    assert "A:" not in text


# ---------------------------------------------------------------------------
# _make_memory_unit
# ---------------------------------------------------------------------------

def test_make_memory_unit_has_expected_fields():
    mod = load_module()
    conv = make_conversation()
    unit = mod._make_memory_unit(conv)

    assert unit["source"] == "claude"
    assert unit["source_type"] == "conversation"
    assert unit["conversation_uuid"] == "550e8400-e29b-41d4-a716-446655440000"
    assert unit["conversation_name"] == "Test Conversation"
    assert unit["timestamp"] == "2025-08-15T14:30:00Z"
    assert unit["updated_at"] == "2025-08-15T15:30:00Z"


def test_make_memory_unit_includes_summary():
    mod = load_module()
    conv = make_conversation(summary="Key insight about architecture.")
    unit = mod._make_memory_unit(conv)
    assert "Key insight about architecture." in unit["content"]


def test_make_memory_unit_includes_exchange():
    mod = load_module()
    conv = make_conversation(
        summary="Testing discussion.",
        messages=[
            {
                "uuid": "m1", "sender": "human",
                "text": "What is the answer?", "content": [],
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "attachments": [], "files": [],
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
            },
            {
                "uuid": "m2", "sender": "assistant",
                "text": "The answer is 42.", "content": [],
                "created_at": "2025-01-01T00:00:01Z",
                "updated_at": "2025-01-01T00:00:01Z",
                "attachments": [], "files": [],
                "parent_message_uuid": "m1",
            },
        ],
    )
    unit = mod._make_memory_unit(conv)
    assert "What is the answer?" in unit["content"]
    assert "The answer is 42." in unit["content"]


def test_make_memory_unit_untitled_conversation():
    mod = load_module()
    conv = make_conversation(name="", summary="")
    unit = mod._make_memory_unit(conv)
    assert "(untitled)" in unit["conversation_name"]
    assert "Conversation:" in unit["content"]


def test_make_memory_unit_with_content_override():
    mod = load_module()
    conv = make_conversation()
    unit = mod._make_memory_unit(conv, content_override="Custom content")
    assert unit["content"] == "Custom content"


# ---------------------------------------------------------------------------
# parse_memories
# ---------------------------------------------------------------------------

def test_parse_memories_conversations_memory(tmp_path):
    mod = load_module()
    mem = [
        {
            "conversations_memory": "User prefers concise documentation.",
            "project_memories": {},
            "account_uuid": "acct-001",
        }
    ]
    path = tmp_path / "memories.json"
    path.write_text(json.dumps(mem), encoding="utf-8")

    units = mod.parse_memories(str(path))
    assert len(units) == 1
    assert units[0]["source_type"] == "conversation_memory"
    assert "concise documentation" in units[0]["content"]


def test_parse_memories_project_memories(tmp_path):
    mod = load_module()
    mem = [
        {
            "conversations_memory": "",
            "project_memories": {
                "proj-001": "***Purpose & context***\nProject about building a mesh network.",
            },
            "account_uuid": "acct-001",
        }
    ]
    path = tmp_path / "memories.json"
    path.write_text(json.dumps(mem), encoding="utf-8")

    units = mod.parse_memories(str(path))
    assert len(units) == 1
    assert units[0]["source_type"] == "project_memory"
    assert "mesh network" in units[0]["content"]


def test_parse_memories_empty_file(tmp_path):
    mod = load_module()
    path = tmp_path / "memories.json"
    path.write_text("[]", encoding="utf-8")
    assert mod.parse_memories(str(path)) == []


def test_parse_memories_missing_file():
    mod = load_module()
    assert mod.parse_memories("/nonexistent/memories.json") == []


def test_parse_memories_skips_empty_text(tmp_path):
    mod = load_module()
    mem = [
        {
            "conversations_memory": "",
            "project_memories": {},
            "account_uuid": "acct-001",
        }
    ]
    path = tmp_path / "memories.json"
    path.write_text(json.dumps(mem), encoding="utf-8")
    assert mod.parse_memories(str(path)) == []


# ---------------------------------------------------------------------------
# parse_export — end-to-end integration
# ---------------------------------------------------------------------------

def test_parse_export_basic(tmp_path):
    mod = load_module()
    convs = [
        make_conversation(
            uuid="aa-001", name="First Chat",
            summary="Initial brainstorming.",
        ),
        make_conversation(
            uuid="bb-002", name="Deep Dive",
            summary="Technical deep dive into architecture.",
        ),
    ]
    export_dir = write_synthetic_export(tmp_path, convs)

    units = mod.parse_export(str(export_dir), include_memories=False)
    assert len(units) == 2
    assert units[0]["conversation_name"] == "First Chat"
    assert units[1]["conversation_name"] == "Deep Dive"


def test_parse_export_includes_memories(tmp_path):
    mod = load_module()
    convs = [make_conversation(uuid="cc-001", name="Solo Chat")]
    export_dir = write_synthetic_export(tmp_path, convs)

    # Also write memories.json
    mem = [
        {
            "conversations_memory": "User likes dark mode.",
            "project_memories": {},
            "account_uuid": "acct-001",
        }
    ]
    (export_dir / "memories.json").write_text(json.dumps(mem), encoding="utf-8")

    units = mod.parse_export(str(export_dir), include_memories=True)
    # 1 conversation + 1 memory
    assert len(units) == 2
    memory_units = [u for u in units if u["source_type"] == "conversation_memory"]
    assert len(memory_units) == 1


def test_parse_export_detects_batch_subdirectory(tmp_path):
    mod = load_module()
    # Create a data-* batch subdirectory structure
    batch_dir = tmp_path / "claude-export" / "data-batch-001"
    batch_dir.mkdir(parents=True)
    convs = [make_conversation(uuid="dd-001", name="Batch Chat")]
    (batch_dir / "conversations.json").write_text(json.dumps(convs), encoding="utf-8")

    units = mod.parse_export(str(tmp_path / "claude-export"), include_memories=False)
    assert len(units) == 1
    assert units[0]["conversation_name"] == "Batch Chat"


def test_parse_export_missing_file(tmp_path):
    mod = load_module()
    # Should sys.exit(1) — catch it
    try:
        mod.parse_export(str(tmp_path / "nonexistent"), include_memories=False)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_long_human_message_truncated():
    mod = load_module()
    long_text = "A" * 1000
    messages = [
        {
            "uuid": "m1", "sender": "human",
            "text": long_text, "content": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
    ]
    text = mod._extract_key_exchanges(messages)
    assert len(text) <= 510  # 500 + "Q: " + "…"


def test_conversation_with_no_messages():
    mod = load_module()
    conv = make_conversation(messages=[])
    unit = mod._make_memory_unit(conv)
    assert unit["content"] is not None
    # No exchange text, just the name/summary
    assert "Test Conversation" in unit["content"]


def test_conversation_with_only_human_messages():
    mod = load_module()
    messages = [
        {
            "uuid": "m1", "sender": "human",
            "text": "I need help with something.", "content": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
        {
            "uuid": "m2", "sender": "human",
            "text": "Actually never mind.", "content": [],
            "created_at": "2025-01-01T00:00:01Z",
            "updated_at": "2025-01-01T00:00:01Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "m1",
        },
    ]
    conv = make_conversation(messages=messages)
    unit = mod._make_memory_unit(conv)
    assert "I need help" in unit["content"]
    # Only human messages appear
    assert "A:" not in unit["content"]


def test_rich_content_blocks():
    mod = load_module()
    messages = [
        {
            "uuid": "m1", "sender": "human",
            "text": "Tell me something interesting.",
            "content": [],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
        {
            "uuid": "m2", "sender": "assistant",
            "text": "",
            "content": [
                {"type": "text", "text": "Rich block answer."},
            ],
            "created_at": "2025-01-01T00:00:01Z",
            "updated_at": "2025-01-01T00:00:01Z",
            "attachments": [], "files": [],
            "parent_message_uuid": "m1",
        },
    ]
    conv = make_conversation(messages=messages)
    unit = mod._make_memory_unit(conv)
    assert "Rich block answer" in unit["content"]
