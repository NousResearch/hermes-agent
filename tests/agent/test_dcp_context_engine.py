from __future__ import annotations

import json

from agent.dcp_context_engine import DCPContextEngine


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_range_tool_schema_is_exposed_by_default():
    engine = DCPContextEngine(config={}, context_length=200000)

    schemas = engine.get_tool_schemas()

    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "compress"
    assert schema["parameters"]["required"] == ["topic", "content"]
    item = schema["parameters"]["properties"]["content"]["items"]
    assert item["required"] == ["startId", "endId", "summary"]


def test_message_tool_schema_when_configured():
    engine = DCPContextEngine(config={"compress": {"mode": "message"}}, context_length=200000)

    schema = engine.get_tool_schemas()[0]

    item = schema["parameters"]["properties"]["content"]["items"]
    assert item["required"] == ["messageId", "topic", "summary"]


def test_deny_permission_hides_compress_tool():
    engine = DCPContextEngine(config={"compress": {"permission": "deny"}}, context_length=200000)

    assert engine.get_tool_schemas() == []


def test_disabled_engine_exposes_no_tool_and_returns_original_api_messages():
    engine = DCPContextEngine(config={"enabled": False}, context_length=200000)
    api_messages = [{"role": "user", "content": "hello"}]

    transformed = engine.transform_api_messages(
        api_messages,
        canonical_messages=[{"role": "user", "content": "hello"}],
        system_prompt="",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )

    assert engine.get_tool_schemas() == []
    assert transformed is api_messages
    assert api_messages == [{"role": "user", "content": "hello"}]


def test_transform_does_not_mutate_canonical_messages_and_adds_refs():
    engine = DCPContextEngine(config={}, context_length=200000)
    canonical = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    original = [msg.copy() for msg in canonical]
    api_messages = [{"role": "system", "content": "sys"}] + [msg.copy() for msg in canonical]

    transformed = engine.transform_api_messages(
        api_messages,
        canonical_messages=canonical,
        system_prompt="sys",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )

    assert canonical == original
    assert '<dcp-ref id="m0001" />' in transformed[1]["content"]
    assert '<dcp-ref id="m0002" />' in transformed[2]["content"]
    assert "DCP context management is active" in transformed[0]["content"]


def test_range_compress_creates_block_and_transform_applies_placeholder():
    engine = DCPContextEngine(config={}, context_length=200000)
    canonical = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "old work"},
        {"role": "user", "content": "new task"},
    ]
    engine._ensure_refs(canonical)

    result = json.loads(
        engine.handle_tool_call(
            "compress",
            {
                "topic": "old work",
                "content": [{"startId": "m0001", "endId": "m0002", "summary": "Old work summary."}],
            },
            messages=canonical,
        )
    )

    assert result["ok"] is True
    assert result["created_blocks"] == [1]
    transformed = engine.transform_api_messages(
        [msg.copy() for msg in canonical],
        canonical_messages=canonical,
        system_prompt="",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )
    assert '<dcp-compressed-block id="b1" topic="old work">' in transformed[0]["content"]
    assert "content moved into compressed block b1" in transformed[1]["content"]
    assert "new task" in transformed[2]["content"]


def test_range_compress_consumes_overlapping_active_blocks():
    engine = DCPContextEngine(config={}, context_length=200000)
    canonical = [
        {"role": "user", "content": "phase one"},
        {"role": "assistant", "content": "phase one result"},
        {"role": "user", "content": "phase two"},
        {"role": "assistant", "content": "phase two result"},
    ]
    engine._ensure_refs(canonical)

    first = json.loads(
        engine.handle_tool_call(
            "compress",
            {"topic": "phase one", "content": [{"startId": "m0001", "endId": "m0002", "summary": "Phase one summary."}]},
            messages=canonical,
        )
    )
    second = json.loads(
        engine.handle_tool_call(
            "compress",
            {"topic": "both phases", "content": [{"startId": "b1", "endId": "m0004", "summary": "Both phases summary."}]},
            messages=canonical,
        )
    )

    assert first["created_blocks"] == [1]
    assert second["created_blocks"] == [2]
    assert second["deactivated_blocks"] == [1]
    assert engine.state.blocks_by_id[1].active is False
    assert engine.state.blocks_by_id[1].deactivated_by_block_id == 2
    assert engine.state.active_block_ids == {2}


def test_multimodal_messages_get_text_ref_without_mutating_canonical_content():
    engine = DCPContextEngine(config={}, context_length=200000)
    canonical = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
            ],
        }
    ]
    api_messages = [{"role": "system", "content": "sys"}] + [msg.copy() for msg in canonical]

    transformed = engine.transform_api_messages(
        api_messages,
        canonical_messages=canonical,
        system_prompt="sys",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )

    assert canonical[0]["content"] == [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
    ]
    assert transformed[1]["content"][-1] == {"type": "text", "text": '<dcp-ref id="m0001" />'}


def test_message_compress_creates_message_block():
    engine = DCPContextEngine(config={"compress": {"mode": "message"}}, context_length=200000)
    canonical = [{"role": "user", "content": "huge pasted log"}]
    engine._ensure_refs(canonical)

    result = json.loads(
        engine.handle_tool_call(
            "compress",
            {"topic": "logs", "content": [{"messageId": "m0001", "topic": "log", "summary": "Useful log facts."}]},
            messages=canonical,
        )
    )

    assert result["ok"] is True
    assert result["mode"] == "message"
    assert result["created_blocks"] == [1]


def test_deduplication_prunes_older_duplicate_tool_output():
    engine = DCPContextEngine(config={}, context_length=200000)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "a", "content": "old output"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("b", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "b", "content": "new output"},
    ]

    engine._apply_deduplication(messages)

    assert "duplicate tool output removed" in messages[1]["content"]
    assert messages[3]["content"] == "new output"


def test_deduplication_respects_protected_tools():
    engine = DCPContextEngine(config={}, context_length=200000)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a", "patch", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "a", "content": "old output"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("b", "patch", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "b", "content": "new output"},
    ]

    engine._apply_deduplication(messages)

    assert messages[1]["content"] == "old output"


def test_purge_errors_preserves_error_summary():
    engine = DCPContextEngine(config={"strategies": {"purgeErrors": {"turns": 0}}}, context_length=200000)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a", "terminal", {"command": "bad"})]},
        {"role": "tool", "tool_call_id": "a", "content": "ERROR: failed\n" + "x" * 500},
        {"role": "user", "content": "next"},
    ]

    engine._apply_purge_errors(messages)

    assert "old failed tool output pruned" in messages[1]["content"]
    assert "ERROR: failed" in messages[1]["content"]


def test_turn_protection_prevents_dedup_pruning_recent_messages():
    engine = DCPContextEngine(
        config={"turnProtection": {"enabled": True, "turns": 1}},
        context_length=200000,
    )
    messages = [
        {"role": "user", "content": "latest turn"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "a", "content": "old output"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("b", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "b", "content": "new output"},
    ]

    engine._apply_deduplication(messages)

    assert messages[1]["content"] == ""
    assert messages[2]["content"] == "old output"


def test_manual_mode_can_disable_automatic_strategies_in_transform():
    engine = DCPContextEngine(
        config={"manualMode": {"enabled": True, "automaticStrategies": False}},
        context_length=200000,
    )
    canonical = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "a", "content": "old output"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("b", "read_file", {"path": "x"})]},
        {"role": "tool", "tool_call_id": "b", "content": "new output"},
    ]

    transformed = engine.transform_api_messages(
        [msg.copy() for msg in canonical],
        canonical_messages=canonical,
        system_prompt="",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )

    assert transformed[1]["content"].startswith("old output")
    assert transformed[3]["content"].startswith("new output")


def test_manual_compress_request_injects_one_shot_nudge_without_mutating_history():
    engine = DCPContextEngine(config={}, context_length=200000)
    canonical = [
        {"role": "user", "content": "please compact old work"},
        {"role": "assistant", "content": "working"},
    ]

    returned = engine.compress(canonical, current_tokens=1234, focus_topic="old investigation")
    first = engine.transform_api_messages(
        [msg.copy() for msg in canonical],
        canonical_messages=canonical,
        system_prompt="",
        tools=[],
        api_call_count=1,
        model="test-model",
        provider="openai",
        session_id="s1",
    )
    second = engine.transform_api_messages(
        [msg.copy() for msg in canonical],
        canonical_messages=canonical,
        system_prompt="",
        tools=[],
        api_call_count=2,
        model="test-model",
        provider="openai",
        session_id="s1",
    )

    assert returned is canonical
    assert "DCP manual compression requested" in first[0]["content"]
    assert "old investigation" in first[0]["content"]
    assert "DCP manual compression requested" not in second[0]["content"]
    assert canonical == [
        {"role": "user", "content": "please compact old work"},
        {"role": "assistant", "content": "working"},
    ]
