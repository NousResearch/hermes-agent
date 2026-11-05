"""Dedupe/delegate-cap drops must sync provider-native replay carriers too.

The per-turn cap trim already syncs bedrock/anthropic sidecars, but the
earlier _deduplicate_tool_calls / _cap_delegate_task_calls drops remove calls
that never produce results while leaving their toolUse/tool_use blocks in
place. Converters replay those sidecars verbatim, so the next request pairs
the full batch with fewer results. These tests drive the REAL Bedrock and
Anthropic normalizers plus the real request converters, with the REAL
AIAgent._deduplicate_tool_calls.
"""

from types import SimpleNamespace

from tests.agent.test_turn_tool_cap_carrier_trim import (
    _TrimAgent,
    _run_trimmed_round,
)


def _bedrock_msg(blocks):
    from agent.bedrock_adapter import normalize_converse_response
    from agent.transports.bedrock import BedrockTransport

    raw = {
        "output": {"message": {"role": "assistant", "content": blocks}},
        "stopReason": "tool_use",
        "usage": {},
        "modelId": "m",
    }
    adapter_ns = normalize_converse_response(raw)
    normalized = BedrockTransport().normalize_response(adapter_ns)
    return SimpleNamespace(
        role="assistant",
        content=normalized.content,
        tool_calls=[
            SimpleNamespace(
                id=tc.id,
                type="function",
                function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
            )
            for tc in normalized.tool_calls
        ],
        reasoning_content=None,
        reasoning_details=None,
        bedrock_content_blocks=(normalized.provider_data or {}).get(
            "bedrock_content_blocks"
        ),
    )


def _converse_ids(verdict):
    from agent.bedrock_adapter import convert_messages_to_converse

    _system, converse = convert_messages_to_converse([
        {"role": "user", "content": "go"},
        *verdict.messages,
    ])
    assistant_turn = next(m for m in converse if m["role"] == "assistant")
    result_turn = next(
        m for m in converse if m["role"] == "user" and m is not converse[0]
    )
    use_ids = [
        b["toolUse"]["toolUseId"] for b in assistant_turn["content"] if "toolUse" in b
    ]
    result_ids = [
        b["toolResult"]["toolUseId"]
        for b in result_turn["content"]
        if "toolResult" in b
    ]
    return use_ids, result_ids


def test_one_slot_trim_replays_single_bedrock_pair():
    """Reviewer probe: trim [t1, t2] to [t1]; replay must be t1/t1."""
    agent = _TrimAgent(max_iterations=1, valid_tool_names=["one", "two"])
    msg = _bedrock_msg([
        {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
        {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
    ])
    verdict = _run_trimmed_round(agent, msg)
    assert verdict.action == "continue"
    assert verdict.tool_call_count == 1
    use_ids, result_ids = _converse_ids(verdict)
    assert use_ids == ["t1"]
    assert result_ids == use_ids, (
        f"malformed replay: toolUse={use_ids} toolResult={result_ids}"
    )


def test_duplicate_drop_syncs_bedrock_carrier():
    """Dedupe drops the second identical call; its carrier block must go too."""
    from run_agent import AIAgent

    agent = _TrimAgent(max_iterations=10, valid_tool_names=["one", "two", "three"])
    agent._deduplicate_tool_calls = AIAgent._deduplicate_tool_calls
    msg = _bedrock_msg([
        {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
        {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
        {"toolUse": {"toolUseId": "t2dup", "name": "two", "input": {"n": 2}}},
        {"toolUse": {"toolUseId": "t3", "name": "three", "input": {"n": 3}}},
    ])
    assert [tc.id for tc in msg.tool_calls] == ["t1", "t2", "t2dup", "t3"]
    verdict = _run_trimmed_round(agent, msg)
    assert verdict.action == "continue"
    assert verdict.tool_call_count == 3
    persisted = next(
        m
        for m in verdict.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    carrier_ids = [
        b["toolUse"]["toolUseId"]
        for b in persisted["bedrock_content_blocks"]
        if "toolUse" in b
    ]
    assert carrier_ids == ["t1", "t2", "t3"], f"dup block survived: {carrier_ids}"
    use_ids, result_ids = _converse_ids(verdict)
    assert use_ids == ["t1", "t2", "t3"]
    assert result_ids == use_ids, (
        f"malformed replay: toolUse={use_ids} toolResult={result_ids}"
    )


def test_duplicate_drop_syncs_anthropic_carrier():
    """Same duplicate-drop hole on the ordered Anthropic path."""
    from run_agent import AIAgent
    from agent.transports.anthropic import AnthropicTransport
    from agent.anthropic_message_convert import convert_messages_to_anthropic

    def _thinking(text, sig):
        return SimpleNamespace(type="thinking", thinking=text, signature=sig)

    def _tool_use(block_id, name, payload):
        return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)

    response = SimpleNamespace(
        content=[
            _thinking("plan A", "sig-AAA"),
            _tool_use("toolu_1", "read_file", {"path": "a.py"}),
            _tool_use("toolu_2", "read_file", {"path": "b.py"}),
            _tool_use("toolu_2b", "read_file", {"path": "b.py"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )
    normalized = AnthropicTransport().normalize_response(response)
    msg = SimpleNamespace(
        role="assistant",
        content=normalized.content,
        tool_calls=[
            SimpleNamespace(
                id=tc.id,
                type="function",
                function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
            )
            for tc in normalized.tool_calls
        ],
        reasoning_content=None,
        reasoning_details=(normalized.provider_data or {}).get("reasoning_details"),
        anthropic_content_blocks=(normalized.provider_data or {}).get(
            "anthropic_content_blocks"
        ),
    )
    agent = _TrimAgent(max_iterations=10, valid_tool_names=["read_file"])
    agent._deduplicate_tool_calls = AIAgent._deduplicate_tool_calls
    verdict = _run_trimmed_round(agent, msg)
    assert verdict.action == "continue"
    assert verdict.tool_call_count == 2
    persisted = next(
        m
        for m in verdict.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    carrier_ids = [
        b["id"]
        for b in persisted["anthropic_content_blocks"]
        if b.get("type") == "tool_use"
    ]
    assert carrier_ids == ["toolu_1", "toolu_2"], f"dup block survived: {carrier_ids}"

    _system, converted = convert_messages_to_anthropic([
        {"role": "user", "content": "go"},
        *verdict.messages,
    ])
    assistant_turn = next(m for m in converted if m["role"] == "assistant")
    result_turn = next(
        m
        for m in converted
        if m["role"] == "user"
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in m["content"]
        )
    )
    use_ids = [
        b["id"] for b in assistant_turn["content"] if b.get("type") == "tool_use"
    ]
    result_ids = [
        b["tool_use_id"]
        for b in result_turn["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert use_ids == ["toolu_1", "toolu_2"]
    assert result_ids == use_ids, (
        f"malformed replay: tool_use={use_ids} tool_result={result_ids}"
    )
