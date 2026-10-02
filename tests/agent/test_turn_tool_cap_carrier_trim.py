"""A per-turn cap trim must trim provider-native ordered replay carriers too.

Trimming only the normalized assistant_message.tool_calls leaves the full
bedrock_content_blocks / anthropic_content_blocks sidecars in place.
The history builder persists those sidecars verbatim, and the request
converters replay the sidecar (authoritative over tool_calls) -- so a
trimmed batch replays the FULL toolUse batch against FEWER toolResults, a
malformed request. These tests drive the REAL Bedrock normalizer, history
builder, and request converter (plus the ordered Anthropic path) and assert
tool-use IDs == toolResult IDs after a cap trim.
"""

import json
import sys
from types import ModuleType, SimpleNamespace

sys.modules.setdefault("requests", ModuleType("requests"))

from agent.chat_completion_helpers import build_assistant_message
from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.turn_tool_round import run_tool_round


class _TrimAgent:
    """Minimal agent driving run_tool_round's trim path with the REAL history builder."""

    def __init__(self, *, max_iterations, valid_tool_names):
        self.max_iterations = max_iterations
        self.valid_tool_names = list(valid_tool_names)
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix = ""
        self.stream_delta_callback = None
        self.messages = None
        self._invalid_tool_retries = 0
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._mute_post_response = False
        self._thinking_prefill_retries = 0
        self._empty_content_retries = 0
        self._post_tool_empty_retried = False
        self._dropped_toolcall_retries = 0
        self._incremental_persistence_failed = False
        self._tool_guardrail_halt_decision = None
        self._stream_needs_break = False
        self._session_messages = None
        self.context_compressor = SimpleNamespace(
            awaiting_real_usage_after_compression=True
        )
        self.iteration_budget = SimpleNamespace(refund=lambda: None)

    # -- validation hooks --
    def _uniquify_tool_call_ids(self, tool_calls):
        pass

    def _repair_tool_call(self, name):
        return None

    def _buffer_vprint(self, *a, **k):
        pass

    def _vprint(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    # -- cap/dedupe hooks (identity: names are valid, ids unique) --
    def _deduplicate_tool_calls(self, calls):
        return list(calls)

    def _cap_delegate_task_calls(self, calls):
        return list(calls)

    # -- REAL history builder --
    def _build_assistant_message(self, assistant_message, finish_reason):
        return build_assistant_message(self, assistant_message, finish_reason)

    def _extract_reasoning(self, message):
        return getattr(message, "reasoning_content", None)

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, value):
        return value, None

    def _deterministic_call_id(self, name, args, index):
        return f"call-{index}"

    def _derive_responses_function_call_id(self, value, response_item_id=None):
        return value

    # -- persistence / emit --
    def _flush_messages_to_session_db(self, messages, conversation_history):
        return True

    def _emit_interim_assistant_message(self, msg):
        pass

    def _has_content_after_think_block(self, content):
        return bool(content and str(content).strip())

    def _should_emit_quiet_tool_messages(self):
        return False

    def _has_stream_consumers(self):
        return False

    def _interim_assistant_visible_text(self, msg):
        return ""

    def _execute_tool_calls(
        self, assistant_message, messages, effective_task_id, api_call_count
    ):
        for tc in assistant_message.tool_calls:
            append_message(
                messages,
                {
                    "role": "tool",
                    "name": tc.function.name,
                    "tool_call_id": coalesce_tool_call_id(tc),
                    "content": json.dumps({"ok": True}),
                },
            )

    def _touch_activity(self, *a, **k):
        pass


def _run_trimmed_round(agent, assistant_message):
    import agent.turn_tool_round as ttr

    real_compress = ttr.compress_after_tool_results
    monkey_holder = {}

    def _passthrough(a, **kw):
        return SimpleNamespace(
            end_turn=False,
            messages=kw["messages"],
            active_system_prompt=kw["active_system_prompt"],
            conversation_history=kw["conversation_history"],
            compression_attempts=kw["compression_attempts"],
            final_response=kw["final_response"],
            turn_exit_reason=kw["turn_exit_reason"],
            current_turn_user_idx=kw["current_turn_user_idx"],
        )

    ttr.compress_after_tool_results = _passthrough
    try:
        return run_tool_round(
            agent,
            assistant_message=assistant_message,
            finish_reason="tool_calls",
            messages=[],
            conversation_history=[],
            api_call_count=1,
            effective_task_id="task",
            user_message="hi",
            system_message=None,
            active_system_prompt=None,
            compression_attempts=0,
            max_compression_attempts=3,
            final_response=None,
            failed=False,
            _turn_exit_reason="unknown",
            truncated_tool_call_retries=0,
            current_turn_user_idx=0,
            tool_call_count=0,
        )
    finally:
        ttr.compress_after_tool_results = real_compress


def _bedrock_assistant_message():
    from agent.bedrock_adapter import normalize_converse_response
    from agent.transports.bedrock import BedrockTransport

    raw = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "working"},
                    {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
                    {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
                    {
                        "toolUse": {
                            "toolUseId": "t3",
                            "name": "three",
                            "input": {"n": 3},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {},
        "modelId": "m",
    }
    adapter_ns = normalize_converse_response(raw)
    normalized = BedrockTransport().normalize_response(adapter_ns)
    assert [tc.id for tc in normalized.tool_calls] == ["t1", "t2", "t3"]
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
        reasoning_details=None,
        bedrock_content_blocks=(normalized.provider_data or {}).get(
            "bedrock_content_blocks"
        ),
    )
    assert msg.bedrock_content_blocks is not None
    return msg


def test_trimmed_batch_replays_matching_bedrock_tool_use_and_results():
    from agent.bedrock_adapter import convert_messages_to_converse

    agent = _TrimAgent(max_iterations=2, valid_tool_names=["one", "two", "three"])
    verdict = _run_trimmed_round(agent, _bedrock_assistant_message())
    assert verdict.action == "continue"
    assert verdict.tool_call_count == 2

    persisted = next(
        m
        for m in verdict.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert [tc["id"] for tc in persisted["tool_calls"]] == ["t1", "t2"]
    carrier_ids = [
        b["toolUse"]["toolUseId"]
        for b in persisted["bedrock_content_blocks"]
        if "toolUse" in b
    ]
    assert carrier_ids == ["t1", "t2"], f"native carrier not trimmed: {carrier_ids}"

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
    assert use_ids == ["t1", "t2"]
    assert result_ids == use_ids, (
        f"malformed replay: toolUse={use_ids} toolResult={result_ids}"
    )


def _anthropic_assistant_message():
    from agent.transports.anthropic import AnthropicTransport

    def _thinking(text, sig):
        return SimpleNamespace(type="thinking", thinking=text, signature=sig)

    def _tool_use(block_id, name, payload):
        return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)

    response = SimpleNamespace(
        content=[
            _thinking("plan A", "sig-AAA"),
            _tool_use("toolu_1", "read_file", {"path": "a.py"}),
            _thinking("plan B", "sig-BBB"),
            _tool_use("toolu_2", "read_file", {"path": "b.py"}),
            _thinking("plan C", "sig-CCC"),
            _tool_use("toolu_3", "read_file", {"path": "c.py"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )
    normalized = AnthropicTransport().normalize_response(response)
    assert [tc.id for tc in normalized.tool_calls] == ["toolu_1", "toolu_2", "toolu_3"]
    blocks = (normalized.provider_data or {}).get("anthropic_content_blocks")
    assert blocks is not None, "ordered carrier required for this regression"
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
        reasoning_details=(normalized.provider_data or {}).get("reasoning_details"),
        anthropic_content_blocks=blocks,
    )


def test_trimmed_batch_replays_matching_anthropic_tool_use_and_results():
    from agent.anthropic_message_convert import convert_messages_to_anthropic

    agent = _TrimAgent(max_iterations=2, valid_tool_names=["read_file"])
    verdict = _run_trimmed_round(agent, _anthropic_assistant_message())
    assert verdict.action == "continue"
    assert verdict.tool_call_count == 2

    persisted = next(
        m
        for m in verdict.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert [tc["id"] for tc in persisted["tool_calls"]] == ["toolu_1", "toolu_2"]
    carrier_ids = [
        b["id"]
        for b in persisted["anthropic_content_blocks"]
        if b.get("type") == "tool_use"
    ]
    assert carrier_ids == ["toolu_1", "toolu_2"], (
        f"native carrier not trimmed: {carrier_ids}"
    )

    _system, converted = convert_messages_to_anthropic([
        {"role": "user", "content": "go"},
        *verdict.messages,
    ])
    assistant_turn = next(m for m in converted if m["role"] == "assistant")
    result_turn = next(
        m
        for m in converted
        if m["role"] == "user"
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
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
    kinds = [b["type"] for b in assistant_turn["content"]]
    assert kinds.index("tool_use") > kinds.index("thinking"), (
        "interleaved order must survive the trim"
    )
