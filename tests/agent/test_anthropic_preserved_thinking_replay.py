from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.message_sanitization import (
    native_anthropic_accounting_projection,
    stale_thinking_reaches_wire,
)
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_native_anthropic_messages_tokens_rough,
    estimate_native_anthropic_request_tokens_rough,
)


def _signed_turn(question: str, answer: str, sig: str, *, thinking: str | None = None):
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": answer,
            "reasoning": thinking or f"thought-{sig}",
            "reasoning_details": [
                {"type": "thinking", "thinking": thinking or f"thought-{sig}", "signature": sig}
            ],
        },
    ]


def _assistant(messages, index: int):
    return [m for m in messages if m["role"] == "assistant"][index]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-4", False),
        ("claude-opus-4-5", True),
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-5", False),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-20250514", False),
        ("claude-sonnet-4-20250514", False),
        ("claude-opus-4-5-20251101", True),
        ("claude-opus-4-6-20250414", True),
        ("claude-sonnet-4-5-20250929", False),
        ("claude-opus-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-fable-5-1", True),
        ("claude-mythos-5", True),
        ("claude-mythos-preview", True),
        ("claude-haiku-4-5", False),
    ],
)
def test_native_anthropic_wire_truth_tracks_preserved_thinking_models(model, expected):
    assert stale_thinking_reaches_wire(
        "anthropic_messages", "anthropic", model, "https://api.anthropic.com"
    ) is expected


def test_formerly_latest_turn_stays_byte_stable_on_preserved_thinking_model():
    prefix = _signed_turn("Q1", "A1", "sig_1") + _signed_turn("Q2", "A2", "sig_2")
    _, short = convert_messages_to_anthropic(prefix, model="claude-opus-4-6")
    _, long = convert_messages_to_anthropic(
        prefix + _signed_turn("Q3", "A3", "sig_3"), model="claude-opus-4-6"
    )

    assert _assistant(short, 1) == _assistant(long, 1)
    assert _assistant(long, 1)["content"][0]["signature"] == "sig_2"


def test_older_claude_keeps_latest_only_policy():
    messages = _signed_turn("Q1", "A1", "sig_1") + _signed_turn("Q2", "A2", "sig_2")
    _, converted = convert_messages_to_anthropic(messages, model="claude-sonnet-4-5")

    first, second = (_assistant(converted, 0), _assistant(converted, 1))
    assert not any(
        isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}
        for block in first["content"]
    )
    assert any(
        isinstance(block, dict) and block.get("type") == "thinking"
        for block in second["content"]
    )


def test_preserved_thinking_accounting_charges_historical_turns():
    thinking = "x" * 8000
    messages = (
        _signed_turn("Q1", "A1", "sig_1", thinking=thinking)
        + _signed_turn("Q2", "A2", "sig_2", thinking=thinking)
        + _signed_turn("Q3", "A3", "sig_3", thinking=thinking)
    )
    keep_all = estimate_messages_tokens_rough(messages, charge_stale_thinking=True)
    latest_only = estimate_messages_tokens_rough(messages, charge_stale_thinking=False)

    assert keep_all - latest_only >= 3500


class _ConfigDB:
    def __init__(self):
        self.config = {}

    def patch_session_model_config(self, session_id, patch):
        self.config.update(patch)

    def get_session_model_config_value(self, session_id, key, default=None):
        return self.config.get(key, default)


def _agent(db):
    return SimpleNamespace(
        api_mode="anthropic_messages",
        provider="anthropic",
        model="claude-opus-4-6",
        base_url="https://api.anthropic.com",
        session_id="session-1",
        _session_db=db,
        _persist_disabled=False,
    )


def _carrier_message():
    return {
        "role": "assistant",
        "content": "answer",
        "reasoning": "secret chain",
        "reasoning_content": "secret chain",
        "reasoning_details": [
            {"type": "thinking", "thinking": "secret chain", "signature": "sig_bad"},
            {"type": "redacted_thinking", "data": "red_bad"},
        ],
        "anthropic_content_blocks": [
            {"type": "thinking", "thinking": "secret chain", "signature": "sig_bad"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "x"}},
            {"type": "redacted_thinking", "data": "red_bad"},
        ],
        "tool_calls": [
            {
                "id": "tool_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{\"q\":\"x\"}"},
            }
        ],
    }


def test_rejected_signature_is_removed_from_every_carrier_and_persists_across_resume():
    from agent.anthropic_thinking_replay import (
        apply_rejected_thinking_suppression,
        remember_rejected_thinking,
    )

    db = _ConfigDB()
    db.config["_usage_anchor"] = {"prompt_tokens": 999}
    first_agent = _agent(db)
    first_agent._usage_anchor = {"prompt_tokens": 999}
    request = [_carrier_message()]

    removed = remember_rejected_thinking(first_agent, request)
    assert removed >= 4
    assert "reasoning_details" not in request[0]
    assert "reasoning" not in request[0]
    assert "reasoning_content" not in request[0]
    assert first_agent._usage_anchor is None
    assert db.config["_usage_anchor"] is None
    assert [b["type"] for b in request[0]["anthropic_content_blocks"]] == ["text", "tool_use"]

    resumed_agent = _agent(db)
    rebuilt = [_carrier_message()]
    apply_rejected_thinking_suppression(resumed_agent, rebuilt)
    assert "reasoning_details" not in rebuilt[0]
    assert "reasoning" not in rebuilt[0]
    assert "reasoning_content" not in rebuilt[0]
    assert [b["type"] for b in rebuilt[0]["anthropic_content_blocks"]] == ["text", "tool_use"]


def test_build_api_messages_applies_persisted_rejection_suppression():
    from agent.anthropic_thinking_replay import remember_rejected_thinking
    from agent.turn_context import build_api_messages

    db = _ConfigDB()
    first_agent = _agent(db)
    rejected = [_carrier_message()]
    remember_rejected_thinking(first_agent, rejected)

    resumed = _agent(db)
    resumed._current_turn_timestamp = 1.0
    resumed.ephemeral_system_prompt = ""
    resumed._copy_reasoning_content_for_api = lambda source, target: None
    resumed._should_sanitize_tool_calls = lambda: False

    history = [
        {"role": "user", "content": "Q"},
        _carrier_message(),
        {"role": "user", "content": "continue"},
    ]
    api_messages, _ = build_api_messages(
        resumed,
        copy.deepcopy(history),
        current_turn_user_idx=2,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="",
    )

    assistant = next(m for m in api_messages if m.get("role") == "assistant")
    assert "reasoning_details" not in assistant
    assert [b["type"] for b in assistant["anthropic_content_blocks"]] == ["text", "tool_use"]


def test_nous_portal_uses_same_preserved_thinking_capability_boundary():
    assert stale_thinking_reaches_wire(
        "anthropic_messages",
        "nous",
        "claude-opus-4-6",
        "https://inference-api.nousresearch.com/v1/messages",
    )
    assert not stale_thinking_reaches_wire(
        "anthropic_messages",
        "openrouter",
        "claude-opus-4-6",
        "https://openrouter.ai/api/v1",
    )




def test_turn_recovery_repairs_recognized_anthropic_signature_rejection():
    from agent.error_classifier import FailoverReason
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState

    db = _ConfigDB()
    agent = _agent(db)
    agent.log_prefix = ""
    agent._vprint = lambda *args, **kwargs: None

    canonical = [_carrier_message()]
    canonical_before = copy.deepcopy(canonical)
    request = copy.deepcopy(canonical)
    retry = TurnRetryState()
    classified = SimpleNamespace(reason=FailoverReason.thinking_signature)

    assert _recover_format_errors(
        agent,
        RuntimeError("Invalid signature in thinking block"),
        classified,
        retry,
        canonical,
        request,
    )
    assert retry.thinking_sig_retry_attempted
    assert canonical == canonical_before
    assert "reasoning_details" not in request[0]
    assert [block["type"] for block in request[0]["anthropic_content_blocks"]] == [
        "text",
        "tool_use",
    ]

    rebuilt = copy.deepcopy(canonical)
    from agent.anthropic_thinking_replay import apply_rejected_thinking_suppression
    apply_rejected_thinking_suppression(_agent(db), rebuilt)
    assert "reasoning_details" not in rebuilt[0]
    assert [block["type"] for block in rebuilt[0]["anthropic_content_blocks"]] == [
        "text",
        "tool_use",
    ]


def test_dated_claude_4_snapshots_keep_latest_only_policy():
    for model in ("claude-opus-4-20250514", "claude-sonnet-4-20250514"):
        messages = _signed_turn("Q1", "A1", "sig_1") + _signed_turn("Q2", "A2", "sig_2")
        _, converted = convert_messages_to_anthropic(messages, model=model)
        first, second = (_assistant(converted, 0), _assistant(converted, 1))
        assert not any(
            isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}
            for block in first["content"]
        )
        assert any(
            isinstance(block, dict) and block.get("type") == "thinking"
            for block in second["content"]
        )


def _assembly_agent(db=None):
    agent = _agent(db or _ConfigDB())
    agent._current_turn_timestamp = 1.0
    agent.ephemeral_system_prompt = ""
    agent.prefill_messages = []
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api

    agent._needs_thinking_reasoning_pad = lambda: False
    agent._copy_reasoning_content_for_api = (
        lambda source, target: copy_reasoning_content_for_api(agent, source, target)
    )
    agent._should_sanitize_tool_calls = lambda: False
    agent._sanitize_api_messages = lambda value: value
    agent._emit_warning = lambda *args, **kwargs: None
    agent._drop_thinking_only_and_merge_users = lambda value, **kwargs: value
    agent.tools = []
    agent._use_prompt_caching = False
    agent._usage_anchor = None
    agent.context_compressor = SimpleNamespace()
    agent._extract_reasoning = lambda message: getattr(message, "reasoning", None)
    agent._strip_think_blocks = lambda text: text
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _native_stored_assistant(agent, size, signature):
    from agent.chat_completion_helpers import build_assistant_message
    from agent.transports.anthropic import AnthropicTransport

    transport = AnthropicTransport()
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="x" * size, signature=signature),
            SimpleNamespace(type="text", text="A"),
        ],
        stop_reason="end_turn",
        stop_details=None,
    )
    normalized = transport.normalize_response(response)
    return build_assistant_message(agent, normalized, normalized.finish_reason)


def _native_history(agent, size):
    return [
        {"role": "user", "content": "Q1"},
        _native_stored_assistant(agent, size, "sig_1"),
        {"role": "user", "content": "Q2"},
        _native_stored_assistant(agent, size, "sig_2"),
        {"role": "user", "content": "continue"},
    ]


def _patch_assembly_loop(monkeypatch, selector=None):
    import agent.conversation_loop as loop

    monkeypatch.setattr(
        loop,
        "_apply_context_engine_selection",
        selector or (lambda agent, api, history, incoming, logger=None: api),
    )
    monkeypatch.setattr(loop, "_canonicalize_api_tool_calls", lambda messages: None)
    monkeypatch.setattr(
        loop,
        "_midturn_request_pressure_tokens",
        lambda agent, messages, system, approx: approx,
    )
    monkeypatch.setattr(loop, "_pressure_with_real_floor", lambda compressor, value: value)


def _assemble(agent, history):
    from agent.turn_request_assembly import assemble_api_request

    return assemble_api_request(
        agent,
        messages=history,
        current_turn_user_idx=len(history) - 1,
        _ext_prefetch_cache="",
        _plugin_user_context="",
        moa_config=None,
        active_system_prompt="",
        original_user_message=history[-1]["content"],
        pending_moa_prepared_request=None,
        request_logger=logging.getLogger("anthropic-preserved-thinking-test"),
    )


def test_full_producer_to_wire_prices_surviving_non_tool_thinking(monkeypatch):
    from agent.transports.anthropic import AnthropicTransport

    _patch_assembly_loop(monkeypatch)
    agent = _assembly_agent()

    small_request = _assemble(agent, _native_history(agent, 1))
    large_request = _assemble(agent, _native_history(agent, 8000))
    assert large_request.approx_tokens - small_request.approx_tokens >= 3500
    assert large_request.request_pressure_tokens == large_request.approx_tokens

    kwargs = AnthropicTransport().build_kwargs(
        agent.model,
        large_request.api_messages,
        tools=[],
        base_url=agent.base_url,
    )
    emitted = sum(
        len(block.get("thinking", ""))
        for message in kwargs["messages"]
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "thinking"
    )
    assert emitted == 16000


def test_context_selection_canonical_clone_does_not_double_count_thinking(monkeypatch):
    from agent.transports.anthropic import AnthropicTransport

    agent = _assembly_agent()
    history = _native_history(agent, 8000)

    _patch_assembly_loop(monkeypatch)
    normal = _assemble(agent, copy.deepcopy(history))

    def select_canonical(agent, api, canonical, incoming, logger=None):
        return copy.deepcopy(canonical)

    _patch_assembly_loop(monkeypatch, selector=select_canonical)
    selected = _assemble(agent, copy.deepcopy(history))

    assert selected.approx_tokens == normal.approx_tokens
    assert selected.request_pressure_tokens == normal.request_pressure_tokens

    transport = AnthropicTransport()
    normal_kwargs = transport.build_kwargs(
        agent.model, normal.api_messages, tools=[], base_url=agent.base_url
    )
    selected_kwargs = transport.build_kwargs(
        agent.model, selected.api_messages, tools=[], base_url=agent.base_url
    )
    assert selected_kwargs["messages"] == normal_kwargs["messages"]


def test_canonical_preflight_dedupes_ordered_thinking_carrier():
    from agent.turn_context import _preflight_request_tokens

    agent = _assembly_agent()
    thinking = "x" * 8000
    assistant = _carrier_message()
    assistant["reasoning"] = thinking
    assistant["reasoning_details"][0]["thinking"] = thinking
    assistant["anthropic_content_blocks"][0]["thinking"] = thinking
    assistant["tool_calls"] = [
        {
            "id": "tool_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"q\":\"x\"}"},
        }
    ]
    canonical = [
        {"role": "user", "content": "Q"},
        assistant,
        {"role": "user", "content": "continue"},
    ]

    preflight = _preflight_request_tokens(agent, copy.deepcopy(canonical), "")

    request_copy = copy.deepcopy(canonical)
    request_copy[1].pop("reasoning", None)
    expected = estimate_native_anthropic_request_tokens_rough(request_copy)
    assert preflight == expected


def test_native_accounting_projection_dedupes_ordered_carrier_and_opaque_bytes():
    base = _carrier_message()
    base["reasoning_details"][0]["thinking"] = "x" * 8000
    base["anthropic_content_blocks"][0]["thinking"] = "x" * 8000

    huge_opaque = copy.deepcopy(base)
    huge_opaque["reasoning_details"][0]["signature"] = "s" * 20000
    huge_opaque["reasoning_details"][1]["data"] = "r" * 20000
    huge_opaque["anthropic_content_blocks"][0]["signature"] = "s" * 20000
    huge_opaque["anthropic_content_blocks"][-1]["data"] = "r" * 20000

    ordered_only = copy.deepcopy(huge_opaque)
    ordered_only.pop("reasoning_details")

    projected_huge, readable = native_anthropic_accounting_projection([huge_opaque])

    assert "_anthropic_readable_thinking_estimate" not in repr(projected_huge)
    assert readable == ("x" * 8000,)
    assert estimate_native_anthropic_messages_tokens_rough(
        [huge_opaque]
    ) == estimate_native_anthropic_messages_tokens_rough([ordered_only])
    assert estimate_native_anthropic_messages_tokens_rough(
        [huge_opaque]
    ) == estimate_native_anthropic_messages_tokens_rough([base])


def test_context_selection_cannot_restore_rejected_thinking(monkeypatch):
    from agent.anthropic_thinking_replay import remember_rejected_thinking

    db = _ConfigDB()
    first = _assembly_agent(db)
    remember_rejected_thinking(first, [_carrier_message()])

    def select_canonical(agent, api, history, incoming, logger=None):
        return copy.deepcopy(history)

    _patch_assembly_loop(monkeypatch, selector=select_canonical)
    resumed = _assembly_agent(db)
    history = [
        {"role": "user", "content": "Q"},
        _carrier_message(),
        {"role": "user", "content": "continue"},
    ]
    assembled = _assemble(resumed, history)
    assistant = next(m for m in assembled.api_messages if m.get("role") == "assistant")

    assert "reasoning_details" not in assistant
    assert "reasoning" not in assistant
    assert "reasoning_content" not in assistant
    assert [b["type"] for b in assistant["anthropic_content_blocks"]] == ["text", "tool_use"]
    _, native = convert_messages_to_anthropic(assembled.api_messages, model=resumed.model)
    assert "sig_bad" not in repr(native)
    assert "secret chain" not in repr(native)


def test_preflight_ignores_persisted_rejected_thinking(monkeypatch):
    from agent.anthropic_thinking_replay import remember_rejected_thinking
    from agent.turn_context import _preflight_request_tokens

    db = _ConfigDB()
    first = _assembly_agent(db)
    remember_rejected_thinking(first, [copy.deepcopy(_carrier_message())])

    resumed = _assembly_agent(db)
    history = [
        {"role": "user", "content": "Q"},
        _carrier_message(),
        {"role": "user", "content": "continue"},
    ]
    preflight = _preflight_request_tokens(resumed, copy.deepcopy(history), "")

    _patch_assembly_loop(monkeypatch)
    assembled = _assemble(resumed, copy.deepcopy(history))
    assert preflight == assembled.approx_tokens

def test_usage_anchor_still_overrides_projected_rough_pressure(monkeypatch):
    import agent.turn_request_assembly as assembly

    _patch_assembly_loop(monkeypatch)
    monkeypatch.setattr(assembly, "anchored_context_tokens", lambda messages, anchor: 1234)
    agent = _assembly_agent()
    agent._usage_anchor = object()
    history = (
        _signed_turn("Q1", "A1", "sig_1", thinking="x" * 8000)
        + [{"role": "user", "content": "continue"}]
    )

    assembled = _assemble(agent, history)
    assert assembled.approx_tokens > 1000
    assert assembled.request_pressure_tokens == 1234
    assert agent._request_pressure_anchored is True
