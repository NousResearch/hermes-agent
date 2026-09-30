from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.message_sanitization import stale_thinking_reaches_wire
from agent.model_metadata import estimate_messages_tokens_rough


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
    }


def test_rejected_signature_is_removed_from_every_carrier_and_persists_across_resume():
    from agent.anthropic_thinking_replay import (
        apply_rejected_thinking_suppression,
        remember_rejected_thinking,
    )

    db = _ConfigDB()
    first_agent = _agent(db)
    request = [_carrier_message()]

    removed = remember_rejected_thinking(first_agent, request)
    assert removed >= 4
    assert "reasoning_details" not in request[0]
    assert [b["type"] for b in request[0]["anthropic_content_blocks"]] == ["text", "tool_use"]

    resumed_agent = _agent(db)
    rebuilt = [_carrier_message()]
    apply_rejected_thinking_suppression(resumed_agent, rebuilt)
    assert "reasoning_details" not in rebuilt[0]
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
