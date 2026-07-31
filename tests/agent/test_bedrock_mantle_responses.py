"""Regression tests for Bedrock Mantle multi-turn stability (Issue #75471).

Two provider-scoped defects on the Mantle Responses endpoint:

1. Mantle reuses response-local indexed pairing IDs (``call_0``) across
   responses. Hermes stores them verbatim, so the pre-call duplicate
   sanitizer drops later call/result pairs and freezes the tool trajectory.

2. Mantle accepts replayed ``encrypted_content`` reasoning items but
   degrades the model (hidden-reasoning-only turns, repeated tool calls,
   leaked tool-call scaffolding). The HTTP 400 ``invalid_encrypted_content``
   recovery switch cannot catch accepted-but-degraded responses, so the
   capability must be decided before dispatch.

These tests pin the strict repair boundary: remint indexed pairing IDs at
Mantle response ingestion, suppress encrypted reasoning replay at history
conversion and final preflight, and preserve every non-Mantle contract.
"""

import re
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import sanitize_api_messages
from agent.chat_completion_helpers import build_assistant_message
from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_api_kwargs,
    _preflight_codex_input_items,
)
from agent.transports import get_transport

MANTLE_URL = "https://bedrock-mantle.us-west-2.api.aws/v1"


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


def _mantle_response(call_id: str = "call_0", item_id: str = "fc_1", name: str = "terminal", arguments: str = "{}"):
    return SimpleNamespace(
        status="completed",
        id="resp_mantle",
        output=[
            SimpleNamespace(
                type="function_call",
                id=item_id,
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        ],
    )


class _StubAgent:
    """Minimal agent stub for ``build_assistant_message`` (Issue #75471)."""

    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    @staticmethod
    def _extract_reasoning(assistant_message):
        return getattr(assistant_message, "reasoning", None)

    @staticmethod
    def _needs_thinking_reasoning_pad():
        return False

    @staticmethod
    def _strip_think_blocks(content):
        return content

    @staticmethod
    def _split_responses_tool_id(raw_id):
        from agent.codex_responses_adapter import _split_responses_tool_id
        return _split_responses_tool_id(raw_id)

    @staticmethod
    def _derive_responses_function_call_id(call_id, response_item_id=None):
        from agent.codex_responses_adapter import _derive_responses_function_call_id
        return _derive_responses_function_call_id(call_id, response_item_id)

    @staticmethod
    def _deterministic_call_id(fn_name, arguments, index=0):
        from agent.codex_responses_adapter import _deterministic_call_id
        return _deterministic_call_id(fn_name, arguments, index)


# ---------------------------------------------------------------------------
# Endpoint predicate (Requirement 1)
# ---------------------------------------------------------------------------


def _mantle_predicate():
    from agent.transports.codex import _is_bedrock_mantle_base_url
    return _is_bedrock_mantle_base_url


def test_mantle_predicate_positive_matches():
    predicate = _mantle_predicate()
    for base_url in (
        "https://bedrock-mantle.us-west-2.api.aws/v1",
        "https://bedrock-mantle.eu-central-1.api.aws",
        "https://bedrock-mantle.ap-southeast-2.api.aws/responses",
    ):
        assert predicate(base_url) is True, base_url


def test_mantle_predicate_ignores_port_path_query_fragment_case_trailing_dot():
    predicate = _mantle_predicate()
    assert predicate("https://bedrock-mantle.us-west-2.api.aws:8443/v1") is True
    assert predicate("BEDROCK-MANTLE.US-WEST-2.API.AWS/v1") is True
    assert predicate("https://bedrock-mantle.us-west-2.api.aws./v1") is True
    assert predicate("https://bedrock-mantle.us-west-2.api.aws/v1?x=1#frag") is True


def test_mantle_predicate_rejects_lookalikes():
    predicate = _mantle_predicate()
    for base_url in (
        "https://api.openai.com/v1",
        "https://bedrock-mantle..api.aws/v1",
        "https://bedrock-mantle.api.aws/v1",
        "https://bedrock-mantle.us-west-2.api.aws.example/v1",
        "https://example.com/bedrock-mantle.us-west-2.api.aws/v1",
        "https://bedrock-mantle.us-west-2.example.com/v1",
        "https://my-bedrock-mantle.us-west-2.api.aws/v1",
        "https://us-west-2.bedrock-mantle.api.aws/v1",
    ):
        assert predicate(base_url) is False, base_url


def test_mantle_predicate_empty_and_invalid_inputs():
    predicate = _mantle_predicate()
    assert predicate(None) is False
    assert predicate("") is False
    assert predicate("not a url") is False


# ---------------------------------------------------------------------------
# Remint unit tests (Requirement 2 / Requirement 3)
# ---------------------------------------------------------------------------


def _remint(*args, **kwargs):
    from agent.codex_responses_adapter import _remint_mantle_indexed_call_id
    return _remint_mantle_indexed_call_id(*args, **kwargs)


@pytest.mark.parametrize("raw_call_id", ["call_0", "call_999999", "fc_1"])
def test_remint_indexed_ids_produce_mantle_surrogate(raw_call_id):
    reminted = _remint(
        raw_call_id,
        request_scope="req-scope-1",
        provider_response_id="resp_mantle",
        output_ordinal=0,
    )
    assert reminted.startswith("call_mtl_")
    assert len(reminted) <= 64
    assert re.fullmatch(r"call_mtl_[0-9a-f]{40}", reminted), reminted


def test_remint_ignores_outside_indexed_grammar():
    for raw_call_id in ("call_1000000", "call_ab", "fc_", "call_", "msg_abc", "call_mtl_x"):
        assert _remint(
            raw_call_id,
            request_scope="req-scope-1",
            provider_response_id="resp_mantle",
            output_ordinal=0,
        ) == raw_call_id


def test_remint_same_input_produces_same_output():
    a = _remint("call_0", request_scope="scope", provider_response_id="resp", output_ordinal=0)
    b = _remint("call_0", request_scope="scope", provider_response_id="resp", output_ordinal=0)
    assert a == b


def test_remint_distinct_scopes_produce_distinct_output():
    a = _remint("call_0", request_scope="scope-1", provider_response_id="resp", output_ordinal=0)
    b = _remint("call_0", request_scope="scope-2", provider_response_id="resp", output_ordinal=0)
    assert a != b


def test_remint_scope_separates_even_when_response_id_repeats():
    a = _remint("call_0", request_scope="scope-1", provider_response_id="same", output_ordinal=0)
    b = _remint("call_0", request_scope="scope-2", provider_response_id="same", output_ordinal=0)
    assert a != b


def test_remint_ordinal_distinguishes_malformed_duplicates_in_one_response():
    a = _remint("call_0", request_scope="scope", provider_response_id="resp", output_ordinal=0)
    b = _remint("call_0", request_scope="scope", provider_response_id="resp", output_ordinal=1)
    assert a != b


def test_remint_missing_or_blank_response_id_keeps_scope_separation():
    a = _remint("call_0", request_scope="scope", provider_response_id="", output_ordinal=0)
    b = _remint("call_0", request_scope="scope-2", provider_response_id="", output_ordinal=0)
    assert a != b


# ---------------------------------------------------------------------------
# Multi-turn round-trip through the real chain (Requirement 4)
# ---------------------------------------------------------------------------


def test_mantle_multiturn_preserves_both_call_result_pairs(transport):
    """Two Mantle responses each reusing ``call_0`` must survive
    normalization, assistant-message building, pre-call sanitization, and
    Responses serialization with BOTH call/result pairs intact.

    Pre-fix: both normalize to ``call_0``, the sanitizer drops the later
    pair, and only one call/output survives. Post-fix: the reminted pairing
    keys are distinct, so both pairs replay.
    """
    result_a = transport.normalize_response(
        _mantle_response(call_id="call_0", item_id="fc_1", name="terminal", arguments="{}"),
        base_url=MANTLE_URL,
        request_scope="scope-a",
    )
    result_b = transport.normalize_response(
        _mantle_response(call_id="call_0", item_id="fc_1", name="read_file", arguments="{}"),
        base_url=MANTLE_URL,
        request_scope="scope-b",
    )

    stub = _StubAgent()
    msg_a = build_assistant_message(stub, result_a, result_a.finish_reason)
    msg_b = build_assistant_message(stub, result_b, result_b.finish_reason)

    call_a = msg_a["tool_calls"][0]
    call_b = msg_b["tool_calls"][0]

    history = [
        {"role": "user", "content": "task A"},
        msg_a,
        {"role": "tool", "tool_call_id": call_a["id"], "content": "result A"},
        {"role": "user", "content": "task B"},
        msg_b,
        {"role": "tool", "tool_call_id": call_b["id"], "content": "result B"},
    ]

    sanitized = sanitize_api_messages(history)

    assistant_calls = [
        m["tool_calls"][0]["id"]
        for m in sanitized
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    result_ids = [m["tool_call_id"] for m in sanitized if m.get("role") == "tool"]

    assert len(assistant_calls) == 2
    assert len(result_ids) == 2
    assert set(assistant_calls) == set(result_ids)

    items = _chat_messages_to_responses_input(sanitized)
    function_calls = [i for i in items if i.get("type") == "function_call"]
    function_outputs = [i for i in items if i.get("type") == "function_call_output"]

    assert len(function_calls) == 2
    assert len(function_outputs) == 2
    assert {i["call_id"] for i in function_calls} == {i["call_id"] for i in function_outputs}


def test_mantle_normalized_tool_call_preserves_response_item_id(transport):
    """A valid non-empty ``fc_*`` provider item id survives reminting as
    ``response_item_id``; the canonical pairing key is the surrogate."""
    result = transport.normalize_response(
        _mantle_response(call_id="call_0", item_id="fc_1", name="terminal", arguments="{}"),
        base_url=MANTLE_URL,
        request_scope="scope-a",
    )
    tc = result.tool_calls[0]
    assert tc.call_id.startswith("call_mtl_")
    assert tc.call_id != "call_0"
    assert tc.response_item_id == "fc_1"


def test_non_mantle_response_keeps_indexed_call_id(transport):
    """A generic relay or OpenAI-style endpoint supplying ``call_0`` must
    keep the provider ID untouched (Requirement 7)."""
    result = transport.normalize_response(
        _mantle_response(call_id="call_0", item_id="fc_1", name="terminal", arguments="{}"),
        base_url="https://responses.example.com/v1",
    )
    tc = result.tool_calls[0]
    assert tc.call_id == "call_0"


def test_sanitizer_still_cleans_damaged_duplicate_pairing_ids(transport):
    """The generic duplicate-defense must remain active for genuinely
    damaged history that already contains raw duplicate pairing IDs."""
    history = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_x", "call_id": "call_x", "response_item_id": "fc_1",
                "type": "function",
                "function": {"name": "a", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": "A"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_x", "call_id": "call_x", "response_item_id": "fc_1",
                "type": "function",
                "function": {"name": "b", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": "B"},
    ]
    sanitized = sanitize_api_messages(history)
    calls = [m for m in sanitized if m.get("role") == "assistant" and m.get("tool_calls")]
    results = [m for m in sanitized if m.get("role") == "tool"]
    assert len(calls) == 1
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Encrypted reasoning gate (Requirement 5)
# ---------------------------------------------------------------------------


def _reasoning_history():
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "thinking",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "opaque-blob", "summary": []},
            ],
        },
        {"role": "user", "content": "follow up"},
    ]


def test_mantle_request_omits_historical_encrypted_reasoning(transport):
    kw = transport.build_kwargs(
        model="gpt-5.6",
        messages=_reasoning_history(),
        tools=[],
        base_url=MANTLE_URL,
        reasoning_config={"effort": "high", "enabled": True},
    )
    reasoning_input = [i for i in kw["input"] if i.get("type") == "reasoning"]
    include = kw.get("include") or []
    assert reasoning_input == []
    assert "reasoning.encrypted_content" not in include
    # Visible context and reasoning effort are preserved.
    assert kw["reasoning"]["effort"] == "high"
    assert any("follow up" in str(i) for i in kw["input"])


def test_openai_codex_request_keeps_reasoning_replay(transport):
    kw = transport.build_kwargs(
        model="gpt-5.6",
        messages=_reasoning_history(),
        tools=[],
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning_config={"effort": "high", "enabled": True},
    )
    reasoning_input = [i for i in kw["input"] if i.get("type") == "reasoning"]
    include = kw.get("include") or []
    assert len(reasoning_input) == 1
    assert "reasoning.encrypted_content" in include


def test_mantle_preflight_filters_override_injected_reasoning(transport):
    """request_overrides merge after history conversion; the final preflight
    boundary must remove injected reasoning and the encrypted include value
    while preserving other include values (Requirement 5.5 / 5.6)."""
    api_kwargs = {
        "model": "gpt-5.6",
        "instructions": "You are Hermes.",
        "store": False,
        "input": [
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "encrypted_content": "injected", "summary": []},
        ],
        "include": ["reasoning.encrypted_content", "reasoning.summary_text"],
    }
    normalized = transport.preflight_kwargs(api_kwargs, base_url=MANTLE_URL)
    assert not [i for i in normalized["input"] if i.get("type") == "reasoning"]
    assert "reasoning.encrypted_content" not in (normalized.get("include") or [])
    assert "reasoning.summary_text" in (normalized.get("include") or [])
    # Visible assistant/user content stays.
    assert normalized["input"][0]["role"] == "user"


def test_non_mantle_preflight_keeps_override_reasoning(transport):
    api_kwargs = {
        "model": "gpt-5.6",
        "instructions": "You are Hermes.",
        "store": False,
        "input": [
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "encrypted_content": "injected", "summary": []},
        ],
        "include": ["reasoning.encrypted_content", "reasoning.summary_text"],
    }
    normalized = transport.preflight_kwargs(
        api_kwargs, base_url="https://chatgpt.com/backend-api/codex"
    )
    assert len([i for i in normalized["input"] if i.get("type") == "reasoning"]) == 1
    assert "reasoning.encrypted_content" in (normalized.get("include") or [])
