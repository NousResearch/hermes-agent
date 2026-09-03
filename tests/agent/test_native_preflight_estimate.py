"""Native Responses preflight must count the checkpoint-pruned wire (#96155)."""

from types import SimpleNamespace

from agent.codex_responses_adapter import estimate_native_responses_preflight_tokens
from agent.model_metadata import estimate_request_tokens_rough
from agent.turn_context import _preflight_request_tokens


def _codex_agent(**over):
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.6",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        context_compressor=SimpleNamespace(threshold_tokens=765_000),
        tools=None,
    )
    for key, value in over.items():
        setattr(agent, key, value)
    return agent


def _history_with_checkpoint():
    # Pre-checkpoint assistant/tool rows are dropped from the wire; user
    # asks are retained. A durable estimate that counts those dropped rows
    # is what falsely tripped local compression in #96155.
    pre = []
    for i in range(30):
        pre.append({"role": "user", "content": f"ask {i}"})
        pre.append({"role": "assistant", "content": "working " + ("tool output " * 400)})
        pre.append(
            {
                "role": "tool",
                "content": "result " + ("payload " * 400),
                "tool_call_id": f"call-{i}",
            }
        )
    checkpoint_turn = {
        "role": "assistant",
        "content": "checkpointed turn",
        "codex_reasoning_items": [
            {
                "type": "compaction",
                "encrypted_content": "blob",
                "_issuer_kind": "codex_backend",
            }
        ],
    }
    tail = [{"role": "user", "content": "follow-up after checkpoint"}]
    return pre + [checkpoint_turn] + tail


def test_returns_none_when_api_mode_is_not_codex_responses():
    agent = _codex_agent(api_mode="chat_completions")
    assert estimate_native_responses_preflight_tokens(agent, _history_with_checkpoint()) is None


def test_returns_none_when_native_compaction_is_disabled():
    agent = _codex_agent(codex_responses_native_compaction=False)
    assert estimate_native_responses_preflight_tokens(agent, _history_with_checkpoint()) is None


def test_returns_none_when_model_is_outside_gpt56_family():
    agent = _codex_agent(model="gpt-5.2")
    assert estimate_native_responses_preflight_tokens(agent, _history_with_checkpoint()) is None


def test_pruned_estimate_is_far_below_durable_transcript():
    agent = _codex_agent()
    messages = _history_with_checkpoint()
    generic = estimate_request_tokens_rough(messages)
    native = estimate_native_responses_preflight_tokens(agent, messages)

    assert native is not None
    assert generic > native * 2
    assert native < 8_000


def test_preflight_wrapper_uses_pruned_estimate_when_eligible():
    agent = _codex_agent()
    messages = _history_with_checkpoint()
    native = estimate_native_responses_preflight_tokens(agent, messages)
    wrapped = _preflight_request_tokens(agent, messages, "")

    assert native is not None
    assert wrapped == native


def test_preflight_wrapper_falls_back_to_generic_when_ineligible():
    agent = _codex_agent(api_mode="chat_completions")
    messages = _history_with_checkpoint()
    generic = estimate_request_tokens_rough(messages)

    assert _preflight_request_tokens(agent, messages, "") == generic


# ── Mid-turn pre-API guard parity (#96995) ────────────────────────────────
# The mid-turn guard in conversation_loop must measure the same pruned wire
# payload the turn-prologue preflight does (#96644/#96155); before #96995 it
# used the generic durable-history estimate and false-tripped 600s local
# compression on compacted native-Codex sessions.


def test_midturn_pressure_uses_pruned_estimate_when_eligible():
    from agent.conversation_loop import (
        _midturn_request_pressure_tokens,
        estimate_messages_tokens_rough,
    )

    agent = _codex_agent()
    messages = [{"role": "system", "content": "be brief"}] + _history_with_checkpoint()
    native = estimate_native_responses_preflight_tokens(
        agent, messages, system_prompt="be brief"
    )
    generic = estimate_messages_tokens_rough(messages)

    assert native is not None
    assert generic > native * 2
    # The assembled api_messages carry the system row; the helper must not
    # double-count it (converter skips system rows, system_prompt adds it once).
    assert _midturn_request_pressure_tokens(
        agent, messages, "be brief", generic
    ) == native


def test_midturn_pressure_falls_back_to_generic_plus_tools_when_ineligible():
    from agent.conversation_loop import (
        _estimate_tools_tokens_rough,
        _midturn_request_pressure_tokens,
        estimate_messages_tokens_rough,
    )

    agent = _codex_agent(
        api_mode="chat_completions",
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
    )
    messages = [{"role": "system", "content": "be brief"}] + _history_with_checkpoint()
    approx = estimate_messages_tokens_rough(messages)

    assert _midturn_request_pressure_tokens(
        agent, messages, "be brief", approx
    ) == approx + _estimate_tools_tokens_rough(agent.tools)
