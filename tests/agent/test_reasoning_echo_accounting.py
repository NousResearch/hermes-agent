"""Legacy explicit echo must price the historical text the active route sends.

Real config/resolver/init and send projection; no server or live profile access.
"""
from copy import deepcopy
from unittest.mock import patch

import json

import pytest

from agent.agent_runtime_helpers import reasoning_route_fingerprint
from agent.context_compressor import _estimate_msg_budget_tokens
from agent.model_metadata import estimate_request_tokens_rough
from agent.turn_context import _preflight_request_tokens
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = {
        "model": {"default": "neutral-model", "provider": "neutral", "reasoning_echo": True,
                  "context_length": 64000},
        "providers": {"neutral": {"base_url": "https://neutral.example/v1", "api_key": "test-key",
                                  "reasoning_replay_field": "none"}},
    }

    def write_config(echo=True):
        config["model"]["reasoning_echo"] = echo
        (home / "config.yaml").write_text(json.dumps(config))

    def create(echo=True, fallback=None, replay_field="none"):
        config["providers"]["neutral"]["reasoning_replay_field"] = replay_field
        write_config(echo)
        route = resolve_runtime_provider(requested="neutral")
        assert route["provider"] == "custom"
        return AIAgent(
            api_key="test-key", base_url=route["base_url"], provider="custom:neutral",
            model="neutral-model", api_mode="chat_completions", quiet_mode=True,
            skip_context_files=True, skip_memory=True, fallback_model=fallback,
        )

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata._query_local_context_length", return_value=None),
    ):
        yield create, write_config


def history(agent, turns=30) -> list[dict]:
    route = reasoning_route_fingerprint(agent.provider, agent.model, agent.base_url, agent.api_mode)
    rows = [{"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Complete the synthetic task"}]
    for i in range(turns):
        rows.extend([
            {"role": "assistant", "content": f"step {i}",
             "reasoning_content": "SYNTHETIC legacy echo accounting trace " * 200,
             "_reasoning_route": route,
             "tool_calls": [{"id": f"call_{i}", "type": "function",
                             "function": {"name": "test_tool", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"result {i}"},
        ])
    return rows


def project(agent, rows):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_request_assembly import build_api_messages

    _reset_per_turn_agent_state(agent)
    projected, _ = build_api_messages(
        agent, rows, current_turn_user_idx=1, ext_prefetch_cache="", plugin_user_context="",
        moa_config=None, active_system_prompt="",
    )
    return projected


def send_request(agent, rows):
    import logging

    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_request_assembly import assemble_api_request

    _reset_per_turn_agent_state(agent)
    return assemble_api_request(
        agent, messages=rows, current_turn_user_idx=1, _ext_prefetch_cache="",
        _plugin_user_context="", moa_config=None, active_system_prompt="",
        original_user_message=rows[1]["content"], pending_moa_prepared_request=None,
        request_logger=logging.getLogger(__name__),
    )


def test_explicit_echo_prices_the_wire_in_preflight_and_tail(runtime):
    create, _ = runtime
    agent = create()
    rows = history(agent)
    before = deepcopy(rows)
    request = send_request(agent, rows)
    wire = request.api_messages
    assert not any("_reasoning_route" in m for m in wire)
    assert agent._reasoning_echo_flag is True
    assert agent._reasoning_replay_field_for_api() is None
    assert all(m["reasoning_content"] == rows[i]["reasoning_content"]
               for i, m in enumerate(wire) if m["role"] == "assistant")
    assert rows == before
    expected = estimate_request_tokens_rough(wire, charge_stale_thinking=True)
    # Canonical provenance metadata can conservatively add a little overhead;
    # it must never hide the historical reasoning bulk or multiply its charge.
    assert expected <= _preflight_request_tokens(agent, rows, "") <= expected * 1.05
    assert request.request_pressure_tokens == expected
    assert agent.context_compressor._stale_thinking_on_wire() is True
    cut = agent.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000)
    retained = sum(_estimate_msg_budget_tokens(m, charge_stale_thinking=True) for m in rows[cut:])
    assert cut > 2
    # Backward alignment can admit one atomic assistant/tool pair above the ceiling.
    pair_tokens = sum(_estimate_msg_budget_tokens(m, charge_stale_thinking=True) for m in rows[-2:])
    assert retained <= agent.context_compressor._tail_soft_ceiling(5000) + pair_tokens
    assert rows[cut]["role"] == "assistant"
    assert rows[cut + 1]["tool_call_id"] == rows[cut]["tool_calls"][0]["id"]


@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("model,preserves_history", [
    ("claude-sonnet-4-5", False),
    ("claude-sonnet-4-6", True),
    ("claude-opus-4-5", True),
])
def test_native_anthropic_echo_prices_only_converted_historical_thinking(
    runtime, echo, model, preserves_history,
):
    from types import SimpleNamespace

    from agent.anthropic_message_convert import convert_messages_to_anthropic
    from agent.model_metadata import estimate_tokens_rough
    from agent.transports.anthropic import AnthropicTransport

    create, _ = runtime
    agent = create(echo)
    agent.switch_model(model, "anthropic", api_key="test-key",
                       base_url="https://api.anthropic.com", api_mode="anthropic_messages")
    assert agent._reasoning_echo_flag is echo

    def assistant(size, signature):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="x" * size, signature=signature),
                SimpleNamespace(type="text", text="answer"),
            ],
            stop_reason="end_turn", stop_details=None,
        )
        normalized = AnthropicTransport().normalize_response(response)
        # Exercise the real producer; never repair/restamp its provenance.
        message = agent._build_assistant_message(normalized, normalized.finish_reason)
        assert message["_reasoning_route"] == reasoning_route_fingerprint(
            agent.provider, agent.model, agent.base_url, agent.api_mode
        )
        return message

    def measure(size):
        rows = [
            {"role": "user", "content": "first question"}, assistant(size, "sig_first"),
            {"role": "user", "content": "second question"}, assistant(1, "sig_second"),
            {"role": "user", "content": "continue"},
        ]
        before = deepcopy(rows)
        preflight = _preflight_request_tokens(agent, rows, "")
        request = send_request(agent, rows)
        _, wire = convert_messages_to_anthropic(
            deepcopy(request.api_messages), base_url=agent.base_url, model=agent.model,
        )
        thinking = [block for message in wire if message["role"] == "assistant"
                    for block in message["content"] if block["type"] == "thinking"]
        assert [block["signature"] for block in thinking] == (
            ["sig_first", "sig_second"] if preserves_history else ["sig_second"]
        )
        cc = agent.context_compressor
        tail_tokens = cc._walk_tail_budget(rows, 0, 10**9, 0, cut_at_break=False)[1]
        tail_cut = cc._find_tail_cut_by_tokens(rows, 0, token_budget=1000)
        assert rows == before
        return {
            "wire": sum(estimate_tokens_rough(block["thinking"]) for block in thinking),
            "preflight": preflight, "tail": tail_tokens,
            "assembled": request.approx_tokens, "pressure": request.request_pressure_tokens,
        }, tail_cut

    small, small_cut = measure(1)
    large, large_cut = measure(8000)
    expected = (estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x")) if preserves_history else 0
    assert {key: large[key] - small[key] for key in small} == dict.fromkeys(small, expected)
    if not preserves_history:
        assert small_cut == large_cut


def test_strict_fallback_drops_primary_echo_accounting(runtime):
    create, _ = runtime
    agent = create(fallback=[{
        "provider": "custom", "model": "gpt-4o-mini", "api_key": "test-key",
        "base_url": "https://fallback.example/v1", "api_mode": "chat_completions",
        "reasoning_replay_field": "none",
    }])
    primary = history(agent)
    assert agent._try_activate_fallback() is True
    assert agent._reasoning_echo_flag is False
    assert all("reasoning_content" not in m for m in project(agent, primary))
    assert agent.context_compressor._stale_thinking_on_wire() is False
    fallback_rows = history(agent)
    assert _preflight_request_tokens(agent, fallback_rows, "") == estimate_request_tokens_rough(
        fallback_rows, charge_stale_thinking=False
    )
    assert agent._restore_primary_runtime() is True
    assert agent._reasoning_echo_flag is True
    assert agent.context_compressor._stale_thinking_on_wire() is True
    assert _preflight_request_tokens(agent, primary, "") == estimate_request_tokens_rough(
        project(agent, primary), charge_stale_thinking=True
    )


def test_failed_model_switch_restores_echo_accounting(runtime):
    create, write_config = runtime
    agent = create()
    write_config(False)
    agent.switch_model("gpt-4o-mini", "custom:neutral", api_key="test-key",
                       base_url="https://neutral.example/v1", api_mode="chat_completions")
    assert agent._reasoning_echo_flag is False
    assert agent.context_compressor._stale_thinking_on_wire() is False
    rows = history(agent)
    assert all("reasoning_content" not in m for m in project(agent, rows))
    before = _preflight_request_tokens(agent, rows, "")
    write_config(True)
    with patch.object(agent, "_create_openai_client", side_effect=RuntimeError("test rebuild failed")):
        with pytest.raises(RuntimeError, match="test rebuild failed"):
            agent.switch_model("gpt-4o", "custom:neutral", api_key="test-key",
                               base_url="https://neutral.example/v1", api_mode="chat_completions")
    assert agent.model == "gpt-4o-mini"
    assert agent._reasoning_echo_flag is False
    assert agent.context_compressor._stale_thinking_on_wire() is False
    assert _preflight_request_tokens(agent, rows, "") == before
    assert all("reasoning_content" not in m for m in project(agent, rows))


def test_echo_on_and_off_select_different_real_tail_budgets(runtime):
    create, write_config = runtime
    on = create(True)
    off = create(False)
    rows = history(on)
    on_pressure = _preflight_request_tokens(on, rows, "")
    off_pressure = _preflight_request_tokens(off, rows, "")
    assert on_pressure > 10 * off_pressure
    assert on.context_compressor.should_compress(on_pressure) is True
    assert off.context_compressor.should_compress(off_pressure) is False
    cut_on = on.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000)
    cut_off = off.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000)
    assert cut_on > cut_off
    assert rows[cut_on]["role"] == rows[cut_off]["role"] == "assistant"
    assert all("reasoning_content" not in m for m in send_request(off, rows).api_messages)
    # The compressor consumes resolved session state, never a fresh config read.
    write_config(False)
    assert on.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000) == cut_on
    assert on._reasoning_echo_flag is True


def test_echo_enabled_fallback_restores_disabled_primary(runtime):
    create, _ = runtime
    agent = create(False, fallback=[{
        "provider": "custom", "model": "gpt-4o-mini", "api_key": "test-key",
        "base_url": "https://fallback.example/v1", "api_mode": "chat_completions",
        "reasoning_echo": True, "reasoning_replay_field": "none",
    }])
    primary_rows = history(agent)
    assert agent._try_activate_fallback() is True
    fallback_rows = history(agent)
    assert agent.context_compressor._stale_thinking_on_wire() is True
    wire = send_request(agent, fallback_rows).api_messages
    assert any(m.get("reasoning_content", "").startswith("SYNTHETIC") for m in wire)
    # Real old-producer history must not be restamped after a route transition.
    foreign = send_request(agent, primary_rows).api_messages
    assert all("SYNTHETIC" not in m.get("reasoning_content", "") for m in foreign)
    assert agent._restore_primary_runtime() is True
    assert agent._reasoning_echo_flag is False
    assert agent.context_compressor._stale_thinking_on_wire() is False
    assert all("reasoning_content" not in m for m in send_request(agent, primary_rows).api_messages)


def test_successful_model_switch_reloads_echo_and_prices_new_route(runtime):
    create, write_config = runtime
    agent = create(False)
    old_rows = history(agent)
    write_config(True)
    agent.switch_model("gpt-4o-mini", "custom:neutral", api_key="test-key",
                       base_url="https://neutral.example/v1", api_mode="chat_completions")
    assert agent._reasoning_echo_flag is True
    assert agent.context_compressor._stale_thinking_on_wire() is True
    assert all("SYNTHETIC" not in m.get("reasoning_content", "")
               for m in send_request(agent, old_rows).api_messages)
    rows = history(agent)
    assert _preflight_request_tokens(agent, rows, "") == estimate_request_tokens_rough(
        project(agent, rows), charge_stale_thinking=True
    )
    assert agent.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000) > 2


@pytest.mark.parametrize("initial_echo", [True, False])
def test_same_model_reselection_refreshes_echo_wire_policy(runtime, initial_echo):
    create, write_config = runtime
    agent = create(initial_echo)
    rows = history(agent)
    send_request(agent, rows)  # materialize the provider pad cache before switching
    write_config(not initial_echo)
    agent.switch_model(agent.model, agent.provider, api_key="test-key",
                       base_url=agent.base_url, api_mode=agent.api_mode)
    expected_echo = not initial_echo
    assert agent._reasoning_echo_flag is expected_echo
    assert agent.context_compressor._stale_thinking_on_wire() is expected_echo
    wire = send_request(agent, rows).api_messages
    assert all(bool(m.get("reasoning_content")) is expected_echo
               for m in wire if m["role"] == "assistant")
    assert _preflight_request_tokens(agent, rows, "") == estimate_request_tokens_rough(
        rows, charge_stale_thinking=expected_echo
    )


def test_duplicate_and_unequal_reasoning_aliases_are_charged_once(runtime):
    create, _ = runtime
    agent = create()
    rows = history(agent)
    both = deepcopy(rows)
    for m in both:
        if m["role"] == "assistant":
            m["reasoning"] = m["reasoning_content"]
    unequal = deepcopy(both)
    for m in unequal:
        if m["role"] == "assistant":
            m["reasoning"] = "short duplicate alias"
    pressure = _preflight_request_tokens(agent, rows, "")
    cut = agent.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000)
    for variant in (both, unequal):
        assert _preflight_request_tokens(agent, variant, "") == pressure
        assert agent.context_compressor._find_tail_cut_by_tokens(variant, 2, token_budget=5000) == cut
        assert sum(map(agent.context_compressor._tail_row_pricer(variant), range(len(variant)))) == sum(
            map(agent.context_compressor._tail_row_pricer(rows), range(len(rows)))
        )
        wire = send_request(agent, variant).api_messages
        assert all("reasoning" not in m for m in wire)
        assert [m.get("reasoning_content") for m in wire] == [m.get("reasoning_content") for m in rows]


def test_codex_conversion_excludes_text_even_with_resolved_echo(runtime):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.turn_context import _agent_stale_thinking_on_wire

    create, _ = runtime
    agent = create()
    agent.switch_model("gpt-4o-mini", "custom:neutral", api_key="test-key",
                       base_url="https://neutral.example/v1", api_mode="codex_responses")
    assert agent._reasoning_echo_flag is True
    assert _agent_stale_thinking_on_wire(agent) is False
    assert agent.context_compressor._stale_thinking_on_wire() is False
    rows = history(agent)
    for m in rows:
        if m["role"] == "assistant":
            m["reasoning"] = m["reasoning_content"]
    rows[2]["codex_reasoning_items"] = [{
        "type": "reasoning", "id": "synthetic_encrypted_item",
        "encrypted_content": "synthetic opaque continuity " * 100, "summary": [],
    }]
    without_text = [{k: v for k, v in m.items() if k not in {"reasoning", "reasoning_content"}}
                    for m in rows]
    converted = _chat_messages_to_responses_input(send_request(agent, rows).api_messages)
    assert converted == _chat_messages_to_responses_input(send_request(agent, without_text).api_messages)
    assert any(item.get("encrypted_content") for item in converted)
    assert "SYNTHETIC legacy echo" not in json.dumps(converted)
    # Generic accounting intentionally reserves the newest assistant's thinking;
    # only historical text is excluded when native checkpoint pricing is unavailable.
    without_stale = deepcopy(rows)
    for m in without_stale[2:-2]:
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
    assert _preflight_request_tokens(agent, rows, "") == _preflight_request_tokens(agent, without_stale, "")
    cc = agent.context_compressor
    assert cc._find_tail_cut_by_tokens(rows, 2, token_budget=5000) == cc._find_tail_cut_by_tokens(
        without_stale, 2, token_budget=5000
    )
    assert sum(map(cc._tail_row_pricer(rows), range(len(rows)))) == sum(
        map(cc._tail_row_pricer(without_stale), range(len(rows)))
    )
    without_opaque = deepcopy(rows)
    without_opaque[2].pop("codex_reasoning_items")
    assert _preflight_request_tokens(agent, rows, "") > _preflight_request_tokens(agent, without_opaque, "")
    larger_ciphertext = deepcopy(rows)
    larger_ciphertext[2]["codex_reasoning_items"][0]["encrypted_content"] *= 100
    assert _preflight_request_tokens(agent, larger_ciphertext, "") == _preflight_request_tokens(agent, rows, "")
    assert sum(map(cc._tail_row_pricer(larger_ciphertext), range(len(rows)))) == sum(
        map(cc._tail_row_pricer(rows), range(len(rows)))
    )


@pytest.mark.parametrize("provenance", [None, "foreign-synthetic-producer"])
def test_echo_keeps_fail_closed_provenance_and_non_assistant_controls(runtime, provenance):
    create, _ = runtime
    agent = create()
    rows = history(agent, turns=2)
    if provenance is None:
        rows[2].pop("_reasoning_route")
    else:
        rows[2]["_reasoning_route"] = provenance
    rows[1]["reasoning_content"] = "SYNTHETIC user injection"
    rows[3]["reasoning_content"] = "SYNTHETIC tool injection"
    wire = send_request(agent, rows).api_messages
    assert all("_reasoning_route" not in m for m in wire)
    assert wire[2].get("reasoning_content", "").strip() == ""
    assert "reasoning_content" not in wire[1]
    assert "reasoning_content" not in wire[3]
    assert wire[4]["reasoning_content"] == rows[4]["reasoning_content"]


def test_family_required_pad_survives_disabled_explicit_echo(runtime):
    create, _ = runtime
    agent = create(False)
    agent.switch_model("deepseek-reasoner", "deepseek", api_key="test-key",
                       base_url="https://api.deepseek.com", api_mode="chat_completions")
    assert agent._reasoning_echo_flag is False
    assert agent.context_compressor._stale_thinking_on_wire() is True
    rows = history(agent, turns=2)
    wire = send_request(agent, rows).api_messages
    assert wire[2]["reasoning_content"] == rows[2]["reasoning_content"]
    rows[2].pop("_reasoning_route")
    assert send_request(agent, rows).api_messages[2]["reasoning_content"].strip() == ""


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_explicit_soft_replay_still_prices_history_without_legacy_echo(runtime, field):
    create, _ = runtime
    agent = create(False, replay_field=field)
    assert agent._reasoning_echo_flag is False
    assert agent._reasoning_replay_field_for_api() == field
    assert agent.context_compressor._stale_thinking_on_wire() is True
    rows = history(agent)
    wire = send_request(agent, rows).api_messages
    assert all(m[field] == rows[i]["reasoning_content"]
               for i, m in enumerate(wire) if m["role"] == "assistant")
    expected = estimate_request_tokens_rough(wire, charge_stale_thinking=True)
    assert expected <= _preflight_request_tokens(agent, rows, "") <= expected * 1.05
    assert agent.context_compressor._find_tail_cut_by_tokens(rows, 2, token_budget=5000) > 2
