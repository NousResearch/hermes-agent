"""Provider fallback must not select a backend that cannot read this request."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client():
    client = MagicMock()
    client.base_url = "https://fallback.example/v1"
    client.api_key = "fallback-key"
    return client


def test_fallback_skips_context_too_small_for_current_request():
    agent = _make_agent(
        [
            {"provider": "small", "model": "small-model"},
            {"provider": "large", "model": "large-model"},
        ]
    )
    # This is the preflight estimate for the already assembled wire request,
    # not a prior-session aggregate. It must fit before an unavailable primary
    # can redirect the retry to a smaller backend.
    agent._active_request_tokens = 120_000

    with (
        patch(
            "agent.model_metadata.get_model_context_length",
            # The selected fallback resolves its context once more when its
            # compressor is rebound after activation.
            side_effect=[65_536, 262_144, 262_144],
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "resolved-model"),
        ) as resolve,
    ):
        assert agent._try_activate_fallback() is True

    assert agent.model == "large-model"
    assert agent.provider == "large"
    assert resolve.call_count == 1


def test_request_assembly_refreshes_fallback_request_measurement(monkeypatch):
    from types import SimpleNamespace

    import agent.turn_request_assembly as assembly

    agent = SimpleNamespace(
        prefill_messages=[], api_mode="chat_completions", tools=[],
        _use_prompt_caching=False, provider="openai", model="model",
        base_url="https://example.test", _active_request_tokens=999_999,
        _usage_anchor=None, context_compressor=object(),
        _sanitize_api_messages=lambda messages: messages,
        _drop_thinking_only_and_merge_users=lambda messages, **_: messages,
        _emit_warning=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(assembly, "build_api_messages", lambda *args, **kwargs: ([{"role": "user", "content": "request"}], ""))
    monkeypatch.setattr(assembly, "anchored_context_tokens", lambda *_: None)
    monkeypatch.setattr("agent.turn_context._agent_stale_thinking_on_wire", lambda _: False)
    monkeypatch.setattr("agent.context_compressor.evict_stale_outbound_tool_images", lambda _: None)
    monkeypatch.setattr("agent.anthropic_thinking_replay.apply_rejected_thinking_suppression", lambda *_: None)
    monkeypatch.setattr("agent.conversation_loop._apply_context_engine_selection", lambda _a, api, *_args, **_kwargs: api)
    monkeypatch.setattr("agent.conversation_loop._canonicalize_api_tool_calls", lambda _: None)
    monkeypatch.setattr("agent.conversation_loop._midturn_request_pressure_tokens", lambda _a, _m, _s, approx: approx)
    monkeypatch.setattr("agent.conversation_loop._pressure_with_real_floor", lambda _c, value: value)
    monkeypatch.setattr("agent.model_metadata.estimate_messages_tokens_rough", lambda *_args, **_kwargs: 1234)

    assembled = assembly.assemble_api_request(
        agent, messages=[{"role": "user", "content": "request"}], current_turn_user_idx=0,
        _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
        active_system_prompt="", original_user_message="request",
        pending_moa_prepared_request=None, request_logger=None,
    )

    assert assembled.approx_tokens == 1234
    assert agent._active_request_tokens == 1234
