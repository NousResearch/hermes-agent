"""Native Gemini output-token reservation at agent init (#57275 claim 4).

When ``model.max_tokens`` is unset, the native generateContent adapter still
sends ``maxOutputTokens=65,535`` (GEMINI_DEFAULT_MAX_OUTPUT_TOKENS) — Gemini
treats an omitted cap as a low internal default, not "unlimited", so the
adapter always sends an explicit cap.  The compressor's trigger is
``pct × (window − max_tokens)``; constructing it with ``max_tokens=None``
reserved 0 while the wire reserved 65,535: on a 128K window the trigger
landed at ~96K against a real safe input budget of ~65K, and the provider
400'd before compaction ever fired.

These tests assert the compressor's reservation mirrors the adapter default
on the native Gemini path, and ONLY there.
"""




from unittest.mock import patch



import agent.context_compressor as cc_mod
from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS


CFG = {"agent": {}}


def _build_agent(model, base_url, provider="", max_tokens=None, window=131072):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=CFG),
        patch(
            "agent.model_metadata.get_model_context_length", return_value=window,
        ),
        patch.object(cc_mod, "get_model_context_length", return_value=window),
    ):
        from run_agent import AIAgent

        return AIAgent(
            model=model,
            api_key="test-key-1234567890",
            base_url=base_url,
            provider=provider,
            max_tokens=max_tokens,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_native_gemini_unset_max_tokens_reserves_adapter_default():
    agent = _build_agent(
        "gemma-3-27b-it",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    cc = agent.context_compressor
    assert cc.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    # Trigger must sit at/below the real safe input budget the wire leaves.
    assert cc.threshold_tokens <= cc.context_length - GEMINI_DEFAULT_MAX_OUTPUT_TOKENS


def test_gemini_provider_name_also_reserves_default():
    agent = _build_agent(
        "gemini-3.7-flash", "https://example-proxy.invalid/v1", provider="google",
    )
    assert agent.context_compressor.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS


def test_explicit_max_tokens_wins_over_adapter_default():
    agent = _build_agent(
        "gemma-3-27b-it",
        "https://generativelanguage.googleapis.com/v1beta",
        max_tokens=8192,
    )
    assert agent.context_compressor.max_tokens == 8192


def test_non_gemini_paths_keep_no_reservation():
    agent = _build_agent(
        "openai/gpt-4.1", "https://openrouter.ai/api/v1",
    )
    assert agent.context_compressor.max_tokens is None


def test_gemini_openai_compat_endpoint_not_treated_as_native():
    # The /openai compatibility endpoint does not use the native adapter.
    agent = _build_agent(
        "gemma-3-27b-it",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    assert agent.context_compressor.max_tokens is None


def test_reservation_follows_the_route_across_model_switch():
    """Regression: ``update_model()`` keeps a reservation it is not given, so a live /model
    switch carried the construction route's cap onto the new route. Into native Gemini the
    trigger stayed above the safe budget the wire leaves (the #57275 400, via /model); out of it
    the 64K reservation halved the next model's trigger."""
    agent = _build_agent("openai/gpt-4.1", "https://openrouter.ai/api/v1", provider="openrouter")
    window = 131072
    with (
        patch("hermes_cli.config.load_config", return_value=CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=CFG),
        patch("agent.model_metadata.get_model_context_length", return_value=window),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent.switch_model(
            "gemma-3-27b-it", "gemini", api_key="test-key-1234567890",
            base_url="https://generativelanguage.googleapis.com/v1beta", api_mode="chat_completions",
        )
        cc = agent.context_compressor
        assert cc.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        assert cc.threshold_tokens <= window - GEMINI_DEFAULT_MAX_OUTPUT_TOKENS

        agent.switch_model(
            "openai/gpt-4.1", "openrouter", api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
        )
        assert agent.context_compressor.max_tokens is None


def test_provider_profile_cap_is_reserved_on_the_chat_completions_wire():
    """The chat-completions wire sends the profile's cap when max_tokens is unset, so the
    compressor must reserve that same cap (the generic half of #43547)."""
    from providers import get_provider_profile

    model = "meta/llama-3.1-8b-instruct"
    agent = _build_agent(model, "https://integrate.api.nvidia.com/v1", provider="nvidia")
    wire_cap = get_provider_profile("nvidia").get_max_tokens(model)
    assert wire_cap
    assert agent.context_compressor.max_tokens == wire_cap
