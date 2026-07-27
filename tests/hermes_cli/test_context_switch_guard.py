"""Tests for hermes_cli.context_switch_guard."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.context_switch_guard import merge_preflight_compression_warning
from hermes_cli.model_switch import ModelSwitchResult


def _result(*, model: str = "small-model") -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="k",
        base_url="https://example.com/v1",
        api_mode="chat_completions",
        provider_label="openrouter",
        model_info={"context_length": 32_000},
    )


def _compressor(monkeypatch, *, context_length: int = 200_000):
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *a, **k: context_length,
    )
    return ContextCompressor(
        model="big-model",
        threshold_percent=0.5,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=context_length,
    )


def test_no_warning_when_below_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    cc.last_prompt_tokens = 10_000
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert not result.warning_message


def test_warns_when_estimate_exceeds_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert result.warning_message
    assert "preflight compression" in result.warning_message
    assert "shrinks" in result.warning_message


def test_merge_appends_to_existing_warning(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result()
    result.warning_message = "expensive"
    merge_preflight_compression_warning(result, agent=agent)
    assert "expensive" in result.warning_message
    assert "preflight compression" in result.warning_message


def test_cross_route_switch_does_not_inherit_current_context_pin(monkeypatch):
    def _resolve_metadata(*_args, **kwargs):
        return kwargs["config_context_length"] or 32_000

    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        _resolve_metadata,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    cc = _compressor(monkeypatch, context_length=1_048_576)
    agent = SimpleNamespace(
        model="shared-model",
        provider="custom",
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="https://large.example/v1",
        api_key="",
    )
    result = _result(model="shared-model")

    merge_preflight_compression_warning(
        result,
        agent=agent,
        config_context_length=1_048_576,
    )

    assert "preflight compression" in result.warning_message


def test_custom_provider_context_avoids_false_shrink_warning(monkeypatch):
    """Classic CLI used to omit custom_providers from the shrink warning.

    Repro: switch onto a custom endpoint with models.<id>.context_length=1M
    while session ~147k. Probe fails → hardcoded catalog match on "qwen"
    (131072) → false "Context window shrinks (... → 131,072)" warning, even
    though /model confirmation and the status bar correctly show 1M.
    """
    custom_provs = [
        {
            "name": "qwen-token-plan",
            "base_url": "https://token-plan.example/compatible-mode/v1",
            "models": {
                "qwen3.8-max-preview": {"context_length": 1_048_576},
            },
        }
    ]
    # Force the probe-down path that hit the "qwen" → 131072 catalog match
    # when custom_providers was not threaded through.
    monkeypatch.setattr(
        "agent.model_metadata._resolve_endpoint_context_length",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "agent.model_metadata._query_ollama_api_show",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 147_053,
    )
    cc = _compressor(monkeypatch, context_length=1_000_000)
    agent = SimpleNamespace(
        model="MiniMax-M3",
        provider="minimax",
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="https://api.minimax.example/v1",
        api_key="",
        _custom_providers=custom_provs,
    )
    result = ModelSwitchResult(
        success=True,
        new_model="qwen3.8-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )

    # Explicit custom_providers — no false shrink warning (1M > 147k*2).
    merge_preflight_compression_warning(
        result,
        agent=agent,
        custom_providers=custom_provs,
    )
    assert not result.warning_message

    # Agent snapshot alone (classic CLI historically forgot to pass the kwarg).
    result2 = ModelSwitchResult(
        success=True,
        new_model="qwen3.8-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )
    merge_preflight_compression_warning(result2, agent=agent)
    assert not result2.warning_message

    # Without any custom_providers source, catalog match still warns (131k).
    agent_no_cp = SimpleNamespace(
        model="MiniMax-M3",
        provider="minimax",
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="https://api.minimax.example/v1",
        api_key="",
        _custom_providers=None,
    )
    result3 = ModelSwitchResult(
        success=True,
        new_model="qwen3.8-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )
    merge_preflight_compression_warning(result3, agent=agent_no_cp)
    assert result3.warning_message
    assert "preflight compression" in result3.warning_message
    assert "shrinks" in result3.warning_message
    # Must not honor the unused 1M custom override when no providers were passed.
    assert "1,048,576" not in result3.warning_message
