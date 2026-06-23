from types import SimpleNamespace

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_switch_model_normalizes_provider_colon_syntax_even_when_current_provider_is_not_aggregator(
    monkeypatch,
):
    seen = {}

    def fake_detect_provider_for_model(model, current_provider):
        seen["model"] = model
        seen["current_provider"] = current_provider
        return None

    def fake_resolve_provider_full(provider, user_providers=None, custom_providers=None):
        if provider == "alibaba":
            return SimpleNamespace(id="alibaba", name="Alibaba", base_url="", source="built-in")
        return None

    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_provider_full",
        fake_resolve_provider_full,
    )
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        fake_detect_provider_for_model,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: _MOCK_VALIDATION,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_info",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *a, **k: None,
    )

    result = switch_model(
        raw_input="Alibaba:qwen3.6-plus",
        current_provider="alibaba",
        current_model="qwen3.6-plus",
        current_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        current_api_key="test-key",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert seen["model"] == "alibaba/qwen3.6-plus"
    assert seen["current_provider"] == "alibaba"


def test_switch_model_does_not_rewrite_variant_suffix_when_model_is_already_provider_slash_model(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: _MOCK_VALIDATION,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_info",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *a, **k: None,
    )

    result = switch_model(
        raw_input="qwen/qwen3-coder:free",
        current_provider="openrouter",
        current_model="openai/gpt-5",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="test-key",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.new_model == "qwen/qwen3-coder:free"