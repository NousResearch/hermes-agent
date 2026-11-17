"""Regression tests for auxiliary fallback model isolation (#155982).

Verifies that auxiliary fallback chains do not leak the primary provider's model
identifier to fallback providers. When a fallback candidate specifies an explicit model,
that model must be honoured. When no model is specified, the fallback provider resolves
to its catalog default model rather than inheriting the main agent's model name.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.auxiliary_client import (
    _FallbackDestination,
    _fallback_chain_entry,
    _fallback_destination_from_entry,
    _scoped_main_model_for_aux,
    _to_async_client,
    _try_main_agent_model_fallback,
    call_llm,
)


@pytest.fixture(autouse=True)
def _setup_test_environment(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test-opencode-key")
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def test_auxiliary_fallback_explicit_model_precedence():
    """Fallback entry's configured model takes precedence over primary's model."""
    mock_config = {
        "model": {
            "provider": "radiance-local",
            "default": "radiance-local/qwen35",
        },
        "auxiliary": {
            "vision": {
                "provider": "radiance-local",
                "model": "qwen35",
                "fallback_chain": [
                    {
                        "provider": "opencode-go",
                        "model": "deepseek-v4.1-flash",
                    }
                ],
            }
        },
        "providers": {
            "radiance-local": {
                "base_url": "http://127.0.0.1:8100/v1",
                "api_mode": "chat_completions",
                "model": "qwen35",
            }
        },
    }

    recorded_models = []

    def mock_create(*args, **kwargs):
        recorded_models.append(kwargs.get("model"))
        if len(recorded_models) == 1:
            raise TimeoutError("Primary timed out")
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "OK from fallback"
        resp.choices = [choice]
        return resp

    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config), \
         patch("hermes_cli.config.load_config", return_value=mock_config), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("openai.resources.chat.completions.Completions.create", side_effect=mock_create):
        res = call_llm(
            task="vision",
            messages=[{"role": "user", "content": "analyze diagram"}],
        )
        assert res.choices[0].message.content == "OK from fallback"

    assert len(recorded_models) == 2
    assert recorded_models[0] == "qwen35"
    assert recorded_models[1] == "deepseek-v4.1-flash"


def test_auxiliary_fallback_bare_provider_uses_catalog_default():
    """A bare fallback entry without explicit model resolves provider default, not primary model."""
    mock_config = {
        "model": {
            "provider": "radiance-local",
            "default": "radiance-local/qwen35",
        },
        "auxiliary": {
            "vision": {
                "provider": "radiance-local",
                "model": "qwen35",
                "fallback_chain": [
                    {
                        "provider": "opencode-go",
                    }
                ],
            }
        },
        "providers": {
            "radiance-local": {
                "base_url": "http://127.0.0.1:8100/v1",
                "api_mode": "chat_completions",
                "model": "qwen35",
            }
        },
    }

    recorded_models = []

    def mock_create(*args, **kwargs):
        recorded_models.append(kwargs.get("model"))
        if len(recorded_models) == 1:
            raise TimeoutError("Primary timed out")
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "OK from default"
        resp.choices = [choice]
        return resp

    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config), \
         patch("hermes_cli.config.load_config", return_value=mock_config), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("openai.resources.chat.completions.Completions.create", side_effect=mock_create):
        res = call_llm(
            task="vision",
            messages=[{"role": "user", "content": "analyze diagram"}],
        )
        assert res.choices[0].message.content == "OK from default"

    assert len(recorded_models) == 2
    assert recorded_models[0] == "qwen35"
    # Invariant: must NOT leak primary model, and must match provider catalog default
    expected_default = ac._get_aux_model_for_provider("opencode-go")
    assert recorded_models[1] == expected_default
    assert recorded_models[1] != "radiance-local/qwen35"


def test_try_main_agent_model_fallback_strips_vendor_prefix_same_provider():
    """_try_main_agent_model_fallback strips provider vendor prefix from qualified model names."""
    with patch("agent.auxiliary_client._read_main_provider", return_value="radiance-local"), \
         patch("agent.auxiliary_client._read_main_model", return_value="radiance-local/qwen35"), \
         patch("agent.auxiliary_client._custom_health_base_url", return_value="http://127.0.0.1:8100/v1"), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
        mock_resolve.return_value = (MagicMock(), "qwen35")
        client, model, label = _try_main_agent_model_fallback("different-provider", task="vision")
        assert client is not None
        assert model == "qwen35"
        assert label == "main-agent(radiance-local)"
        mock_resolve.assert_called_once_with(provider="radiance-local", model="qwen35")


def test_try_main_agent_model_fallback_strips_vendor_prefix_custom_provider():
    """_try_main_agent_model_fallback switches to named custom provider when main_provider is custom."""
    mock_entry = {"base_url": "http://127.0.0.1:8100/v1", "model": "qwen35"}
    with patch("agent.auxiliary_client._read_main_provider", return_value="custom"), \
         patch("agent.auxiliary_client._read_main_model", return_value="radiance-local/qwen35"), \
         patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=mock_entry), \
         patch("agent.auxiliary_client._custom_health_base_url", return_value="http://127.0.0.1:8100/v1"), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
        mock_resolve.return_value = (MagicMock(), "qwen35")
        client, model, label = _try_main_agent_model_fallback("different-provider", task="vision")
        assert client is not None
        assert model == "qwen35"
        assert label == "main-agent(radiance-local)"
        mock_resolve.assert_called_once_with(provider="radiance-local", model="qwen35")



def test_main_agent_fallback_preserves_unknown_custom_model_prefix():
    """An unregistered model namespace is part of a custom endpoint's wire model ID."""
    model_id = "org/private-vision-v1"
    with patch("agent.auxiliary_client._read_main_provider", return_value="custom"), \
         patch("agent.auxiliary_client._read_main_model", return_value=model_id), \
         patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None), \
         patch("agent.auxiliary_client._custom_health_base_url", return_value="http://127.0.0.1:8100/v1"), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(MagicMock(), model_id)) as resolve:
        client, resolved_model, label = _try_main_agent_model_fallback(
            "other-provider", task="compression",
        )

    assert client is not None
    assert resolved_model == model_id
    assert label == "main-agent(custom)"
    resolve.assert_called_once_with(provider="custom", model=model_id)


def test_scoped_main_model_for_aux():
    """_scoped_main_model_for_aux only returns main model when provider matches main provider."""
    with patch("agent.auxiliary_client._read_main_provider", return_value="radiance-local"), \
         patch("agent.auxiliary_client._read_main_model", return_value="radiance-local/qwen35"):
        # Matches main provider
        assert _scoped_main_model_for_aux("radiance-local") == "radiance-local/qwen35"
        # Different provider: must not inherit
        assert _scoped_main_model_for_aux("opencode-go") is None
        assert _scoped_main_model_for_aux("openai") is None
        assert _scoped_main_model_for_aux("auto") is None
        assert _scoped_main_model_for_aux("") is None


def test_fallback_chain_entry_resolves_fallback_providers_syntax():
    """_fallback_chain_entry resolves top-level fallback_providers entries."""
    mock_config = {
        "fallback_providers": [
            {"provider": "opencode-go", "model": "deepseek-v4.1-flash"},
            {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        ]
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
        entry_0 = _fallback_chain_entry(None, "fallback_providers[0](opencode-go)")
        assert entry_0 == {"provider": "opencode-go", "model": "deepseek-v4.1-flash"}

        entry_1 = _fallback_chain_entry("vision", "fallback_providers[1](openrouter)")
        assert entry_1 == {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"}

        entry_out_of_bounds = _fallback_chain_entry(None, "fallback_providers[99](none)")
        assert entry_out_of_bounds is None


def test_fallback_destination_from_entry_prefers_resolved_fb_model():
    """_fallback_destination_from_entry prefers normalized fb_model over raw entry model."""
    entry = {"provider": "opencode-go", "model": "raw-config-model", "base_url": "https://api.opencode.ai/v1"}
    dest = _fallback_destination_from_entry(entry, MagicMock(), fb_model="normalized-candidate-model")
    assert dest.model == "normalized-candidate-model"
    assert dest.provider == "opencode-go"


def test_fallback_destination_from_entry_falls_back_to_entry_model():
    """_fallback_destination_from_entry uses entry model when fb_model is None."""
    entry = {"provider": "opencode-go", "model": "configured-entry-model", "base_url": "https://api.opencode.ai/v1"}
    dest = _fallback_destination_from_entry(entry, MagicMock(), fb_model=None)
    assert dest.model == "configured-entry-model"
    assert dest.provider == "opencode-go"


def test_async_openai_conversion_keeps_fallback_destination():
    """Real OpenAI sync client exercises the wrapper path and preserves fallback destination."""
    from openai import OpenAI

    sync_client = OpenAI(api_key="test-key", base_url="https://api.opencode.ai/v1")
    dest = _FallbackDestination("opencode-go", "https://api.opencode.ai/v1", "chat_completions", "deepseek-v4.1-flash")
    sync_client._hermes_fallback_destination = dest

    async_client, _ = _to_async_client(sync_client, "deepseek-v4.1-flash")

    assert async_client is not sync_client
    assert async_client._hermes_fallback_destination == dest


def test_async_codex_conversion_keeps_fallback_destination():
    """CodexAuxiliaryClient preserves _hermes_fallback_destination across async wrapper."""
    from agent.auxiliary_client import CodexAuxiliaryClient

    dest = _FallbackDestination("opencode-go", "https://api.opencode.ai/v1", "chat_completions", "deepseek-v4.1-flash")
    codex_sync = MagicMock(spec=CodexAuxiliaryClient)
    codex_sync.chat = MagicMock()
    codex_sync.api_key = "test-key"
    codex_sync.base_url = "https://codex.endpoint"
    codex_sync._hermes_fallback_destination = dest
    codex_async, _ = _to_async_client(codex_sync, "codex-model")
    assert hasattr(codex_async, "_hermes_fallback_destination")
    assert codex_async._hermes_fallback_destination == dest


def test_explicit_auxiliary_route_does_not_walk_main_fallback_providers():
    """An explicit auxiliary task route does not inherit top-level fallback_providers (R1)."""
    mock_config = {
        "model": {
            "provider": "custom",
            "default": "radiance-local/qwen35",
        },
        "auxiliary": {
            "vision": {
                "provider": "radiance-local",
                "model": "qwen35",
                # No task fallback_chain configured
            }
        },
        "fallback_providers": [
            {"provider": "opencode-go", "model": "deepseek-v4.1-flash"}
        ],
        "providers": {
            "radiance-local": {
                "base_url": "http://127.0.0.1:8100/v1",
                "model": "qwen35",
            }
        },
    }

    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Endpoint timeout")

    with patch("hermes_cli.config.load_config_readonly", return_value=mock_config), \
         patch("hermes_cli.config.load_config", return_value=mock_config), \
         patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
         patch("openai.resources.chat.completions.Completions.create", side_effect=mock_create):
        with pytest.raises(Exception):
            call_llm(
                task="vision",
                messages=[{"role": "user", "content": "analyze diagram"}],
            )

    # For an explicitly pinned route without task fallback_chain:
    # 1 try for primary, 1 try for main-agent fallback (which also fails)
    # Crucially, it must NOT fall through to fallback_providers[0] (opencode-go)!
    assert call_count <= 2
