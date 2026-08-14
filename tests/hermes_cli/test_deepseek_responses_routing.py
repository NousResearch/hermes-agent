"""DeepSeek model capability and Responses routing contracts."""

from types import SimpleNamespace

import pytest

from hermes_cli import providers as provider_registry
from hermes_cli import runtime_provider as rp
from hermes_cli.providers import (
    DeepSeekModelCapabilities,
    deepseek_api_mode,
    deepseek_model_capabilities,
    deepseek_native_web_search_models,
    deepseek_supports_native_web_search,
    deepseek_supports_responses,
    determine_api_mode,
    normalize_deepseek_base_url,
)


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-flash",
        "DEEPSEEK-V4-FLASH",
        "deepseek/deepseek-v4-flash",
        "deepseek-v4-pro",
        "DEEPSEEK-V4-PRO",
        "deepseek/deepseek-v4-pro",
    ],
)
def test_documented_v4_models_support_responses_and_native_search(model):
    assert deepseek_supports_responses(model)
    assert deepseek_supports_native_web_search(model)
    assert deepseek_api_mode(model) == "codex_responses"
    assert determine_api_mode("deepseek", "https://api.deepseek.com/v1", model) == "codex_responses"


@pytest.mark.parametrize(
    "model",
    [
        "",
        "deepseek-v4-pro-20260801",
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-flash-20260423",
        "unknown",
    ],
)
def test_other_deepseek_models_stay_on_chat_completions(model):
    assert not deepseek_supports_responses(model)
    assert not deepseek_supports_native_web_search(model)
    assert deepseek_api_mode(model) == "chat_completions"


def test_v4_pro_has_ga_responses_and_native_search_capabilities():
    assert deepseek_model_capabilities("deepseek-v4-pro") == DeepSeekModelCapabilities(
        responses_api=True,
        native_web_search=True,
    )
    enabled = deepseek_native_web_search_models()
    assert "deepseek-v4-flash" in enabled
    assert "deepseek-v4-pro" in enabled


def test_native_search_capability_fails_closed_without_responses(monkeypatch):
    monkeypatch.setitem(
        provider_registry._DEEPSEEK_MODEL_CAPABILITIES,
        "deepseek-v4-future",
        DeepSeekModelCapabilities(
            responses_api=False,
            native_web_search=True,
        ),
    )

    assert not deepseek_supports_native_web_search("deepseek-v4-future")
    assert deepseek_api_mode("deepseek-v4-future") == "chat_completions"


@pytest.mark.parametrize(
    ("mode", "value", "expected"),
    [
        ("codex_responses", "https://api.deepseek.com/v1", "https://api.deepseek.com"),
        ("codex_responses", "https://api.deepseek.com/", "https://api.deepseek.com"),
        ("chat_completions", "https://api.deepseek.com", "https://api.deepseek.com/v1"),
        ("chat_completions", "https://api.deepseek.com/v1/", "https://api.deepseek.com/v1"),
    ],
)
def test_official_base_url_is_normalized_by_wire(mode, value, expected):
    assert normalize_deepseek_base_url("deepseek", mode, value) == expected


def test_custom_deepseek_proxy_is_not_rewritten():
    value = "https://deepseek-proxy.example/v1/"
    assert normalize_deepseek_base_url("deepseek", "codex_responses", value) == value


def test_non_deepseek_url_is_not_changed():
    value = "https://another-provider.example/v1/"
    assert normalize_deepseek_base_url("custom", "codex_responses", value) == value


def test_lookalike_official_host_is_not_rewritten():
    value = "https://api.deepseek.com.attacker.test/v1"
    assert normalize_deepseek_base_url("deepseek", "codex_responses", value) == value


@pytest.mark.parametrize(
    "value",
    [
        "https://api.deepseek.com/v1?tenant=one",
        "https://api.deepseek.com/v1#fragment",
        "https://api.deepseek.com/v1?redirect=https://example.test/",
        "https://api.deepseek.com/v1#section/",
        "https://user@api.deepseek.com/v1",
        "https://api.deepseek.com/custom/v1",
    ],
)
def test_noncanonical_official_urls_are_not_rewritten(value):
    assert normalize_deepseek_base_url("deepseek", "codex_responses", value) == value


@pytest.fixture
def deepseek_runtime(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {})
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "deepseek",
        "default": "deepseek-v4-pro",
        "api_mode": "chat_completions",
    })
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "deepseek",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "source": "env",
        },
    )


def test_runtime_target_flash_overrides_stale_persisted_chat_mode(deepseek_runtime):
    resolved = rp.resolve_runtime_provider(
        requested="deepseek", target_model="deepseek-v4-flash"
    )
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://api.deepseek.com"


def test_runtime_normalizes_retired_alias_before_wire_selection(deepseek_runtime):
    resolved = rp.resolve_runtime_provider(
        requested="deepseek", target_model="deepseek-chat"
    )
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://api.deepseek.com"


def test_runtime_target_pro_overrides_stale_persisted_chat_mode(deepseek_runtime, monkeypatch):
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "deepseek",
        "default": "deepseek-v4-flash",
        "api_mode": "chat_completions",
    })
    resolved = rp.resolve_runtime_provider(
        requested="deepseek", target_model="deepseek-v4-pro"
    )
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://api.deepseek.com"


@pytest.mark.parametrize(
    ("target_model", "stale_mode", "entry_url", "expected_mode", "expected_url"),
    [
        (
            "deepseek-v4-flash",
            "chat_completions",
            "https://api.deepseek.com/v1",
            "codex_responses",
            "https://api.deepseek.com",
        ),
        (
            "deepseek-v4-pro",
            "chat_completions",
            "https://api.deepseek.com/v1",
            "codex_responses",
            "https://api.deepseek.com",
        ),
    ],
)
def test_pool_runtime_recomputes_wire_and_official_root(
    target_model, stale_mode, entry_url, expected_mode, expected_url
):
    entry = SimpleNamespace(
        runtime_api_key="pool-key",
        access_token="pool-key",
        runtime_base_url=entry_url,
        base_url=entry_url,
        source="credential_pool",
    )
    resolved = rp._resolve_runtime_from_pool_entry(
        provider="deepseek",
        entry=entry,
        requested_provider="deepseek",
        model_cfg={
            "provider": "deepseek",
            "default": "deepseek-v4-pro",
            "api_mode": stale_mode,
        },
        target_model=target_model,
    )
    assert resolved["api_mode"] == expected_mode
    assert resolved["base_url"] == expected_url


def test_explicit_runtime_uses_model_aware_wire_and_official_root(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {})
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    resolved = rp.resolve_runtime_provider(
        requested="deepseek",
        explicit_api_key="sk-test",
        explicit_base_url="https://api.deepseek.com/v1",
        target_model="deepseek-v4-flash",
    )
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://api.deepseek.com"


def test_direct_agent_init_routes_flash_to_responses(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("run_agent.OpenAI", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        provider="deepseek",
        api_mode="chat_completions",
        model="deepseek-v4-flash",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.base_url == "https://api.deepseek.com"


def test_direct_agent_init_normalizes_alias_before_wire_selection(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("run_agent.OpenAI", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        provider="deepseek",
        api_mode="chat_completions",
        model="deepseek-chat",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent.model == "deepseek-v4-flash"
    assert agent.api_mode == "codex_responses"
    assert agent.base_url == "https://api.deepseek.com"


def test_direct_agent_init_routes_pro_to_responses(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("run_agent.OpenAI", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        provider="deepseek",
        api_mode="chat_completions",
        model="deepseek-v4-pro",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.base_url == "https://api.deepseek.com"


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_main_request_builder_uses_explicit_native_search_backend(
    monkeypatch, tmp_path, model
):
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("run_agent.OpenAI", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr(
        "agent.web_search_registry._read_config_key",
        lambda *path: "deepseek" if path == ("web", "search_backend") else None,
    )
    monkeypatch.setattr(
        "agent.web_search_registry.get_active_search_provider",
        lambda: SimpleNamespace(name="deepseek"),
    )
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        provider="deepseek",
        model=model,
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    kwargs = agent._build_api_kwargs(
        [{"role": "user", "content": "latest news"}],
        tools_for_api=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )
    assert kwargs["tools"] == [{"type": "web_search"}]
    assert "prompt_cache_key" not in kwargs
    assert "include" not in kwargs
