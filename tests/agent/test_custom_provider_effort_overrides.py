"""custom provider effort_overrides: config survives normalization, wire effort is rewritten."""
import logging
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_init import (
    _apply_custom_provider_effort_overrides,
    _custom_provider_effort_overrides_for_agent,
    _store_custom_provider_effort_overrides,
)
from agent.chat_completion_helpers import _build_api_kwargs_for_mode
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli.config import (
    _PROVIDER_NORMALIZE_WARNED,
    _VALID_CUSTOM_PROVIDER_FIELDS,
    _normalize_custom_provider_entry,
)
from providers import get_provider_profile
import agent.agent_runtime_helpers as arh


def test_named_route_normalizes_case_and_ignores_non_strings():
    got = _custom_provider_effort_overrides_for_agent(
        provider="custom:relay",
        model="kimi-k3",
        base_url="https://relay.example/v1/",
        custom_providers=[{
            "provider_key": "relay",
            "name": "Relay",
            "base_url": "https://relay.example/v1",
            "model": "kimi-k3",
            "effort_overrides": {"XHigh": "Max", "low": 1, "": "high"},
        }],
    )
    assert got == {"xhigh": "max"}


def test_non_custom_and_url_mismatch_are_absent():
    entry = {
        "name": "relay",
        "base_url": "https://relay.example/v1",
        "effort_overrides": {"xhigh": "max"},
    }
    assert _custom_provider_effort_overrides_for_agent(
        provider="openrouter", model="m", base_url="https://relay.example/v1",
        custom_providers=[entry],
    ) is None
    assert _custom_provider_effort_overrides_for_agent(
        provider="custom", model="m", base_url="https://other.example/v1",
        custom_providers=[entry],
    ) is None


def test_model_specific_map_beats_the_url_fallback():
    providers = [
        {
            "name": "relay",
            "base_url": "https://relay.example/v1",
            "effort_overrides": {"xhigh": "high"},
        },
        {
            "name": "relay",
            "base_url": "https://relay.example/v1",
            "model": "kimi-k3",
            "effort_overrides": {"xhigh": "max"},
        },
    ]
    assert _custom_provider_effort_overrides_for_agent(
        provider="custom", model="kimi-k3", base_url="https://relay.example/v1",
        custom_providers=providers,
    ) == {"xhigh": "max"}
    assert _custom_provider_effort_overrides_for_agent(
        provider="custom", model="other", base_url="https://relay.example/v1",
        custom_providers=providers,
    ) == {"xhigh": "high"}


def test_store_is_empty_dict_when_the_key_is_absent():
    agent = SimpleNamespace(provider="custom", model="m", base_url="https://relay.example/v1")
    _store_custom_provider_effort_overrides(agent, [{
        "name": "relay", "base_url": "https://relay.example/v1",
    }])
    assert agent._custom_provider_effort_overrides == {}


def test_apply_rewrites_a_copy_and_leaves_the_session_config():
    original = {"enabled": True, "effort": "XHigh"}
    agent = SimpleNamespace(_custom_provider_effort_overrides={"xhigh": "max"})
    sent = _apply_custom_provider_effort_overrides(agent, original)
    assert sent == {"enabled": True, "effort": "max"}
    assert original["effort"] == "XHigh"
    assert _apply_custom_provider_effort_overrides(agent, {"enabled": True, "effort": "low"}) is not None
    assert _apply_custom_provider_effort_overrides(
        agent, {"enabled": True, "effort": "low"},
    )["effort"] == "low"


def test_normalizer_keeps_the_key_without_an_unknown_warning(caplog):
    _PROVIDER_NORMALIZE_WARNED.clear()
    entry = {
        "name": "relay",
        "base_url": "https://relay.example/v1",
        "effortOverrides": {"xhigh": "max"},
    }
    with caplog.at_level(logging.WARNING):
        result = _normalize_custom_provider_entry(dict(entry), provider_key="relay")
    assert result is not None
    assert result["effort_overrides"] == {"xhigh": "max"}
    assert "effort_overrides" in _VALID_CUSTOM_PROVIDER_FIELDS
    assert not [r for r in caplog.records if "unknown config keys" in r.message.lower()]
    _PROVIDER_NORMALIZE_WARNED.clear()


class _Agent:
    def __init__(self):
        self.reasoning_config = {"enabled": True, "effort": "xhigh"}
        self.provider = "custom:relay"
        self.model = "kimi-k3"
        self.api_mode = "chat_completions"
        self.base_url = "http://relay.example/v1"
        self.tools = []
        self.request_overrides = None
        self.service_tier = None
        self._fast_until = 0.0
        self._ephemeral_reasoning_off = False
        self._reasoning_effort_rejected = False
        self._reasoning_disable_rejected = False
        self._ollama_num_ctx = None
        self._custom_provider_effort_overrides = {"xhigh": "max"}

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_wire_builder_sees_the_mapped_effort_and_the_session_keeps_xhigh():
    agent = _Agent()

    def _capture(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
        return {"reasoning_config": reasoning_config}

    with patch("agent.models_dev.get_model_capabilities", return_value=None), \
         patch("agent.chat_completion_helpers._build_chat_completions_kwargs", _capture):
        sent = _build_api_kwargs_for_mode(agent, [], [])["reasoning_config"]
    assert sent["effort"] == "max"
    assert agent.reasoning_config["effort"] == "xhigh"
    kwargs = ChatCompletionsTransport().build_kwargs(
        "kimi-k3", [{"role": "user", "content": "hi"}], tools=None,
        provider_profile=get_provider_profile("custom:relay"),
        reasoning_config=sent, base_url="http://relay.example/v1",
    )
    assert kwargs.get("reasoning_effort") == "max"


def test_switch_refreshes_the_map_for_the_destination_key():
    agent = SimpleNamespace(
        provider="custom",
        model="kimi-k3",
        base_url="https://relay.example/v1",
        request_overrides={},
        _custom_providers=[
            {
                "provider_key": "other",
                "name": "other",
                "base_url": "https://relay.example/v1",
                "model": "kimi-k3",
                "effort_overrides": {"xhigh": "high"},
            },
            {
                "provider_key": "relay",
                "name": "relay",
                "base_url": "https://relay.example/v1",
                "model": "kimi-k3",
                "effort_overrides": {"xhigh": "max"},
            },
        ],
    )
    arh._apply_switched_provider_request_overrides(agent, "custom:relay")
    assert agent._custom_provider_effort_overrides == {"xhigh": "max"}
