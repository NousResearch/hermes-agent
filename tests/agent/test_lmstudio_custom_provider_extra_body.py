"""Explicit LMStudio registrations apply only to their endpoint and exact model."""
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from agent.agent_init import _custom_provider_extra_body_for_agent, _merge_custom_provider_extra_body
from agent.chat_completion_helpers import build_api_kwargs, _rescope_fallback_extra_body
from run_agent import AIAgent

MODEL = "huihui-qwen3.8-27b-abliterated"
URL = "http://127.0.0.1:1234/v1"
ENTRY = {"name": "LM Studio LAN", "provider": "lmstudio", "model": MODEL, "base_url": URL,
         "extra_body": {"temperature": 0}}


def resolve(entry=ENTRY, *, provider="lmstudio", model=MODEL, base_url=URL):
    return _custom_provider_extra_body_for_agent(
        provider=provider, model=model, base_url=base_url, custom_providers=[entry])


def test_explicit_lmstudio_model_registration():
    assert resolve(base_url=URL + "/") == {"temperature": 0}


@pytest.mark.parametrize("change", [
    {"provider": "custom"}, {"provider": "openai"}, {"provider": ""},
    {"model": ""}, {"model": "*"}, {"model": "other-local"},
    {"model": MODEL.upper()}, {"base_url": "http://localhost:1234/v1"},
    {"models": [MODEL], "model": "other-local"},
])
def test_lmstudio_registration_requires_explicit_provider_endpoint_exact_model(change):
    assert resolve({**ENTRY, **change}) is None


@pytest.mark.parametrize("provider", ["openai", "openrouter", "zai", "ollama"])
def test_non_custom_cloud_and_other_local_providers_unchanged(provider):
    assert resolve(provider=provider) is None


def test_legacy_custom_wildcard_still_applies():
    assert resolve({"base_url": URL, "extra_body": {"top_k": 20}},
                   provider="custom", model="any-model") == {"top_k": 20}


def test_lmstudio_caller_extra_body_takes_precedence():
    agent = SimpleNamespace(provider="lmstudio", model=MODEL, base_url=URL,
                            request_overrides={"extra_body": {"temperature": 0.4, "caller": True}})
    _merge_custom_provider_extra_body(agent, [ENTRY])
    assert agent.request_overrides == {"extra_body": {"temperature": 0.4, "caller": True}}


@pytest.mark.parametrize("caller", [None, 0, 0.4])
def test_switching_away_removes_only_model_contribution(caller):
    eb = {"caller": True}
    if caller is not None:
        eb["temperature"] = caller
    agent = SimpleNamespace(provider="lmstudio", model=MODEL, base_url=URL,
                            _custom_providers=[ENTRY], request_overrides={})
    _merge_custom_provider_extra_body(agent, [ENTRY])
    # Set after the registration was merged and recorded: positively the caller's.
    agent.request_overrides["extra_body"] = {**agent.request_overrides["extra_body"], **eb}
    agent.model = "other-local"
    _rescope_fallback_extra_body(agent, MODEL, "lmstudio", URL)
    expected = {"caller": True}
    # The registration's own 0 was recorded when merged; a caller 0 equals what the route being left
    # sets (upstream's rule). Either way it is config-owned; only a differing caller value is kept.
    if caller not in (None, ENTRY["extra_body"]["temperature"]):
        expected["temperature"] = caller
    assert agent.request_overrides == {"extra_body": expected}


@pytest.mark.parametrize("after_init", [False, True])
@pytest.mark.parametrize("representation", ["legacy", "converted"])
@pytest.mark.parametrize("destination", ["lmstudio", "openrouter"])
@pytest.mark.parametrize("caller", [None, 0, 0.4])
def test_full_agent_fallback_builder_loads_model_registration(
        tmp_path, monkeypatch, caller, destination, representation, after_init):
    home = tmp_path / "home"
    home.mkdir()
    from hermes_cli.config_providers import _custom_provider_entry_to_provider_config
    config = {"custom_providers": [ENTRY]} if representation == "legacy" else {
        "providers": {"lan": _custom_provider_entry_to_provider_config(ENTRY)}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    overrides = {} if caller is None else {"extra_body": {"temperature": caller}}
    # No inference required: the real init/config/fallback/builder run; only client resolution
    # and remote model metadata are isolated from network I/O.
    client = SimpleNamespace(base_url=URL, api_key="test")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI"),
          patch("agent.auxiliary_client.resolve_provider_client",
                side_effect=lambda provider, model, **kw: (client, model)),
          patch("agent.model_metadata.get_model_context_length", return_value=128_000),
          # URL is LM Studio's default port: never probe or preload a live local server.
          patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", return_value=None),
          patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
          patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden"))):
        agent = AIAgent(model="primary", provider="custom", api_key="test",
                        base_url="http://primary.invalid/v1", enabled_toolsets=[], quiet_mode=True,
                        skip_context_files=True, skip_memory=True, skip_background_review=True,
                        request_overrides={} if after_init else overrides,
                        fallback_model=[
                            {"provider": "lmstudio", "model": MODEL, "base_url": URL},
                            {"provider": destination, "model": "other-local", "base_url": URL},
                        ])
        if after_init:
            agent.request_overrides = dict(overrides)
        assert agent._try_activate_fallback()
        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], [])
        assert agent._try_activate_fallback()
        switched = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], [])
    # A constructor body is config-owned while config defines extra_body (every Hermes path passes
    # runtime-resolved config there), so it never reaches a fallback. A value set after construction
    # is the caller's: it wins over the registration and moves on, unless it equals what the route
    # being left sets (a caller 0 leaves with the pinned model, like the registration's own 0).
    caller_owned = after_init and caller is not None
    assert kwargs["extra_body"]["temperature"] == (caller if caller_owned else 0)
    assert switched.get("extra_body", {}).get("temperature") == (caller if caller_owned and caller != 0 else None)
    assert "reasoning_effort" not in kwargs
    assert "reasoning_effort" not in kwargs.get("extra_body", {})
