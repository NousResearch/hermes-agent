"""Opaque Messages routes preserve explicit effort without claiming a Claude identity.

Regressions for #66244 and the gateway consequence of #61334.
"""

from types import SimpleNamespace

import pytest

from agent.transports.anthropic import AnthropicTransport
from hermes_cli.config import atomic_config_write


BASE_URL = "https://relay.example/anthropic"
MESSAGES = [{"role": "user", "content": "Reply briefly."}]


def configure_route(home, models, *, legacy=False):
    entry = {"api": BASE_URL, "transport": "anthropic_messages", "models": models}
    config = {"providers": {"relay": entry}}
    if legacy:
        config = {"custom_providers": [{
            "name": "relay", "base_url": BASE_URL, "api_mode": "anthropic_messages", "models": models,
        }]}
    atomic_config_write(home / "config.yaml", config)


def request(model="gpt-6.1-sol", effort="max", base_url=BASE_URL, **params):
    return AnthropicTransport().build_kwargs(
        model=model, messages=MESSAGES, reasoning_config={"enabled": True, "effort": effort},
        base_url=base_url, **params,
    )


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "opus", "auto", "haiku-route"])
@pytest.mark.parametrize("legacy", [False, True])
def test_declared_adaptive_route_preserves_max(tmp_path, monkeypatch, model, legacy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {model: {"supports_adaptive_thinking": True}}, legacy=legacy)

    kwargs = request(model)

    assert kwargs.get("output_config") == {"effort": "max"}
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["model"] == model
    assert "temperature" not in kwargs  # Do not inherit manual thinking's forced temperature.


@pytest.mark.parametrize("accepts_xhigh, expected", [(True, "xhigh"), (False, "max")])
def test_declared_route_uses_its_xhigh_capability(tmp_path, monkeypatch, accepts_xhigh, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {
        "supports_adaptive_thinking": True, "supports_xhigh_effort": accepts_xhigh,
    }})

    assert request(effort="xhigh")["output_config"] == {"effort": expected}


@pytest.mark.parametrize("model, capability", [
    ("minimax-m2", {}),
    ("gpt-6.1-sol", {}),
    ("gpt-6.1-sol", {"supports_adaptive_thinking": "true"}),
    ("gpt-6.1-sol", {"supports_adaptive_thinking": False}),
    ("gpt-6.1-sol", {"supports_xhigh_effort": True}),
])
def test_manual_and_undeclared_routes_keep_their_wire(tmp_path, monkeypatch, model, capability):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {model: capability})

    kwargs = request(model, effort="high")

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert "output_config" not in kwargs
    assert kwargs["temperature"] == 1


@pytest.mark.parametrize("model, base_url", [
    ("gpt-6.1-sol", "https://relay.example/other"),
    ("gpt-6.1-sol", "https://relay.example:8443/anthropic"),
    ("other-model", BASE_URL),
    ("gpt-6.1-sol", None),
])
def test_capability_is_scoped_to_the_exact_model_and_route(tmp_path, monkeypatch, model, base_url):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": True}})

    assert "output_config" not in request(model, base_url=base_url)


def test_lookup_uses_the_routing_id_before_wire_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"anthropic/gpt-6.1-sol": {
        "supports_adaptive_thinking": True, "supports_xhigh_effort": True,
    }})

    kwargs = request("anthropic/gpt-6.1-sol", effort="xhigh")

    assert kwargs["model"] == "gpt-6.1-sol"
    assert kwargs["output_config"] == {"effort": "xhigh"}


def test_route_switch_and_retry_resolve_the_current_declaration(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import _build_anthropic_kwargs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {
        "gpt-6.1-sol": {"supports_adaptive_thinking": True},
        "budget-route": {"supports_adaptive_thinking": False},
    })
    agent = SimpleNamespace(
        model="gpt-6.1-sol", provider="custom:relay", max_tokens=16384,
        context_compressor=SimpleNamespace(context_length=8192),
        _is_anthropic_oauth=False, _anthropic_base_url=BASE_URL,
        _get_transport=AnthropicTransport, _anthropic_preserve_dots=lambda: False,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
    )
    reasoning = {"enabled": True, "effort": "max"}

    first = _build_anthropic_kwargs(agent, MESSAGES, None, reasoning, {})
    retry = _build_anthropic_kwargs(agent, MESSAGES, None, reasoning, {})
    assert first == retry
    assert first["output_config"] == {"effort": "max"}
    assert first["max_tokens"] == 8191

    agent.model = "budget-route"
    manual = _build_anthropic_kwargs(agent, MESSAGES, None, {"effort": "high"}, {})
    assert manual["thinking"]["type"] == "enabled"
    assert "output_config" not in manual

    agent.model = "gpt-6.1-sol"
    agent._anthropic_base_url = "https://different-relay.example/anthropic"
    assert "output_config" not in _build_anthropic_kwargs(agent, MESSAGES, None, reasoning, {})


def test_auxiliary_adapter_keeps_max_and_structured_output(tmp_path, monkeypatch):
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": True}})
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn", usage=None)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    adapter = _AnthropicCompletionsAdapter(client, "gpt-6.1-sol", base_url=BASE_URL)
    adapter.create(
        model="gpt-6.1-sol", messages=MESSAGES, _reasoning_config={"enabled": True, "effort": "max"},
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    )

    assert captured["thinking"]["type"] == "adaptive"
    assert captured["output_config"]["effort"] == "max"
    assert captured["output_config"]["format"]["schema"] == {"type": "object"}


def test_real_sdk_serialization_preserves_the_explicit_effort(tmp_path, monkeypatch):
    import json

    import anthropic
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": True}})
    captured = {}

    def receive(req):
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "gpt-6.1-sol",
            "content": [{"type": "text", "text": "fixture"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    with anthropic.Anthropic(
        api_key="fixture-key", base_url=BASE_URL, http_client=httpx.Client(transport=httpx.MockTransport(receive)),
    ) as client:
        client.messages.create(**request())

    assert captured["model"] == "gpt-6.1-sol"
    assert captured["output_config"] == {"effort": "max"}
    assert "budget_tokens" not in captured["thinking"]
    assert "_reasoning_config" not in captured


@pytest.mark.parametrize("model, base_url, capability, expected_type", [
    ("claude-sonnet-5", "https://api.anthropic.com", False, "adaptive"),
    ("claude-sonnet-5", BASE_URL, False, "adaptive"),
    ("claude-haiku-4-5", "https://api.anthropic.com", True, None),
    ("claude-haiku-4-5", BASE_URL, True, None),
    ("kimi-k2.5", BASE_URL, False, "adaptive"),
])
def test_known_family_contracts_cannot_be_overridden(tmp_path, monkeypatch, model, base_url, capability, expected_type):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": base_url, "transport": "anthropic_messages",
        "models": {model: {"supports_adaptive_thinking": capability, "supports_xhigh_effort": True}},
    }}})

    kwargs = request(model, effort="high", base_url=base_url)

    assert (kwargs.get("thinking") or {}).get("type") == expected_type
    assert "temperature" not in kwargs


@pytest.mark.parametrize("model, base_url", [
    ("gpt-6.1-sol", "https://api.anthropic.com"),
    ("claude-sonnet-5", BASE_URL),
    ("kimi-k2.5", BASE_URL),
    ("gpt-6.1-sol", None),
])
def test_unrelated_family_and_native_routes_do_not_read_custom_config(monkeypatch, model, base_url):
    import agent.anthropic_adapter as adapter
    from hermes_cli import config

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("unrelated native/family route performed a custom capability lookup")

    monkeypatch.setattr(config, "get_custom_provider_model_capability", unexpected_lookup)
    # Catch the old direct binding too; after the fix the re-export is the only surface.
    monkeypatch.setattr(adapter, "get_custom_provider_model_capability", unexpected_lookup, raising=False)
    request(model, base_url=base_url)


def test_explicit_snapshot_wins_over_the_ambient_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": False}})
    snapshot = [{"name": "session-relay", "base_url": BASE_URL, "models": {
        "gpt-6.1-sol": {"supports_adaptive_thinking": True},
    }}]

    kwargs = request(custom_providers=snapshot)

    assert kwargs["output_config"] == {"effort": "max"}
    assert "output_config" not in request(custom_providers=[])


def build_agent(**overrides):
    agent = SimpleNamespace(
        model="gpt-6.1-sol", provider="custom:relay", max_tokens=16384,
        context_compressor=SimpleNamespace(context_length=8192),
        _is_anthropic_oauth=False, _anthropic_base_url=BASE_URL, _custom_providers=None,
        _get_transport=AnthropicTransport, _anthropic_preserve_dots=lambda: False,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def agent_kwargs(agent, reasoning=None):
    from agent.chat_completion_helpers import _build_anthropic_kwargs
    return _build_anthropic_kwargs(
        agent, MESSAGES, None, reasoning or {"enabled": True, "effort": "max"}, {})


def test_main_agent_uses_its_own_snapshot_over_the_ambient_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": False}})
    agent = build_agent(_custom_providers=[{"name": "session-relay", "base_url": BASE_URL, "models": {
        "gpt-6.1-sol": {"supports_adaptive_thinking": True},
    }}])

    assert agent_kwargs(agent)["output_config"] == {"effort": "max"}

    agent._custom_providers = []  # an explicit empty snapshot is authoritative, not "no snapshot"
    assert "output_config" not in agent_kwargs(agent)


def test_standalone_builder_reads_config_at_most_once_per_request(tmp_path, monkeypatch):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {
        "supports_adaptive_thinking": True, "supports_xhigh_effort": True,
    }})
    real = config.load_config_readonly
    reads = []

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(config, "load_config_readonly", counting)
    assert request(effort="xhigh")["output_config"] == {"effort": "xhigh"}
    assert len(reads) <= 1, f"resolved the capability list {len(reads)} times"


def test_lookup_resolves_through_the_hermes_cli_config_reexport(tmp_path, monkeypatch):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": True}})
    seen = []

    def fake_lookup(*args, **kwargs):
        seen.append((args[0], args[2]))
        return False

    monkeypatch.setattr(config, "get_custom_provider_model_capability", fake_lookup)
    kwargs = request(effort="max")

    assert seen, "the adapter bypassed the hermes_cli.config re-export"
    assert "output_config" not in kwargs


@pytest.mark.parametrize("label, model, base_url", [
    ("oauth", "gpt-6.1-sol", BASE_URL),
    ("nous-portal", "gpt-6.1-sol", "https://inference-api.nousresearch.com/v1"),
])
def test_oauth_and_portal_routes_skip_the_custom_lookup(monkeypatch, tmp_path, label, model, base_url):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {model: {"supports_adaptive_thinking": True}}, legacy=label == "oauth")

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError(f"{label} request performed a custom capability lookup")

    monkeypatch.setattr(config, "get_custom_provider_model_capability", unexpected_lookup)
    params = {"is_oauth": True} if label == "oauth" else {}
    kwargs = request(model, base_url=base_url, **params)

    assert kwargs["model"] == model
    assert "output_config" not in kwargs


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-6"])
def test_known_family_xhigh_declarations_are_ignored(tmp_path, monkeypatch, model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": BASE_URL, "transport": "anthropic_messages",
        "models": {model: {"supports_adaptive_thinking": True, "supports_xhigh_effort": True}},
    }}})

    assert request(model, effort="xhigh")["output_config"] == {"effort": "max"}


def test_duplicate_legacy_and_provider_entries_resolve_the_same_route(tmp_path, monkeypatch):
    """The existing legacy-first view and first-boolean lookup keep their precedence."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {
        "custom_providers": [{"name": "relay", "base_url": BASE_URL, "api_mode": "anthropic_messages",
                             "models": {"gpt-6.1-sol": {"supports_adaptive_thinking": True}}}],
        "providers": {"relay": {"api": BASE_URL, "transport": "anthropic_messages",
                                "models": {"gpt-6.1-sol": {"supports_adaptive_thinking": False}}}},
    })

    assert request()["output_config"] == {"effort": "max"}


def test_iteration_summary_retries_use_the_owning_agent_snapshot(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configure_route(tmp_path, {"gpt-6.1-sol": {"supports_adaptive_thinking": False}})
    agent = build_agent(
        reasoning_config={"enabled": True, "effort": "max"},
        _custom_providers=[{"base_url": BASE_URL, "models": {
            "gpt-6.1-sol": {"supports_adaptive_thinking": True},
        }}],
        _interruptible_api_call=lambda kwargs: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="summary")], stop_reason="end_turn", usage=None,
        ),
    )
    captured = []

    def managed_call(agent, api_request_id, request, callback, *, retry_count):
        captured.append(request)
        return callback(request)

    monkeypatch.setattr(helpers, "_managed_summary_call", managed_call)
    attempt = helpers._anthropic_summary_attempt(agent, MESSAGES, "fixture-summary")
    assert attempt(0) == attempt(1) == "summary"
    assert all(kwargs["output_config"] == {"effort": "max"} for kwargs in captured)
