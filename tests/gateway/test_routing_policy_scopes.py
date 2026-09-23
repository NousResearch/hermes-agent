"""Regression coverage for owner-scoped terminal routing policy enforcement."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def routed_runner(tmp_path, monkeypatch):
    """A real multiplex runner with different default and beta route policy."""
    from agent import secret_scope
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (beta / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    source_a = SessionSource(platform=Platform.SLACK, chat_id="A", chat_type="dm", profile="default")
    source_b = SessionSource(platform=Platform.SLACK, chat_id="B", chat_type="dm", profile="beta")

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield runner, source_a, source_b
    finally:
        secret_scope.set_multiplex_active(previous)


def _install_complete_override(runner, source):
    session_key = runner._session_key_for_source(source)
    runner._session_state(session_key).conversation.model_override = {
        "provider": "openrouter",
        "model": "z-ai/glm-5.2",
        "api_key": "x",
    }


def test_gateway_runtime_resolution_follows_source_profile_a_to_b_to_a(routed_runner, monkeypatch):
    """A beta denial does not leak into the following default-profile resolution."""
    from hermes_cli.routing_policy import RoutingPolicyError

    runner, source_a, source_b = routed_runner
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda _provider: None)
    _install_complete_override(runner, source_a)
    _install_complete_override(runner, source_b)

    assert runner._resolve_session_agent_runtime(source=source_a)[0] == "z-ai/glm-5.2"
    with pytest.raises(RoutingPolicyError, match="selected model"):
        runner._resolve_session_agent_runtime(source=source_b)
    assert runner._resolve_session_agent_runtime(source=source_a)[0] == "z-ai/glm-5.2"


class _RecordingClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return "sent"


def test_final_wire_policy_denies_beta_without_sending_then_recovers_a(routed_runner):
    """A final wire guard runs before client construction and cannot leak across scopes."""
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli.routing_policy import RoutingPolicyError

    runner, source_a, source_b = routed_runner
    client = _RecordingClient()
    agent = SimpleNamespace(
        provider="openrouter",
        model="z-ai/glm-5.2",
        base_url="",
        api_mode="chat_completions",
    )
    request = {"model": "z-ai/glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    with runner._profile_scope_for_source(source_b):
        with pytest.raises(RoutingPolicyError, match="selected model"):
            _dispatch_nonstreaming_api_request(agent, dict(request), make_client=lambda *_args, **_kwargs: client)
    assert client.calls == []

    with runner._profile_scope_for_source(source_a):
        assert _dispatch_nonstreaming_api_request(agent, dict(request), make_client=lambda *_args, **_kwargs: client) == "sent"
    assert len(client.calls) == 1

    with runner._profile_scope_for_source(source_b):
        with pytest.raises(RoutingPolicyError, match="selected model"):
            _dispatch_nonstreaming_api_request(agent, dict(request), make_client=lambda *_args, **_kwargs: client)
    assert len(client.calls) == 1


def test_openai_final_wire_policy_uses_extra_body_model_override(monkeypatch):
    """SDK ``extra_body`` wins over top-level model at the actual OpenAI wire."""
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli.routing_policy import RoutingPolicyError

    client = _RecordingClient()
    agent = SimpleNamespace(provider="openrouter", model="allowed", base_url="", api_mode="chat_completions")
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["denied"]},
    })

    with pytest.raises(RoutingPolicyError, match="selected model"):
        _dispatch_nonstreaming_api_request(agent, {
            "model": "allowed", "extra_body": {"model": "denied"}, "messages": [],
        }, make_client=lambda *_args, **_kwargs: client)

    assert client.calls == []


def test_native_anthropic_final_wire_policy_uses_extra_body_model_override(monkeypatch):
    """Anthropic SDK merges ``extra_body`` after the checked payload too."""
    from agent.client_lifecycle import ClientLifecycleMixin
    from hermes_cli.routing_policy import RoutingPolicyError

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_final_message(self):
            return SimpleNamespace(content=[])

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **_kwargs: _Stream(), create=lambda **_kwargs: None))
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="anthropic", model="allowed", base_url="",
        _anthropic_client=client, _capture_anthropic_response_headers=lambda _response: None,
    )
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["denied"]},
    })

    with pytest.raises(RoutingPolicyError, match="selected model"):
        ClientLifecycleMixin._anthropic_messages_create(agent, {
            "model": "allowed", "extra_body": {"model": "denied"}, "messages": [],
        }, client=client)


def test_openai_stream_final_wire_policy_denies_before_sdk_send(monkeypatch):
    """A denied streaming request must not reach ``chat.completions.create``."""
    from unittest.mock import MagicMock
    from agent.chat_completion_helpers import _StreamingCall
    from hermes_cli.routing_policy import RoutingPolicyError

    client = _RecordingClient()
    agent = SimpleNamespace(
        provider="openrouter", model="denied", base_url="", _touch_activity=MagicMock(),
        _create_request_openai_client=lambda **_kwargs: client,
    )
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = agent
    call.clients = SimpleNamespace(set_client=lambda value: value)
    call.last_chunk_time = {}
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["denied"]},
    })

    with pytest.raises(RoutingPolicyError, match="selected model"):
        call._open_chat_stream({"model": "denied", "messages": []})

    assert client.calls == []


def test_codex_stream_final_wire_policy_denies_before_sdk_send(monkeypatch):
    """A denied Responses stream must not reach ``responses.create``."""
    from unittest.mock import MagicMock
    from run_agent import AIAgent
    from hermes_cli.routing_policy import RoutingPolicyError

    client = MagicMock()
    client.responses.create.side_effect = AssertionError("Responses SDK send must not happen")
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="denied",
        provider="openrouter", quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["denied"]},
    })

    with pytest.raises(RoutingPolicyError, match="selected model"):
        agent._run_codex_stream({"model": "denied", "input": []}, client=client)

    client.responses.create.assert_not_called()


def test_native_anthropic_stream_final_wire_policy_denies_before_sdk_send(monkeypatch):
    """A denied native stream must not reach ``messages.stream`` or fallback ``create``."""
    from unittest.mock import MagicMock
    from agent.client_lifecycle import ClientLifecycleMixin
    from hermes_cli.routing_policy import RoutingPolicyError

    client = MagicMock()
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="anthropic", model="denied", base_url="",
        _capture_anthropic_response_headers=lambda _response: None,
    )
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["denied"]},
    })

    with pytest.raises(RoutingPolicyError, match="selected model"):
        ClientLifecycleMixin._anthropic_messages_create(agent, {"model": "denied", "messages": []}, client=client)

    client.messages.stream.assert_not_called()
    client.messages.create.assert_not_called()
