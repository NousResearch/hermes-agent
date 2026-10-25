"""Fast-mode wire parameters follow the active route, not the session's primary."""

from types import SimpleNamespace

import pytest

from agent.fast_mode import begin_turn, effective_request_overrides


def _agent(**kwargs):
    values = dict(
        model="claude-opus-5-5", provider="anthropic",
        base_url="https://api.anthropic.com", api_mode="anthropic_messages",
        service_tier="priority", request_overrides={"speed": "fast", "extra_body": {"keep": 1}},
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tier", ["priority", "auto", "cold"])
def test_fast_parameters_follow_fallback_and_primary_restore(tier):
    agent = _agent(service_tier=tier)
    begin_turn(agent, [])
    original = agent.request_overrides.copy()
    assert effective_request_overrides(agent) == {"speed": "fast", "extra_body": {"keep": 1}}

    agent.model, agent.provider = "gpt-6-astra", "openai-codex"
    agent.base_url, agent.api_mode = "https://chatgpt.com/backend-api/codex", "codex_responses"
    # The old Anthropic client can remain on the agent after a route swap.
    agent._anthropic_base_url = "https://api.anthropic.com"
    assert effective_request_overrides(agent) == {"service_tier": "priority", "extra_body": {"keep": 1}}

    agent.model, agent.provider = "glm-5.3", "zai"
    agent.base_url, agent.api_mode = "https://api.z.ai/api/coding/paas/v4/", "chat_completions"
    assert effective_request_overrides(agent) == {"extra_body": {"keep": 1}}

    agent.model, agent.provider = "claude-opus-5-5", "anthropic"
    agent.base_url, agent.api_mode = "https://api.anthropic.com", "anthropic_messages"
    assert effective_request_overrides(agent) == original
    assert agent.request_overrides == original


@pytest.mark.parametrize("provider,base_url", [
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("custom", "https://proxy.example/v1"),
    ("anthropic", "https://proxy.example/v1"),
])
def test_static_fast_never_leaks_to_unsupported_route(provider, base_url):
    agent = _agent(provider=provider, base_url=base_url)
    assert effective_request_overrides(agent) == {"extra_body": {"keep": 1}}


def test_ultrafast_is_not_downgraded_when_fallback_does_not_support_it():
    agent = _agent(
        model="gpt-6-astra", provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", service_tier="ultrafast",
        request_overrides={"service_tier": "ultrafast", "temperature": 0.2},
    )
    assert effective_request_overrides(agent) == agent.request_overrides
    agent.model = "gpt-6-sol"
    assert effective_request_overrides(agent) == {"temperature": 0.2}
    agent.model, agent.provider, agent.api_mode = "claude-opus-5-5", "anthropic", "anthropic_messages"
    agent.base_url = "https://api.anthropic.com"
    assert effective_request_overrides(agent) == {"temperature": 0.2}


def test_task_only_glm_override_has_no_fast_parameters_on_existing_cli():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    stub = SimpleNamespace(
        model="glm-5.3", provider="zai", api_key="test-key",
        base_url="https://api.z.ai/api/coding/paas/v4/", api_mode="chat_completions",
        acp_command=None, acp_args=[], _credential_pool=None, service_tier="priority",
    )
    route = CLIAgentSetupMixin._resolve_turn_agent_config(stub, "test")
    assert route["model"] == "glm-5.3"
    assert route["runtime"]["provider"] == "zai"
    assert route["request_overrides"] is None


def test_real_fallback_binding_and_sdk_requests_do_not_carry_primary_speed(monkeypatch):
    import json

    import httpx
    from openai import OpenAI
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    from run_agent import AIAgent

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        if request.url.host == "chatgpt.com":
            return httpx.Response(200, json={
                "id": "offline-response", "object": "response", "created_at": 0,
                "status": "completed", "output": [],
            })
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "offline fixture"},
                "finish_reason": "stop",
            }],
        })

    clients = {
        provider: OpenAI(api_key="offline-test-key", base_url=url,
                         http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        for provider, url in (
            ("openai-codex", "https://chatgpt.com/backend-api/codex/"),
            ("zai", "https://api.z.ai/api/coding/paas/v4/"),
        )
    }
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                        lambda provider, **kwargs: (clients[provider], kwargs["model"]))
    agent = AIAgent(
        model="claude-opus-5-5", provider="anthropic", api_mode="anthropic_messages",
        api_key="offline-test-key", base_url="https://api.anthropic.com",
        service_tier="priority", request_overrides={"speed": "fast"},
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, fallback_model=[
            {"provider": "openai-codex", "model": "gpt-6-astra"},
            {"provider": "zai", "model": "glm-5.3"},
        ],
    )
    messages = [{"role": "user", "content": "offline request fixture"}]
    assert agent._build_api_kwargs(messages, [])["extra_body"]["speed"] == "fast"
    assert agent._try_activate_fallback()
    assert (agent.provider, agent.api_mode) == ("openai-codex", "codex_responses")
    codex_kwargs = _preflight_codex_api_kwargs(agent._build_api_kwargs(messages, []))
    clients["openai-codex"].responses.create(**codex_kwargs)
    assert requests[-1][0] == "chatgpt.com"
    assert "speed" not in requests[-1][1]
    assert requests[-1][1]["service_tier"] == "priority"

    assert agent._try_activate_fallback()
    assert (agent.provider, agent.api_mode) == ("zai", "chat_completions")
    clients["zai"].chat.completions.create(**agent._build_api_kwargs(messages, []))
    assert requests[-1][0] == "api.z.ai"
    assert not {"speed", "service_tier"} & requests[-1][1].keys()
    assert agent.request_overrides == {"speed": "fast"}

    assert agent._restore_primary_runtime()
    assert (agent.provider, agent.model, agent.api_mode) == ("anthropic", "claude-opus-5-5", "anthropic_messages")
    assert agent._build_api_kwargs(messages, [])["extra_body"]["speed"] == "fast"
    for client in clients.values():
        client.close()
