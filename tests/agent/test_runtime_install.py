"""Atomic installation contract for a live AIAgent runtime."""

from types import MappingProxyType

import pytest

from agent.runtime_bundle import ClientBundle, ResolvedRuntime, build_client_bundle
from run_agent import AIAgent


class _RecordingLock:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock-enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append("lock-exit")


def _bare_agent(events):
    agent = AIAgent.__new__(AIAgent)
    agent.model = "old-model"
    agent.provider = "old-provider"
    agent.requested_provider = "old-provider"
    agent.base_url = "https://old.example/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "old-key"
    agent.client = object()
    agent._anthropic_client = None
    agent._client_kwargs = {
        "api_key": "old-key",
        "base_url": "https://old.example/v1",
    }
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = ""
    agent._is_anthropic_oauth = False
    agent._client_lock = _RecordingLock(events)
    agent._transport_cache = {"chat_completions": object()}
    agent._cached_system_prompt = "stable prompt"
    agent._session_messages = [{"role": "user", "content": "keep me"}]
    return agent


def test_install_runtime_commits_under_lock_then_retires_old_client():
    events = []
    agent = _bare_agent(events)
    old_client = agent.client
    new_client = object()
    retired = []

    def _retire(client, *, reason):
        events.append("retire")
        retired.append((client, reason))

    agent._retire_shared_openai_client = _retire
    runtime = ResolvedRuntime.from_mapping(
        {
            "provider": "new-provider",
            "requested_provider": "custom:new-provider",
            "model": "new-model",
            "api_mode": "codex_responses",
            "api_key": "new-key",
            "base_url": "https://new.example/v1",
        }
    )
    bundle = ClientBundle(
        runtime=runtime,
        client=new_client,
        client_kwargs=MappingProxyType(
            {
                "api_key": "new-key",
                "base_url": "https://new.example/v1",
                "timeout": 42.0,
            }
        ),
    )

    installed = agent.install_runtime(bundle, reason="test")

    assert installed is runtime
    assert events == ["lock-enter", "lock-exit", "retire"]
    assert retired == [(old_client, "install:test")]
    assert agent.model == "new-model"
    assert agent.provider == "new-provider"
    assert agent.requested_provider == "custom:new-provider"
    assert agent.base_url == "https://new.example/v1"
    assert agent.api_mode == "codex_responses"
    assert agent.api_key == "new-key"
    assert agent.client is new_client
    assert agent._anthropic_client is None
    assert agent._client_kwargs["timeout"] == 42.0
    assert agent._transport_cache == {}
    assert agent._resolved_runtime is runtime
    assert agent._cached_system_prompt == "stable prompt"
    assert agent._session_messages == [{"role": "user", "content": "keep me"}]


def test_bundle_build_failure_leaves_live_runtime_untouched():
    events = []
    agent = _bare_agent(events)
    before = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": agent.api_key,
        "client": agent.client,
        "client_kwargs": dict(agent._client_kwargs),
        "prompt": agent._cached_system_prompt,
        "messages": agent._session_messages,
    }
    runtime = ResolvedRuntime.from_mapping(
        {
            "provider": "broken-provider",
            "model": "broken-model",
            "api_mode": "chat_completions",
            "api_key": "broken-key",
            "base_url": "https://broken.example/v1",
        }
    )

    with pytest.raises(RuntimeError, match="build exploded"):
        build_client_bundle(
            runtime,
            openai_builder=lambda _kwargs: (_ for _ in ()).throw(
                RuntimeError("build exploded")
            ),
        )

    assert events == []
    assert agent.model == before["model"]
    assert agent.provider == before["provider"]
    assert agent.requested_provider == before["requested_provider"]
    assert agent.base_url == before["base_url"]
    assert agent.api_mode == before["api_mode"]
    assert agent.api_key == before["api_key"]
    assert agent.client is before["client"]
    assert agent._client_kwargs == before["client_kwargs"]
    assert agent._cached_system_prompt is before["prompt"]
    assert agent._session_messages is before["messages"]
