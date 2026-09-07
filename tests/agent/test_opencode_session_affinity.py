"""x-opencode-session rides on every OpenCode request, on every transport."""

from __future__ import annotations

import pytest

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


def _agent(provider, model, base_url, api_mode=None, requested_provider=None):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="sess-affinity-1",
        requested_provider=requested_provider,
    )
    if api_mode:
        agent.api_mode = api_mode
        agent._transport = None
        agent._anthropic_base_url = base_url
    return agent


@pytest.mark.parametrize(
    "provider, model, base_url, api_mode",
    [
        ("opencode-go", "glm-5", "https://opencode.ai/zen/go/v1", None),  # chat_completions
        ("opencode-go", "gpt-5.6-luna", "https://opencode.ai/zen/go/v1", None),  # codex_responses
        ("opencode-go", "minimax-m2.7", "https://opencode.ai/zen/go/v1", "anthropic_messages"),
        ("opencode-free", "laguna-s-2.1-free", "https://opencode.ai/zen/v1", None),
        ("custom", "glm-5", "https://opencode.ai/zen/go/v1", None),  # URL-only detection
    ],
)
def test_main_turn_sends_stable_session_header_on_every_transport(provider, model, base_url, api_mode):
    agent = _agent(provider, model, base_url, api_mode)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    assert first == second == "sess-affinity-1"

    other = _agent("openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1")
    assert "x-opencode-session" not in (build_api_kwargs(other, _MSGS).get("extra_headers") or {})


@pytest.mark.parametrize("requested_provider", ["opencode-go-proxy", "custom:opencode-go-proxy"])
def test_named_custom_opencode_gateway_uses_requested_provider_identity(
    requested_provider, tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CUSTOM_OPENCODE_API_KEY", "test-key")
    (hermes_home / "config.yaml").write_text(
        """
model:
  provider: custom:opencode-go-proxy
  default: glm-5
providers:
  opencode-go-proxy:
    base_url: https://gateway.example/v1
    key_env: CUSTOM_OPENCODE_API_KEY
    transport: chat_completions
"""
    )

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested=requested_provider, target_model="glm-5")
    assert runtime["provider"] == "custom"
    agent = _agent(
        runtime["provider"], "glm-5", runtime["base_url"],
        requested_provider=runtime["requested_provider"],
    )
    assert build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"] == "sess-affinity-1"


def test_auxiliary_calls_share_the_main_turn_session_key():
    token = aux.set_runtime_main(
        "opencode-go", "glm-5", base_url="https://opencode.ai/zen/go/v1", session_id="sess-affinity-1"
    )
    try:
        kwargs = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1")
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
        other = aux._build_call_kwargs("openrouter", "x", _MSGS, base_url="https://openrouter.ai/api/v1")
        assert "x-opencode-session" not in (other.get("extra_headers") or {})
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_auxiliary_calls_inherit_named_custom_opencode_identity_only_on_main_route():
    token = aux.set_runtime_main(
        "custom", "glm-5",
        requested_provider="custom:opencode-go-proxy",
        base_url="https://gateway.example/v1",
        session_id="sess-affinity-1",
    )
    try:
        kwargs = aux._build_call_kwargs(
            "custom", "glm-5", _MSGS, base_url="https://gateway.example/v1"
        )
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-affinity-1"

        fallback = aux._build_call_kwargs(
            "custom", "glm-5", _MSGS, base_url="https://fallback.example/v1"
        )
        assert "x-opencode-session" not in (fallback.get("extra_headers") or {})
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)
