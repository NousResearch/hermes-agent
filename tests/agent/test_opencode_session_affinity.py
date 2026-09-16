"""x-opencode-session rides on every OpenCode request, on every transport."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import title_generator, turn_context
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


def _agent(provider, model, base_url, api_mode=None):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="sess-affinity-1",
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
        ("opencode-free", "nemotron-3.5-lightning-free", "https://opencode.ai/zen/v1", None),
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


def test_auto_title_uses_explicit_main_runtime_session(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                model="glm-5",
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
        base_url="https://opencode.ai/zen/v1",
    )
    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: (
            "opencode-zen", "glm-5", "https://opencode.ai/zen/v1", "test-key", None,
        ),
    )
    monkeypatch.setattr(aux, "_get_cached_client", lambda *_args, **_kwargs: (client, "glm-5"))
    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "_kanban_task_title", lambda: None)
    monkeypatch.setattr(title_generator, "apply_instant_title", lambda *_args, **_kwargs: None)

    def spawn_immediately(target, *, name, args, kwargs):
        def start():
            token = aux._RUNTIME_MAIN_CONTEXT.set(None)
            try:
                target(*args, **kwargs)
            finally:
                aux._RUNTIME_MAIN_CONTEXT.reset(token)

        return SimpleNamespace(start=start)

    monkeypatch.setattr("agent.memory_provider.spawn_context_thread", spawn_immediately)

    session_db = SimpleNamespace(
        get_session_title_source=lambda _session_id: None,
        get_conversation_root=lambda session_id: session_id,
        set_auto_title=lambda *_args, **_kwargs: True,
    )
    agent = _agent("opencode-zen", "glm-5", "https://opencode.ai/zen/v1")
    agent._session_db = session_db
    agent._session_db_created = True

    turn_context._maybe_title_session_at_turn_start(agent, _MSGS)

    assert captured["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
