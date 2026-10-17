"""x-opencode-session rides on every OpenCode request, on every transport."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
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


@pytest.fixture
def out_of_turn():
    """No ambient turn context: runtime binding, conversation root and declared affinity scope all unset.

    Earlier AIAgent-driven tests in this process can leave those contextvars set, which would hand the
    header to an unfixed tree through ``get_conversation_context()`` instead of the explicit runtime."""
    from agent import portal_tags

    tokens = (
        aux._RUNTIME_MAIN_CONTEXT.set(None),
        portal_tags.set_conversation_context(None),
        portal_tags.set_affinity_scope(None),
    )
    try:
        yield
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(tokens[0])
        portal_tags.reset_conversation_context(tokens[1])
        portal_tags.reset_affinity_scope(tokens[2])


_OPENCODE_RUNTIME = {
    "provider": "opencode-zen", "model": "glm-5", "base_url": "https://opencode.ai/zen/v1",
    "api_key": "test-key", "session_id": "sess-affinity-1",
}


def _route_to_fake_opencode_client(monkeypatch, captured, *, async_mode):
    """Pin the aux resolver on an OpenCode route served by a fake SDK client that records its kwargs."""
    if async_mode:
        async def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], model="glm-5")
    else:
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], model="glm-5")
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), base_url="https://opencode.ai/zen/v1",
    )
    monkeypatch.setattr(aux, "_resolve_task_provider_model",
                        lambda *_a, **_k: ("opencode-zen", "glm-5", "https://opencode.ai/zen/v1", "test-key", None))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *_a, **_k: (client, "glm-5"))


def test_sync_out_of_turn_call_binds_the_explicit_main_runtime_session(monkeypatch, out_of_turn):
    """#112717: a background/out-of-turn ``call_llm(main_runtime=...)`` (title, approval, skills hub,
    ``/btw``) must send the conversation's ``x-opencode-session`` exactly like the main turn does."""
    captured = {}
    _route_to_fake_opencode_client(monkeypatch, captured, async_mode=False)

    aux.call_llm(task="title_generation", main_runtime=_OPENCODE_RUNTIME, messages=_MSGS)

    assert captured["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
    assert aux._RUNTIME_MAIN_CONTEXT.get() is None  # the explicit binding does not leak past the call


def test_async_out_of_turn_call_binds_the_explicit_main_runtime_session(monkeypatch, out_of_turn):
    """Same contract on the async twin (#112717)."""
    captured = {}
    _route_to_fake_opencode_client(monkeypatch, captured, async_mode=True)

    asyncio.run(aux.async_call_llm(task="approval", main_runtime=_OPENCODE_RUNTIME, messages=_MSGS))

    assert captured["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
    assert aux._RUNTIME_MAIN_CONTEXT.get() is None


def test_tui_gateway_oneshot_runtime_snapshot_carries_the_session(monkeypatch, out_of_turn):
    """The Desktop/TUI-gateway ``llm.oneshot`` path builds its explicit ``main_runtime`` from the live
    agent; without ``session_id`` an OpenCode title request sends no ``x-opencode-session`` (#112717)."""
    from tui_gateway.server import _main_runtime_from_agent

    agent = SimpleNamespace(
        provider="opencode-zen", model="glm-5", base_url="https://opencode.ai/zen/v1", api_key="test-key",
        api_mode="chat_completions", auth_mode="", session_id="sess-desktop-1",
    )
    captured = {}
    _route_to_fake_opencode_client(monkeypatch, captured, async_mode=False)

    aux.call_llm(task="title_generation", main_runtime=_main_runtime_from_agent(agent), messages=_MSGS)

    assert captured["extra_headers"]["x-opencode-session"] == "sess-desktop-1"


def test_stateless_oneshot_still_sends_an_opencode_session_header(out_of_turn):
    """A one-shot with no live session (Desktop commit-message generation from the review panel with
    no active chat, standalone aux calls) has no conversation identity at all, yet the relay rejects
    header-less requests with 400 MissingSessionID (#105841). It must carry an ephemeral key instead
    of nothing; non-OpenCode targets stay untouched."""
    from agent.opencode_affinity import opencode_session_headers

    kwargs = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1")
    assert kwargs["extra_headers"]["x-opencode-session"]

    assert opencode_session_headers("opencode-go", None, session_id=None).get("x-opencode-session")
    assert opencode_session_headers("openrouter", "https://openrouter.ai/api/v1", session_id=None) == {}
