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


def test_auxiliary_thread_reuses_main_turn_session_key(out_of_turn, monkeypatch):
    """#115000: auxiliary calls (goal judge, compression, vision, etc.) run on threads whose
    contextvars are empty, so ``resolve_affinity_key`` returns "" and the relay-affinity header
    would either fall back to a fresh ``oneshot-`` (losing backend cache locality) or, when the
    caller path is the legacy ``merge_opencode_session_headers`` named in the issue, omit the
    header entirely and trip HTTP 400 ``MissingSessionID`` on OpenCode Go. The main turn's
    resolved key must be reused for any later OpenCode request — even when the caller has no
    ambient context — so the goal judge etc. land on the conversation's warm backend."""
    from agent import opencode_affinity, portal_tags
    from agent.opencode_affinity import opencode_session_headers

    # Reset the per-profile fallback cache so this test starts with no leftover state.
    monkeypatch.setattr(opencode_affinity, "_LAST_OPENCODE_SESSION_KEYS", {}, raising=False)

    base_url = "https://opencode.ai/zen/go/v1"

    # 1. Main turn: set the conversation context and resolve the header.
    main_token = portal_tags.set_conversation_context("conv-affinity-115000")
    try:
        main_header = opencode_session_headers("opencode-go", base_url, session_id=None)
    finally:
        portal_tags.reset_conversation_context(main_token)
    assert main_header["x-opencode-session"] == "conv-affinity-115000"

    # 2. Auxiliary thread (empty contextvars): must reuse the main turn's key, not a
    #    fresh ``oneshot-`` ephemeral, and must NOT return the empty-dict legacy path.
    #    ``reuse_cached_key`` is what a conversation-bound chain (the goal judge) passes —
    #    a stateless caller must not get this key (#131047).
    aux_header = opencode_session_headers("opencode-go", base_url, session_id=None, reuse_cached_key=True)
    assert aux_header["x-opencode-session"] == "conv-affinity-115000", (
        f"auxiliary thread lost the conversation key; got {aux_header!r}"
    )
    assert not aux_header["x-opencode-session"].startswith("oneshot-"), (
        "fallback path generated a fresh oneshot- instead of reusing the main turn key"
    )

    # 3. Explicit ``session_id`` on an auxiliary call still wins over the cache (the cache
    #    only fills gaps), so an external caller pinning a key is preserved.
    aux_pinned = opencode_session_headers("opencode-go", base_url, session_id="pinned-sess")
    assert aux_pinned["x-opencode-session"] == "pinned-sess"

    # 4. Non-OpenCode targets stay untouched regardless of cache state.
    assert opencode_session_headers("openrouter", "https://openrouter.ai/api/v1", session_id=None) == {}


def test_first_opencode_call_without_any_main_key_still_sends_a_header(out_of_turn, monkeypatch):
    """The legacy ``oneshot-`` fallback (#105841) must be preserved for the truly stateless
    case (no main turn has ever resolved a key). A brand-new process with no conversation
    context and no cached key still gets a non-empty ``x-opencode-session`` so OpenCode Go
    does not 400; non-OpenCode targets still get ``{}`` (#115000)."""
    from agent import opencode_affinity
    from agent.opencode_affinity import opencode_session_headers

    # Reset every source of state this function reads so the test is hermetic regardless of
    # which other tests in this file ran first.
    monkeypatch.setattr(opencode_affinity, "_LAST_OPENCODE_SESSION_KEYS", {}, raising=False)

    base_url = "https://opencode.ai/zen/go/v1"
    header = opencode_session_headers("opencode-go", base_url, session_id=None)
    assert header.get("x-opencode-session")
    assert header["x-opencode-session"].startswith("oneshot-")

    assert opencode_session_headers("openrouter", "https://openrouter.ai/api/v1", session_id=None) == {}


def test_cached_key_never_crosses_profiles(out_of_turn, monkeypatch):
    """Root ``AGENTS.md`` § Code Shape Rules: a module global holds the *launch* profile's state, so
    an unbound read is a silent default-profile leak. The fallback cache is slotted by
    ``hermes_home_key()``: profile B's off-context auxiliary call must not ride profile A's
    conversation key — it keeps its own ``oneshot-`` until B's own main turn resolves one."""
    import hermes_constants

    from agent import opencode_affinity, portal_tags
    from agent.opencode_affinity import opencode_session_headers

    monkeypatch.setattr(opencode_affinity, "_LAST_OPENCODE_SESSION_KEYS", {}, raising=False)
    home = {"v": "home-A"}
    monkeypatch.setattr(hermes_constants, "hermes_home_key", lambda *a, **k: home["v"])
    base_url = "https://opencode.ai/zen/go/v1"

    token = portal_tags.set_conversation_context("conv-profile-A")
    try:
        assert opencode_session_headers(
            "opencode-go", base_url, reuse_cached_key=True)["x-opencode-session"] == "conv-profile-A"
    finally:
        portal_tags.reset_conversation_context(token)

    home["v"] = "home-B"
    header = opencode_session_headers("opencode-go", base_url, reuse_cached_key=True)
    assert header["x-opencode-session"].startswith("oneshot-"), (
        f"profile B inherited profile A's conversation key: {header!r}"
    )


def test_stateless_call_after_a_main_turn_keeps_its_own_oneshot_key(out_of_turn, monkeypatch):
    """#131047 review: the cached last-main-turn key may only be borrowed by a chain that is
    conversation-bound by construction. A stateless call (commit-message generation, cron summary,
    standalone prompt) has no conversation identity at all, so it must keep a fresh ``oneshot-`` key
    per call instead of riding the previous conversation's backend — while the goal judge, which
    inherited no runtime context (#115000), still reaches that same backend."""
    from agent import opencode_affinity, portal_tags

    monkeypatch.setattr(opencode_affinity, "_LAST_OPENCODE_SESSION_KEYS", {}, raising=False)
    base_url = "https://opencode.ai/zen/go/v1"

    # 1. A real main turn resolves the conversation key and leaves it cached.
    token = portal_tags.set_conversation_context("conv-real")
    try:
        main = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url=base_url)
    finally:
        portal_tags.reset_conversation_context(token)
    assert main["extra_headers"]["x-opencode-session"] == "conv-real"

    # 2. Stateless: no explicit runtime, no conversation-bound task -> a fresh key per call.
    keys = tuple(
        aux._build_call_kwargs(
            "opencode-go", "glm-5", _MSGS, base_url=base_url, task="title_generation",
        )["extra_headers"]["x-opencode-session"]
        for _ in range(2)
    )
    assert all(k.startswith("oneshot-") for k in keys), keys
    assert keys[0] != keys[1], f"stateless calls reused one key: {keys!r}"

    # 3. The goal judge is off-context by construction and still borrows the conversation's backend.
    judge = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url=base_url, task="goal_judge")
    assert judge["extra_headers"]["x-opencode-session"] == "conv-real"
