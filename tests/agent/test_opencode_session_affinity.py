"""x-opencode-session rides on every OpenCode request, on every transport."""

from __future__ import annotations

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
        (
            "opencode-go",
            "glm-5",
            "https://opencode.ai/zen/go/v1",
            None,
        ),  # chat_completions
        (
            "opencode-go",
            "gpt-5.6-luna",
            "https://opencode.ai/zen/go/v1",
            None,
        ),  # codex_responses
        (
            "opencode-go",
            "minimax-m2.7",
            "https://opencode.ai/zen/go/v1",
            "anthropic_messages",
        ),
        ("opencode-free", "laguna-s-2.1-free", "https://opencode.ai/zen/v1", None),
        (
            "custom",
            "glm-5",
            "https://opencode.ai/zen/go/v1",
            None,
        ),  # URL-only detection
    ],
)
def test_main_turn_sends_stable_session_header_on_every_transport(
    provider, model, base_url, api_mode
):
    agent = _agent(provider, model, base_url, api_mode)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    assert first == second == "sess-affinity-1"

    other = _agent(
        "openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1"
    )
    assert "x-opencode-session" not in (
        build_api_kwargs(other, _MSGS).get("extra_headers") or {}
    )


def test_auxiliary_calls_share_the_main_turn_session_key():
    token = aux.set_runtime_main(
        "opencode-go",
        "glm-5",
        base_url="https://opencode.ai/zen/go/v1",
        session_id="sess-affinity-1",
    )
    try:
        kwargs = aux._build_call_kwargs(
            "opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1"
        )
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
        other = aux._build_call_kwargs(
            "openrouter", "x", _MSGS, base_url="https://openrouter.ai/api/v1"
        )
        assert "x-opencode-session" not in (other.get("extra_headers") or {})
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_out_of_turn_aux_calls_get_stable_fallback_key(monkeypatch):
    """No turn scope and no readable session_id → header still sent with a stable key (#105011)."""
    from agent import opencode_affinity
    from agent import portal_tags

    monkeypatch.setattr(portal_tags, "get_affinity_scope", lambda: "")
    monkeypatch.setattr(portal_tags, "get_conversation_context", lambda: "")
    monkeypatch.setattr(opencode_affinity, "_FALLBACK_SESSION_KEY", None)

    first = opencode_affinity.opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", None
    )
    second = opencode_affinity.opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", None
    )
    assert first == second
    assert first["x-opencode-session"].startswith("hermes-aux-")

    # Explicit session_id still wins over the fallback.
    pinned = opencode_affinity.opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", "sess-1"
    )
    assert pinned["x-opencode-session"] == "sess-1"

    # Non-OpenCode targets stay untouched.
    assert (
        opencode_affinity.opencode_session_headers(
            "openrouter", "https://openrouter.ai/api/v1", None
        )
        == {}
    )


def test_fallback_key_is_per_install(monkeypatch):
    from agent import opencode_affinity

    monkeypatch.setattr(opencode_affinity, "_FALLBACK_SESSION_KEY", None)
    monkeypatch.setenv("HERMES_HOME", "/tmp/install-a")
    key_a = opencode_affinity._fallback_session_key()
    monkeypatch.setattr(opencode_affinity, "_FALLBACK_SESSION_KEY", None)
    monkeypatch.setenv("HERMES_HOME", "/tmp/install-b")
    key_b = opencode_affinity._fallback_session_key()
    assert key_a != key_b
    assert key_a.startswith("hermes-aux-") and key_b.startswith("hermes-aux-")


def test_aux_call_from_plain_thread_carries_the_header():
    """HTTP-handler threads (kanban Specify/Decompose) have no turn scope — the built
    kwargs must still carry the affinity header instead of omitting it."""
    import threading

    from agent import auxiliary_client as aux

    result = {}

    def run():
        kwargs = aux._build_call_kwargs(
            "opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1"
        )
        result["key"] = (kwargs.get("extra_headers") or {}).get(
            "x-opencode-session", ""
        )

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert result["key"].startswith("hermes-aux-")
