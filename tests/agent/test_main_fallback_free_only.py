"""Regression tests: ``auxiliary.free_only`` must not veto the chat-lane fallback (#75803).

Incident (2026-11-14, kevin-core): the Codex subscription answered HTTP 429 "usage limit".
The configured ``fallback_providers`` entry ``openrouter / deepseek/deepseek-v4.1-flash``
was then rejected by ``auxiliary.free_only`` — an *auxiliary* spend policy consulted on the
chat lane. ``bind_route_entry`` received ``(None, None)``, logged "provider not configured",
marked the entry unusable, and the turn ended on the primary provider instead of continuing.

The fix separates the lanes through the existing task context: the chat lane marks its
``resolve_provider_client`` call with ``task=MAIN_FALLBACK_TASK``, which the resolver strips
before any task-specific routing sees it. Auxiliary calls keep the gate, so paid auxiliary
calls stay blocked and the cost guard is preserved.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    MAIN_FALLBACK_TASK,
    _resolve_strict_vision_backend,
    _try_openrouter,
    resolve_provider_client,
)

PAID_MODEL = "deepseek/deepseek-v4.1-flash"


# ── auxiliary lane: the gate must stay closed ────────────────────────────


def test_auxiliary_call_without_task_is_gated(free_only_without_client):
    """No lane marker → auxiliary lane → free_only keeps rejecting a paid model."""
    assert _try_openrouter() == (None, None)


def test_auxiliary_call_with_task_is_gated(free_only_without_client):
    """A named auxiliary task keeps the gate; the diagnostic names the task."""
    assert _try_openrouter(task="compression") == (None, None)


def test_vision_backend_keeps_the_auxiliary_gate(free_only_without_client):
    """Vision is an auxiliary task, so a paid OpenRouter vision model stays blocked."""
    client, _model = _resolve_strict_vision_backend("openrouter")
    assert client is None


# ── chat lane: the configured fallback must resolve ──────────────────────


def test_chat_lane_marker_allows_the_configured_paid_model(free_only_with_client):
    client, model = _try_openrouter(main_fallback=True)

    assert client is free_only_with_client
    assert model == PAID_MODEL


def test_resolve_provider_client_without_marker_reproduces_the_incident(
        free_only_without_client):
    """The reported failure: identical provider/model/free_only, but chat lane unmarked."""
    assert resolve_provider_client("openrouter", model=PAID_MODEL) == (None, None)


def test_resolve_provider_client_with_marker_resolves(free_only_with_client):
    client, model = resolve_provider_client("openrouter", model=PAID_MODEL,
                                            task=MAIN_FALLBACK_TASK)

    assert client is free_only_with_client
    assert model == PAID_MODEL


def test_marker_is_stripped_before_task_specific_routing(monkeypatch):
    """The lane marker must never reach task routing, cache keys or auto-discovery policy."""
    captured = {}

    def _capture(req):
        captured["task"] = req.task
        captured["main_fallback"] = req.main_fallback
        return None, None

    monkeypatch.setitem(aux._EXPLICIT_PROVIDER_BRANCHES, "openrouter", _capture)

    resolve_provider_client("openrouter", model=PAID_MODEL, task=MAIN_FALLBACK_TASK)

    assert captured == {"task": None, "main_fallback": True}


def test_free_only_false_is_unaffected(monkeypatch, openrouter_client):
    """No free_only policy configured → auxiliary calls resolve as before (no regression)."""
    monkeypatch.setattr(aux, "_aux_openrouter_settings", lambda: (False, PAID_MODEL))

    client, model = _try_openrouter(task="compression")

    assert client is openrouter_client
    assert model == PAID_MODEL


# ── the 429 switch and the continuation of the running turn ──────────────


def test_429_switch_binds_openrouter_and_continues(monkeypatch, free_only_with_client):
    """End-to-end: a rate-limit failover switches onto the configured fallback and stays there,
    and the swap reaches the resolver as the chat lane rather than as an auxiliary task."""
    from run_agent import AIAgent
    from agent.error_classifier import FailoverReason

    resolved = []
    real_resolve = aux.resolve_provider_client

    def _recording_resolve(provider, **kwargs):
        resolved.append((provider, kwargs.get("task")))
        return real_resolve(provider, **kwargs)

    monkeypatch.setattr(aux, "resolve_provider_client", _recording_resolve)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url="https://chatgpt.com/backend-api/codex",
            provider="openai-codex",
            model="gpt-5.6-sol",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "openrouter", "model": PAID_MODEL, "api_key": "or-key"},
        )
    agent.client = MagicMock()

    switched = agent._try_activate_fallback(FailoverReason.rate_limit)

    assert switched is True, "the chat lane must not be blocked by auxiliary.free_only"
    assert agent.provider == "openrouter"
    assert agent.model == PAID_MODEL
    assert agent._provider_fallback_active is True
    assert ("openrouter", MAIN_FALLBACK_TASK) in resolved



# ── integration: a 429 inside an already running turn ────────────────────


def _simulated_429():
    """The failure shape the primary OAuth provider produced: HTTP 429 usage limit."""
    from types import SimpleNamespace

    error = Exception("Error code: 429 - usage limit reached")
    error.status_code = 429
    error.response = SimpleNamespace(headers={}, status_code=429)
    error.body = {"error": {"message": "usage limit reached"}}
    return error


def _agent_with_openrouter_fallback(fallback_chain=None):
    """A running agent whose primary is the Codex OAuth route and whose chain is openrouter."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url="https://chatgpt.com/backend-api/codex",
            provider="openai-codex",
            model="gpt-5.6-sol",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_chain or {
                "provider": "openrouter", "model": PAID_MODEL, "api_key": "or-key"},
        )
    agent.client = MagicMock(name="codex_client")
    return agent


def _run_turn_error_handler(agent, *, retry_count, max_retries, api_error=None,
                            messages=None, api_messages=None):
    """Drive the real turn error handler instead of the fallback primitive."""
    from agent.turn_api_error import handle_api_error
    from agent.turn_retry_state import TurnRetryState

    return handle_api_error(
        agent,
        api_error=api_error or _simulated_429(),
        _retry=TurnRetryState(),
        thinking_spinner=None,
        messages=messages if messages is not None else [{"role": "user", "content": "repariere bitte die Sicherung"}],
        api_messages=api_messages if api_messages is not None else [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "repariere bitte die Sicherung"},
        ],
        api_kwargs={"model": agent.model, "messages": api_messages or []},
        system_message="sys",
        active_system_prompt="sys",
        conversation_history=[],
        approx_tokens=0,
        retry_count=retry_count,
        max_retries=max_retries,
        compression_attempts=0,
        max_compression_attempts=1,
        api_call_count=1,
        api_request_id="req-1",
        api_start_time=0.0,
        effective_task_id="task-1",
        turn_id="turn-1",
    )


def test_turn_continues_after_429_without_a_new_user_request(free_only_with_client):
    """The running job survives the primary 429: the same turn continues on OpenRouter."""
    from types import SimpleNamespace

    agent = _agent_with_openrouter_fallback()
    codex_client = agent.client
    codex_client.chat.completions.create = MagicMock(side_effect=_simulated_429())
    messages = [{"role": "user", "content": "repariere bitte die Sicherung"}]
    api_messages = [{"role": "system", "content": "sys"}] + messages
    before = list(messages)

    # The turn's first attempt fails on the primary exactly once.
    with pytest.raises(Exception):
        codex_client.chat.completions.create(model="gpt-5.6-sol", messages=api_messages)
    assert codex_client.chat.completions.create.call_count == 1

    verdict = _run_turn_error_handler(agent, retry_count=3, max_retries=3,
                                      messages=messages, api_messages=api_messages)

    # "break" is the loop's "fallback bound, rebuild and retry the same turn" signal.
    assert verdict.action == "break"
    assert agent.provider == "openrouter"
    assert agent.model == PAID_MODEL
    assert agent._provider_fallback_active is True
    # Continuation, not a new request: the conversation is untouched and retries restarted.
    assert messages == before
    assert verdict.retry_count == 0
    assert codex_client.chat.completions.create.call_count == 1, "Codex must not be retried after the switch"

    # The next attempt of the same turn is served by the fallback client.
    agent.client.chat.completions.create = MagicMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Sicherung repariert."))],
    ))
    response = agent.client.chat.completions.create(model=agent.model, messages=api_messages)

    assert response.choices[0].message.content == "Sicherung repariert."
    assert agent.client is free_only_with_client, "the bound client must be the fallback one"


def test_no_repeated_primary_request_after_the_switch(free_only_with_client):
    """A second 429 must not send the turn back to Codex: no loop, no repeated primary call."""
    agent = _agent_with_openrouter_fallback()
    codex_client = agent.client
    codex_client.chat.completions.create = MagicMock(side_effect=_simulated_429())
    with pytest.raises(Exception):
        codex_client.chat.completions.create(model="gpt-5.6-sol", messages=[])
    assert codex_client.chat.completions.create.call_count == 1

    first = _run_turn_error_handler(agent, retry_count=3, max_retries=3)
    assert first.action == "break"
    assert codex_client.chat.completions.create.call_count == 1  # the turn never re-asks Codex

    # The fallback provider now fails too: the turn must end, not bounce back to the primary.
    second = _run_turn_error_handler(agent, retry_count=3, max_retries=3)

    assert second.action == "return"
    assert codex_client.chat.completions.create.call_count == 1
    assert agent.provider == "openrouter"


def test_missing_or_unknown_task_context_keeps_the_cost_gate(free_only_without_client):
    """No task, empty task and an unknown task name all stay on the gated auxiliary lane."""
    assert _try_openrouter() == (None, None)
    assert _try_openrouter(task="") == (None, None)
    assert _try_openrouter(task="some_unknown_task") == (None, None)
    assert resolve_provider_client("openrouter", model=PAID_MODEL) == (None, None)
    assert resolve_provider_client("openrouter", model=PAID_MODEL, task="") == (None, None)



# ── streaming, multi-step chains, reset window, full turn ────────────────


def _chat_response(content):
    from types import SimpleNamespace

    return SimpleNamespace(
        id="resp-1",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )


def test_streaming_branch_raises_into_the_same_fallback(monkeypatch, free_only_with_client):
    """A streaming attempt that fails with 429 must reach the same chat-lane fallback."""
    from agent.turn_api_call import perform_api_call
    from agent.turn_retry_state import TurnRetryState

    agent = _agent_with_openrouter_fallback()
    agent.api_mode = "chat_completions"  # the mocked transport speaks chat completions
    error = _simulated_429()
    agent._interruptible_streaming_api_call = MagicMock(side_effect=error)
    # Force the streaming branch; the transport is mocked, so no bytes ever move.
    monkeypatch.setattr("agent.turn_api_call._should_stream", lambda _agent: True)

    with pytest.raises(Exception) as raised:
        perform_api_call(
            agent,
            api_kwargs={"model": agent.model, "messages": [], "stream": True},
            _original_api_kwargs={},
            _llm_middleware_trace={},
            _moa_prepared_request=None,
            _retry=TurnRetryState(),
            thinking_spinner=None,
            retry_count=3,
            api_call_count=1,
            api_request_id="req-stream",
            effective_task_id="task-1",
            turn_id="turn-1",
            interrupted=None,
        )

    assert raised.value is error
    assert agent._interruptible_streaming_api_call.call_count == 1

    verdict = _run_turn_error_handler(agent, retry_count=3, max_retries=3, api_error=error)

    assert verdict.action == "break"
    assert agent.provider == "openrouter"
    assert agent._has_pending_fallback() is False


def test_multi_step_chain_advances_then_terminates(monkeypatch, free_only_with_client):
    """Two-entry chain: openrouter first, a custom relay second, then a controlled end."""
    from types import SimpleNamespace

    clients = {}

    def _client_for(**kwargs):
        base_url = kwargs.get("base_url") or "https://openrouter.ai/api/v1"
        client = clients.setdefault(base_url, MagicMock(name=f"client for {base_url}"))
        client.base_url = base_url
        client.chat.completions.create = MagicMock(return_value=_chat_response("ok"))
        return client

    monkeypatch.setattr(aux, "OpenAI", _client_for)
    monkeypatch.setattr(aux, "_aux_openrouter_settings", lambda: (True, PAID_MODEL))
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a, **k: (False, None))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    agent = _agent_with_openrouter_fallback(
        fallback_chain=[
            {"provider": "openrouter", "model": PAID_MODEL, "api_key": "or-key"},
            {"provider": "custom", "model": "relay-model", "base_url": "https://relay.example.com/v1",
             "api_key": "relay-key"},
        ]
    )

    first = _run_turn_error_handler(agent, retry_count=3, max_retries=3)
    assert first.action == "break"
    assert agent.provider == "openrouter"

    second = _run_turn_error_handler(agent, retry_count=3, max_retries=3)
    assert second.action == "break"
    assert agent.provider == "custom", "the chain must advance to the second entry"
    assert agent.base_url.rstrip("/") == "https://relay.example.com/v1"

    third = _run_turn_error_handler(agent, retry_count=3, max_retries=3)
    assert third.action == "return", "an exhausted chain ends the turn instead of looping"


def test_reset_window_defers_the_switch_and_ends_the_turn(monkeypatch, free_only_without_client):
    """fallback.min_switch_reset_seconds: a near reset stays on the primary, no silent hang."""
    import time as _time

    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda *a, **k: {"fallback": {"min_switch_reset_seconds": 600}})
    agent = _agent_with_openrouter_fallback()
    monkeypatch.setattr(agent, "_extract_api_error_context",
                        lambda error: {"reset_at": _time.time() + 60})

    verdict = _run_turn_error_handler(agent, retry_count=3, max_retries=3)

    assert verdict.action == "return", "the turn must end with the reset information"
    assert agent.provider == "openai-codex", "no switch inside the deferral window"
    assert agent._fallback_activated is False
    # The cost guard is untouched by the deferral path: no paid auxiliary client was built.
    assert _try_openrouter(task="compression") == (None, None)
    assert _try_openrouter() == (None, None)


def test_full_turn_continues_after_429(monkeypatch, free_only_with_client):
    """The whole run_conversation path: primary 429, fallback answers, the turn completes."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url="https://api.deepinfra.com/v1/openai",
            provider="deepinfra",
            model="deepseek-ai/DeepSeek-V4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "openrouter", "model": PAID_MODEL, "api_key": "or-key"},
        )
    agent._disable_streaming = True
    primary_client = MagicMock(name="primary_client")
    primary_client.base_url = "https://api.deepinfra.com/v1/openai"
    primary_client.chat.completions.create = MagicMock(side_effect=_simulated_429())
    agent.client = primary_client
    free_only_with_client.chat.completions.create = MagicMock(
        return_value=_chat_response("Sicherung repariert."))

    monkeypatch.setattr("agent.turn_api_error.compute_error_backoff", lambda *a, **k: 0)
    monkeypatch.setattr("agent.turn_api_error.time.sleep", lambda *a, **k: None)

    with (
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.compression_enabled = False
        agent.save_trajectories = False
        result = agent.run_conversation("repariere bitte die Sicherung")

    assert "Sicherung repariert." in str(result)
    assert agent.provider == "openrouter"
    assert primary_client.chat.completions.create.call_count == 1


# ── fixtures ─────────────────────────────────────────────────────────────


def _free_only_settings():
    return True, PAID_MODEL


def _install_openrouter_client(monkeypatch):
    """Patch the OpenRouter lane so a resolved client is returned without any network call."""
    client = MagicMock(name="openrouter_client")
    client.base_url = "https://openrouter.ai/api/v1"
    monkeypatch.setattr(aux, "_aux_openrouter_settings", _free_only_settings)
    monkeypatch.setattr(aux, "OpenAI", lambda *a, **k: client)
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a, **k: (False, None))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    return client


@pytest.fixture
def free_only_without_client(monkeypatch):
    """free_only configured; building an OpenRouter client here is a hard failure."""
    monkeypatch.setattr(aux, "_aux_openrouter_settings", _free_only_settings)
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a, **k: (False, None))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    forbidden = MagicMock(name="must_not_be_built")
    monkeypatch.setattr(aux, "OpenAI", forbidden)
    yield forbidden
    assert forbidden.call_count == 0, "the free_only gate must reject before any client is built"


@pytest.fixture
def free_only_with_client(monkeypatch):
    """free_only configured, but the chat lane may build the paid client."""
    client = _install_openrouter_client(monkeypatch)
    yield client


@pytest.fixture
def openrouter_client(monkeypatch):
    """No free_only policy; a client is built and returned."""
    client = _install_openrouter_client(monkeypatch)
    yield client