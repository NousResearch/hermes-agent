"""A ``503 Service Unavailable`` that carries ``Retry-After`` is the server telling
the client WHEN to come back (RFC 9110 §15.6.4): a gateway draining for a
deploy, a maintenance window, a restarting backend. It is provider-wide (every
model behind that endpoint answers the same 503) and self-clearing.

Contract:
  * a 503 + Retry-After is waited out on the SAME provider/model inside a
    wall-clock budget (``agent.unavailable_wait_seconds``), honouring the
    header, and the waits do not consume the generic retry attempts;
  * once the budget is spent the fallback chain runs, but it SKIPS entries on
    the failing provider (same ``base_url``, or same provider name when either
    side has no base_url): another model behind the same 503 only collects it;
  * a server asking for longer than the remaining budget fails over at once;
  * a 503 WITHOUT Retry-After, and the budget set to 0, keep the old policy.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent

BASE = "https://gateway.example.com/v1"
OTHER_BASE = "https://openrouter.ai/api/v1"


def _unavailable(retry_after: str | None = "0.05", status: int = 503) -> openai.APIStatusError:
    req = httpx.Request("POST", f"{BASE}/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    body = {"error": {"message": "Service Unavailable", "type": "service_unavailable"}}
    resp = httpx.Response(status, request=req, json=body, headers=headers)
    return openai.InternalServerError(
        f"Error code: {status} - {body}", response=resp, body=body["error"],
    )


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="m", usage=None)


FB_CHAIN = [
    # Same endpoint, different model: a MODEL fallback behind the same 503.
    {"provider": "custom", "model": "model-b", "base_url": BASE},
    {"provider": "openrouter", "model": "fallback-model", "base_url": OTHER_BASE},
]


def _make_agent(chain=FB_CHAIN):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url=BASE,
            provider="custom",
            model="model-a",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=list(chain),
        )
    agent.client = MagicMock()
    agent._api_max_retries = 3
    return agent


def _run(agent, fake_api_call, *, wait_s):
    fb_client = MagicMock()
    fb_client.api_key = "primary-key-abcdef12"
    fb_client._custom_headers = None
    fb_client.default_headers = None

    def _resolve(provider, model=None, **kw):
        fb_client.base_url = kw.get("explicit_base_url") or BASE
        return fb_client, model

    agent._unavailable_wait_s = wait_s
    activate = MagicMock(wraps=agent._try_activate_fallback)
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_try_activate_fallback", activate),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.retry_utils.jittered_backoff", return_value=0.01),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("hello")
    return result, activate


def test_503_retry_after_classifies_server_unavailable_retryable():
    r = classify_api_error(_unavailable(), provider="custom", model="model-a", base_url=BASE)
    assert r.reason is FailoverReason.overloaded
    assert r.retryable is True


def test_503_retry_after_that_clears_inside_budget_stays_on_same_model():
    """More 503s than the generic 3-attempt budget, then the server is back."""
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) <= 4:
            raise _unavailable()
        return _mock_response("served by model-a")

    result, activate = _run(agent, fake_api_call, wait_s=30.0)
    assert calls == [("custom", "model-a")] * 5
    assert result["final_response"] == "served by model-a"
    activate.assert_not_called()


def test_503_outliving_budget_skips_same_provider_fallback():
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.provider == "custom":
            raise _unavailable()
        return _mock_response("served by openrouter")

    result, activate = _run(agent, fake_api_call, wait_s=0.3)
    assert ("custom", "model-b") not in calls
    assert calls[-1] == ("openrouter", "fallback-model")
    assert result["final_response"] == "served by openrouter"


def test_retry_after_longer_than_budget_fails_over_at_once_off_the_endpoint():
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.provider == "custom":
            raise _unavailable(retry_after="5")
        return _mock_response("served by openrouter")

    # The server asks for 5s, only 0.5s of budget: fail over now, not after sleeping.
    result, _ = _run(agent, fake_api_call, wait_s=0.5)
    assert calls == [("custom", "model-a"), ("openrouter", "fallback-model")]
    assert result["final_response"] == "served by openrouter"


@pytest.mark.parametrize("retry_after,wait_s", [(None, 120.0), ("0.05", 0.0)])
def test_503_without_retry_after_or_with_wait_disabled_keeps_old_policy(retry_after, wait_s):
    """No header (or knob 0): one retry, then the chain's first entry, same endpoint or not."""
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.model == "model-a":
            raise _unavailable(retry_after=retry_after)
        return _mock_response("served by model-b")

    result, _ = _run(agent, fake_api_call, wait_s=wait_s)
    assert calls == [("custom", "model-a"), ("custom", "model-a"), ("custom", "model-b")]
    assert result["final_response"] == "served by model-b"


def test_non_503_5xx_with_retry_after_is_not_waited_out():
    agent = _make_agent()
    calls = []

    def fake_api_call(api_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.model == "model-a":
            raise _unavailable(status=502)
        return _mock_response("served by model-b")

    _run(agent, fake_api_call, wait_s=120.0)
    assert ("custom", "model-b") in calls
    assert calls.count(("custom", "model-a")) <= 3


@pytest.mark.parametrize("retry_after,waited,budget,expected", [
    (15.0, 0.0, 120.0, 15.0),
    (15.0, 100.0, 120.0, 15.0),
    (15.0, 110.0, 120.0, None),   # would overrun the budget: fail over now
    (15.0, 120.0, 120.0, None),
    (None, 0.0, 120.0, None),
    (0.0, 0.0, 120.0, None),
    (15.0, 0.0, 0.0, None),
])
def test_unavailable_retry_wait(retry_after, waited, budget, expected):
    from agent.retry_utils import unavailable_retry_wait

    assert unavailable_retry_wait(retry_after, waited_s=waited, budget_s=budget) == expected


def test_endpoint_scope_skips_sibling_models_on_the_same_endpoint():
    from agent.backend_identity import BackendIdentity, FailureScope, should_skip_candidate

    failed = BackendIdentity.build(provider="custom", model="model-a", base_url=BASE)
    sibling = BackendIdentity.build(provider="custom", model="model-b", base_url=BASE)
    elsewhere = BackendIdentity.build(provider="openrouter", model="model-b", base_url=OTHER_BASE)
    assert not should_skip_candidate(sibling, failed, FailureScope.MODEL)
    assert should_skip_candidate(sibling, failed, FailureScope.ENDPOINT)
    assert not should_skip_candidate(elsewhere, failed, FailureScope.ENDPOINT)


def test_unavailable_wait_seconds_default_is_in_config_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["unavailable_wait_seconds"] == 120
