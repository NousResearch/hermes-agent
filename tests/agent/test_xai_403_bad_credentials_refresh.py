"""xAI reports an expired OAuth access token as HTTP 403 with body code
``unauthenticated:bad-credentials`` (not 401). The 401-only refresh trigger
never fired for it, so a long-lived worker kept its dead in-memory token and
aborted every turn (#82052). These tests pin the 403 variant: bad-credentials
triggers one forced refresh plus retry, while an unrelated 403 still aborts
without touching credentials.
"""
import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


@pytest.fixture(autouse=True)
def _no_codex_backoff(monkeypatch):
    """Short-circuit retry backoff so these tests don't block on real
    wall-clock waits (same guard as test_run_agent_codex_responses.py)."""
    import time as _time
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


class _XaiBadCredentials403(Exception):
    status_code = 403

    def __init__(self):
        super().__init__(
            'HTTP 403: {"code":"unauthenticated:bad-credentials",'
            '"error":"The OAuth2 access token could not be validated."}'
        )


class _UnrelatedForbidden403(Exception):
    status_code = 403

    def __init__(self):
        super().__init__('HTTP 403: {"error":"forbidden by policy"}')


class _XaiSpendingLimit403(Exception):
    """xAI's other 403: a billing wall that no refresh can clear."""
    status_code = 403

    def __init__(self):
        super().__init__(
            'HTTP 403: {"code":"personal-team-blocked:spending-limit",'
            '"error":"Your team has reached its spending limit."}'
        )


class _XaiBadCredentialsBody403(Exception):
    """SDK variants attach the parsed body; str(exc) still carries the
    marker, which is what the narrow match keys on."""
    status_code = 403

    def __init__(self):
        body = {
            "code": "unauthenticated:bad-credentials",
            "error": "The OAuth2 access token could not be validated.",
        }
        self.body = body
        super().__init__(f"HTTP 403: {body}")


def _codex_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="completed",
        model="grok-4.5",
    )


def _build_xai_agent(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

    agent = run_agent.AIAgent(
        model="grok-4.5",
        provider="xai-oauth",
        api_mode="codex_responses",
        base_url="https://api.x.ai/v1",
        api_key="stale",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_xai_403_bad_credentials_refreshes_and_retries(monkeypatch):
    agent = _build_xai_agent(monkeypatch)

    calls = {"api": 0}

    def _api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _XaiBadCredentials403()
        return _codex_message_response("OK")

    refreshes = {"count": 0, "force": None}

    def _refresh(force=False):
        refreshes["count"] += 1
        refreshes["force"] = force
        agent.api_key = "fresh"
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 1
    assert refreshes["force"] is True
    assert calls["api"] == 2
    assert result["completed"] is True
    assert result["final_response"] == "OK"


def test_xai_403_refresh_consumes_the_shared_once_flag(monkeypatch):
    """The 403 variant shares ``codex_auth_retry_attempted`` with the 401
    path: one credential refresh per turn, whichever status spends it. A
    401 arriving after a consumed 403-refresh must not refresh again."""
    agent = _build_xai_agent(monkeypatch)

    class _Unauthorized401(Exception):
        status_code = 401

        def __init__(self):
            super().__init__("HTTP 401: unauthorized")

    calls = {"api": 0}

    def _api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _XaiBadCredentials403()
        raise _Unauthorized401()

    refreshes = {"count": 0}

    def _refresh(force=False):
        refreshes["count"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 1
    assert result.get("completed") is not True


def test_openai_codex_403_stays_non_retryable(monkeypatch):
    """The helper is scoped to xai-oauth: an openai-codex 403 whose body
    happens to contain the marker words must not trigger a refresh."""
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="tok",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None

    def _api_call(api_kwargs):
        raise _XaiBadCredentials403()

    refreshes = {"count": 0}

    def _refresh(force=False):
        refreshes["count"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 0
    assert result.get("completed") is not True


def test_xai_403_spending_limit_does_not_refresh(monkeypatch):
    """The billing-wall 403 (personal-team-blocked:spending-limit) carries
    no bad-credentials marker, so the narrow match excludes it by
    construction: refreshing cannot clear a spending limit and retrying
    it would spin."""
    agent = _build_xai_agent(monkeypatch)

    def _api_call(api_kwargs):
        raise _XaiSpendingLimit403()

    refreshes = {"count": 0}

    def _refresh(force=False):
        refreshes["count"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 0
    assert result.get("completed") is not True


def test_xai_403_structured_body_variant_refreshes(monkeypatch):
    """An SDK exception carrying the parsed body dict still exposes the
    marker through str(exc), so the structured variant recovers too."""
    agent = _build_xai_agent(monkeypatch)

    calls = {"api": 0}

    def _api_call(api_kwargs):
        calls["api"] += 1
        if calls["api"] == 1:
            raise _XaiBadCredentialsBody403()
        return _codex_message_response("OK")

    refreshes = {"count": 0}

    def _refresh(force=False):
        refreshes["count"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 1
    assert calls["api"] == 2
    assert result["completed"] is True


def test_xai_403_unrelated_body_does_not_touch_credentials(monkeypatch):
    agent = _build_xai_agent(monkeypatch)

    def _api_call(api_kwargs):
        raise _UnrelatedForbidden403()

    refreshes = {"count": 0}

    def _refresh(force=False):
        refreshes["count"] += 1
        return True

    monkeypatch.setattr(agent, "_interruptible_api_call", _api_call)
    monkeypatch.setattr(
        agent, "_try_refresh_codex_client_credentials", _refresh
    )

    result = agent.run_conversation("Say OK")

    assert refreshes["count"] == 0
    assert result.get("completed") is not True
