"""OpenCode Go 429s: a plan-window wall benches the key, a per-model upstream throttle does not."""

from __future__ import annotations

import httpx
import openai
import pytest

import model_tools  # registers the bundled provider profiles
from agent import auxiliary_client as ac
from agent.error_classifier import FailoverReason, classify_api_error

_UPSTREAM = {"error": {"type": "server_error",
                       "message": "Upstream request failed: Rate limit exceeded. Please try again later."}}
_PLAN_WALL = {"type": "error", "error": {"type": "GoUsageLimitError", "message": "Go usage limit exceeded"},
              "metadata": {"workspace": "wrk_mock", "limitName": "weekly"}}


def _go_429(body: dict) -> openai.RateLimitError:
    resp = httpx.Response(429, json=body, request=httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions"))
    return openai.RateLimitError(f"Error code: 429 - {body}", response=resp, body=body)


@pytest.mark.parametrize("body, reason", [
    (_UPSTREAM, FailoverReason.upstream_rate_limit),
    (_PLAN_WALL, FailoverReason.billing),  # quota wall: benches the key and rotates
])
def test_go_429_classification_separates_model_throttle_from_plan_wall(body, reason):
    verdict = classify_api_error(_go_429(body), provider="opencode-go", model="mock-model")
    assert verdict.reason == reason
    assert verdict.should_fallback is True


@pytest.mark.parametrize("body, benches", [(_UPSTREAM, False), (_PLAN_WALL, True)])
def test_auxiliary_pool_recovery_keeps_key_on_model_throttle(monkeypatch, body, benches):
    class _StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()

    pool = _StubPool()
    monkeypatch.setattr(ac, "load_pool", lambda provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)
    assert ac._recover_provider_pool("opencode-go", _go_429(body), failed_api_key="sk-mock") is benches
    assert len(pool.rotate_calls) == (1 if benches else 0)
