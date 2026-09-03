"""Behavior contract for ``rate_limit`` in the post_api_request hook payload.

``agent/rate_limit_tracker.py`` parses the full ``x-ratelimit-*`` schema into
``agent._rate_limit_state`` on every streamed response, but the state was
unreachable from plugins: the ``post_api_request`` payload carried a
token-usage summary only. ``_rate_limit_state_for_hook`` is a pure
passthrough of that already-computed state.

Contract under test:

- With a captured ``RateLimitState``, the payload value is its dict form,
  each bucket carrying the tracker's own derived ``usage_pct``.
- No captured state (``None``), an agent without the getter, a getter that
  raises, and a non-dataclass return all yield ``None`` — the hook payload
  never breaks the loop and never fabricates state.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from agent.conversation_loop import _rate_limit_state_for_hook
from agent.rate_limit_tracker import RateLimitBucket, RateLimitState


def _agent_with_state(state):
    return SimpleNamespace(get_rate_limit_state=lambda: state)


def test_captured_state_serializes_with_bucket_usage_pct():
    state = RateLimitState(
        requests_min=RateLimitBucket(limit=100, remaining=40, reset_seconds=30.0),
        tokens_hour=RateLimitBucket(limit=200_000, remaining=50_000, reset_seconds=900.0),
        captured_at=time.time(),
        provider="openai",
    )
    payload = _rate_limit_state_for_hook(_agent_with_state(state))

    assert isinstance(payload, dict)
    assert payload["provider"] == "openai"
    assert payload["captured_at"] == state.captured_at
    # Raw bucket fields survive asdict...
    assert payload["requests_min"]["limit"] == 100
    assert payload["requests_min"]["remaining"] == 40
    # ...and each bucket carries the tracker's own derived percentage, so a
    # consumer does not re-implement the arithmetic.
    assert payload["requests_min"]["usage_pct"] == state.requests_min.usage_pct
    assert payload["tokens_hour"]["usage_pct"] == state.tokens_hour.usage_pct
    # Untouched buckets serialize too (defaults), rather than being dropped.
    assert "tokens_min" in payload and "requests_hour" in payload


def test_no_captured_state_is_none():
    assert _rate_limit_state_for_hook(_agent_with_state(None)) is None


def test_agent_without_getter_is_none():
    assert _rate_limit_state_for_hook(SimpleNamespace()) is None


def test_raising_getter_is_none():
    def boom():
        raise RuntimeError("tracker unavailable")

    assert _rate_limit_state_for_hook(SimpleNamespace(get_rate_limit_state=boom)) is None


def test_non_dataclass_state_is_none():
    assert _rate_limit_state_for_hook(_agent_with_state({"limit": 1})) is None
