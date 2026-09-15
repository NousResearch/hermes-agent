"""Regression tests for #86444 - large-context watchdog accommodations for codex_responses transports.

Ensures:
1. xAI Responses transport (api_mode="codex_responses", provider="xai-oauth") receives
   the large-context stale timeout floor (e.g. 1200s at >100K tokens, 900s at >50K tokens)
   in interruptible_api_call. A baseline short timeout of 0.2s is elevated so a 0.5s call succeeds without stale_call_kill.
2. Large requests scale TTFB timeout for xai-oauth (codex_responses) instead of killing at a short TTFB cutoff.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_mock_agent(provider="xai-oauth", api_mode="codex_responses", base_url="https://api.x.ai/v1"):
    agent = MagicMock()
    agent.provider = provider
    agent.api_mode = api_mode
    agent.base_url = base_url
    agent._base_url_lower = base_url.lower()
    agent._base_url_hostname = "api.x.ai"
    agent._emit_status = lambda *a, **k: None
    agent._buffer_status = lambda *a, **k: None
    agent._touch_activity = lambda *a, **k: None
    agent._codex_stream_last_event_ts = None
    agent._interrupt_requested = False
    agent.timeout = 180.0
    agent._compute_non_stream_stale_timeout = lambda api_kwargs: 0.2
    return agent


def test_xai_responses_receives_stale_timeout_floor_in_api_call(monkeypatch):
    """interruptible_api_call elevates stale timeout to context floor (>=900s) for xai-oauth codex_responses.
    
    When baseline _compute_non_stream_stale_timeout is 0.2s and TTFB watchdog is disabled (0),
    a 0.5s execution would fail with stale_call_kill on base (stale timeout at 0.2s), but passes when the floor elevates it to >=900s.
    """
    from agent import chat_completion_helpers as h

    agent = _make_mock_agent(provider="xai-oauth", api_mode="codex_responses", base_url="https://api.x.ai/v1")
    assert h._is_openai_codex_backend(agent) is False

    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")
    closes = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: SimpleNamespace())
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **k: None)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    # 65k tokens estimate -> floor is >=900.0s
    large_text = "word " * 65_000
    api_kwargs = {"messages": [{"role": "user", "content": large_text}], "stream": False}

    sentinel = SimpleNamespace(ok=True)
    stop = {"flag": False}

    def fake_hang_or_delay(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 0.5
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        if "stale_call_kill" in closes:
            raise RuntimeError("aborted by stale kill")
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang_or_delay)

    try:
        res = h.interruptible_api_call(agent, api_kwargs)
        assert res is sentinel
        assert "stale_call_kill" not in closes
    finally:
        stop["flag"] = True


def test_xai_responses_ttfb_scaled_for_large_requests(monkeypatch):
    """Large requests scale TTFB timeout for xai-oauth (codex_responses) instead of killing at small TTFB cutoff."""
    from agent import chat_completion_helpers as h

    agent = _make_mock_agent(provider="xai-oauth", api_mode="codex_responses", base_url="https://api.x.ai/v1")
    assert h._is_openai_codex_backend(agent) is False

    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.2")

    closes = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: SimpleNamespace())
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **k: None)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    # 1. Small request (<10k tokens) should trip TTFB watchdog at 0.2s
    stop_small = {"flag": False}

    def fake_hang_small(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 5
        while time.time() < deadline and not stop_small["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("hang")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang_small)

    try:
        with pytest.raises(TimeoutError):
            h.interruptible_api_call(agent, {"messages": [{"role": "user", "content": "hello"}], "stream": True})
        assert "codex_ttfb_kill" in closes
    finally:
        stop_small["flag"] = True

    # 2. Large request (>50k tokens) scales TTFB timeout to 120s, so it does NOT trip at 0.2s
    closes.clear()
    large_text = "word " * 65_000
    stop_large = {"flag": False}

    def fake_stream_delayed(api_kwargs, client=None, on_first_delta=None):
        # Sleeps for 0.5s (longer than the initial 0.2s TTFB cutoff)
        time.sleep(0.5)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream_delayed)

    result = h.interruptible_api_call(agent, {"messages": [{"role": "user", "content": large_text}], "stream": True})
    assert result.ok is True
    assert "codex_ttfb_kill" not in closes
