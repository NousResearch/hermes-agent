"""Config-only fast reconnect policy and finite-stale ordering."""
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _resolve_nonstream_watchdogs


@pytest.mark.parametrize("stale", [0.1, 3, 5, 10, 90, float("inf")])
def test_fast_reconnect_ordering(tmp_path, monkeypatch, stale):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  codex:\n    ttfb_fast_reconnect_seconds: 7\n")
    monkeypatch.setenv("HERMES_CODEX_TTFB_FAST_RECONNECT_SECONDS", "999")
    monkeypatch.setenv("HERMES_CODEX_TTFB_BELOW_STALE_MARGIN_SECONDS", "0")
    monkeypatch.setenv("HERMES_CODEX_TTFB_BELOW_STALE", "0")
    agent = SimpleNamespace(api_mode="codex_responses", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _compute_non_stream_stale_timeout=lambda _: stale)
    result = _resolve_nonstream_watchdogs(agent, {"input": "hi"})
    assert 0 < result.ttfb_timeout <= 7
    assert result.ttfb_timeout < stale


@pytest.mark.parametrize("size", [40_000, 240_000, 480_000])
def test_large_prefill_policy_not_shortened(tmp_path, monkeypatch, size):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  codex:\n    ttfb_fast_reconnect_seconds: 1\n")
    agent = SimpleNamespace(api_mode="codex_responses", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _compute_non_stream_stale_timeout=lambda _: 90)
    result = _resolve_nonstream_watchdogs(agent, {"input": "x" * size})
    assert result.ttfb_timeout == 120
