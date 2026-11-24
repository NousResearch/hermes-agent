"""Codex app-server usage keeps the cache-write bucket the protocol reports.

codex-rs ``TokenUsageBreakdown`` carries ``cacheWriteInputTokens`` (from the Responses
``input_tokens_details.cache_write_tokens``); like ``cachedInputTokens`` it is inside
``inputTokens``. GPT-5.6+ bills those writes at 1.25x, so dropping them under-reports cost.
"""

from __future__ import annotations

from unittest.mock import patch

import run_agent
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult


def test_cache_write_tokens_are_read_and_not_double_counted(monkeypatch):
    def fake_run_turn(self, user_input: str, **kwargs):
        return TurnResult(
            final_text="done", projected_messages=[{"role": "assistant", "content": "done"}],
            turn_id="turn-w-1", thread_id="thread-w-1",
            token_usage_last={"totalTokens": 130, "inputTokens": 100, "cachedInputTokens": 20,
                              "cacheWriteInputTokens": 50, "outputTokens": 30, "reasoningOutputTokens": 0},
            model_context_window=200000,
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-w-1")
    agent = run_agent.AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                              api_mode="codex_app_server", quiet_mode=True, skip_context_files=True, skip_memory=True)
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello")

    assert (result["input_tokens"], result["cache_read_tokens"], result["cache_write_tokens"]) == (30, 20, 50)
    assert result["prompt_tokens"] == 100  # uncached + read + write == inputTokens, never more
    assert agent.session_cache_write_tokens == 50
    # The app-server sets its own cache options: the clock records the activity as an explicit
    # unknown, so no older window outlives it and /context names the reason.
    from agent.prompt_cache_clock import cache_window, render_prompt_cache_line
    window = cache_window(agent)
    assert (window.confidence, window.reason) == ("unknown", "codex_app_server")
    assert "runtime sets its own cache options" in render_prompt_cache_line(window.__dict__)
