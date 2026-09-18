"""Codex non-stream watchdogs honour the request's reasoning effort (#112909).

GPT-5-family models at ``reasoning_effort=high``/``xhigh`` think server-side for 100-170s before
their first substantive SSE event even on a ~6KB prompt, while the token-sized watchdog tiers hand
such a prompt 12s (event-idle) / 120s (TTFB) / 90s (stale) fuses — the client killed healthy
requests three times in a row and reported the provider as unavailable. High effort now raises the
IMPLICIT defaults to one shared silence floor; explicit operator values still win.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SMALL_PROMPT = {"model": "gpt-5.5", "input": [{"role": "user", "content": "x" * 6000}]}


def _codex_agent(tmp_path: Path, monkeypatch, effort: str, *, enabled: bool = True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    for var in ("HERMES_API_CALL_STALE_TIMEOUT", "HERMES_CODEX_TTFB_TIMEOUT_SECONDS",
                "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "HERMES_CODEX_TTFB_MAX_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5", provider="openai-codex", api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex", quiet_mode=True,
        skip_context_files=True, skip_memory=True, platform="cli",
    )
    agent.api_mode = "codex_responses"
    agent.reasoning_config = {"enabled": enabled, "effort": effort}
    return agent


@pytest.mark.parametrize("effort", ["high", "xhigh", "max", "ultra"])
def test_high_effort_small_prompt_gets_the_silence_floor_on_all_three_fuses(tmp_path, monkeypatch, effort):
    from agent.chat_completion_helpers import HIGH_EFFORT_SILENCE_FLOOR_SECONDS, _resolve_nonstream_watchdogs

    wd = _resolve_nonstream_watchdogs(_codex_agent(tmp_path, monkeypatch, effort), _SMALL_PROMPT)

    assert wd.idle_enabled and wd.ttfb_enabled
    assert wd.idle_timeout >= HIGH_EFFORT_SILENCE_FLOOR_SECONDS
    assert wd.ttfb_timeout >= HIGH_EFFORT_SILENCE_FLOOR_SECONDS
    assert wd.stale_timeout >= HIGH_EFFORT_SILENCE_FLOOR_SECONDS


def test_default_effort_tiers_and_explicit_operator_values_are_untouched(tmp_path, monkeypatch):
    """Control: medium effort (and disabled reasoning) keep the small-prompt tiers; an explicit
    env value for any fuse is never raised by the floor."""
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    medium = _resolve_nonstream_watchdogs(_codex_agent(tmp_path, monkeypatch, "medium"), _SMALL_PROMPT)
    assert (medium.idle_timeout, medium.ttfb_timeout, medium.stale_timeout) == (12.0, 120.0, 90.0)

    disabled = _resolve_nonstream_watchdogs(_codex_agent(tmp_path, monkeypatch, "high", enabled=False), _SMALL_PROMPT)
    assert (disabled.idle_timeout, disabled.ttfb_timeout, disabled.stale_timeout) == (12.0, 120.0, 90.0)

    agent = _codex_agent(tmp_path, monkeypatch, "high")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "75")
    explicit = _resolve_nonstream_watchdogs(agent, _SMALL_PROMPT)
    assert (explicit.idle_timeout, explicit.ttfb_timeout, explicit.stale_timeout) == (20.0, 45.0, 75.0)


def test_effort_floor_never_outlives_the_run_budget_cap(tmp_path, monkeypatch):
    """A run budget (cron/kanban) caps the IMPLICIT stale timeout at half the remaining budget; the
    high-effort floor must be applied before that cap, not override it."""
    import time

    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    agent = _codex_agent(tmp_path, monkeypatch, "high")
    agent.run_budget_seconds = 100
    agent._run_budget_started_at = time.time() - 5

    assert _resolve_nonstream_watchdogs(agent, _SMALL_PROMPT).stale_timeout == 60.0
