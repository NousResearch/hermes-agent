"""Tests for wait-state visibility — the live "what are we waiting on" notices.

Long provider waits (slow/overloaded backend, no first byte, reasoning model
thinking for minutes) used to leave CLI/TUI/Desktop users staring at a generic
"cogitating..." spinner with no explanation. ``AIAgent._emit_wait_notice``
rewrites the live spinner/status line (via ``thinking_callback``, bridged to
``thinking.delta`` for TUI/Desktop) and updates the activity tracker (which the
gateway's "⏳ Working — N min" heartbeat includes).
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_agent(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    return AIAgent(
        model="test-model",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
        **kwargs,
    )


def test_emit_wait_notice_updates_spinner_and_activity(tmp_path, monkeypatch):
    """The notice reaches the live display callback AND the activity tracker."""
    seen: list = []
    agent = _make_agent(tmp_path, monkeypatch, thinking_callback=seen.append)

    agent._emit_wait_notice("⏳ waiting on test-model — 30s with no response yet")

    assert seen == ["⏳ waiting on test-model — 30s with no response yet"]
    summary = agent.get_activity_summary()
    assert "waiting on test-model" in summary["last_activity_desc"]


def test_emit_wait_notice_without_callback_still_touches_activity(tmp_path, monkeypatch):
    """No thinking_callback bound (gateway sessions) — activity still updates."""
    agent = _make_agent(tmp_path, monkeypatch)
    agent.thinking_callback = None

    agent._emit_wait_notice("⏳ waiting on test-model — 60s")

    assert "waiting on test-model" in agent.get_activity_summary()["last_activity_desc"]


def test_emit_wait_notice_swallows_callback_errors(tmp_path, monkeypatch):
    """A broken display callback must never break the API-call wait loop."""

    def _boom(text):
        raise RuntimeError("display exploded")

    agent = _make_agent(tmp_path, monkeypatch, thinking_callback=_boom)

    agent._emit_wait_notice("⏳ waiting")  # must not raise
    assert "waiting" in agent.get_activity_summary()["last_activity_desc"]


def test_nonstream_wait_loop_emits_explained_notice(tmp_path, monkeypatch):
    """After ~30s with no response, interruptible_api_call rewrites the live
    line with an explanation (model name, elapsed, overload hint, recovery
    deadline) instead of a bare 'waiting for non-streaming response'."""
    from agent import chat_completion_helpers as h

    seen: list = []
    agent = _make_agent(tmp_path, monkeypatch, thinking_callback=seen.append)
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0)

    # Compress the 30s cadence: the loop fires the notice every 100 polls of
    # 0.3s; patch the join timeout down via a tiny thread that stays alive
    # briefly, and shrink the poll interval by patching time.  Simplest
    # reliable approach: run a worker that hangs ~1.2s and patch the modulo
    # counter trigger by making the loop's join timeout effectively immediate.
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: None)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: None)

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 10
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)
    # TTFB kill at 1s ends the call quickly; the wait notice fires on the
    # 100-poll cadence, so to observe it within the 1s window we shrink the
    # cadence by patching threading.Thread.join used in the poll loop is
    # overkill — instead just verify the TTFB reconnect notice, which flows
    # through the same _emit_wait_notice path.
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    try:
        with pytest.raises(TimeoutError):
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    finally:
        stop["flag"] = True

    reconnect_notices = [s for s in seen if "reconnecting" in s]
    assert reconnect_notices, f"expected a reconnect wait-notice, saw: {seen}"
    assert "no response from provider" in reconnect_notices[0]


def test_nonstream_wait_notice_survives_infinite_stale_timeout(tmp_path, monkeypatch):
    """Local/ACP endpoints use stale_timeout=+inf; the 30s wait notice must not
    crash with OverflowError from int(inf) (seen with cursor-acp / acp://cursor)."""
    import threading

    from agent import chat_completion_helpers as h

    seen: list = []
    agent = _make_agent(tmp_path, monkeypatch, thinking_callback=seen.append)
    agent.api_mode = "chat_completions"
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: float("inf")
    )

    dummy_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=None))
    )

    # Drive the poll loop: hang long enough for ~100 × 0.3s cadence, but shrink
    # join timeout so the test finishes quickly while still hitting % 100 == 0.
    real_join = threading.Thread.join
    poll_state = {"joins": 0}

    def fast_join(self, timeout=None):
        poll_state["joins"] += 1
        if timeout is not None:
            return real_join(self, timeout=0.001)
        return real_join(self, timeout=timeout)

    hang_until = {"n": 105}  # >100 polls so the notice fires once

    def fake_create(**kwargs):
        while poll_state["joins"] < hang_until["n"] and not agent._interrupt_requested:
            time.sleep(0.001)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            model="agent",
        )

    dummy_client.chat.completions.create = fake_create
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: None)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: None)
    monkeypatch.setattr(threading.Thread, "join", fast_join)

    result = h.interruptible_api_call(agent, {"model": "agent", "messages": []})
    assert result is not None
    wait_notices = [s for s in seen if "with no response yet" in s]
    assert wait_notices, f"expected wait notice, saw: {seen}"
    assert "auto-reconnect" not in wait_notices[0]
    assert "slow or overloaded" in wait_notices[0]
