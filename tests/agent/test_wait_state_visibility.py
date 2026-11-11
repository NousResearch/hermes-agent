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
import types

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
