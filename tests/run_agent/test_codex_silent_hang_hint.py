"""Tests for the ``_codex_silent_hang_hint`` heuristic.

The helper substitutes an actionable hint into the stale-call timeout
warning when the request matches a known Codex silent-reject pattern
(gpt-5.5 family on the ChatGPT Codex backend).  See issue #21444 for
symptom history. The recommended workaround for ChatGPT Codex OAuth
accounts is `gpt-5.4` / `gpt-5.3-codex`, not `gpt-5.4-codex`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


@pytest.fixture(autouse=True)
def _isolate_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")


# ── positive cases: hint fires ─────────────────────────────────────────────


def test_hint_fires_for_bare_gpt_5_5_on_codex(tmp_path):
    agent = _make_agent(tmp_path)
    agent.api_mode = "codex_responses"
    hint = agent._codex_silent_hang_hint(model="gpt-5.5")
    assert hint is not None
    assert "gpt-5.4" in hint
    assert "gpt-5.3-codex" in hint
    assert "gpt-5.4-codex" in hint
    assert "fallback chain" in hint


def test_hint_fires_for_vendor_prefixed_gpt_5_5(tmp_path):
    agent = _make_agent(tmp_path, model="openai/gpt-5.5")
    agent.api_mode = "codex_responses"
    hint = agent._codex_silent_hang_hint(model="openai/gpt-5.5")
    assert hint is not None






# ── negative cases: hint stays None ────────────────────────────────────────












