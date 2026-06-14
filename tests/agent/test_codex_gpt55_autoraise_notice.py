"""Regression tests for silent Codex gpt-5.5 compaction autoraise."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from run_agent import AIAgent


_CODEX_CFG = {
    "model": {"context_length": 272_000},
    "compression": {
        "enabled": True,
        "threshold": 0.50,
        "target_ratio": 0.20,
        "codex_gpt55_autoraise": True,
    },
    "memory": {"memory_enabled": False, "user_profile_enabled": False},
    "agent": {},
    "skills": {},
}


def _make_codex_gpt55_agent(*, quiet_mode: bool) -> Any:
    with patch("hermes_cli.config.load_config", return_value=_CODEX_CFG):
        return AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-codex-token",
            quiet_mode=quiet_mode,
            skip_context_files=True,
            skip_memory=True,
        )


def test_codex_gpt55_autoraise_changes_threshold_without_cli_notice(capsys):
    agent = _make_codex_gpt55_agent(quiet_mode=False)

    output = capsys.readouterr().out

    assert agent.context_compressor.threshold_percent == 0.85
    assert agent.context_compressor.threshold_tokens == int(272_000 * 0.85)
    assert "compress at 85%" in output
    assert "Codex gpt-5.5 caps context" not in output
    assert "auto-compaction was raised" not in output
    assert "codex_gpt55_autoraise" not in output


def test_codex_gpt55_autoraise_does_not_queue_gateway_lifecycle_notice():
    agent = _make_codex_gpt55_agent(quiet_mode=True)

    events = []
    agent.status_callback = lambda event, message: events.append((event, message))
    agent._replay_compression_warning()

    assert agent.context_compressor.threshold_percent == 0.85
    assert agent._compression_warning is None
    assert events == []
