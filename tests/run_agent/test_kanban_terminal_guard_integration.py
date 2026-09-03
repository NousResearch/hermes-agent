"""Integration coverage for fail-closed Kanban turn finalization."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *args, **kwargs: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _plain_stop_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="The work is done.",
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="test-model",
    )


def test_plain_text_stops_exhaust_to_structured_protocol_failure(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    from hermes_cli.kanban_exit_codes import (
        KANBAN_PROTOCOL_EXIT_CODE,
        single_query_exit_code,
    )
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy",
        base_url="https://example.invalid/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent._disable_streaming = True
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: _plain_stop_response(),
    )
    # Set worker identity only after construction so this test exercises the
    # stop guard without requiring a real board during agent initialization.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_guard")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")

    result = agent.run_conversation("finish the assigned task")

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["failure_reason"] == "kanban_protocol"
    assert result["turn_exit_reason"] == "kanban_protocol_violation"
    assert single_query_exit_code(result, kanban_worker=True) == (
        KANBAN_PROTOCOL_EXIT_CODE
    )
    assert result["api_calls"] == 3
    assert "protocol violation" in result["final_response"].lower()
    assert result["error"] == result["final_response"]
