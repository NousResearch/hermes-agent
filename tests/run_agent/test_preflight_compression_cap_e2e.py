"""E2E: compression.max_attempts=6 drives a 4th+ preflight compaction pass.

The turn-start preflight loop in ``agent/turn_context.py`` was hardcoded to
``range(3)``: even when every pass made real progress and the request stayed
over threshold, the 4th pass never ran, regardless of configuration.  The
loop now sizes itself from the same resolved ``compression.max_attempts`` cap
as the conversation loop's compression sites.

This test builds a real ``AIAgent`` from a config with
``compression.max_attempts: 6`` (the config-driven path through
``agent_init``), then drives a full ``run_conversation()`` turn in which the
estimated request size keeps shrinking ~10% per compaction but stays above
threshold — the exact "progress, but not enough yet" shape that legitimately
needs more than three rounds.  With cap=6 the preflight must run a 4th pass
(and ultimately all six).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent


def _config(max_attempts) -> dict:
    return {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            "max_attempts": max_attempts,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _stop_response():
    msg = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(monkeypatch, tmp_path: Path, *, max_attempts) -> AIAgent:
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: _config(max_attempts)
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            model="test/model",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            session_db=db,
            session_id="preflight-cap-e2e",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    return agent


def test_preflight_runs_fourth_compaction_pass_at_cap_six(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, max_attempts=6)
    # Config-driven attach seam (agent_init) resolved the raised cap.
    assert agent.max_compression_attempts == 6

    # Keep the request permanently over threshold while every compaction
    # makes material (~10% > the 5% progress floor) headway.
    compressor = agent.context_compressor
    compressor.threshold_tokens = 50_000

    estimate_state = {"tokens": 1_000_000.0, "calls": 0}

    def _shrinking_estimate(*_args, **_kwargs):
        if estimate_state["calls"]:
            estimate_state["tokens"] *= 0.9
        estimate_state["calls"] += 1
        return int(estimate_state["tokens"])

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    # 60 messages > protect_first_n + protect_last_n + 1, so the cheap
    # preflight count gate opens without patching internals.
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(60)
    ]
    agent.client.chat.completions.create.return_value = _stop_response()

    with (
        patch(
            "agent.turn_context.estimate_request_tokens_rough",
            side_effect=_shrinking_estimate,
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)

    assert result["completed"] is True
    # The old hardcoded range(3) made a 4th pass impossible; cap=6 must
    # deliver it (and, with steady progress over threshold, all six).
    assert len(compress_calls) >= 4, (
        f"expected a 4th preflight compaction pass at cap=6, "
        f"got {len(compress_calls)} passes"
    )
    assert len(compress_calls) == 6


def test_preflight_still_stops_at_default_three(monkeypatch, tmp_path):
    """Unset compression.max_attempts keeps the historical 3-pass behavior."""
    agent = _make_agent(monkeypatch, tmp_path, max_attempts=None)
    assert agent.max_compression_attempts == 3

    compressor = agent.context_compressor
    compressor.threshold_tokens = 50_000

    estimate_state = {"tokens": 1_000_000.0, "calls": 0}

    def _shrinking_estimate(*_args, **_kwargs):
        if estimate_state["calls"]:
            estimate_state["tokens"] *= 0.9
        estimate_state["calls"] += 1
        return int(estimate_state["tokens"])

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(60)
    ]
    agent.client.chat.completions.create.return_value = _stop_response()

    with (
        patch(
            "agent.turn_context.estimate_request_tokens_rough",
            side_effect=_shrinking_estimate,
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)

    assert result["completed"] is True
    assert len(compress_calls) == 3
