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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import (
    _context_admission_input_limit,
    _context_admission_pressure,
)
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

    monkeypatch.setattr(
        config_mod, "load_config_readonly", lambda: _config(max_attempts)

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


@pytest.mark.parametrize(
    ("context_limit", "output_reserve", "expected"),
    [
        (272_000, 0, 267_904),
        (200_000, 10_000, 190_000),
        (32_000, 0, 31_488),
        (0, 0, 0),
    ],
)
def test_context_admission_limit(context_limit, output_reserve, expected):
    assert _context_admission_input_limit(
        context_limit,
        reserved_output_tokens=output_reserve,
    ) == expected


def test_context_admission_pressure_removes_only_proven_fixed_overcount():
    compressor = SimpleNamespace(
        last_rough_tokens_when_real_prompt_fit=200_000,
        last_real_prompt_tokens=160_000,
        threshold_tokens=204_000,
    )

    assert _context_admission_pressure(compressor, 270_000) == 230_000


def test_context_admission_pressure_fails_closed_without_valid_calibration():
    compressor = SimpleNamespace(
        last_rough_tokens_when_real_prompt_fit=200_000,
        last_real_prompt_tokens=210_000,
        threshold_tokens=204_000,
    )

    assert _context_admission_pressure(compressor, 270_000) == 270_000


def test_pre_provider_admission_blocks_uncompressible_request(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, max_attempts=3)
    agent.context_compressor.context_length = 200_000
    agent.max_tokens = 0
    safe_limit = _context_admission_input_limit(200_000)

    with (
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=safe_limit,
        ),
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=AssertionError("provider call must remain local"),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is False
    assert result["provider_call_blocked"] is True
    assert result["api_calls"] == 0
    assert result["safe_input_limit"] == safe_limit


def test_safe_boundary_overrides_preflight_deferral(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, max_attempts=3)
    compressor = agent.context_compressor
    compressor.context_length = 200_000
    compressor.threshold_tokens = 100_000
    agent.max_tokens = 0
    safe_limit = _context_admission_input_limit(200_000)
    defer = MagicMock(return_value=True)
    compressor.should_defer_preflight_to_real_usage = defer
    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, system_message

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(40)
    ]

    with (
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=safe_limit,
        ),
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=AssertionError("provider call must remain local"),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)

    assert len(compress_calls) == 1
    # The turn-start estimator may consult deferral for its small real input;
    # the oversized loop request must not.
    assert all(call.args[0] < safe_limit for call in defer.call_args_list)
    assert result["provider_call_blocked"] is True


def test_request_below_safe_boundary_reaches_provider(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, max_attempts=3)
    agent.context_compressor.context_length = 200_000
    agent.max_tokens = 0
    safe_limit = _context_admission_input_limit(200_000)

    with (
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=safe_limit - 1,
        ),
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=_stop_response(),
        ) as provider_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    provider_call.assert_called_once()


def test_provider_proven_calibration_avoids_false_local_block(monkeypatch, tmp_path):
    agent: Any = _make_agent(monkeypatch, tmp_path, max_attempts=3)
    compressor = agent.context_compressor
    compressor.context_length = 200_000
    compressor.threshold_tokens = 150_000
    compressor.last_rough_tokens_when_real_prompt_fit = 180_000
    compressor.last_real_prompt_tokens = 100_000
    agent.compression_enabled = False
    agent.max_tokens = 0

    with (
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=270_000,
        ),
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=_stop_response(),
        ) as provider_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert _context_admission_pressure(compressor, 270_000) == 190_000
    assert result["completed"] is True
    provider_call.assert_called_once()
