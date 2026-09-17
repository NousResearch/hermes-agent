"""Routed background reviews honor ``auxiliary.background_review.reasoning_effort`` (#94825).

The review fork is a full AIAgent, not an auxiliary_client call. The routed branch deliberately
skips the PARENT's reasoning_config (its effort vocabulary may be invalid for the routed model),
but an explicitly configured per-task effort must win over provider defaults — mirroring how every
other auxiliary task folds the same key into ``extra_body.reasoning``.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import run_agent
import agent.background_review as bg_review
from agent.background_review import build_cache_parity_fork

from tests.agent.test_background_review_cache_parity import _make_agent_stub, _make_recorder_class

ROUTED_RUNTIME = {
    "provider": "openrouter", "model": "aux-cheap-model", "api_key": "test-key",
    "base_url": None, "api_mode": None, "credential_pool": None, "request_overrides": {},
    "max_tokens": None, "command": None, "args": [], "routed": True,
}


def _routed_fork_kwargs(task_cfg):
    captured = {}
    agent = _make_agent_stub(run_agent.AIAgent)
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    with patch.object(run_agent, "AIAgent", _make_recorder_class(captured)), \
            patch.object(bg_review, "_resolve_review_runtime", return_value=ROUTED_RUNTIME):
        _fork, _rt, routed = build_cache_parity_fork(agent, task_cfg, max_iterations=5)
    assert routed
    return captured["init_kwargs"]


def test_routed_review_applies_configured_effort_not_parents():
    kwargs = _routed_fork_kwargs({"reasoning_effort": "xhigh"})
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    # ``none`` disables thinking on the routed fork, same vocabulary as every other aux task.
    assert _routed_fork_kwargs({"reasoning_effort": "none"})["reasoning_config"] == {"enabled": False}


def test_routed_review_falls_back_to_provider_default(caplog):
    # Unset: provider default, and the parent's ``high`` is NOT smuggled across the route.
    assert "reasoning_config" not in _routed_fork_kwargs({"reasoning_effort": ""})
    assert "reasoning_config" not in _routed_fork_kwargs({})
    with caplog.at_level(logging.WARNING):
        assert "reasoning_config" not in _routed_fork_kwargs({"reasoning_effort": "ludicrous"})
    assert "ludicrous" in caplog.text
