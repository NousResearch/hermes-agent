"""Tests for agent.brain_host — Phase 3 Brain Host seam.

Coverage:
  * parity     — BrainHost.build_agent forwards kwargs identically to direct AIAgent().
  * singleton  — BrainHost.get() always returns the same instance.
  * flag-gate  — HERMES_BRAIN_HOST=0/unset → brain_host never imported by _make_agent;
                  HERMES_BRAIN_HOST=1 → routes through BrainHost.build_agent.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder_class():
    """Return a fresh recorder class (captures __init__ kwargs) and a list
    that accumulates every instance created."""
    instances: list = []

    class RecorderAgent:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            instances.append(self)

    return RecorderAgent, instances


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def test_brain_host_parity():
    """BrainHost.build_agent must pass exactly the same kwargs as a direct call.

    Monkeypatch run_agent.AIAgent with a recorder so we can compare the two
    construction paths without importing the real AIAgent.
    """
    from agent.brain_host import AgentSpec, BrainHost

    RecorderAgent, instances = _make_recorder_class()

    test_kwargs = {
        "model": "claude-opus-4-6",
        "provider": "anthropic",
        "api_key": "sk-test",
        "quiet_mode": True,
    }

    with patch("run_agent.AIAgent", RecorderAgent):
        # --- path 1: via BrainHost ---
        result_hosted = BrainHost.get().build_agent(
            AgentSpec(intent="test", kwargs=test_kwargs)
        )
        # --- path 2: direct ---
        from run_agent import AIAgent as _Direct  # noqa: PLC0415

        result_direct = _Direct(**test_kwargs)

    assert len(instances) == 2
    hosted_kwargs = instances[0]._kwargs
    direct_kwargs = instances[1]._kwargs
    assert hosted_kwargs == direct_kwargs, (
        f"kwargs mismatch:\nhosted={hosted_kwargs}\ndirect={direct_kwargs}"
    )
    # Both return the recorder instance (not None, not a mock wrapper).
    assert isinstance(result_hosted, RecorderAgent)
    assert isinstance(result_direct, RecorderAgent)


# ---------------------------------------------------------------------------
# Singleton test
# ---------------------------------------------------------------------------

def test_brain_host_singleton():
    """BrainHost.get() must return the same object on repeated calls."""
    from agent.brain_host import BrainHost

    a = BrainHost.get()
    b = BrainHost.get()
    assert a is b


# ---------------------------------------------------------------------------
# Flag-gate tests — default OFF
# ---------------------------------------------------------------------------

def _call_make_agent_with_env(env_patch: dict, monkeypatch_patches: dict):
    """Import tui_gateway.server._make_agent under a controlled set of
    patches, call it, and return (mock_ai_agent, sys_modules_snapshot)."""
    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "claude-opus-4-6"}}

    mock_ai_agent = MagicMock()

    # Remove brain_host from sys.modules to get a clean import-detection slate.
    sys.modules.pop("agent.brain_host", None)

    ctx_managers = [
        patch.dict(os.environ, env_patch, clear=False),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch("tui_gateway.server._load_fallback_model", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent", mock_ai_agent),
    ]
    # Apply any extra patches the caller wants.
    for target, new_val in monkeypatch_patches.items():
        ctx_managers.append(patch(target, new_val))

    # Enter all context managers.
    entered = []
    try:
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        from tui_gateway.server import _make_agent  # noqa: PLC0415

        _make_agent("sid-gate", "key-gate")
        modules_after = set(sys.modules.keys())
    finally:
        for cm, result in zip(reversed(ctx_managers), reversed(entered)):
            cm.__exit__(None, None, None)

    return mock_ai_agent, modules_after


def test_flag_gate_off_by_default():
    """When HERMES_BRAIN_HOST is unset, agent.brain_host must NOT be imported."""
    env = {}
    os.environ.pop("HERMES_BRAIN_HOST", None)

    _, mods = _call_make_agent_with_env(env, {})
    assert "agent.brain_host" not in mods, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST is unset"
    )


def test_flag_gate_off_when_zero():
    """When HERMES_BRAIN_HOST=0, agent.brain_host must NOT be imported."""
    _, mods = _call_make_agent_with_env({"HERMES_BRAIN_HOST": "0"}, {})
    assert "agent.brain_host" not in mods, (
        "agent.brain_host was imported even though HERMES_BRAIN_HOST=0"
    )


def test_flag_gate_on_routes_through_brain_host():
    """When HERMES_BRAIN_HOST=1, _make_agent must call BrainHost.build_agent."""
    from agent.brain_host import BrainHost  # noqa: PLC0415 — ensure loaded

    build_agent_mock = MagicMock(return_value=MagicMock())

    with (
        patch.dict(os.environ, {"HERMES_BRAIN_HOST": "1"}),
        patch("agent.brain_host.BrainHost.build_agent", build_agent_mock),
    ):
        fake_runtime = {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-test",
            "api_mode": "anthropic_messages",
            "command": None,
            "args": None,
            "credential_pool": None,
        }
        fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "claude-opus-4-6"}}

        with (
            patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
            patch("tui_gateway.server._get_db", return_value=MagicMock()),
            patch("tui_gateway.server._load_reasoning_config", return_value=None),
            patch("tui_gateway.server._load_service_tier", return_value=None),
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
            patch("tui_gateway.server._load_fallback_model", return_value=None),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=fake_runtime,
            ),
            patch("run_agent.AIAgent", MagicMock()),
        ):
            from tui_gateway.server import _make_agent  # noqa: PLC0415

            _make_agent("sid-on", "key-on")

    build_agent_mock.assert_called_once()
    spec = build_agent_mock.call_args.args[0]
    assert spec.intent == "tui_gateway"
    assert isinstance(spec.kwargs, dict)
    # Spot-check a stable key that the gate path must forward.
    assert spec.kwargs.get("quiet_mode") is True
