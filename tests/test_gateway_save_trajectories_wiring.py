"""Regression tests for save_trajectories wiring through the gateway AIAgent paths.

The gateway builds AIAgent in three places:
  1. _run_background_task — fresh agent per background-task turn.
  2. _run_agent_inner (fresh path) — when no cached agent exists for the session.
  3. _run_agent_inner (cached path) — when a cached agent is reused across turns.

The cached path is the most subtle: even after fixing the constructor calls,
a cached agent born with save_trajectories=False would stay False forever
without an explicit per-turn refresh alongside max_iterations. These tests
mirror test_cached_agent_max_iterations.py, extending the same invariants to
save_trajectories.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest


def _make_cached_agent(save_trajectories: bool = False) -> SimpleNamespace:
    """Minimal stand-in for a cached agent with the attributes the helpers touch."""
    return SimpleNamespace(
        _last_activity_ts=time.time() - 1000,
        _last_activity_desc="previous turn",
        _api_call_count=42,
        _last_flushed_db_idx=5,
        save_trajectories=save_trajectories,
    )


def test_init_cached_agent_for_turn_does_not_touch_save_trajectories():
    """The per-turn reset helper must leave save_trajectories untouched.

    The gateway refreshes save_trajectories explicitly right after calling
    this helper (mirroring max_iterations). If the helper ever reset it to
    False, that refresh would be undone and trajectory saving would silently
    break for all cached sessions regardless of config.
    """
    from gateway.run import GatewayRunner

    agent = _make_cached_agent(save_trajectories=True)
    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

    assert agent._api_call_count == 0                       # per-turn state was reset …
    assert agent._last_activity_desc == "starting new turn (cached)"
    assert agent.save_trajectories is True                  # … but save_trajectories was not


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a writable config.yaml and a clean module cache.

    Re-importing gateway.run re-evaluates ``_hermes_home = get_hermes_home()``
    so ``_load_gateway_config()`` reads from the tmp directory.  Module state
    is restored on teardown so the fixture does not leak into siblings.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _saved = {
        k: v for k, v in sys.modules.items()
        if k.startswith(("hermes_cli", "gateway"))
    }

    def write_cfg(body: str) -> None:
        (hermes_home / "config.yaml").write_text(textwrap.dedent(body))

    def fresh_gateway():
        for mod in list(sys.modules.keys()):
            if mod.startswith(("hermes_cli", "gateway")):
                del sys.modules[mod]
        return importlib.import_module("gateway.run")

    try:
        yield write_cfg, fresh_gateway
    finally:
        for k in list(sys.modules.keys()):
            if k.startswith(("hermes_cli", "gateway")):
                del sys.modules[k]
        sys.modules.update(_saved)


def test_cached_agent_refresh_picks_up_save_trajectories_from_config(isolated_home):
    """Per-turn refresh propagates save_trajectories=True from live config to a
    cached agent that was born with save_trajectories=False.

    This mirrors the max_iterations refresh contract: a long-lived cached agent
    must always run with the *current* config value, not the one it was built with.
    """
    write_cfg, fresh_gateway = isolated_home
    write_cfg("agent:\n  save_trajectories: true\n")
    gw = fresh_gateway()

    user_config = gw._load_gateway_config()
    agent_cfg_local = user_config.get("agent") or {}

    agent = _make_cached_agent(save_trajectories=False)
    # Simulate the per-turn refresh (gateway/run.py, after _init_cached_agent_for_turn):
    #   agent.save_trajectories = agent_cfg_local.get("save_trajectories", False)
    agent.save_trajectories = agent_cfg_local.get("save_trajectories", False)

    assert agent.save_trajectories is True


def test_cached_agent_refresh_reverts_stale_true_when_config_omits_key(isolated_home):
    """When config omits save_trajectories, a cached agent with True is corrected to False.

    Prevents a scenario where save_trajectories was enabled, trajectories were
    saved, then the option was removed from config — the cached agent should
    immediately stop saving.
    """
    write_cfg, fresh_gateway = isolated_home
    write_cfg("agent: {}\n")
    gw = fresh_gateway()

    user_config = gw._load_gateway_config()
    agent_cfg_local = user_config.get("agent") or {}

    agent = _make_cached_agent(save_trajectories=True)
    agent.save_trajectories = agent_cfg_local.get("save_trajectories", False)

    assert agent.save_trajectories is False
