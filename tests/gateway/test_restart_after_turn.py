"""Unit tests for in-band restart after-turn deferral helpers (#77184)."""

from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    parse_restart_after_turn_timeout,
    resolve_restart_exit_wait_budget,
)
from gateway.run import GatewayRunner


def test_parse_restart_after_turn_timeout_defaults_and_clamps():
    assert parse_restart_after_turn_timeout("") == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout(None) == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout("bogus") == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout(0) == 0.0
    assert parse_restart_after_turn_timeout("-5") == 0.0
    assert parse_restart_after_turn_timeout("120") == 120.0


def test_default_restart_after_turn_timeout_is_human_tolerable():
    """The shipped default must not make interactive restarts block for hours.

    A wedged turn must not pin `hermes gateway restart` for 6h — the
    default is a safety valve for hung agents, not a target latency
    (#79133). 900-1800s protects long autonomous turns while keeping
    worst-case interactive restart in human-tolerable territory.
    """
    assert 900 <= DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT <= 1800
    # An interactive restart's printed wait budget stays under ~32 min.
    budget = resolve_restart_exit_wait_budget(
        60, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT, headroom=15
    )
    assert budget <= 1875


def test_resolve_restart_exit_wait_budget_covers_both_phases():
    assert resolve_restart_exit_wait_budget(0, 0, headroom=15) == 15.0
    assert resolve_restart_exit_wait_budget(180, 21600, headroom=15) == 180 + 21600 + 15
    assert resolve_restart_exit_wait_budget("bad", "bad", headroom="x") == 0.0


def test_load_restart_after_turn_timeout_preserves_zero(tmp_path, monkeypatch):
    """Config/env ``0`` must disable after-turn wait, not fall back to default."""
    import gateway.run as gateway_run

    monkeypatch.delenv("HERMES_RESTART_AFTER_TURN_TIMEOUT", raising=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_after_turn_timeout: 0\n",
        encoding="utf-8",
    )
    assert GatewayRunner._load_restart_after_turn_timeout() == 0.0

    monkeypatch.setenv("HERMES_RESTART_AFTER_TURN_TIMEOUT", "0")
    assert GatewayRunner._load_restart_after_turn_timeout() == 0.0


def test_active_work_count_includes_async_delegations(monkeypatch):
    """A planned restart must not kill detached subagents after the parent turn."""
    import tools.async_delegation as async_delegation

    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(runner, "_running_agent_count", lambda: 0)
    monkeypatch.setattr(runner, "_active_cron_job_count", lambda: 0)
    monkeypatch.setattr(runner, "_active_api_run_count", lambda: 0)
    monkeypatch.setattr(runner, "_active_deferred_agent_worker_count", lambda: 0)
    monkeypatch.setattr(async_delegation, "active_count", lambda: 1)

    assert runner._active_work_count() == 1
