"""The startup "Agent budget" line reports the budget the turn loop enforces (#116888)."""

import logging

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _bridge_max_turns_to_env, _current_max_iterations
from hermes_cli.config import TURN_LIMIT_UNLIMITED


@pytest.mark.parametrize(
    ("max_turns", "expected"),
    [({"max_turns": 50}, "50"), ({}, "90")],
)
def test_agent_budget_line_matches_enforced_limit(monkeypatch, max_turns, expected):
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "90")  # .env value: config wins when set, else it is the legacy bridge
    monkeypatch.setattr("gateway.run._reload_runtime_env_preserving_config_authority", lambda: None)
    monkeypatch.setattr("gateway.run.get_hermes_home_override", lambda: None)
    _bridge_max_turns_to_env(max_turns)
    enforced = _current_max_iterations()
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    startup_logger = logging.getLogger("gateway.run")
    startup_logger.addHandler(handler)
    old_level = startup_logger.level
    startup_logger.setLevel(logging.INFO)
    try:
        GatewayRunner._log_agent_budget()
    finally:
        startup_logger.removeHandler(handler)
        startup_logger.setLevel(old_level)
    line = [r.getMessage() for r in records if "Agent budget" in r.getMessage()]
    assert line, records
    assert f"max_iterations={expected}" in line[0]
    assert ("unlimited" in line[0]) == (enforced == TURN_LIMIT_UNLIMITED)


@pytest.mark.asyncio
async def test_gateway_start_logs_unlimited_agent_budget(monkeypatch, tmp_path, caplog):
    """Drive the real startup path: ``GatewayRunner.start()`` must emit the budget line, and with the
    bridged ``unlimited`` spelling it says so instead of ``int()`` raising inside ``suppress()``
    and logging nothing (#116888)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "unlimited")  # what _bridge_max_turns_to_env writes for max_turns: unlimited
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert await runner.start() is True
    lines = [r.getMessage() for r in caplog.records if "Agent budget" in r.getMessage()]
    assert lines and "max_iterations=unlimited" in lines[0], caplog.text
