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
