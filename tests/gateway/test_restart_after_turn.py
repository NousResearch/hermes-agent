"""Unit tests for in-band restart after-turn deferral helpers (#77184)."""

from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    parse_restart_after_turn_timeout,
    resolve_restart_exit_wait_budget,
    resolve_systemd_timeout_stop_sec,
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
    assert resolve_restart_exit_wait_budget(0, 0, 0, headroom=15) == resolve_systemd_timeout_stop_sec(0, 0) + 15
    assert resolve_restart_exit_wait_budget(180, 21600, 0, headroom=15) == 21600 + resolve_systemd_timeout_stop_sec(180, 0) + 15
    for chat, cron in ((2, 80), (80, 2), (2, 0), (0, 0)):
        stop_envelope = resolve_systemd_timeout_stop_sec(chat, cron)
        assert resolve_restart_exit_wait_budget(chat, 3, cron, headroom=15) == 3 + stop_envelope + 15
    # A bounded observer uses the longer stop path, not the sum of independent drains.
    assert resolve_restart_exit_wait_budget(80, 3, 80, headroom=15) == 3 + resolve_systemd_timeout_stop_sec(80, 80) + 15
    assert resolve_restart_exit_wait_budget(2, 3, 0, headroom=15) < resolve_restart_exit_wait_budget(2, 3, 80, headroom=15)
    assert resolve_restart_exit_wait_budget("bad", "bad", 0, headroom="x") == 60.0


def test_cli_restart_wait_covers_configured_cron_drain(tmp_path, monkeypatch):
    import hermes_cli.gateway as gateway_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("HERMES_RESTART_DRAIN_TIMEOUT", "HERMES_RESTART_AFTER_TURN_TIMEOUT", "HERMES_CRON_DRAIN_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_drain_timeout: 2\n  restart_after_turn_timeout: 3\n  cron_drain_timeout: 80\n",
        encoding="utf-8",
    )
    assert gateway_cli._get_restart_exit_wait_budget() == 3 + resolve_systemd_timeout_stop_sec(2, 80) + 15
    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_drain_timeout: 2\n  restart_after_turn_timeout: 3\n  cron_drain_timeout: 0\n",
        encoding="utf-8",
    )
    assert gateway_cli._get_restart_exit_wait_budget() == 3 + resolve_systemd_timeout_stop_sec(2, 0) + 15


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
