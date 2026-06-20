"""Regression tests for gateway transport-liveness classification.

Background: the gateway process can be alive (pid present, gateway_state=running)
while a platform transport (e.g. Telegram long-poll) has been auto-paused after
repeated reconnect failures. Without transport-level classification the
operator sees "process healthy" and assumes messages flow - they don't.

These tests pin the contract for two helpers under `gateway.status`:

  - ``classify_gateway_transport_liveness(state, stale_after_seconds=900, now=...)``
  - ``classify_gateway_log_line(line)``

Tests are deterministic - no live gateway, no clock dependence.
"""

from datetime import datetime, timedelta, timezone

import pytest

from gateway import status as gateway_status


def _state(platforms, **extra):
    base = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["python", "-m", "hermes_cli.main", "gateway"],
        "start_time": 100,
        "gateway_state": "running",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 1,
        "platforms": platforms,
        "updated_at": "2026-06-20T12:00:00+00:00",
    }
    base.update(extra)
    return base


class TestClassifyGatewayTransportLiveness:
    def test_running_process_with_paused_telegram_fails_red(self):
        """Process alive + telegram paused = red verdict with HERMES_TELEGRAM_PAUSED.

        This is the incident shape: pid record present, gateway_state=running,
        but the long-poll transport has been parked. Liveness must catch it.
        """
        state = _state({
            "telegram": {
                "state": "paused",
                "error_code": None,
                "error_message": "auto-paused after 10 consecutive reconnect failures",
                "updated_at": "2026-06-20T12:00:00+00:00",
            }
        })

        result = gateway_status.classify_gateway_transport_liveness(state)

        assert result["healthy"] is False, result
        assert result["code"] == "HERMES_TELEGRAM_PAUSED"
        # platforms list must call out telegram specifically so operators
        # don't have to grep the summary for what's broken.
        platforms = result.get("platforms") or []
        assert any(
            (isinstance(p, str) and p == "telegram")
            or (isinstance(p, dict) and p.get("platform") == "telegram")
            for p in platforms
        ), platforms

    def test_connected_telegram_with_fresh_poll_is_healthy(self):
        """state=connected + last_successful_poll_at within window = green."""
        now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = _state({
            "telegram": {
                "state": "connected",
                "error_code": None,
                "error_message": None,
                "last_successful_poll_at": (now - timedelta(seconds=10)).isoformat(),
                "updated_at": (now - timedelta(seconds=10)).isoformat(),
            }
        })

        result = gateway_status.classify_gateway_transport_liveness(
            state, stale_after_seconds=300, now=now
        )

        assert result["healthy"] is True, result

    def test_default_stale_threshold_allows_max_heartbeat_interval(self):
        """Default stale threshold must stay above the heartbeat max clamp."""
        now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = _state({
            "telegram": {
                "state": "connected",
                "last_successful_poll_at": (now - timedelta(seconds=600)).isoformat(),
            }
        })

        result = gateway_status.classify_gateway_transport_liveness(state, now=now)

        assert result["healthy"] is True, result

    def test_stale_telegram_poll_beyond_threshold_fails_red(self):
        """Connected but last_successful_poll_at is stale -> red, telegram-stale code."""
        now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = _state({
            "telegram": {
                "state": "connected",
                "error_code": None,
                "error_message": None,
                "last_successful_poll_at": (now - timedelta(seconds=900)).isoformat(),
                "updated_at": (now - timedelta(seconds=900)).isoformat(),
            }
        })

        result = gateway_status.classify_gateway_transport_liveness(
            state, stale_after_seconds=300, now=now
        )

        assert result["healthy"] is False, result
        code = result["code"]
        assert "TELEGRAM" in code.upper() and "STALE" in code.upper(), code


class TestClassifyGatewayLogLine:
    def test_recognizes_telegram_pause_phrase(self):
        """The canonical pause phrase must classify as HERMES_TELEGRAM_PAUSED,
        and the recommended repair must point at restarting the Hermes gateway
        (or profile) ONLY - not Qwen, Ollama, Postgres, or any other healthy
        subsystem."""
        line = (
            "2026-06-20T12:00:00 [WARNING] gateway: "
            "Telegram paused after 10 consecutive reconnect failures"
        )

        result = gateway_status.classify_gateway_log_line(line)

        assert result is not None, "pause phrase must classify, not return None"
        assert result["code"] == "HERMES_TELEGRAM_PAUSED"

        repair = (result.get("recommended_repair") or "").lower()
        assert "hermes" in repair, repair
        assert ("gateway" in repair) or ("profile" in repair), repair
        # Repair MUST NOT recommend touching healthy subsystems.
        for forbidden in ("qwen", "ollama", "postgres", "postgresql", "api server"):
            assert forbidden not in repair, (
                f"recommended_repair must not mention {forbidden!r}: {repair!r}"
            )

    def test_unrelated_line_returns_none(self):
        """Liveness classifier should not pattern-match unrelated log lines."""
        line = "2026-06-20T12:00:00 [INFO] gateway: telegram polling resumed"
        result = gateway_status.classify_gateway_log_line(line)
        assert result is None, result
