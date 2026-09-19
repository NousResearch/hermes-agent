"""Regression: #115439 — API-server-only gateways (no messaging platforms) must not
emit the 'No env user allowlists configured' startup warning. The message is
only relevant when a messaging adapter is actually connected.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.run_startup import GatewayStartupMixin


def _runner(connected: list[Platform]) -> GatewayStartupMixin:
    runner = object.__new__(GatewayStartupMixin)
    cfg = SimpleNamespace(get_connected_platforms=lambda: connected)
    runner.config = cfg
    return runner


def test_api_only_no_warning(monkeypatch, caplog):
    """Only API_SERVER + WEBHOOK connected → no allowlist warning."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    runner = _runner([Platform.API_SERVER, Platform.WEBHOOK])
    with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
        assert runner._start_check_access_policy() is False
    assert "No env user allowlists configured" not in caplog.text


def test_messaging_connected_still_warns(monkeypatch, caplog):
    """TELEGRAM connected → warning still fires when no allowlist configured."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    runner = _runner([Platform.TELEGRAM, Platform.API_SERVER])
    with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
        assert runner._start_check_access_policy() is False
    assert "No env user allowlists configured" in caplog.text


def test_local_only_no_warning(monkeypatch, caplog):
    """LOCAL only (CLI) → no allowlist warning."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner = _runner([Platform.LOCAL])
    with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
        assert runner._start_check_access_policy() is False
    assert "No env user allowlists configured" not in caplog.text


def test_messaging_with_allowlist_no_warning(monkeypatch, caplog):
    """TELEGRAM connected but allowlist set → no warning."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
    runner = _runner([Platform.TELEGRAM])
    with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
        assert runner._start_check_access_policy() is False
    assert "No env user allowlists configured" not in caplog.text


def test_get_connected_platforms_fails_still_warns(monkeypatch, caplog):
    """If get_connected_platforms raises → fall back to legacy behavior (warn)."""
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner = _runner([])
    with patch.object(
        runner.config, "get_connected_platforms", side_effect=RuntimeError("boom")
    ):
        with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
            assert runner._start_check_access_policy() is False
    assert "No env user allowlists configured" in caplog.text
