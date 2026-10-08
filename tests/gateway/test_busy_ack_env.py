"""``HERMES_GATEWAY_BUSY_ACK_ENABLED``: shared truthy set, and unset/blank keeps the default (on)."""

import pytest

from gateway.run_busy import busy_ack_env_enabled


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_keeps_ack_on(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    else:
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raw)

    assert busy_ack_env_enabled() is True


@pytest.mark.parametrize("raw, expected", [
    ("true", True), ("True", True), (" true ", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_values_use_shared_truthy_set(monkeypatch, raw, expected):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raw)

    assert busy_ack_env_enabled() is expected
