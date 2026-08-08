"""Webhook adapter must tolerate malformed int config fields."""

from __future__ import annotations

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_PORT,
    DEFAULT_RATE_LIMIT,
    WebhookAdapter,
)


def _make_config(**extra):
    data = {
        "host": "127.0.0.1",
        "port": DEFAULT_PORT,
        "routes": {},
        "secret": "test-secret",
    }
    data.update(extra)
    return PlatformConfig(enabled=True, extra=data)


def test_malformed_port_falls_back_to_default():
    adapter = WebhookAdapter(_make_config(port="not-a-port"))
    assert adapter._port == DEFAULT_PORT


def test_malformed_rate_limit_falls_back_to_default():
    adapter = WebhookAdapter(_make_config(rate_limit="nope"))
    assert adapter._rate_limit == DEFAULT_RATE_LIMIT


def test_malformed_max_body_bytes_falls_back_to_default():
    adapter = WebhookAdapter(_make_config(max_body_bytes="oops"))
    assert adapter._max_body_bytes == DEFAULT_MAX_BODY_BYTES


def test_valid_port_is_honored():
    adapter = WebhookAdapter(_make_config(port=9001))
    assert adapter._port == 9001
