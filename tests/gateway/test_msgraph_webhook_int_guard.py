"""MS Graph webhook adapter must tolerate malformed int config fields."""

from gateway.config import PlatformConfig
from gateway.platforms.msgraph_webhook import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_SEEN_RECEIPTS,
    DEFAULT_PORT,
    MSGraphWebhookAdapter,
)


def _make_adapter(**extra_overrides) -> MSGraphWebhookAdapter:
    extra = {
        "host": "127.0.0.1",
        "client_state": "expected-client-state",
        "accepted_resources": ["communications/onlineMeetings"],
    }
    extra.update(extra_overrides)
    return MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def test_malformed_port_falls_back_to_default():
    adapter = _make_adapter(port="bad-port")
    assert adapter._port == DEFAULT_PORT


def test_malformed_receipt_limit_falls_back_to_default():
    adapter = _make_adapter(max_seen_receipts="bad-limit")
    assert adapter._max_seen_receipts == DEFAULT_MAX_SEEN_RECEIPTS


def test_malformed_body_limit_falls_back_to_default():
    adapter = _make_adapter(max_body_bytes="bad-size")
    assert adapter._max_body_bytes == DEFAULT_MAX_BODY_BYTES


def test_valid_port_is_honored():
    adapter = _make_adapter(port=9006)
    assert adapter._port == 9006
