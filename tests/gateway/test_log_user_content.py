"""Gateway logs at INFO+ carry no verbatim user content (#46006): `hermes debug share` uploads them."""

import logging

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.helpers import hash_for_log
from gateway.platforms.webhook import WebhookAdapter

_SECRET_TEXT = "my bank PIN is 4321, call me at alice@example.com"


def test_hash_for_log_is_stable_short_and_hides_the_value():
    digest = hash_for_log("room:!secret-room-id")
    assert digest == hash_for_log("room:!secret-room-id") != hash_for_log("room:!other")
    assert "secret" not in digest and len(digest) <= 12
    assert hash_for_log(None) == hash_for_log("") == "<none>"


@pytest.mark.asyncio
@pytest.mark.parametrize("deliver", ["_deliver_to", "_direct_deliver"])
async def test_webhook_log_delivery_keeps_the_body_out_of_info(caplog, deliver):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 0, "routes": {}}))
    caplog.set_level(logging.DEBUG, logger="gateway.platforms.webhook")
    args = ({"deliver": "log"}, "webhook:x", _SECRET_TEXT) if deliver == "_deliver_to" else (
        _SECRET_TEXT, {"deliver": "log"})
    result = await getattr(adapter, deliver)(*args)
    assert result.success
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert info and not any("PIN" in m or "alice@" in m for m in info), info
    assert any(_SECRET_TEXT in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
