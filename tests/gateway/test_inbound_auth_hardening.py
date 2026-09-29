"""Inbound webhook/callback auth: every remote-supplied secret or signature goes through a
timing-safe compare that fails closed on hostile input, and a signed timestamp outside the
replay window is refused even when the signature is valid."""

import asyncio
import hashlib
import hmac
import json
import os
import time
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig

SECRET = "s3cret"
_AES_KEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
# A byte that is not UTF-8 reaches handlers as a lone surrogate (aiohttp surrogate-escapes header
# bytes; JSON decodes "\udcff" escapes to the same), so it is the hostile input that matters.
_HOSTILE = "tök\udcff"


def _wecom_crypt():
    from plugins.platforms.wecom.wecom_crypto import WXBizMsgCrypt
    return WXBizMsgCrypt(SECRET, _AES_KEY, "ww1234567890")


def _wecom_accepts(timestamp: str, presented_signature=None) -> bool:
    from plugins.platforms.wecom.wecom_crypto import SignatureError
    crypt = _wecom_crypt()
    root = ET.fromstring(crypt.encrypt("<xml/>", nonce="n", timestamp=timestamp))
    try:
        crypt.decrypt(presented_signature or root.findtext("MsgSignature"), timestamp, "n", root.findtext("Encrypt"))
    except SignatureError:
        return False
    return True


def _feishu_accepts(timestamp: str) -> bool:
    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import FeishuAdapter
    env = {"FEISHU_APP_ID": "cli", "FEISHU_APP_SECRET": "sec", "FEISHU_ENCRYPT_KEY": SECRET,
           "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict(os.environ, env, clear=True):
        adapter = FeishuAdapter(PlatformConfig())
    body = b'{"type":"event"}'
    sig = hashlib.sha256(f"{timestamp}n{SECRET}".encode() + body).hexdigest()
    headers = {"x-lark-request-timestamp": timestamp, "x-lark-request-nonce": "n", "x-lark-signature": sig}
    return adapter._is_webhook_signature_valid(headers, body)


@pytest.mark.parametrize("accepts", [_wecom_accepts, _feishu_accepts], ids=["wecom", "feishu"])
def test_validly_signed_request_outside_replay_window_is_refused(accepts):
    now = int(time.time())
    assert accepts(str(now))
    assert not accepts(str(now - 600))
    assert not accepts(str(now + 600))


def _bluebubbles(presented: str) -> bool:
    pytest.importorskip("aiohttp")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": SECRET}))

    class _Request:
        query = {"password": presented}
        headers: dict = {}

        async def read(self):
            return json.dumps({"type": "typing-indicator"}).encode()

    return asyncio.run(adapter._handle_webhook(_Request())).status != 401


def _google_meet(presented: str) -> bool:
    from plugins.google_meet.node import protocol
    msg = {"type": "ping", "id": "1", "token": presented, "payload": {}}
    return protocol.validate_request(msg, SECRET)[0]


def _wecom_signature(presented: str) -> bool:
    return _wecom_accepts(str(int(time.time())), None if presented == SECRET else presented)


def _a2a(presented: str) -> bool:
    from plugins.platforms.a2a.security import A2ASecurityContext
    ctx = A2ASecurityContext(bearer_token=SECRET, peer_tokens=(), trusted_peers=frozenset(),
                             allow_all_users=False, requested_host="0.0.0.0", push_secret=SECRET)
    return ctx.authenticate(f"Bearer {presented}", "10.0.0.1") is not None


@pytest.mark.parametrize("accepts", [_bluebubbles, _google_meet, _wecom_signature, _a2a],
                         ids=["bluebubbles", "google_meet", "wecom", "a2a"])
def test_presented_secret_is_compared_timing_safe_and_fails_closed(accepts, monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))

    assert accepts(SECRET)
    assert calls, "the secret was not compared with hmac.compare_digest"
    assert not accepts(SECRET + "x")
    assert not accepts(_HOSTILE)
