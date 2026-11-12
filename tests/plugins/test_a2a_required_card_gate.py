"""A2A outbound signature enforcement when the peer card cannot be fetched."""

import json
import urllib.error

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import algorithms

from plugins.platforms.a2a import security, tools


def test_required_card_fetch_failure_refuses_send(monkeypatch):
    """Requiring a signed card must fail closed before posting to the peer RPC URL."""
    public_key = ec.generate_private_key(ec.SECP256R1()).public_key()
    jwk = json.loads(algorithms.get_default_algorithms()["ES256"].to_jwk(public_key))
    jwk["kid"] = "trusted-peer"
    settings = {
        "A2A_TRUSTED_SIGNERS": json.dumps({"keys": [jwk]}),
        "A2A_REQUIRE_CARD_SIGNATURE": "true",
    }
    monkeypatch.setattr(security, "_startup_env", lambda name: settings.get(name, ""))

    def card_unavailable(*args, **kwargs):
        raise urllib.error.URLError("card endpoint unavailable")

    posted_urls = []

    def record_post(url, *args, **kwargs):
        posted_urls.append(url)
        return {"result": {}}

    monkeypatch.setattr(tools, "_fetch_card", card_unavailable)
    monkeypatch.setattr(tools, "_http_post_json", record_post)
    monkeypatch.setattr(security, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools.protocol, "persist_message", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Agent Card rejected"):
        tools._send_task("peer", {"url": "https://peer.example"}, "hello", "")
    assert posted_urls == []
