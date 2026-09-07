"""Lockstep contract with the connector's F-004 change (gateway-gateway #223).

`POST /relay/provision` used to return the stored per-gateway secret on every
replay, to anyone who reached the endpoint with the right gatewayId. The
connector now returns credential material ONLY to a caller that proves ownership
or possession, and answers ``secretIssued: false`` otherwise, with
``secret``/``deliveryKey`` absent rather than empty.

These are behaviour contracts on how this gateway must treat that response, not
snapshots of it:

  1. a secret-less response is a VALUE, not an error — the multi-platform boot
     loop must keep going, because every platform after the first shares one
     gatewayId and is therefore a replay of an unchanged binding;
  2. credentials are only ever written from a response that actually carries
     them, so a withheld response cannot blank creds already in hand;
  3. a genuinely malformed body still fails loudly.
"""

from __future__ import annotations

import os

import pytest

import gateway.relay as relay


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "GATEWAY_RELAY_URL",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "GATEWAY_RELAY_ENDPOINT",
        "GATEWAY_RELAY_ROUTE_KEYS",
        "GATEWAY_RELAY_PLATFORM",
        "GATEWAY_RELAY_BOT_ID",
        "GATEWAY_RELAY_INSTANCE_ID",
        "GATEWAY_RELAY_WAKE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)


class _Resp:
    """Minimal stand-in for the urlopen context manager `_post_provision` uses."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _post_returning(monkeypatch, body: str):
    import json as _json

    def _fake_json_post(url, token, payload, timeout):  # noqa: ANN001
        return _Resp(body.encode())

    monkeypatch.setattr(relay, "_json_post", _fake_json_post, raising=True)
    return _json


def test_secretless_provision_response_is_not_an_error(monkeypatch):
    """A withheld secret must be returned to the caller, not raised.

    This is the leg that gates the whole lockstep: the connector answers
    `secretIssued:false` on a replay by a caller that proved neither ownership
    nor possession, and on every platform after the first in a multi-platform
    boot. Raising here aborts a legitimate provision.
    """
    _post_returning(
        monkeypatch,
        '{"secretIssued": false, "tenant": "org-x", "gatewayId": "gw-1", "routeKeys": ["k"]}',
    )

    payload = relay._post_provision(
        provision_url="https://connector.example/relay/provision",
        access_token="tok",
        gateway_id="gw-1",
        platform="telegram",
        bot_id="BOT",
        gateway_endpoint="",
        route_keys=["k"],
    )

    assert payload["secretIssued"] is False
    assert "secret" not in payload
    assert payload["gatewayId"] == "gw-1"


def test_malformed_response_still_raises(monkeypatch):
    """No secret AND no `secretIssued` key is a pre-F-004 malformed body.

    The permissive branch must not swallow the failure it replaced.
    """
    _post_returning(monkeypatch, '{"tenant": "org-x", "gatewayId": "gw-1"}')

    with pytest.raises(RuntimeError, match="no secret"):
        relay._post_provision(
            provision_url="https://connector.example/relay/provision",
            access_token="tok",
            gateway_id="gw-1",
            platform="telegram",
            bot_id="BOT",
            gateway_endpoint="",
            route_keys=["k"],
        )


def test_multiplatform_boot_survives_a_secretless_second_platform(monkeypatch):
    """The credentials from platform 1 must survive platform 2 withholding them.

    All platforms share one gatewayId, so exactly one POST mints the secret and
    the rest are replays. The invariant: creds are written only from a response
    that carries them, and every platform is still provisioned.
    """
    calls: list[str] = []

    def _fake_post(**kwargs):
        calls.append(kwargs["platform"])
        if len(calls) == 1:
            return {
                "secretIssued": True,
                "secret": "a" * 64,
                "deliveryKey": "b" * 64,
                "tenant": "org-x",
                "gatewayId": kwargs["gateway_id"],
                "routeKeys": kwargs["route_keys"],
            }
        # Replay of an unchanged binding: no credential material.
        return {
            "secretIssued": False,
            "tenant": "org-x",
            "gatewayId": kwargs["gateway_id"],
            "routeKeys": kwargs["route_keys"],
        }

    monkeypatch.setattr(relay, "_post_provision", _fake_post, raising=True)
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", lambda: "tok", raising=True)
    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://connector.example")
    monkeypatch.setattr(
        relay,
        "relay_platform_identities",
        lambda: [("telegram", "BOT_T"), ("discord", "BOT_D")],
        raising=False,
    )

    ok = relay.self_provision_relay()

    assert ok is True
    assert calls == ["telegram", "discord"], "the secret-less platform must not abort the loop"
    # THE POINT: platform 2's withheld response must not blank platform 1's creds.
    assert os.environ["GATEWAY_RELAY_SECRET"] == "a" * 64
    assert os.environ["GATEWAY_RELAY_DELIVERY_KEY"] == "b" * 64


def test_a_secretless_first_platform_does_not_write_empty_creds(monkeypatch):
    """A withheld FIRST response must leave the env untouched for the next one.

    The old guard tested the same variable it wrote, so an empty write read as
    "still unset" on the next pass while deliveryKey had already been clobbered.
    """
    calls: list[str] = []

    def _fake_post(**kwargs):
        calls.append(kwargs["platform"])
        if len(calls) == 1:
            # Withheld, and it names a DIFFERENT gatewayId so we can see which
            # response actually stamped the env.
            return {
                "secretIssued": False,
                "tenant": "org-x",
                "gatewayId": "gw-from-withheld",
                "routeKeys": kwargs["route_keys"],
            }
        return {
            "secretIssued": True,
            "secret": "c" * 64,
            "deliveryKey": "d" * 64,
            "tenant": "org-x",
            "gatewayId": "gw-self",
            "routeKeys": kwargs["route_keys"],
        }

    monkeypatch.setattr(relay, "_post_provision", _fake_post, raising=True)
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", lambda: "tok", raising=True)
    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://connector.example")
    monkeypatch.setattr(
        relay,
        "relay_platform_identities",
        lambda: [("telegram", "BOT_T"), ("discord", "BOT_D")],
        raising=False,
    )

    ok = relay.self_provision_relay()

    assert ok is True
    # The later real credentials must land, not be shadowed by an empty write.
    assert os.environ["GATEWAY_RELAY_SECRET"] == "c" * 64
    assert os.environ["GATEWAY_RELAY_DELIVERY_KEY"] == "d" * 64
    # And the withheld FIRST response must not have written anything at all.
    # The old guard wrote `str(result.get("secret") or "")` unconditionally, so
    # GATEWAY_RELAY_ID was stamped from a response carrying no credentials — the
    # secret stayed falsy (masking it), but the id and delivery key were already
    # set from the wrong response. Assert the id came from the credential-bearing
    # response, which is the only one entitled to name the gateway.
    assert os.environ["GATEWAY_RELAY_ID"] == "gw-self"


def test_a_withheld_response_never_overwrites_credentials_in_hand(monkeypatch):
    """Creds already held must survive a later withheld response.

    Directly pins the write rule: only a response carrying a secret may touch
    the credential env vars. Under the old guard the env var it tested was the
    same one it wrote, so an empty write looked "unset" and the next pass
    re-entered — clobbering deliveryKey with the withheld response's absent one.
    """
    calls: list[str] = []

    def _fake_post(**kwargs):
        calls.append(kwargs["platform"])
        # EVERY platform withholds; nothing may be written.
        return {
            "secretIssued": False,
            "tenant": "org-x",
            "gatewayId": "gw-connector-chosen",
            "routeKeys": kwargs["route_keys"],
        }

    monkeypatch.setattr(relay, "_post_provision", _fake_post, raising=True)
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", lambda: "tok", raising=True)
    monkeypatch.setenv("GATEWAY_RELAY_URL", "https://connector.example")
    monkeypatch.setattr(
        relay,
        "relay_platform_identities",
        lambda: [("telegram", "BOT_T"), ("discord", "BOT_D")],
        raising=False,
    )

    relay.self_provision_relay()

    assert calls == ["telegram", "discord"]
    # Nothing was written, because nothing was issued.
    assert not os.environ.get("GATEWAY_RELAY_SECRET")
    assert not os.environ.get("GATEWAY_RELAY_DELIVERY_KEY")
    assert not os.environ.get("GATEWAY_RELAY_ID")
