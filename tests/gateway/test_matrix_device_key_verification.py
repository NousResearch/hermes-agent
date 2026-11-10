"""Regression tests: Matrix device-record verification must validate the full
server-side record, not just string-match the ed25519 identity key.

A record that advertises a diverging curve25519 encryption key — or whose
self-signature does not verify against the advertised ed25519 key — must fail
verification. Otherwise peers encrypt Megolm sessions to a key this
installation does not hold, and inbound messages become undecryptable with no
local error.
"""

import pytest
from mautrix.crypto.account import OlmAccount
from mautrix.types import DeviceID, UserID

USER_ID = UserID("@hermes-test:example.org")
DEVICE_ID = DeviceID("HERMESTEST")


def _signed_device_keys():
    """A real Olm account and its genuinely signed DeviceKeys record."""
    account = OlmAccount()
    return account, account.get_device_keys(USER_ID, DEVICE_ID)


class _SerializedRecord:
    """Carries a (possibly tampered) serialized DeviceKeys payload with the
    two access patterns the adapter uses: ``.keys`` and ``.serialize()``."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.keys = payload.get("keys", {})

    def serialize(self) -> dict:
        return dict(self._payload)


def test_self_signature_valid_for_real_signed_device_keys():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    account, device_keys = _signed_device_keys()

    assert MatrixAdapter._has_valid_device_self_signature(
        device_keys, str(USER_ID), str(DEVICE_ID), account.identity_keys["ed25519"]
    )
    # Both identity keys extract from the real record.
    assert (
        MatrixAdapter._extract_server_ed25519(device_keys)
        == account.identity_keys["ed25519"]
    )
    assert (
        MatrixAdapter._extract_server_curve25519(device_keys)
        == account.identity_keys["curve25519"]
    )


def test_self_signature_rejects_tampered_encryption_key():
    """Swapping curve25519 for another account's key must invalidate the record."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    account, device_keys = _signed_device_keys()
    other = OlmAccount()

    payload = device_keys.serialize()
    payload["keys"][f"curve25519:{DEVICE_ID}"] = other.identity_keys["curve25519"]
    tampered = _SerializedRecord(payload)

    # Raw ed25519 still matches — string comparison alone would accept this.
    assert MatrixAdapter._extract_server_ed25519(tampered) == account.identity_keys["ed25519"]
    assert not MatrixAdapter._has_valid_device_self_signature(
        tampered, str(USER_ID), str(DEVICE_ID), account.identity_keys["ed25519"]
    )


def test_self_signature_rejects_wrong_device_and_user():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    account, device_keys = _signed_device_keys()
    ed25519 = account.identity_keys["ed25519"]

    assert not MatrixAdapter._has_valid_device_self_signature(
        device_keys, str(USER_ID), "OTHERDEVICE", ed25519
    )
    assert not MatrixAdapter._has_valid_device_self_signature(
        device_keys, "@someone-else:example.org", str(DEVICE_ID), ed25519
    )


def test_self_signature_fails_closed_on_malformed_record():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    account, device_keys = _signed_device_keys()

    payload = device_keys.serialize()
    payload.pop("signatures", None)
    unsigned_record = _SerializedRecord(payload)

    assert not MatrixAdapter._has_valid_device_self_signature(
        unsigned_record, str(USER_ID), str(DEVICE_ID), account.identity_keys["ed25519"]
    )
