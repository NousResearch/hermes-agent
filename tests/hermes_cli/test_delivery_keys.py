"""Reproduce the spec §4.10 worked vector and pin the derivation formulas.

Every formula in ``hermes_cli/delivery_keys.py`` must reproduce the spec's
single concrete id set bit-for-bit; this test is the gate (§7 acceptance
criterion, "A coder MUST be able to reproduce this table before writing code").
"""

import re

from hermes_cli.delivery_keys import (
    body_sha256,
    delivery_fingerprint,
    delivery_id_from,
    derive_delivery_key,
)


VECTOR = {
    "sender_profile": "yoyodine-majordomo",
    "target_profile": "yoyodine-coder",
    "session_id": "20260904_095123_f7675a",
    "message_body": "please run the R1 audit",
    "now_epoch_seconds": 1789167519,
    "scope": "a0798feeec13171b4f272365e9934d712f56844a63fd5ffac802e48cd6c549ae",
    "body_hash": "26e0cb723da52e955fa1c9ba7cf53d3b0e109b495ccb734fe0826fd6d67cd39b",
    "bucket": 1987963,
    "idempotency_key": "auto:d8559a73fbed247d9a25452e:1987963",
    "fingerprint": "1ff6f45b5e4a8eca1cb401179af90dfadf262cc492a16e80892da6f6aa30c759",
    "delivery_id": "9030e165fe4a499de11e29f7a5a8eb4b",
}


def test_body_sha256_vector():
    assert body_sha256(VECTOR["message_body"]) == VECTOR["body_hash"]


def test_derive_delivery_key_vector():
    key = derive_delivery_key(
        VECTOR["sender_profile"],
        VECTOR["target_profile"],
        VECTOR["session_id"],
        VECTOR["message_body"],
        now_epoch_seconds=VECTOR["now_epoch_seconds"],
    )
    assert key == VECTOR["idempotency_key"]


def test_delivery_fingerprint_vector():
    fp = delivery_fingerprint(
        VECTOR["sender_profile"],
        VECTOR["target_profile"],
        VECTOR["session_id"],
        VECTOR["message_body"],
    )
    assert fp == VECTOR["fingerprint"]


def test_delivery_id_vector():
    did = delivery_id_from(VECTOR["scope"], VECTOR["idempotency_key"])
    assert did == VECTOR["delivery_id"]
    assert re.fullmatch(r"[0-9a-f]{32,64}", did)


def test_derived_key_is_deterministic():
    k1 = derive_delivery_key("s", "t", "ses", "body", now_epoch_seconds=1234)
    k2 = derive_delivery_key("s", "t", "ses", "body", now_epoch_seconds=1234)
    assert k1 == k2


def test_derived_key_is_bucket_sensitive():
    # same body in a later bucket is a NEW logical message (B-T8 / §4.1)
    k1 = derive_delivery_key("s", "t", "ses", "body", now_epoch_seconds=1789167519)
    k2 = derive_delivery_key("s", "t", "ses", "body", now_epoch_seconds=1789167519 + 900)
    assert k1 != k2


def test_delivery_id_is_content_derived():
    assert delivery_id_from("scope", "key") == delivery_id_from("scope", "key")
    assert delivery_id_from("scope", "key") != delivery_id_from("scope2", "key")
    assert delivery_id_from("scope", "key") != delivery_id_from("scope", "key2")
