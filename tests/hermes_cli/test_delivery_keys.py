"""Pin the derivation formulas on one concrete, self-consistent worked vector.

Every formula in ``hermes_cli/delivery_keys.py`` must reproduce this vector's
id set bit-for-bit. The profile identifiers here are neutral placeholders; the
derived values follow from exactly these inputs — recompute them with a
standalone implementation of the documented formulas before changing them.
This test is the gate (§7 acceptance criterion, "A coder MUST be able to
reproduce this table before writing code").
"""

import re

from hermes_cli.delivery_keys import (
    body_sha256,
    delivery_fingerprint,
    delivery_id_from,
    derive_delivery_key,
)


VECTOR = {
    "sender_profile": "lane-beta",
    "target_profile": "lane-alpha",
    "session_id": "20260904_095123_f7675a",
    "message_body": "please run the R1 audit",
    "now_epoch_seconds": 1789167519,
    "scope": "a0798feeec13171b4f272365e9934d712f56844a63fd5ffac802e48cd6c549ae",
    "body_hash": "26e0cb723da52e955fa1c9ba7cf53d3b0e109b495ccb734fe0826fd6d67cd39b",
    "bucket": 1987963,
    "idempotency_key": "auto:17438bb8c7d20a3acc72f75e:1987963",
    "fingerprint": "3bca268f451f652721cee40af2ac2b1005c070452ba233fc621d22fd3ce577f8",
    "delivery_id": "bad400fc9cc9365bb92eae4a67980d3d",
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
