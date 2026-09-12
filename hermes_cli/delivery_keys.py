"""Idempotency-key and delivery-id derivation for peer/bot sends.

Single home for the three content-derived formulas that make a retry reproduce
the same logical-message identity (peer-send spec §4):

- ``body_sha256``          — sha256 of the UTF-8 body (§4.1).
- ``derive_delivery_key``  — the derived ``auto:<digest>:<bucket>`` key (§4.1).
- ``delivery_fingerprint`` — the canonical-JSON fingerprint compared with
  ``hmac.compare_digest`` by the run-idempotency store (§4.4).
- ``delivery_id_from``     — the deterministic delivery/run id (§4.3).

Nothing random may enter any of these: a retry with the same logical inputs must
reproduce them bit-for-bit, which is what makes dedup exactly-once. Per-attempt
identity belongs in ``send_id`` (a uuid4 minted per HTTP attempt), never here.

This module is imported by both the peer client (``hermes_cli/subcommands/peer.py``)
and the gateway (which already imports ``hermes_cli.*``), so sender and receiver
agree without sharing state.
"""

from __future__ import annotations

import hashlib
import json
import time

# The derived-key digest seeds the version tag so a future change to the
# derivation shape cannot be silently misread as the same logical message.
_DERIVED_KEY_VERSION = "v1"


def _sha256hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def body_sha256(message_body: str) -> str:
    """SHA-256 (lowercase hex) of the message body's UTF-8 encoding (§4.1)."""
    return _sha256hex(message_body.encode("utf-8"))


def derive_delivery_key(
    sender_profile: str,
    target_profile: str,
    session_id: str,
    message_body: str,
    now_epoch_seconds: int | None = None,
    dedup_window_seconds: int = 900,
) -> str:
    """Derive the ``auto:<digest>:<bucket>`` idempotency key (§4.1).

    The same body from the same sender to the same target session inside one
    ``dedup_window_seconds`` bucket is one logical message (a retry storm
    coalesces); the same body in a later bucket is a new message. The honest
    bound is one bucket — a caller needing a longer guarantee must pass an
    explicit ``Idempotency-Key``.
    """
    now = int(now_epoch_seconds) if now_epoch_seconds is not None else int(time.time())
    body_hash = body_sha256(message_body)
    bucket = now // int(dedup_window_seconds)
    digest_input = (
        _DERIVED_KEY_VERSION
        + "\0"
        + sender_profile
        + "\0"
        + target_profile
        + "\0"
        + session_id
        + "\0"
        + body_hash
    )
    digest = _sha256hex(digest_input.encode("utf-8"))[:24]
    return f"auto:{digest}:{bucket}"


def delivery_fingerprint(
    sender_profile: str,
    target_profile: str,
    session_id: str,
    message_body: str,
) -> str:
    """The canonical-JSON fingerprint stored on the idempotency row (§4.4).

    Mirrors the run path's fingerprint idiom (``sort_keys=True``,
    ``separators=(",", ":")``, ``ensure_ascii=False``). The ``"v": 1`` tag means
    a future fingerprint change cannot be misread as a conflict.
    """
    obj = {
        "v": 1,
        "kind": "bot_send",
        "sender": sender_profile,
        "target": target_profile,
        "session": session_id,
        "body_sha256": body_sha256(message_body),
    }
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256hex(canonical.encode("utf-8"))


def delivery_id_from(scope: str, idempotency_key: str) -> str:
    """The deterministic delivery id (also the idempotency row's ``run_id``, §4.3)."""
    return _sha256hex((scope + "\x00" + idempotency_key).encode("utf-8"))[:32]
