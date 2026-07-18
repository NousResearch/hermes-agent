#!/usr/bin/env python3
"""Receipt signing abstraction for the passkey v2 authority."""

from __future__ import annotations

import base64
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import passkey_v2_protocol as protocol


class PasskeyV2SignerError(RuntimeError):
    """Stable receipt signer boundary failure."""


class ReceiptSigner:
    """In-memory interface to a separately protected Ed25519 receipt key."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise PasskeyV2SignerError("passkey_v2_receipt_signer_invalid")
        self._private_key = private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._private_key.public_key()

    @property
    def key_id(self) -> str:
        return protocol.sha256_bytes(self.public_key.public_bytes_raw())

    def sign(self, unsigned: Mapping[str, Any]) -> Mapping[str, Any]:
        signature = self._private_key.sign(protocol.canonical_json_bytes(unsigned))
        encoded = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
        signed = {**dict(unsigned), "signature_ed25519_b64url": encoded}
        return {**signed, "receipt_sha256": protocol.sha256_json(signed)}
