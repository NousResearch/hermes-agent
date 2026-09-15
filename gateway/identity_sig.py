"""Ed25519 proofs for the gateway identity exported to tool subprocesses.

The private key is deliberately process-local.  Child tools receive only a
signed, minimal principal tuple and its public verification key; authorization
code must use the verified tuple rather than advisory ``HERMES_SESSION_*``
variables.
"""

import base64
import json
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

IDENTITY_ENV = "HERMES_SESSION_IDENTITY"
IDENTITY_PUBLIC_KEY_ENV = "HERMES_SESSION_IDENTITY_PUBLIC_KEY"
_IDENTITY_FIELDS = ("platform", "chat_id", "user_id")
_SIGNING_KEY = Ed25519PrivateKey.generate()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("identity value is missing")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def session_identity_public_key() -> str:
    """Return this gateway process's Ed25519 public key in base64url form."""
    raw = _SIGNING_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64encode(raw)


def sign_session_identity(*, platform: str, chat_id: str, user_id: str) -> tuple[str, str]:
    """Sign the complete, non-empty gateway principal tuple.

    Callers must not produce partial identities: an absent source user is not a
    principal that a tool may authorize.
    """
    values = (platform, chat_id, user_id)
    if not all(isinstance(value, str) and value for value in values):
        return "", ""
    identity = dict(zip(_IDENTITY_FIELDS, values))
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = _SIGNING_KEY.sign(payload)
    return f"v1.{_b64encode(payload)}.{_b64encode(signature)}", session_identity_public_key()


def verify_session_identity(token: str, public_key: str) -> dict[str, str] | None:
    """Return a verified principal tuple, or ``None`` for every invalid input."""
    try:
        version, encoded_payload, encoded_signature = token.split(".")
        if version != "v1":
            return None
        payload = _b64decode(encoded_payload)
        signature = _b64decode(encoded_signature)
        key = Ed25519PublicKey.from_public_bytes(_b64decode(public_key))
        key.verify(signature, payload)
        identity = json.loads(payload)
        if set(identity) != set(_IDENTITY_FIELDS):
            return None
        if not all(isinstance(identity[field], str) and identity[field] for field in _IDENTITY_FIELDS):
            return None
        return {field: identity[field] for field in _IDENTITY_FIELDS}
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, InvalidSignature):
        return None


def verify_session_identity_from_env(env: Mapping[str, str]) -> dict[str, str] | None:
    """Verify the signed child-environment identity without trusting raw vars."""
    return verify_session_identity(env.get(IDENTITY_ENV, ""), env.get(IDENTITY_PUBLIC_KEY_ENV, ""))
