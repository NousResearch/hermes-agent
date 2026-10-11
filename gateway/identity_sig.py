"""Ed25519 proofs for the gateway identity exported to tool subprocesses.

The signing key and current-invocation authority stay in the parent process.
Child tools receive only a signed, minimal principal tuple; authorization code
must verify it at the parent/tool-executor boundary rather than trusting
advisory ``HERMES_SESSION_*`` variables.
"""

import base64
import json
import secrets
from dataclasses import dataclass
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

IDENTITY_ENV = "HERMES_SESSION_IDENTITY"
_IDENTITY_FIELDS = ("platform", "chat_id", "user_id", "generation", "audience")
_SIGNING_KEY = Ed25519PrivateKey.generate()


@dataclass(frozen=True)
class SessionIdentityAuthority:
    """Parent-owned coordinates for accepting an identity proof.

    This object is deliberately never serialized into a child environment.  A
    tool executor keeps it alongside the current invocation and supplies it
    when checking a child-provided proof.
    """

    generation: str
    audience: str


def new_session_identity_authority(*, audience: str = "tool-executor") -> SessionIdentityAuthority:
    """Create a single-invocation authority for the consuming executor."""
    return SessionIdentityAuthority(generation=secrets.token_urlsafe(32), audience=audience)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("identity value is missing")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def sign_session_identity(
    *, platform: str, chat_id: str, user_id: str, authority: SessionIdentityAuthority,
) -> str:
    """Sign the complete, non-empty gateway principal tuple.

    Callers must not produce partial identities: an absent source user is not a
    principal that a tool may authorize.
    """
    values = (platform, chat_id, user_id, authority.generation, authority.audience)
    if not all(isinstance(value, str) and value for value in values):
        return ""
    identity = dict(zip(_IDENTITY_FIELDS, values))
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = _SIGNING_KEY.sign(payload)
    return f"v2.{_b64encode(payload)}.{_b64encode(signature)}"


def verify_session_identity(
    token: str, *, authority: SessionIdentityAuthority,
) -> dict[str, str] | None:
    """Verify a child proof against parent-owned current-invocation authority."""
    try:
        version, encoded_payload, encoded_signature = token.split(".")
        if version != "v2":
            return None
        payload = _b64decode(encoded_payload)
        signature = _b64decode(encoded_signature)
        _SIGNING_KEY.public_key().verify(signature, payload)
        identity = json.loads(payload)
        if set(identity) != set(_IDENTITY_FIELDS):
            return None
        if not all(isinstance(identity[field], str) and identity[field] for field in _IDENTITY_FIELDS):
            return None
        if (identity["generation"], identity["audience"]) != (authority.generation, authority.audience):
            return None
        return {field: identity[field] for field in ("platform", "chat_id", "user_id")}
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, InvalidSignature):
        return None


def verify_session_identity_from_env(
    env: Mapping[str, str], *, authority: SessionIdentityAuthority,
) -> dict[str, str] | None:
    """Verify a child assertion; its env is never accepted as a trust root."""
    return verify_session_identity(env.get(IDENTITY_ENV, ""), authority=authority)
