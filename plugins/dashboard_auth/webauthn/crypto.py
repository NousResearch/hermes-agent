"""fido2 adapter for the dashboard WebAuthn provider.

Everything Yubico-``fido2``-specific lives behind these seams so switching the
underlying WebAuthn library is one module, not a ceremony rewrite. Imported
lazily by the provider plugin (``fido2`` is an optional extra, not a core
dependency); importing this module pulls ``fido2`` in.

Pins verified against the library: ``fido2>=2.2.1,<3`` — the extra carries the
rationale (pure Python, single runtime dep, ``cryptography<52`` ceiling
compatible with the repo's exact ``cryptography==50.0.1`` pin, first-party RP
server API).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence

from fido2 import cbor
from fido2.server import Fido2Server
from fido2.utils import websafe_decode, websafe_encode
from fido2.webauthn import (
    AttestedCredentialData, AuthenticatorData, PublicKeyCredentialRpEntity,
    PublicKeyCredentialUserEntity, ResidentKeyRequirement, UserVerificationRequirement)

# WebAuthn auth-data byte layout (fixed by §6.1 of the spec):
# rp_id_hash(32) | flags(1) | sign_count(4) | [attested credential data ...]
_AUTH_DATA_COUNTER_OFFSET = 33
_USER_HANDLE_ID = b"hermes-operator"
_USER_NAME = "hermes-operator"


def sign_count(auth_data: AuthenticatorData) -> int:
    """The authenticator's monotonically increasing counter (raw auth-data bytes)."""
    return sign_count_from_bytes(bytes(auth_data))


def sign_count_from_bytes(raw: bytes) -> int:
    """Same counter, read from raw auth-data bytes (registration path)."""
    return int.from_bytes(raw[_AUTH_DATA_COUNTER_OFFSET:_AUTH_DATA_COUNTER_OFFSET + 4], "big")


def parked_to_state(challenge: str, user_verification: str) -> dict:
    """Rebuild fido2's per-ceremony state dict from the parked store entry."""
    if not user_verification:
        return {"challenge": challenge, "user_verification": None}
    return {"challenge": challenge, "user_verification": UserVerificationRequirement(user_verification)}


def cose_key_bytes(public_key) -> bytes:
    """Serialise a COSE public key (CBOR) for the credential store."""
    return cbor.encode(public_key)


def jsonable(value: Any) -> Any:
    """Convert a fido2 options/response mapping graph into plain JSON types.

    fido2's webauthn dataclasses are ``Mapping`` views whose ``bytes`` members
    already websafe-encode on access, but nested containers need full recursion
    before ``json.dumps``/``JSONResponse`` accept the structure.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return websafe_encode(value)
    if isinstance(value, Mapping):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [jsonable(v) for v in value]
    raise TypeError(f"unserialisable WebAuthn option value: {type(value).__name__}")


def verify_rp_id(rp_id: str, origin: str) -> bool:
    """The W3C RP-ID↔origin pairing rule, enforced by fido2 (fido2.rpid)."""
    from fido2.rpid import verify_rp_id as _verify

    return _verify(rp_id, origin)


class Fido2Adapter:
    """Server-side WebAuthn RP: registration + assertion ceremonies with
    attestation ignored (``none``) — a metadata-service dependency is irrelevant
    for a single-operator RP (design doc §9.3)."""

    def __init__(self, *, rp_id: str, rp_name: str) -> None:
        self._rp = PublicKeyCredentialRpEntity(id=rp_id, name=rp_name)
        self._rp_id_hash = hashlib.sha256(rp_id.encode("utf-8")).digest()
        self._server = Fido2Server(self._rp)  # attestation default None; origin
        # verification = fido2.rpid.verify_rp_id(rp.id, ...) via the default
        # closure — the W3C rule, verified in fido2's source.

    def register_options(
        self, *, user_verification: str,
        credentials: Sequence[AttestedCredentialData],
        challenge: Optional[bytes] = None) -> tuple[dict, dict]:
        """``CredentialCreationOptions`` JSON + the fido2 state dict to park."""
        options, state = self._server.register_begin(
            user=PublicKeyCredentialUserEntity(
                id=_USER_HANDLE_ID, name=_USER_NAME, display_name=_USER_NAME),
            credentials=credentials,
            resident_key_requirement=ResidentKeyRequirement.REQUIRED,  # discoverable: username-less login
            user_verification=_uv(user_verification), challenge=challenge)
        return jsonable(options), _state_to_parked(state)

    def complete_registration(
        self, state: Mapping[str, Any], response: Mapping[str, Any]) -> tuple[
            AttestedCredentialData, bool, bool, bytes]:
        """Verify a registration response against the parked ``state``; return
        ``(attested_credential_data, user_present, user_verified, raw_auth_data)``.
        Raises ``ValueError`` on ANY rejection (origin/rp-id-hash/challenge/UP)."""
        auth_data = self._server.register_complete(state, response)
        return (
            auth_data.credential_data, auth_data.is_user_present(),
            auth_data.is_user_verified(), bytes(auth_data))

    def authenticate_options(
        self, *, user_verification: str,
        credentials: Sequence[AttestedCredentialData],
        challenge: Optional[bytes] = None) -> tuple[dict, dict]:
        """``CredentialRequestOptions`` JSON + the fido2 state dict to park.

        ``credentials`` empty ⇒ discoverable-credential flow (``allowCredentials``
        omitted → the browser offers any passkey for this RP ID).
        """
        options, state = self._server.authenticate_begin(
            credentials=credentials, user_verification=_uv(user_verification),
            challenge=challenge)
        return jsonable(options), _state_to_parked(state)

    def complete_authentication(
        self, state: Mapping[str, Any], credentials: Sequence[AttestedCredentialData],
        response: Mapping[str, Any]) -> tuple[str, int, bool]:
        """Verify an assertion against the parked ``state`` + stored ``credentials``.

        Returns ``(credential_id_hex, new_sign_count, user_verified)``. fido2 has
        already enforced origin, challenge, RP-ID hash, UP, (UV when the parked
        state required it) and the signature; the flags/counter come from the
        matched credential's raw auth-data bytes (rp_id_hash|flags|counter|…).
        Raises ``ValueError`` on any rejection.
        """
        matched = self._server.authenticate_complete(state, credentials, response)
        raw = bytes(matched)
        flags = raw[32]
        return matched.credential_id.hex(), sign_count_from_bytes(raw), bool(flags & 0x04)

    def rebuild_credential(
        self, credential_id: bytes, public_key_cbor: bytes) -> AttestedCredentialData:
        """Rehydrate a stored credential (COSE public key → AttestedCredentialData)."""
        return AttestedCredentialData.create(
            bytes(16), credential_id, cbor.decode(public_key_cbor))


def _state_to_parked(state: Mapping[str, Any]) -> dict:
    """Plain-JSON snapshot of fido2's per-ceremony state (the ONLY server-held
    state between begin and complete)."""
    return {
        "challenge": state["challenge"],
        "user_verification": str(getattr(state.get("user_verification"), "value", None) or ""),
    }


def _uv(value: str):
    if value == "required":
        return UserVerificationRequirement.REQUIRED
    if value == "preferred":
        return UserVerificationRequirement.PREFERRED
    raise ValueError(f"unsupported user_verification: {value!r}")