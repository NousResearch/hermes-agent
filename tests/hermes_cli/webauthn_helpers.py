"""Stub WebAuthn authenticator + shared harness for the passkey provider tests.

NOT a conftest.py — an importable helper module (discipline of
tests/hermes_cli/conftest_dashboard_auth.py).

:class:`StubWebAuthnAuthenticator` is a minimal virtual authenticator: it holds
one ES256 key pair, produces spec-shaped registration/assertion responses via
fido2's own dataclasses (no device, no network), and supports the failure modes
the contracts care about (UV absent for a ``required`` config; a foreign origin
in clientDataJSON; a wrong signature; a challenge replay).
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fido2 import cbor
from fido2.webauthn import (
    AttestationObject, AuthenticatorAssertionResponse, AuthenticatorAttestationResponse,
    AuthenticatorData, AttestedCredentialData, CollectedClientData, RegistrationResponse,
    AuthenticationResponse)
from fido2.utils import websafe_encode

FLAG_UP = 0x01
FLAG_UV = 0x04
FLAG_AT = 0x40  # attested credential data included (registration)


class StubWebAuthnAuthenticator:
    """One credential, fully in-memory, driven by explicit choices per ceremony."""

    def __init__(self, origin: str, rp_id: str, credential_id: bytes | None = None,
                 user_handle: bytes = b"hermes-operator"):
        self.origin = origin
        self.rp_id = rp_id
        self.credential_id = credential_id if credential_id is not None else bytes(range(1, 33))
        self.user_handle = user_handle
        self._key = ec.generate_private_key(ec.SECP256R1())
        self.counter = 0

    # ---- response builders --------------------------------------------------

    def register(self, challenge: bytes, *, uv: bool = True) -> RegistrationResponse:
        self.counter += 1
        client_data = self._client_data("webauthn.create", challenge)
        flags = FLAG_UP | (FLAG_UV if uv else 0) | FLAG_AT
        att_obj = AttestationObject.create(
            fmt="none",
            auth_data=hashlib.sha256(self.rp_id.encode()).digest()
            + bytes([flags]) + self.counter.to_bytes(4, "big")
            + bytes(16) + len(self.credential_id).to_bytes(2, "big")
            + self.credential_id + self._cose_public_key(),
            att_stmt={})
        return RegistrationResponse(
            raw_id=self.credential_id,
            response=AuthenticatorAttestationResponse(
                client_data=CollectedClientData(client_data), attestation_object=att_obj))

    def assert_login(self, challenge: bytes, *, uv: bool = True, wrong_key: bool = False,
                     origin: str | None = None, credential_id: bytes | None = None,
                     challenge_override: bytes | None = None) -> AuthenticationResponse:
        self.counter += 1
        origin = origin or self.origin
        client_data_bytes = self._client_data("webauthn.get", challenge_override or challenge,
                                              origin=origin)
        auth_raw = (hashlib.sha256(self.rp_id.encode()).digest()
                    + bytes([FLAG_UP | (FLAG_UV if uv else 0)])
                    + self.counter.to_bytes(4, "big"))
        if wrong_key:
            signer = ec.generate_private_key(ec.SECP256R1())
        else:
            signer = self._key
        signature = signer.sign(
            auth_raw + hashlib.sha256(client_data_bytes).digest(), ec.ECDSA(hashes.SHA256()))
        return AuthenticationResponse(
            raw_id=credential_id if credential_id is not None else self.credential_id,
            response=AuthenticatorAssertionResponse(
                client_data=CollectedClientData(client_data_bytes),
                authenticator_data=AuthenticatorData(auth_raw), signature=signature,
                user_handle=self.user_handle))

    # ---- server-side credential rehydration for the enrolment round trip ----

    def stored_record(self, handle_attested) -> dict:
        """The credential-store record shape for a registered
        ``AttestedCredentialData`` (mirrors complete_webauthn_registration)."""
        from fido2.utils import websafe_encode as enc

        return {"credential_id": enc(handle_attested.credential_data.credential_id),
                "public_key": enc(cbor.encode(handle_attested.credential_data.public_key)),
                "transports": ["internal"], "sign_count": self.counter,
                "label": "stub", "created_at": "2026-10-14T00:00:00Z"}

    def rebuild(self, record: dict) -> AttestedCredentialData:
        from fido2.utils import websafe_decode

        return AttestedCredentialData.create(
            bytes(16), websafe_decode(record["credential_id"]),
            cbor.decode(websafe_decode(record["public_key"])))

    def credential_json(self, cred) -> dict:
        """Browser-standard JSON wire form (camelCase, base64url) of a
        registration or assertion response — exactly what the fetch body carries."""
        import json as _json

        from fido2.utils import websafe_encode as enc

        response = cred.response
        fields = {"clientDataJSON": enc(response.client_data)}
        for snake, camel in (("attestation_object", "attestationObject"),
                             ("authenticator_data", "authenticatorData"),
                             ("signature", "signature"),
                             ("user_handle", "userHandle")):
            value = getattr(response, snake, None)
            if value is not None:
                fields[camel] = enc(value)
        return _json.loads(_json.dumps(
            {"id": enc(cred.raw_id), "rawId": enc(cred.raw_id),
             "type": "public-key", "response": fields}))

    # ---- internals ------------------------------------------------------------

    def _cose_public_key(self) -> bytes:
        pub = self._key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
        # ES256 COSE key: [-2] = x coordinate, [-3] = y coordinate (of the raw point)
        return cbor.encode({1: 2, 3: -7, -1: 1, -2: pub[1:33], -3: pub[33:]})

    def _client_data(self, typ, challenge: bytes, *, origin: str | None = None) -> bytes:
        return json.dumps({"type": typ, "challenge": websafe_encode(challenge),
                           "origin": origin or self.origin, "crossOrigin": False}).encode()


def forwarded_headers(origin: str) -> dict:
    """Headers a TLS edge would forward, so a TestClient request reconstructs the
    origin the passkey ceremony is exercised from."""
    parsed = urlparse(origin)
    return {"X-Forwarded-Proto": parsed.scheme, "X-Forwarded-Host": parsed.netloc}