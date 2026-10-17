"""Entra ID certificate client credentials (RFC 7523 ``client_assertion``).

Azure AD accepts a short-lived JWT in place of a client secret. The JWT is
RS256-signed with the app registration's certificate private key. The header
carries ``x5t`` (SHA-1) or ``x5t#S256`` (SHA-256) — the base64url thumbprint
the portal shows — so the token endpoint can pick the registered certificate.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.serialization import Encoding, load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate

_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
# Entra rejects client assertions whose lifetime is longer than 10 minutes.
_ASSERTION_TTL_SECONDS = 600
_HEX_THUMBPRINT = re.compile(r"[^0-9a-fA-F]")
# A combined PEM starts with the certificate; pull the key block out so the loader
# does not have to accept the whole buffer.
_PRIVATE_KEY_BLOCK = re.compile(
    rb"-----BEGIN ((?:ENCRYPTED |RSA |EC )?)PRIVATE KEY-----"
    rb".*?"
    rb"-----END \1PRIVATE KEY-----",
    re.DOTALL,
)


@dataclass(frozen=True)
class PreparedCertificate:
    """Signing key plus the JWT header claim that names the certificate."""

    private_key: Any
    x5t_header: str
    x5t_value: str


def certificate_configured(cert_path: str, key_path: str, thumbprint: str) -> bool:
    """True when any certificate setting is present (cert then replaces the secret)."""
    return bool(cert_path or key_path or thumbprint)


def prepare_certificate(cert_path: str, key_path: str, thumbprint: str) -> PreparedCertificate:
    """Load the PEM key and resolve ``x5t`` / ``x5t#S256``. Raises ``ValueError`` with the
    env var to fix when the material cannot sign an assertion."""
    cert_pem = _read_pem(cert_path, "TEAMS_CLIENT_CERT_PATH") if cert_path else b""
    key_pem = _read_pem(key_path, "TEAMS_CLIENT_KEY_PATH") if key_path else b""
    private_key = _load_private_key(key_pem, "TEAMS_CLIENT_KEY_PATH") if key_pem else None
    if private_key is None:
        private_key = _load_private_key(cert_pem, "TEAMS_CLIENT_CERT_PATH")
    if private_key is None:
        raise ValueError(
            "Teams certificate auth needs a PEM private key "
            "(TEAMS_CLIENT_KEY_PATH, or a private key inside TEAMS_CLIENT_CERT_PATH)")
    header, value = _x5t_header(cert_pem, thumbprint)
    return PreparedCertificate(private_key, header, value)


def build_client_assertion(
    prepared: PreparedCertificate, *, tenant_id: str, client_id: str, now: int | None = None,
) -> str:
    """Signed client assertion for ``tenant_id``'s v2 token endpoint."""
    issued = int(time.time()) if now is None else int(now)
    claims = {
        "aud": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        "iss": client_id, "sub": client_id, "jti": str(uuid.uuid4()),
        "nbf": issued, "exp": issued + _ASSERTION_TTL_SECONDS,
    }
    token = jwt.encode(
        claims, prepared.private_key, algorithm="RS256",
        headers={prepared.x5t_header: prepared.x5t_value})
    return token if isinstance(token, str) else token.decode("ascii")


def client_assertion_fields(
    cert_path: str, key_path: str, thumbprint: str, *, tenant_id: str, client_id: str,
) -> dict[str, str]:
    """Form fields that replace ``client_secret`` on the token request."""
    prepared = prepare_certificate(cert_path, key_path, thumbprint)
    return {
        "client_assertion_type": _ASSERTION_TYPE,
        "client_assertion": build_client_assertion(prepared, tenant_id=tenant_id, client_id=client_id),
    }


def _read_pem(path: str, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} ({path}) could not be read: {exc.strerror or exc}") from exc


def _private_key_pem(pem: bytes) -> bytes | None:
    match = _PRIVATE_KEY_BLOCK.search(pem or b"")
    return match.group(0) if match else None


def _load_private_key(pem: bytes, label: str):
    """The key object, or ``None`` when ``pem`` has no private-key block."""
    block = _private_key_pem(pem)
    if block is None:
        return None
    try:
        return load_pem_private_key(block, password=None)
    except TypeError as exc:
        raise ValueError(f"{label} is an encrypted private key; provide an unencrypted PEM") from exc
    except ValueError as exc:
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise ValueError(f"{label} is an encrypted private key; provide an unencrypted PEM") from exc
        raise ValueError(f"{label} is not a usable PEM private key") from exc


def _cert_der(pem: bytes) -> bytes | None:
    if b"CERTIFICATE" not in pem:
        return None
    try:
        cert = load_pem_x509_certificate(pem)
    except ValueError as exc:
        raise ValueError("TEAMS_CLIENT_CERT_PATH is not a usable PEM certificate") from exc
    return cert.public_bytes(Encoding.DER)


def _decode_thumbprint(value: str) -> bytes:
    hex_str = _HEX_THUMBPRINT.sub("", value)
    if len(hex_str) not in (40, 64):
        raise ValueError(
            "TEAMS_CLIENT_CERT_THUMBPRINT must be a SHA-1 (40 hex chars) or "
            "SHA-256 (64 hex chars) thumbprint")
    return bytes.fromhex(hex_str)


def _x5t_header(cert_pem: bytes, thumbprint: str) -> tuple[str, str]:
    supplied = _decode_thumbprint(thumbprint) if thumbprint else None
    der = _cert_der(cert_pem) if cert_pem else None
    if supplied is None and der is None:
        raise ValueError(
            "Teams certificate auth needs TEAMS_CLIENT_CERT_PATH or TEAMS_CLIENT_CERT_THUMBPRINT "
            "so the client assertion can name the certificate")
    if supplied is not None and der is not None and supplied != _digest_for(der, len(supplied)):
        raise ValueError("TEAMS_CLIENT_CERT_THUMBPRINT does not match TEAMS_CLIENT_CERT_PATH")
    raw = supplied if supplied is not None else _digest_for(der, 20)
    header = "x5t#S256" if len(raw) == 32 else "x5t"
    return header, base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _digest_for(der: bytes, nbytes: int) -> bytes:
    # ``x5t`` is defined as SHA-1 of the certificate DER (Azure portal "Thumbprint").
    # SHA-256 is only used when the configured thumbprint is 32 bytes (``x5t#S256``).
    if nbytes == 32:
        return hashlib.sha256(der).digest()
    return hashlib.sha1(der, usedforsecurity=False).digest()
