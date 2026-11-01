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
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate

_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
# Entra rejects client assertions whose lifetime is longer than 10 minutes.
_ASSERTION_TTL_SECONDS = 600
_HEX_THUMBPRINT = re.compile(r"[^0-9a-fA-F]")
# A combined PEM starts with the certificate; pull the key block out so the loader
# does not have to accept the whole buffer. EC blocks stay in the pattern so a
# non-RSA key is rejected by name (RS256) instead of looking like a missing key.
_PRIVATE_KEY_BLOCK = re.compile(
    rb"-----BEGIN ((?:ENCRYPTED |RSA |EC )?)PRIVATE KEY-----"
    rb".*?"
    rb"-----END \1PRIVATE KEY-----",
    re.DOTALL,
)
_KEY_LABEL = "TEAMS_CLIENT_KEY_PATH"
_CERT_LABEL = "TEAMS_CLIENT_CERT_PATH"


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
    env var to fix when the material cannot sign an RS256 assertion."""
    cert_pem = _read_pem(cert_path, _CERT_LABEL) if cert_path else b""
    key_pem = _read_pem(key_path, _KEY_LABEL) if key_path else b""
    cert = _load_certificate(cert_pem)
    private_key, key_label = _signing_key(cert_pem, key_pem)
    _require_certificate_match(private_key, cert, key_label)
    header, value = _x5t_header(cert, thumbprint)
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
    try:
        token = jwt.encode(
            claims, prepared.private_key, algorithm="RS256",
            headers={prepared.x5t_header: prepared.x5t_value})
    except (TypeError, jwt.InvalidKeyError) as exc:
        # prepare_certificate rejects a non-RSA key. This keeps a later mint
        # (cron, attachments, the SDK callback) from surfacing a raw TypeError
        # if a PreparedCertificate is built some other way.
        raise ValueError(
            "Teams certificate auth could not sign the client assertion with RS256. "
            f"{_KEY_LABEL} or the key inside {_CERT_LABEL} must be an RSA private key") from exc
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


def _signing_key(cert_pem: bytes, key_pem: bytes):
    """(RSA key, env var it was loaded from). The key file wins over a key embedded in the cert PEM."""
    if key_pem:
        loaded = _load_private_key(key_pem, _KEY_LABEL)
        if loaded is not None:
            return loaded, _KEY_LABEL
    loaded = _load_private_key(cert_pem, _CERT_LABEL)
    if loaded is None:
        raise ValueError(
            "Teams certificate auth needs a PEM private key "
            f"({_KEY_LABEL}, or a private key inside {_CERT_LABEL})")
    return loaded, _CERT_LABEL


def _load_private_key(pem: bytes, label: str):
    """An RSA key, or ``None`` when ``pem`` has no private-key block.

    A non-RSA key (EC, including ``EC PRIVATE KEY``) raises ``ValueError`` naming
    ``label``. Entra client assertions from this adapter are RS256 only.
    """
    block = _private_key_pem(pem)
    if block is None:
        return None
    try:
        key = load_pem_private_key(block, password=None)
    except TypeError as exc:
        raise ValueError(f"{label} is an encrypted private key; provide an unencrypted PEM") from exc
    except ValueError as exc:
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise ValueError(f"{label} is an encrypted private key; provide an unencrypted PEM") from exc
        raise ValueError(f"{label} is not a usable PEM private key") from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError(f"{label} must be an RSA private key; Entra certificate auth signs with RS256")
    return key


def _load_certificate(pem: bytes):
    """The certificate in ``pem``, or ``None`` when it has no certificate block."""
    if not pem or b"CERTIFICATE" not in pem:
        return None
    try:
        return load_pem_x509_certificate(pem)
    except ValueError as exc:
        raise ValueError(f"{_CERT_LABEL} is not a usable PEM certificate") from exc


def _require_certificate_match(private_key, cert, key_label: str) -> None:
    """The signing key and the certificate must be the same pair when a cert PEM is set.

    A rotated key left next to the previous certificate otherwise signs assertions
    Entra rejects (AADSTS700027) on every token request.
    """
    if cert is None:
        return
    cert_public = cert.public_key()
    if isinstance(cert_public, rsa.RSAPublicKey) and cert_public == private_key.public_key():
        return
    if key_label == _CERT_LABEL:
        raise ValueError(f"The private key inside {_CERT_LABEL} does not match the certificate")
    raise ValueError(f"{key_label} does not match the certificate in {_CERT_LABEL}")


def _decode_thumbprint(value: str) -> bytes:
    hex_str = _HEX_THUMBPRINT.sub("", value)
    if len(hex_str) not in (40, 64):
        raise ValueError(
            "TEAMS_CLIENT_CERT_THUMBPRINT must be a SHA-1 (40 hex chars) or "
            "SHA-256 (64 hex chars) thumbprint")
    return bytes.fromhex(hex_str)


def _x5t_header(cert, thumbprint: str) -> tuple[str, str]:
    supplied = _decode_thumbprint(thumbprint) if thumbprint else None
    der = cert.public_bytes(Encoding.DER) if cert is not None else None
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
