"""Test credentials: scrypt bootstrap and opaque access tokens."""
from __future__ import annotations
import hashlib, hmac, secrets

SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
def bootstrap_record(secret: str) -> tuple[str, str]:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(secret.encode(), salt=salt, **SCRYPT)
    return salt.hex(), digest.hex()
def verify_bootstrap(secret: str, salt_hex: str, digest_hex: str) -> bool:
    candidate = hashlib.scrypt(secret.encode(), salt=bytes.fromhex(salt_hex), **SCRYPT)
    return hmac.compare_digest(candidate.hex(), digest_hex)
def new_access_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, hashlib.sha256(token.encode()).hexdigest()
def token_hash(token: str) -> str: return hashlib.sha256(token.encode()).hexdigest()
def verify_access_token(token: str, stored_digest: str) -> bool:
    candidate = token_hash(token)
    if not isinstance(stored_digest, str) or len(stored_digest) != len(candidate):
        return False
    return hmac.compare_digest(candidate, stored_digest)
