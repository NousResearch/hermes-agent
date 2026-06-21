"""Key-derivation functions for passphrase- and recovery-code-based key slots.

A passphrase never encrypts data directly. It derives a 32-byte Key Encryption
Key (KEK) that wraps the Data Encryption Key (DEK) stored in the keystore.

Two KDFs are supported, identified by a one-byte ``kdf_id`` persisted in the
keystore slot so :func:`derive_kek` can reproduce the derivation later:

* ``KDF_ARGON2ID`` — memory-hard, side-channel-resistant. Preferred. Requires
  the optional ``argon2-cffi`` package.
* ``KDF_SCRYPT`` — stdlib ``hashlib.scrypt``. No extra dependency; used as the
  fallback when ``argon2-cffi`` is unavailable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from .errors import DependencyError

KDF_SCRYPT = 1
KDF_ARGON2ID = 2

# Argon2id defaults. memory_cost is in KiB. 128 MiB keeps a one-off unlock well
# under the RAM budget of a small ($5) VPS while remaining expensive to brute
# force; raise ``security.encryption.argon2`` in config.yaml for more headroom.
DEFAULT_ARGON2_PARAMS: Dict[str, int] = {
    "time_cost": 3,
    "memory_cost_kib": 131072,
    "parallelism": 4,
}

# Recovery codes are 160-bit CSPRNG secrets, not operator-chosen passphrases.
# Memory-hard Argon2 protects low-entropy passphrases from offline guessing; it
# does not strengthen a uniformly random 160-bit value. Minimal cost keeps
# recovery unlock practical on small VPS hosts.
RECOVERY_ARGON2_PARAMS: Dict[str, int] = {
    "time_cost": 1,
    "memory_cost_kib": 1024,
    "parallelism": 1,
}

# scrypt cost parameters (n must be a power of two).
_SCRYPT_N = 1 << 15
_SCRYPT_R = 8
_SCRYPT_P = 1
# CPython caps scrypt memory at maxmem; n*r*128 must fit. 64 MiB is ample.
_SCRYPT_MAXMEM = 64 * 1024 * 1024


def argon2_available() -> bool:
    """Return True when the ``argon2-cffi`` package can be imported."""
    try:
        import argon2.low_level  # noqa: F401
    except ImportError:
        return False
    return True


def preferred_kdf_id() -> int:
    """Return ``KDF_ARGON2ID`` when argon2-cffi is installed, else ``KDF_SCRYPT``."""
    return KDF_ARGON2ID if argon2_available() else KDF_SCRYPT


def derive_kek(
    passphrase: bytes,
    salt: bytes,
    kdf_id: int,
    params: Dict[str, Any] | None = None,
) -> bytes:
    """Derive a 32-byte KEK from *passphrase* and *salt* using the named KDF."""
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    if len(salt) < 16:
        raise ValueError("salt must be at least 16 bytes")

    if kdf_id == KDF_ARGON2ID:
        try:
            from argon2.low_level import Type, hash_secret_raw
        except ImportError as exc:
            raise DependencyError(
                "This keystore slot was created with Argon2id but the "
                "'argon2-cffi' package is not installed. Install it with:  "
                "pip install 'hermes-agent[encryption]'"
            ) from exc
        p = {**DEFAULT_ARGON2_PARAMS, **(params or {})}
        return hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=int(p["time_cost"]),
            memory_cost=int(p["memory_cost_kib"]),
            parallelism=int(p["parallelism"]),
            hash_len=32,
            type=Type.ID,
        )

    if kdf_id == KDF_SCRYPT:
        return hashlib.scrypt(
            passphrase,
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            maxmem=_SCRYPT_MAXMEM,
            dklen=32,
        )

    raise ValueError(f"unknown kdf_id {kdf_id}")
