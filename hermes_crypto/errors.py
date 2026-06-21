"""Exception hierarchy for the hermes_crypto encryption-at-rest layer.

Pure stdlib — safe to import from anywhere, including the import-safe
``hermes_crypto.__init__`` and the early credential loaders.
"""

from __future__ import annotations


class HermesCryptoError(Exception):
    """Base class for every error raised by the encryption layer."""


class DependencyError(HermesCryptoError):
    """A required optional dependency is missing.

    Raised when encryption is enabled but ``cryptography`` / ``argon2-cffi`` /
    ``keyring`` / ``sqlcipher3`` is not installed. Carries an actionable
    install hint instead of a bare ``ModuleNotFoundError``.
    """


class DecryptionError(HermesCryptoError):
    """Ciphertext could not be decrypted.

    Wrong key, a corrupted file, or tampering — the AES-GCM tag failed to
    verify. Deliberately does not distinguish the three: an attacker should
    not learn which.
    """


class KeystoreError(HermesCryptoError):
    """The keystore is missing, malformed, or has no usable key slot."""


class LockedError(HermesCryptoError):
    """The data encryption key is not available in this process.

    Raised when a decrypt is attempted but the keystore has not been
    unlocked (e.g. passphrase mode on a headless host with no
    ``HERMES_ENCRYPTION_PASSPHRASE`` set).
    """
