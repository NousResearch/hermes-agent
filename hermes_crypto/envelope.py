"""Self-describing AES-256-GCM envelope for encrypted-at-rest files.

Binary envelope layout::

    magic "HRMSENC" (7 bytes) | version (1 byte) | nonce (12 bytes) | ciphertext+GCM-tag

* The magic + version prefix makes detection trivial — a reader checks the
  first bytes and decrypts only when they match, so a plaintext (un-migrated)
  file and an encrypted file can coexist and migration stays idempotent.
* The header bytes are fed to AES-GCM as associated data, so tampering with
  the version field fails authentication.
* ``.env`` is a special case: it must stay a text file so tools that read it
  directly don't choke on raw bytes. The encrypted form is a marker comment
  line followed by base64 of the binary envelope.

This module imports ``cryptography`` at module load. It is therefore only
imported lazily by ``hermes_crypto.__init__`` so the package stays
import-safe when the optional ``encryption`` extra is not installed.
"""

from __future__ import annotations

import base64
import os

from .detect import ENV_MARKER, ENV_MARKER_BYTES, MAGIC
from .errors import DecryptionError, DependencyError


def _load_crypto():
    """Lazily import ``cryptography`` so this module stays import-safe.

    Importing :mod:`hermes_crypto.envelope` never fails on a missing optional
    dependency — only an actual encrypt/decrypt call does, with an
    actionable :class:`DependencyError`.
    """
    try:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - exercised via DependencyError
        raise DependencyError(
            "The 'cryptography' package is required for encryption-at-rest. "
            "Install it with:  pip install 'hermes-agent[encryption]'"
        ) from exc
    return AESGCM, InvalidTag


VERSION = 1
_HEADER = MAGIC + bytes([VERSION])
NONCE_LEN = 12

__all__ = ["VERSION", "encrypt", "decrypt", "encrypt_env", "decrypt_env"]


def encrypt(plaintext: bytes, dek: bytes) -> bytes:
    """Encrypt *plaintext* under the 32-byte data key, returning a full envelope."""
    if len(dek) != 32:
        raise ValueError("data encryption key must be 32 bytes")
    AESGCM, _ = _load_crypto()
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(dek).encrypt(nonce, plaintext, _HEADER)
    return _HEADER + nonce + ciphertext


def decrypt(envelope: bytes, dek: bytes) -> bytes:
    """Decrypt a binary envelope produced by :func:`encrypt`.

    Raises :class:`DecryptionError` on a wrong key, corruption, or tampering.
    """
    if not envelope.startswith(MAGIC):
        raise DecryptionError("not a hermes_crypto envelope (bad magic)")
    if len(envelope) <= len(_HEADER) + NONCE_LEN:
        # Too short to even hold the header + nonce — catch this before the
        # version-byte read below so a magic-only blob raises DecryptionError,
        # not IndexError.
        raise DecryptionError("truncated envelope")
    version = envelope[len(MAGIC)]
    if version != VERSION:
        raise DecryptionError(f"unsupported envelope version {version}")
    header = envelope[: len(_HEADER)]
    nonce = envelope[len(_HEADER) : len(_HEADER) + NONCE_LEN]
    ciphertext = envelope[len(_HEADER) + NONCE_LEN :]
    if len(nonce) != NONCE_LEN or not ciphertext:
        raise DecryptionError("truncated envelope")
    AESGCM, InvalidTag = _load_crypto()
    try:
        return AESGCM(dek).decrypt(nonce, ciphertext, header)
    except InvalidTag as exc:
        raise DecryptionError(
            "decryption failed — wrong key or the file was tampered with"
        ) from exc


def encrypt_env(plaintext: bytes, dek: bytes) -> bytes:
    """Encrypt ``.env`` content into the text-framed form (marker + base64)."""
    envelope = encrypt(plaintext, dek)
    body = base64.b64encode(envelope).decode("ascii")
    return f"{ENV_MARKER}\n{body}\n".encode("ascii")


def decrypt_env(data: bytes, dek: bytes) -> bytes:
    """Decrypt the text-framed ``.env`` form produced by :func:`encrypt_env`."""
    if not data.startswith(ENV_MARKER_BYTES):
        raise DecryptionError("not an encrypted .env file (missing marker)")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DecryptionError("corrupt encrypted .env (non-ASCII body)") from exc
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise DecryptionError("corrupt encrypted .env (empty body)")
    try:
        # binascii.Error (raised on bad base64) is a subclass of ValueError.
        envelope = base64.b64decode("".join(lines), validate=True)
    except ValueError as exc:
        raise DecryptionError("corrupt encrypted .env (bad base64)") from exc
    return decrypt(envelope, dek)
