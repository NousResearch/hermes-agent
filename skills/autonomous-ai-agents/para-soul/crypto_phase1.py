#!/usr/bin/env python3
"""Para-Soul File Encryption Module (Phase 1 — Single User)

Key derivation: Ed25519 private key → HKDF-SHA256 → AES-256-GCM key.
Context-bound with salt + info to prevent cross-protocol key reuse.

Encryption: AES-256-GCM with random 12-byte IV.
Output format: base64(iv || tag || ciphertext)

Phase 1 only supports single-user — one DID, one encryption key.
Multi-user KEM support planned for Phase 2.
"""

import os
import base64
import hashlib

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# ── Constants ───────────────────────────────────────
SALT = b"para-soul-encryption-v1"       # Fixed salt (not secret — binds context)
INFO = b"file-encryption-key-phase1"     # HKDF info parameter
IV_LENGTH = 12                           # 96 bits for GCM
TAG_LENGTH = 16                          # GCM auth tag


class ParaSoulCrypto:
    """Client-side encryption for Para-Soul files."""

    def __init__(self, key_path: str):
        """Initialize with path to Ed25519 private key PEM file."""
        self._key_path = key_path
        self._aes_key: bytes | None = None

    @property
    def aes_key(self) -> bytes:
        """Derive AES-256 key from Ed25519 private key (lazy, cached)."""
        if self._aes_key is None:
            self._aes_key = self._derive_key()
        return self._aes_key

    def _derive_key(self) -> bytes:
        """Ed25519 private key → HKDF-SHA256 → AES-256 key.

        Uses the raw private key bytes as HKDF input key material (IKM).
        Context-bound with SALT + INFO to prevent key reuse across protocols.
        """
        with open(self._key_path, "rb") as f:
            private_key = load_pem_private_key(f.read(), password=None)

        # Extract raw private key bytes
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"Expected Ed25519 private key, got {type(private_key)}")

        raw_bytes = private_key.private_bytes_raw()

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=SALT,
            info=INFO,
        )
        return hkdf.derive(raw_bytes)

    def encrypt(self, plaintext: bytes) -> str:
        """Encrypt plaintext → base64(iv + tag + ciphertext).

        Returns a URL-safe base64 string suitable for JSON transport.
        Each encryption uses a fresh random IV.
        """
        iv = os.urandom(IV_LENGTH)
        aesgcm = AESGCM(self.aes_key)
        # AESGCM.encrypt returns ciphertext + tag concatenated
        ct_with_tag = aesgcm.encrypt(iv, plaintext, None)

        # Pack: iv (12) + ciphertext+tag (ct+16)
        packed = iv + ct_with_tag
        return base64.urlsafe_b64encode(packed).decode('ascii')

    def decrypt(self, encoded: str) -> bytes:
        """Decrypt base64(iv + tag + ciphertext) → plaintext."""
        packed = base64.urlsafe_b64decode(encoded)
        iv = packed[:IV_LENGTH]
        ct_with_tag = packed[IV_LENGTH:]

        aesgcm = AESGCM(self.aes_key)
        return aesgcm.decrypt(iv, ct_with_tag, None)

    def encrypt_file(self, filepath: str) -> tuple[str, str]:
        """Encrypt a file. Returns (ciphertext_b64, plaintext_hash_hex)."""
        with open(filepath, "rb") as f:
            plaintext = f.read()

        plaintext_hash = hashlib.sha256(plaintext).hexdigest()
        ciphertext = self.encrypt(plaintext)
        return ciphertext, plaintext_hash

    def decrypt_to_file(self, encoded: str, filepath: str, expected_hash: str = "") -> bool:
        """Decrypt and write to file. If expected_hash provided, verify integrity.

        Returns True if integrity check passes, False otherwise.
        """
        plaintext = self.decrypt(encoded)

        if expected_hash:
            actual_hash = hashlib.sha256(plaintext).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity check failed for {filepath}: "
                    f"expected {expected_hash[:12]}..., got {actual_hash[:12]}..."
                )

        with open(filepath, "wb") as f:
            f.write(plaintext)
        return True


def make_crypto_from_env(keys_dir: str = "") -> ParaSoulCrypto | None:
    """Create a ParaSoulCrypto instance from standard key location.

    Returns None if key not found (encryption unavailable — plaintext fallback).
    """
    if not keys_dir:
        keys_dir = os.path.expanduser("~/.config/paragate/keys")
    key_path = os.path.join(keys_dir, "private.pem")
    if not os.path.exists(key_path):
        return None
    try:
        return ParaSoulCrypto(key_path)
    except Exception:
        return None


# ── CLI test ────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 crypto_phase1.py <encrypt|decrypt|test> [file]")
        sys.exit(1)

    cmd = sys.argv[1]
    crypto = make_crypto_from_env()

    if not crypto:
        print("❌ No encryption key found. Place private.pem in ~/.config/paragate/keys/")
        sys.exit(1)

    if cmd == "test":
        plaintext = b"Hello, Para-Soul! This is a test of Phase 1 encryption."
        print(f"Original ({len(plaintext)}B): {plaintext[:50]}...")
        ct = crypto.encrypt(plaintext)
        print(f"Encrypted ({len(ct)}B base64): {ct[:50]}...")
        decrypted = crypto.decrypt(ct)
        print(f"Decrypted: {decrypted}")
        print(f"Match: {plaintext == decrypted}")
        print(f"Key fingerprint: {hashlib.sha256(crypto.aes_key).hexdigest()[:16]}...")

    elif cmd == "encrypt" and len(sys.argv) > 2:
        ct, phash = crypto.encrypt_file(sys.argv[2])
        print(f"Ciphertext: {ct[:80]}...")
        print(f"Plaintext hash: {phash}")

    elif cmd == "decrypt" and len(sys.argv) > 2:
        encoded = sys.stdin.read().strip()
        crypto.decrypt_to_file(encoded, sys.argv[2])
        print(f"✅ Decrypted to {sys.argv[2]}")
