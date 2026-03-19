import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_LENGTH = 32
IV_LENGTH = 12
TAG_LENGTH = 16


def encrypt(data: bytes) -> tuple[bytes, str]:
    """Encrypt data with AES-256-GCM. Returns (encrypted_blob, key_hex)."""
    key = os.urandom(KEY_LENGTH)
    iv = os.urandom(IV_LENGTH)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt returns ciphertext+tag appended
    ct_with_tag = aesgcm.encrypt(iv, data, None)
    # ct_with_tag = ciphertext || tag(16 bytes)
    ciphertext = ct_with_tag[:-TAG_LENGTH]
    tag = ct_with_tag[-TAG_LENGTH:]
    # Output format: [IV:12][AuthTag:16][Ciphertext] — matches JS SDK
    encrypted = iv + tag + ciphertext
    return encrypted, key.hex()


def decrypt(encrypted: bytes, key_hex: str) -> bytes:
    """Decrypt AES-256-GCM encrypted data. Input format: [IV:12][AuthTag:16][Ciphertext]."""
    key = bytes.fromhex(key_hex)
    if len(key) != KEY_LENGTH:
        raise ValueError(f"Invalid key length: {len(key)} (expected {KEY_LENGTH})")
    iv = encrypted[:IV_LENGTH]
    tag = encrypted[IV_LENGTH:IV_LENGTH + TAG_LENGTH]
    ciphertext = encrypted[IV_LENGTH + TAG_LENGTH:]
    aesgcm = AESGCM(key)
    # AESGCM.decrypt expects ciphertext+tag appended
    return aesgcm.decrypt(iv, ciphertext + tag, None)
