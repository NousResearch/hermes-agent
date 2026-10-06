#!/usr/bin/env python3
"""Para-Soul Multi-Agent Encryption (Phase 2)

Key Encapsulation Mechanism (KEM) for multi-agent file sharing.

Design:
  1. Each file gets a random AES-256 file key
  2. File content encrypted with file_key → ciphertext
  3. For each authorized agent, file_key is sealed with that agent's X25519 public key
  4. Upload: ciphertext + {did: sealed_file_key} + plaintext_hash

X25519 keys are derived from Ed25519 DID keys via HKDF (key separation).
No libsodium/nacl dependency — uses only cryptography library (already required).

Protocol: ECDH-X25519 → HKDF → AES-256-GCM for key wrapping (equivalent to
libsodium crypto_box_seal, manually implemented per NIST SP 800-56A).
"""

import os
import base64
import hashlib
import json

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives.serialization import load_pem_private_key, Encoding, PublicFormat

# ── Phase 1 imports (reuse) ──────────────────────────
from crypto_phase1 import ParaSoulCrypto, SALT, INFO as P1_INFO, IV_LENGTH, TAG_LENGTH

# ── Phase 2 constants ─────────────────────────────────
X25519_DERIVE_SALT = b"para-soul-x25519-derivation-v1"
X25519_DERIVE_INFO = b"multi-agent-key-phase2"
SEAL_SALT = b"para-soul-seal-phase2"
SEAL_INFO = b"file-key-wrap"


class MultiAgentCrypto:
    """Multi-agent encryption with per-file keys and per-agent key sealing.

    Uses X25519 keypairs derived from the agent's Ed25519 DID key.
    Supports encrypt-once-decrypt-by-any-authorized-agent pattern.
    """

    def __init__(self, ed25519_private_key_path: str):
        """Initialize with path to Ed25519 private key PEM."""
        self._key_path = ed25519_private_key_path
        self._x25519_private: x25519.X25519PrivateKey | None = None
        self._x25519_public: x25519.X25519PublicKey | None = None
        self._did: str = ""  # Will be set from identity

        # Also init Phase 1 crypto for single-user mode
        self._p1 = ParaSoulCrypto(ed25519_private_key_path)

    @property
    def x25519_private(self) -> x25519.X25519PrivateKey:
        """Derive X25519 private key from Ed25519 private key (lazy)."""
        if self._x25519_private is None:
            self._derive_x25519_keys()
        return self._x25519_private

    @property
    def x25519_public(self) -> x25519.X25519PublicKey:
        """Derive X25519 public key from Ed25519 private key (lazy)."""
        if self._x25519_public is None:
            self._derive_x25519_keys()
        return self._x25519_public

    @property
    def x25519_public_bytes(self) -> bytes:
        """X25519 public key as raw 32 bytes (for sharing with other agents)."""
        return self.x25519_public.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def _derive_x25519_keys(self):
        """Ed25519 private key → HKDF → X25519 private key.

        Uses key separation: different SALT+INFO from Phase 1 AES derivation.
        """
        with open(self._key_path, "rb") as f:
            pk = load_pem_private_key(f.read(), password=None)

        if not isinstance(pk, ed25519.Ed25519PrivateKey):
            raise TypeError(f"Expected Ed25519 private key, got {type(pk)}")

        raw_bytes = pk.private_bytes_raw()

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # X25519 needs 32 bytes
            salt=X25519_DERIVE_SALT,
            info=X25519_DERIVE_INFO,
        )
        derived = hkdf.derive(raw_bytes)

        self._x25519_private = x25519.X25519PrivateKey.from_private_bytes(derived)
        self._x25519_public = self._x25519_private.public_key()

    def _seal_file_key(self, file_key: bytes, recipient_pubkey_bytes: bytes) -> str:
        """Seal (encrypt) a file key for a specific recipient.

        Implements libsodium crypto_box_seal with cryptography library:
          1. Generate ephemeral X25519 keypair
          2. ECDH(ephemeral_sk, recipient_pk) → shared_secret
          3. HKDF(shared_secret) → wrapping_key (KEK)
          4. AES-256-GCM(wrapping_key, file_key) → sealed_key

        Returns: base64(ephemeral_pk || sealed_key)
        """
        # 1. Ephemeral keypair
        eph_sk = x25519.X25519PrivateKey.generate()
        eph_pk = eph_sk.public_key()

        # 2. ECDH
        recipient_pk = x25519.X25519PublicKey.from_public_bytes(recipient_pubkey_bytes)
        shared_secret = eph_sk.exchange(recipient_pk)

        # 3. HKDF → wrapping key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=SEAL_SALT,
            info=SEAL_INFO,
        )
        wrapping_key = hkdf.derive(shared_secret)

        # 4. AES-256-GCM seal
        iv = os.urandom(IV_LENGTH)
        aesgcm = AESGCM(wrapping_key)
        sealed = aesgcm.encrypt(iv, file_key, None)

        # Pack: ephemeral_pk (32) + iv (12) + sealed_key (32 + 16)
        packed = eph_pk.public_bytes(Encoding.Raw, PublicFormat.Raw) + iv + sealed
        return base64.urlsafe_b64encode(packed).decode('ascii')

    def _unseal_file_key(self, sealed_b64: str) -> bytes:
        """Unseal (decrypt) a file key sealed for this agent.

        Reverses _seal_file_key:
          1. Extract ephemeral_pk, iv, sealed from packed data
          2. ECDH(self_sk, ephemeral_pk) → shared_secret
          3. HKDF → wrapping_key
          4. AES-GCM decrypt → file_key
        """
        packed = base64.urlsafe_b64decode(sealed_b64)

        # Extract components
        eph_pk_bytes = packed[:32]
        iv = packed[32:32 + IV_LENGTH]
        sealed = packed[32 + IV_LENGTH:]

        # ECDH
        eph_pk = x25519.X25519PublicKey.from_public_bytes(eph_pk_bytes)
        shared_secret = self.x25519_private.exchange(eph_pk)

        # HKDF → wrapping key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SEAL_SALT,
            info=SEAL_INFO,
        )
        wrapping_key = hkdf.derive(shared_secret)

        # Decrypt
        aesgcm = AESGCM(wrapping_key)
        return aesgcm.decrypt(iv, sealed, None)

    def encrypt_for_agents(self, filepath: str, agent_keys: dict[str, bytes]) -> dict:
        """Encrypt a file for multiple agents.

        Args:
            filepath: Path to file to encrypt
            agent_keys: {did: x25519_public_key_bytes} for each authorized agent

        Returns:
            {
                "ciphertext": base64 encrypted file content,
                "encrypted_keys": {did: base64 sealed file key},
                "plaintext_hash": hex,
                "algorithm": "para-soul-kem-v1"
            }
        """
        with open(filepath, "rb") as f:
            plaintext = f.read()

        # Generate random file key
        file_key = os.urandom(32)  # AES-256

        # Encrypt file content
        iv = os.urandom(IV_LENGTH)
        aesgcm = AESGCM(file_key)
        ct_with_tag = aesgcm.encrypt(iv, plaintext, None)
        ciphertext = base64.urlsafe_b64encode(iv + ct_with_tag).decode('ascii')

        # Compute integrity hash
        plaintext_hash = hashlib.sha256(plaintext).hexdigest()

        # Seal file key for each agent
        encrypted_keys = {}
        for did, pubkey_bytes in agent_keys.items():
            encrypted_keys[did] = self._seal_file_key(file_key, pubkey_bytes)

        return {
            "ciphertext": ciphertext,
            "encrypted_keys": encrypted_keys,
            "plaintext_hash": plaintext_hash,
            "algorithm": "para-soul-kem-v1",
        }

    def decrypt_as_agent(self, data: dict) -> bytes:
        """Decrypt a file as one of the authorized agents.

        Args:
            data: Dict from encrypt_for_agents (with ciphertext + encrypted_keys)

        Returns:
            Plaintext bytes (integrity verified via plaintext_hash)
        """
        encrypted_keys = data.get("encrypted_keys", {})

        # Find our sealed key — try by DID first, then try all
        file_key = None
        # Try each sealed key until one works (we only know our own)
        for did, sealed_b64 in encrypted_keys.items():
            try:
                file_key = self._unseal_file_key(sealed_b64)
                break
            except Exception:
                continue

        if file_key is None:
            raise ValueError("Not authorized: could not decrypt any sealed file key. "
                           "Your DID may not be in the authorized agent list.")

        # Decrypt file content
        packed = base64.urlsafe_b64decode(data["ciphertext"])
        iv = packed[:IV_LENGTH]
        ct_with_tag = packed[IV_LENGTH:]

        aesgcm = AESGCM(file_key)
        plaintext = aesgcm.decrypt(iv, ct_with_tag, None)

        # Verify integrity
        expected_hash = data.get("plaintext_hash", "")
        if expected_hash:
            actual_hash = hashlib.sha256(plaintext).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity check failed: "
                    f"expected {expected_hash[:12]}..., got {actual_hash[:12]}..."
                )

        return plaintext

    # ── Convenience methods ───────────────────────────

    def encrypt_file_for_all(self, filepath: str, authorized_dids: list[str],
                             resolve_pubkey) -> dict:
        """Encrypt for a list of DIDs, resolving public keys via callback.

        Args:
            filepath: File to encrypt
            authorized_dids: List of agent DIDs to authorize
            resolve_pubkey: Callable(did) → x25519 public key bytes
        """
        agent_keys = {}
        for did in authorized_dids:
            try:
                agent_keys[did] = resolve_pubkey(did)
            except Exception as e:
                print(f"  ⚠️  Could not resolve pubkey for {did}: {e}")

        if not agent_keys:
            raise ValueError("No agent keys could be resolved")

        return self.encrypt_for_agents(filepath, agent_keys)

    def get_my_public_key(self) -> bytes:
        """Return this agent's X25519 public key for sharing."""
        return self.x25519_public_bytes


# ── CLI test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 crypto_phase2.py <test|pubkey>")
        sys.exit(1)

    keys_dir = os.path.expanduser("~/.config/paragate/keys")
    key_path = os.path.join(keys_dir, "private.pem")

    if not os.path.exists(key_path):
        print("❌ No private key found")
        sys.exit(1)

    crypto = MultiAgentCrypto(key_path)

    if sys.argv[1] == "pubkey":
        pk = crypto.x25519_public_bytes
        print(f"X25519 public key (32 bytes hex): {pk.hex()}")
        print(f"Base64: {base64.urlsafe_b64encode(pk).decode()}")

    elif sys.argv[1] == "test":
        print("=== Phase 2 Multi-Agent Encryption Test ===")

        plaintext = b"Hello from Para-Soul Phase 2! This file is shared across agents."

        # Simulate: encrypt for ourselves (same machine, same key)
        # In real multi-agent, each agent would have different pubkeys
        agents = {"test-agent": crypto.x25519_public_bytes}

        # Generate random file key manually and encrypt
        file_key = os.urandom(32)
        iv = os.urandom(IV_LENGTH)
        aesgcm = AESGCM(file_key)
        ct = aesgcm.encrypt(iv, plaintext, None)
        ciphertext = base64.urlsafe_b64encode(iv + ct).decode()
        plaintext_hash = hashlib.sha256(plaintext).hexdigest()

        encrypted_keys = {}
        for did, pubkey in agents.items():
            encrypted_keys[did] = crypto._seal_file_key(file_key, pubkey)

        encrypted = {
            "ciphertext": ciphertext,
            "encrypted_keys": encrypted_keys,
            "plaintext_hash": plaintext_hash,
            "algorithm": "para-soul-kem-v1",
        }

        print(f"Original: {plaintext[:50]}...")
        print(f"Ciphertext size: {len(encrypted['ciphertext'])} chars")
        print(f"Authorized agents: {list(encrypted['encrypted_keys'].keys())}")

        decrypted = crypto.decrypt_as_agent(encrypted)
        print(f"Decrypted: {decrypted[:50]}...")
        print(f"Match: {plaintext == decrypted}")
