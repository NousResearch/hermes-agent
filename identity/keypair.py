"""
Ed25519 keypair management for Hermes Agent identity.

Keypair is stored at ~/.hermes/identity/:
  - public.key  (32 bytes, hex encoded)
  - private.key (64 bytes, binary, chmod 600)

The private key file is protected with restrictive permissions
and should never be accessible to the agent's tools.
"""

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey

logger = logging.getLogger(__name__)

IDENTITY_DIR_NAME = "identity"
PUBLIC_KEY_FILE = "public.key"
PRIVATE_KEY_FILE = "private.key"


@dataclass(frozen=True)
class Identity:
    """Agent's Ed25519 identity."""

    signing_key: SigningKey
    verify_key: VerifyKey

    @property
    def pubkey_hex(self) -> str:
        """Public key as hex string (64 chars)."""
        return self.verify_key.encode().hex()

    @property
    def pubkey_bytes(self) -> bytes:
        """Public key as raw bytes (32 bytes)."""
        return bytes(self.verify_key.encode())

    def sign(self, message: bytes) -> bytes:
        """Sign a message, return 64-byte signature."""
        signed = self.signing_key.sign(message)
        return signed.signature


def _get_hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _get_identity_dir() -> Path:
    return _get_hermes_home() / IDENTITY_DIR_NAME


def identity_exists() -> bool:
    """Check if an identity keypair already exists."""
    identity_dir = _get_identity_dir()
    return (
        (identity_dir / PRIVATE_KEY_FILE).is_file()
        and (identity_dir / PUBLIC_KEY_FILE).is_file()
    )


def get_identity() -> Identity:
    """Load existing identity or create a new one.

    On first call, generates a new Ed25519 keypair and saves it.
    Subsequent calls load the existing keypair from disk.
    """
    identity_dir = _get_identity_dir()

    private_key_path = identity_dir / PRIVATE_KEY_FILE
    public_key_path = identity_dir / PUBLIC_KEY_FILE

    if private_key_path.is_file() and public_key_path.is_file():
        return _load_identity(private_key_path, public_key_path)

    return _create_identity(identity_dir, private_key_path, public_key_path)


def _create_identity(
    identity_dir: Path,
    private_key_path: Path,
    public_key_path: Path,
) -> Identity:
    """Generate a new Ed25519 keypair and save to disk."""
    identity_dir.mkdir(parents=True, exist_ok=True)

    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key

    # Write private key with restrictive permissions (owner read-only)
    private_key_path.write_bytes(bytes(signing_key.encode()))
    os.chmod(private_key_path, stat.S_IRUSR)

    # Write public key as hex (human-readable)
    public_key_path.write_text(verify_key.encode().hex() + "\n")
    os.chmod(public_key_path, stat.S_IRUSR | stat.S_IWUSR)

    # Restrict identity directory
    os.chmod(identity_dir, stat.S_IRWXU)

    logger.info(
        "Created new agent identity: %s",
        verify_key.encode().hex()[:16] + "...",
    )

    return Identity(signing_key=signing_key, verify_key=verify_key)


def _load_identity(
    private_key_path: Path,
    public_key_path: Path,
) -> Identity:
    """Load an existing keypair from disk."""
    private_bytes = private_key_path.read_bytes()

    if len(private_bytes) != 32:
        raise ValueError(
            f"Invalid private key size: expected 32 bytes, got {len(private_bytes)}. "
            "Delete ~/.hermes/identity/ and restart to regenerate."
        )

    signing_key = SigningKey(private_bytes)
    verify_key = signing_key.verify_key

    # Verify public key matches
    stored_pubkey = public_key_path.read_text().strip()
    actual_pubkey = verify_key.encode().hex()
    if stored_pubkey != actual_pubkey:
        raise ValueError(
            "Public key mismatch: stored public.key does not match private.key. "
            "Delete ~/.hermes/identity/ and restart to regenerate."
        )

    logger.debug("Loaded agent identity: %s...", actual_pubkey[:16])
    return Identity(signing_key=signing_key, verify_key=verify_key)
