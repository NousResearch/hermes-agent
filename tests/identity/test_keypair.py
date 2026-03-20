"""Tests for identity keypair management."""

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from nacl.signing import VerifyKey

from identity.keypair import (
    Identity,
    get_identity,
    identity_exists,
)


@pytest.fixture
def temp_hermes_home(tmp_path):
    """Create a temporary HERMES_HOME directory."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


class TestIdentityExists:
    def test_returns_false_when_no_identity(self, temp_hermes_home):
        assert identity_exists() is False

    def test_returns_true_when_identity_exists(self, temp_hermes_home):
        ident = get_identity()
        assert ident is not None
        assert identity_exists() is True

    def test_returns_false_when_partial(self, temp_hermes_home):
        identity_dir = temp_hermes_home / "identity"
        identity_dir.mkdir()
        (identity_dir / "public.key").write_text("aabbcc\n")
        assert identity_exists() is False


class TestGetIdentity:
    def test_creates_new_identity(self, temp_hermes_home):
        ident = get_identity()

        assert isinstance(ident, Identity)
        assert len(ident.pubkey_hex) == 64
        assert len(ident.pubkey_bytes) == 32

    def test_creates_identity_files(self, temp_hermes_home):
        get_identity()

        identity_dir = temp_hermes_home / "identity"
        assert identity_dir.is_dir()
        assert (identity_dir / "private.key").is_file()
        assert (identity_dir / "public.key").is_file()

    def test_private_key_has_restrictive_permissions(self, temp_hermes_home):
        get_identity()

        private_key = temp_hermes_home / "identity" / "private.key"
        mode = private_key.stat().st_mode
        assert mode & stat.S_IRUSR  # owner can read
        assert not (mode & stat.S_IWUSR)  # owner cannot write
        assert not (mode & stat.S_IRGRP)  # group cannot read
        assert not (mode & stat.S_IROTH)  # others cannot read

    def test_loads_existing_identity(self, temp_hermes_home):
        ident1 = get_identity()
        ident2 = get_identity()

        assert ident1.pubkey_hex == ident2.pubkey_hex

    def test_public_key_matches_private_key(self, temp_hermes_home):
        ident = get_identity()

        stored_pubkey = (
            temp_hermes_home / "identity" / "public.key"
        ).read_text().strip()
        assert stored_pubkey == ident.pubkey_hex

    def test_raises_on_corrupted_private_key(self, temp_hermes_home):
        identity_dir = temp_hermes_home / "identity"
        identity_dir.mkdir(parents=True)

        (identity_dir / "private.key").write_bytes(b"short")
        os.chmod(identity_dir / "private.key", stat.S_IRUSR)
        (identity_dir / "public.key").write_text("aa" * 32 + "\n")

        with pytest.raises(ValueError, match="Invalid private key size"):
            get_identity()

    def test_raises_on_pubkey_mismatch(self, temp_hermes_home):
        ident = get_identity()

        # Tamper with public key
        pub_path = temp_hermes_home / "identity" / "public.key"
        os.chmod(pub_path, stat.S_IRUSR | stat.S_IWUSR)
        pub_path.write_text("bb" * 32 + "\n")

        with pytest.raises(ValueError, match="Public key mismatch"):
            get_identity()


class TestIdentitySign:
    def test_sign_returns_64_bytes(self, temp_hermes_home):
        ident = get_identity()
        sig = ident.sign(b"hello")
        assert len(sig) == 64

    def test_sign_is_deterministic(self, temp_hermes_home):
        ident = get_identity()
        sig1 = ident.sign(b"test message")
        sig2 = ident.sign(b"test message")
        # Ed25519 is deterministic
        assert sig1 == sig2

    def test_signature_is_verifiable(self, temp_hermes_home):
        ident = get_identity()
        message = b"verify me"
        sig = ident.sign(message)

        verify_key = VerifyKey(ident.pubkey_bytes)
        # Should not raise
        verify_key.verify(message, sig)

    def test_different_messages_different_signatures(self, temp_hermes_home):
        ident = get_identity()
        sig1 = ident.sign(b"message 1")
        sig2 = ident.sign(b"message 2")
        assert sig1 != sig2
