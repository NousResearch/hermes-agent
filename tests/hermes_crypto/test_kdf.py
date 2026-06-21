"""Unit tests for the passphrase key-derivation functions."""

from __future__ import annotations

import os

import pytest

from hermes_crypto import kdf

# Tiny Argon2 parameters keep the test suite fast — production defaults are
# far heavier (see kdf.DEFAULT_ARGON2_PARAMS).
FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


@pytest.mark.parametrize("kdf_id", [kdf.KDF_ARGON2ID, kdf.KDF_SCRYPT])
def test_derivation_is_deterministic(kdf_id):
    salt = os.urandom(16)
    a = kdf.derive_kek(b"correct horse", salt, kdf_id, FAST_ARGON2)
    b = kdf.derive_kek(b"correct horse", salt, kdf_id, FAST_ARGON2)
    assert a == b
    assert len(a) == 32


@pytest.mark.parametrize("kdf_id", [kdf.KDF_ARGON2ID, kdf.KDF_SCRYPT])
def test_different_salt_changes_key(kdf_id):
    a = kdf.derive_kek(b"pw", os.urandom(16), kdf_id, FAST_ARGON2)
    b = kdf.derive_kek(b"pw", os.urandom(16), kdf_id, FAST_ARGON2)
    assert a != b


@pytest.mark.parametrize("kdf_id", [kdf.KDF_ARGON2ID, kdf.KDF_SCRYPT])
def test_different_passphrase_changes_key(kdf_id):
    salt = os.urandom(16)
    a = kdf.derive_kek(b"passphrase-one", salt, kdf_id, FAST_ARGON2)
    b = kdf.derive_kek(b"passphrase-two", salt, kdf_id, FAST_ARGON2)
    assert a != b


def test_argon2_and_scrypt_differ():
    salt = os.urandom(16)
    argon = kdf.derive_kek(b"pw", salt, kdf.KDF_ARGON2ID, FAST_ARGON2)
    scrypt = kdf.derive_kek(b"pw", salt, kdf.KDF_SCRYPT, None)
    assert argon != scrypt


def test_empty_passphrase_rejected():
    with pytest.raises(ValueError):
        kdf.derive_kek(b"", os.urandom(16), kdf.KDF_SCRYPT, None)


def test_short_salt_rejected():
    with pytest.raises(ValueError):
        kdf.derive_kek(b"pw", b"short", kdf.KDF_SCRYPT, None)


def test_unknown_kdf_rejected():
    with pytest.raises(ValueError):
        kdf.derive_kek(b"pw", os.urandom(16), 999, None)


def test_preferred_kdf_is_argon2_when_available():
    # argon2-cffi is part of the encryption extra and is installed for tests.
    assert kdf.argon2_available()
    assert kdf.preferred_kdf_id() == kdf.KDF_ARGON2ID
