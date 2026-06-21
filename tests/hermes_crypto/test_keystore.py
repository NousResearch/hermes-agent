"""Unit tests for the keystore: key slots, unlock, rotation, recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_crypto import audit, keystore
from hermes_crypto import kdf
from hermes_crypto.errors import DecryptionError, KeystoreError, LockedError

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def test_passphrase_init_and_unlock():
    dek = keystore.init_keystore("passphrase", passphrase="hunter2", argon2_params=FAST_ARGON2)
    assert len(dek) == 32
    assert keystore.is_unlocked()

    keystore.lock()
    assert not keystore.is_unlocked()
    assert keystore.unlock(passphrase="hunter2") == dek


def test_wrong_passphrase_rejected():
    keystore.init_keystore("passphrase", passphrase="right", argon2_params=FAST_ARGON2)
    keystore.lock()
    with pytest.raises(DecryptionError):
        keystore.unlock(passphrase="wrong")


def test_keyring_init_and_unlock():
    # The in-memory fake keyring backend (see conftest) is treated as secure.
    assert keystore.keyring_is_secure()
    dek = keystore.init_keystore("keyring")
    keystore.lock()
    assert keystore.unlock() == dek


def test_keyfile_init_and_unlock():
    dek = keystore.init_keystore("keyfile")
    assert keystore.keyfile_path().exists()
    keystore.lock()
    assert keystore.unlock() == dek


def test_recovery_slot_unlocks_same_dek():
    dek = keystore.init_keystore("passphrase", passphrase="primary-pw", argon2_params=FAST_ARGON2)
    code = keystore.add_recovery_slot()
    assert keystore.has_recovery_slot()
    recovery = next(
        s for s in keystore.load_keystore()["slots"] if s.get("type") == "recovery"
    )
    assert recovery["kdf_params"] == kdf.RECOVERY_ARGON2_PARAMS

    keystore.lock()
    assert keystore.unlock(recovery_code=code) == dek
    # The recovery code is also accepted with lowercase / spacing noise.
    keystore.lock()
    assert keystore.unlock(recovery_code=code.lower().replace("-", " ")) == dek


def test_legacy_recovery_slot_with_stored_kdf_params_still_unlocks():
    """Slots created before the fast default persisted heavy params; unlock reads them."""
    dek = keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    legacy_params = {"time_cost": 2, "memory_cost_kib": 16, "parallelism": 1}
    code = keystore.add_recovery_slot(argon2_params=legacy_params)
    recovery = next(
        s for s in keystore.load_keystore()["slots"] if s.get("type") == "recovery"
    )
    assert recovery["kdf_params"] == legacy_params

    keystore.lock()
    assert keystore.unlock(recovery_code=code) == dek


def test_add_recovery_requires_unlock():
    keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    keystore.lock()
    with pytest.raises(LockedError):
        keystore.add_recovery_slot()


def test_replace_data_key_changes_cached_dek():
    dek = keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    new_dek = b"\x01" * 32
    dropped = keystore.replace_data_key(new_dek, passphrase="pw")
    assert dropped == 0
    assert keystore.get_cached_dek() == new_dek
    keystore.lock()
    assert keystore.unlock(passphrase="pw") == new_dek
    assert new_dek != dek


def test_replace_data_key_keyfile_rotates_kek():
    """Full re-key must generate a fresh keyfile KEK, not reuse the old one."""
    dek = keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    new_dek = b"\x02" * 32

    dropped = keystore.replace_data_key(new_dek)
    assert dropped == 0
    assert keystore.get_cached_dek() == new_dek
    assert new_dek != dek

    new_kek = keystore.keyfile_path().read_bytes()
    assert new_kek != old_kek

    keystore.lock()
    assert keystore.unlock() == new_dek

    keystore.lock()
    keystore.keyfile_path().write_bytes(old_kek)
    with pytest.raises(DecryptionError):
        keystore.unlock()


def test_replace_data_key_keyring_rotates_kek():
    """Full re-key must generate a fresh keyring KEK, not reuse the old one.

    Mirrors ``test_replace_data_key_keyfile_rotates_kek`` for the keyring slot,
    pinning the fresh-KEK-on-rekey invariant for the keyring backend ().
    """
    import base64
    import keyring as _keyring_mod

    dek = keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    old_kek_b64 = _keyring_mod.get_password(keystore.KEYRING_SERVICE, username)
    assert old_kek_b64 is not None
    old_kek = base64.b64decode(old_kek_b64)
    assert len(old_kek) == 32

    new_dek = b"\x02" * 32
    dropped = keystore.replace_data_key(new_dek)
    assert dropped == 0
    assert keystore.get_cached_dek() == new_dek
    assert new_dek != dek

    new_kek_b64 = _keyring_mod.get_password(keystore.KEYRING_SERVICE, username)
    assert new_kek_b64 is not None
    new_kek = base64.b64decode(new_kek_b64)
    assert len(new_kek) == 32
    assert new_kek != old_kek

    keystore.lock()
    assert keystore.unlock() == new_dek

    # Restoring the old KEK in the keyring must no longer unlock the keystore:
    # the wrapped DEK in the slot was re-wrapped under the *new* KEK.
    keystore.lock()
    _keyring_mod.set_password(
        keystore.KEYRING_SERVICE, username, base64.b64encode(old_kek).decode("ascii")
    )
    with pytest.raises(DecryptionError):
        keystore.unlock()


def test_rotate_primary_keyfile_overwrites_when_unlink_fails(monkeypatch):
    """rotate_primary must still install a fresh KEK if unlink() of the keyfile fails.

    Regression test for previously the call site swallowed OSError from
    ``keyfile_path().unlink()`` and ``_build_keyfile_slot`` then read the
    old bytes back, silently reusing the same KEK.
    """
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    assert len(old_kek) == 32

    real_unlink = Path.unlink
    keyfile = keystore.keyfile_path()

    def fail_keyfile_unlink(self, *args, **kwargs):
        if self == keyfile:
            raise OSError(13, "Permission denied (handle held)")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_keyfile_unlink)

    # Sanity: unlink really is blocked for the keyfile under the monkeypatch.
    with pytest.raises(OSError):
        keyfile.unlink()
    assert keyfile.exists()

    keystore.rotate_primary("keyfile")

    new_kek = keystore.keyfile_path().read_bytes()
    assert len(new_kek) == 32
    # The fix: the KEK on disk changed despite unlink being blocked.
    assert new_kek != old_kek

    # And the keystore unlocks cleanly with the new KEK in place.
    keystore.lock()
    assert len(keystore.unlock()) == 32


def test_rotate_passphrase_keeps_dek_and_invalidates_old():
    dek = keystore.init_keystore("passphrase", passphrase="old-pw", argon2_params=FAST_ARGON2)
    keystore.rotate_primary("passphrase", new_passphrase="new-pw", argon2_params=FAST_ARGON2)

    keystore.lock()
    assert keystore.unlock(passphrase="new-pw") == dek  # same DEK, new wrapping

    keystore.lock()
    with pytest.raises(DecryptionError):
        keystore.unlock(passphrase="old-pw")


def test_rotate_can_switch_key_source():
    dek = keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    keystore.rotate_primary("keyring")
    assert keystore.primary_slot_type() == "keyring"
    keystore.lock()
    assert keystore.unlock() == dek


def test_rotate_preserves_recovery_slot():
    keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    code = keystore.add_recovery_slot()
    keystore.rotate_primary("passphrase", new_passphrase="pw2", argon2_params=FAST_ARGON2)
    keystore.lock()
    # Recovery code still opens the keystore after a primary rotation.
    assert len(keystore.unlock(recovery_code=code)) == 32


def test_destroy_keystore_removes_everything():
    keystore.init_keystore("keyfile")
    assert keystore.keystore_exists()
    keystore.destroy_keystore()
    assert not keystore.keystore_exists()
    assert not keystore.keyfile_path().exists()
    assert not keystore.is_unlocked()


def test_destroy_keystore_raises_when_unlink_fails(monkeypatch):
    keystore.init_keystore("keyfile")
    real_unlink = Path.unlink

    def fail_keystore_unlink(self, *args, **kwargs):
        if self == keystore.keystore_path():
            raise OSError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_keystore_unlink)
    with pytest.raises(KeystoreError, match="could not remove keystore"):
        keystore.destroy_keystore()
    assert keystore.keystore_exists()
    record = audit.read_recent(1)[0]
    assert record["activity"] == audit.KEYSTORE_DESTROYED
    assert record["outcome"] == audit.FAILURE
    assert str(keystore.keystore_path()) in record["detail"]["paths"]


def test_init_refuses_when_keystore_exists():
    keystore.init_keystore("keyfile")
    with pytest.raises(KeystoreError):
        keystore.init_keystore("keyfile")


def test_load_keystore_rejects_newer_version():
    keystore.init_keystore("keyfile")
    data = keystore.load_keystore()
    data["version"] = keystore.KEYSTORE_VERSION + 1
    keystore.save_keystore(data)
    keystore.lock()
    with pytest.raises(KeystoreError, match="newer than this Hermes"):
        keystore.load_keystore()


def test_load_keystore_rejects_malformed_version():
    keystore.init_keystore("keyfile")
    data = keystore.load_keystore()
    data["version"] = "not-an-int"
    keystore.save_keystore(data)
    keystore.lock()
    with pytest.raises(KeystoreError, match="invalid version"):
        keystore.load_keystore()


def test_unlock_maps_malformed_slot_to_keystore_error():
    keystore.init_keystore("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    data = keystore.load_keystore()
    data["slots"][0].pop("salt")
    keystore.save_keystore(data)
    keystore.lock()
    with pytest.raises(KeystoreError, match="slot is malformed"):
        keystore.unlock(passphrase="pw")


def test_recovery_code_normalization_round_trip():
    code = keystore.generate_recovery_code()
    raw = keystore.normalize_recovery_code(code)
    assert keystore.normalize_recovery_code(code.lower()) == raw
    assert keystore.normalize_recovery_code(code.replace("-", "  ")) == raw


# ─── M2: crash-safe rekey (keyfile + keyring) ──────────────────────────────────
#
# These exercise the stage→commit→promote→rollback protocol: a save_keystore
# failure mid-rekey must leave the OLD KEK + OLD keystore live and recoverable
# (no brick, no leaked staging artifacts), and a crash in the commit→promote gap
# must self-heal from the staged KEK on the next unlock.


def _keyring_get(username):
    import keyring as _keyring_mod

    return _keyring_mod.get_password(keystore.KEYRING_SERVICE, username)


def test_replace_data_key_keyfile_save_failure_is_recoverable(monkeypatch):
    """A save_keystore failure during a keyfile rekey must not brick the DEK."""
    dek = keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    new_dek = b"\x07" * 32

    def boom(_data):
        raise OSError("disk full")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.replace_data_key(new_dek)

    # Live keyfile is untouched (rollback removed the sidecar), no leak.
    assert keystore.keyfile_path().read_bytes() == old_kek
    assert not keystore.keyfile_new_path().exists()

    # The OLD DEK is still recoverable under the OLD KEK.
    keystore.lock()
    assert keystore.unlock() == dek


def test_replace_data_key_keyring_save_failure_is_recoverable(monkeypatch):
    """A save_keystore failure during a keyring rekey must not brick the DEK."""
    dek = keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    old_kek_b64 = _keyring_get(username)
    assert old_kek_b64 is not None
    new_dek = b"\x07" * 32

    def boom(_data):
        raise OSError("disk full")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.replace_data_key(new_dek)

    # Live keyring entry untouched; staged ":new" entry cleaned up.
    assert _keyring_get(username) == old_kek_b64
    assert _keyring_get(keystore._keyring_username_new()) is None

    keystore.lock()
    assert keystore.unlock() == dek


def test_rotate_primary_keyfile_save_failure_is_recoverable(monkeypatch):
    """A save_keystore failure during a keyfile rotate must keep the old slot."""
    dek = keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()

    def boom(_data):
        raise OSError("disk full")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.rotate_primary("keyfile")

    assert keystore.keyfile_path().read_bytes() == old_kek
    assert not keystore.keyfile_new_path().exists()

    keystore.lock()
    assert keystore.unlock() == dek  # DEK unchanged, old KEK still works


def test_rotate_primary_keyring_save_failure_is_recoverable(monkeypatch):
    """A save_keystore failure during a keyring rotate must keep the old slot."""
    dek = keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    old_kek_b64 = _keyring_get(username)

    def boom(_data):
        raise OSError("disk full")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.rotate_primary("keyring")

    assert _keyring_get(username) == old_kek_b64
    assert _keyring_get(keystore._keyring_username_new()) is None

    keystore.lock()
    assert keystore.unlock() == dek


def test_keyfile_power_loss_between_commit_and_promote_is_recoverable(monkeypatch):
    """Commit succeeds, promote dies (power loss): next unlock self-heals.

    True power loss kills the process between save_keystore and promote — the
    function never returns. We simulate the *on-disk* mid-gap state by staging
    + committing the new keystore but skipping the live-keyfile promotion, then
    assert that a fresh ``unlock`` reads the sidecar, returns the new DEK, and
    finishes the promotion.

    (A promote that *raises* rather than aborting the process is covered by the
    M2 regression below — that path must NOT propagate; it returns normally.)
    """
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    new_dek = b"\x09" * 32

    real_stage = keystore._stage_keyfile_slot

    def stage_no_promote(slot_id, dek):
        slot, staged = real_stage(slot_id, dek)
        # Model an aborted process: the keystore commits but the live keyfile is
        # never promoted (the staged sidecar is left in place for the self-heal).
        return slot, keystore._StagedKEK(promote=lambda: None, rollback=staged.rollback)

    monkeypatch.setattr(keystore, "_stage_keyfile_slot", stage_no_promote)
    keystore.replace_data_key(new_dek)

    # Mid-gap state: keystore committed (new KEK), live keyfile still old, sidecar present.
    assert keystore.keyfile_path().read_bytes() == old_kek
    assert keystore.keyfile_new_path().exists()

    # A fresh unlock self-heals from the sidecar and returns the NEW dek.
    keystore.lock()
    assert keystore.unlock() == new_dek

    # Promotion completed: live keyfile is now the new KEK, sidecar gone.
    assert keystore.keyfile_path().read_bytes() != old_kek
    assert not keystore.keyfile_new_path().exists()


def test_keyring_power_loss_between_commit_and_promote_is_recoverable(monkeypatch):
    """Commit succeeds, promote dies: next unlock heals staged→live keyring."""
    keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    staged_username = keystore._keyring_username_new()
    old_kek_b64 = _keyring_get(username)
    new_dek = b"\x09" * 32

    real_stage = keystore._stage_keyring_slot

    def stage_no_promote(slot_id, dek):
        slot, staged = real_stage(slot_id, dek)
        return slot, keystore._StagedKEK(promote=lambda: None, rollback=staged.rollback)

    monkeypatch.setattr(keystore, "_stage_keyring_slot", stage_no_promote)
    keystore.replace_data_key(new_dek)

    # Mid-gap: keystore committed (new KEK), live entry still old, staged present.
    assert _keyring_get(username) == old_kek_b64
    assert _keyring_get(staged_username) is not None

    keystore.lock()
    assert keystore.unlock() == new_dek

    # Healed: live entry holds the new KEK, staged entry deleted.
    assert _keyring_get(username) != old_kek_b64
    assert _keyring_get(staged_username) is None


# ─── M2: post-commit promote failure must NOT raise (data-loss window fix) ─────
#
# The crash-safe refactor calls staged.promote() AFTER save_keystore commits.
# If promote() RAISES (os.replace OSError because AV/backup grabbed keyfile.new;
# a transient keyring backend error), replace_data_key used to propagate it. But
# the keystore is already committed under the NEW KEK, so propagating made
# migrate.full_rekey treat the rekey as "untouched" and roll every artifact back
# to OLD-DEK ciphertext while the keystore yields the NEW DEK → undecryptable.
#
# Contract now: once save_keystore returns, replace_data_key / rotate_primary
# MUST NOT raise. The promote is best-effort; on failure the staged KEK survives
# and unlock()'s _try_staged_unlock self-heals on the next load.


def test_replace_data_key_keyfile_promote_failure_does_not_raise(monkeypatch):
    """A keyfile promote() OSError after commit must NOT propagate.

    Reproduces the reviewer's scenario: save_keystore SUCCEEDS, then promote
    raises (os.replace blocked because AV/backup grabbed keyfile.new). Assert:
    (a) replace_data_key RETURNS normally, (b) the cached DEK is the new DEK,
    (c) unlock() returns the new DEK by self-healing from the surviving sidecar,
    and (d) the sidecar is consumed by the self-heal.
    """
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    new_dek = b"\x11" * 32

    real_stage = keystore._stage_keyfile_slot

    def stage_then_raise_promote(slot_id, dek):
        slot, staged = real_stage(slot_id, dek)

        def raising_promote():
            raise OSError(13, "keyfile.new locked by AV during promote")

        return slot, keystore._StagedKEK(
            promote=raising_promote, rollback=staged.rollback
        )

    monkeypatch.setattr(keystore, "_stage_keyfile_slot", stage_then_raise_promote)

    # (a) MUST NOT raise even though promote() raised after the commit.
    dropped = keystore.replace_data_key(new_dek)
    assert dropped == 0

    # (b) cached DEK is the new one (success path ran).
    assert keystore.get_cached_dek() == new_dek

    # Mid-gap on disk: keystore committed (new KEK), live keyfile still old,
    # sidecar present for the self-heal.
    assert keystore.keyfile_path().read_bytes() == old_kek
    assert keystore.keyfile_new_path().exists()

    # (c) unlock self-heals from the surviving sidecar and returns the new DEK.
    keystore.lock()
    assert keystore.unlock() == new_dek

    # (d) self-heal consumed the sidecar and installed the new live KEK.
    assert keystore.keyfile_path().read_bytes() != old_kek
    assert not keystore.keyfile_new_path().exists()


def test_replace_data_key_keyfile_promote_osreplace_failure_does_not_raise(monkeypatch):
    """Same window, but exercise the REAL promote via a forced os.replace error.

    Rather than swap the whole staged object, let the real promote() run and
    make ``utils.atomic_replace``'s os.replace raise — the concrete failure the
    reviewer named (AV/backup holding keyfile.new). The function must still
    return, and unlock() must self-heal.
    """
    import utils as _utils

    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    new_dek = b"\x12" * 32

    real_replace = _utils.os.replace
    keyfile_new = str(keystore.keyfile_new_path())
    # A toggle, not monkeypatch.undo(): the shared function-scoped monkeypatch
    # also holds autouse-fixture patches (e.g. _live_system_guard), so undo()
    # would revert those too. Flip the flag to let the later self-heal replace
    # land while leaving every other patch intact.
    block_sidecar_replace = {"on": True}

    def fail_replace_of_sidecar(src, dst, *a, **k):
        if block_sidecar_replace["on"] and str(src) == keyfile_new:
            raise OSError(13, "keyfile.new locked")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(_utils.os, "replace", fail_replace_of_sidecar)

    # MUST NOT raise.
    keystore.replace_data_key(new_dek)
    assert keystore.get_cached_dek() == new_dek
    # Live keyfile untouched, sidecar survives for the self-heal.
    assert keystore.keyfile_path().read_bytes() == old_kek
    assert keystore.keyfile_new_path().exists()

    # Stop blocking the sidecar replace so the self-heal can land.
    block_sidecar_replace["on"] = False
    keystore.lock()
    assert keystore.unlock() == new_dek
    assert not keystore.keyfile_new_path().exists()


def test_replace_data_key_keyring_promote_failure_does_not_raise(monkeypatch):
    """A keyring promote() error after commit must NOT propagate (keyring mode).

    Force the keyring ``set_password`` used during promote to raise (transient
    backend error). Assert replace_data_key returns, and unlock self-heals the
    staged→live keyring promotion and returns the new DEK.
    """
    import keyring as _keyring_mod

    keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    staged_username = keystore._keyring_username_new()
    old_kek_b64 = _keyring_get(username)
    new_dek = b"\x13" * 32

    real_set = _keyring_mod.set_password
    # Toggle rather than monkeypatch.undo() — see the keyfile variant for why.
    block_live_set = {"on": True}

    def fail_promote_set(service, name, value, *a, **k):
        # promote() does set_password(live_username, staged_kek). Fail exactly
        # that write; allow the staging set_password(:new, ...) to succeed.
        if block_live_set["on"] and name == username:
            raise RuntimeError("transient keyring backend error during promote")
        return real_set(service, name, value, *a, **k)

    monkeypatch.setattr(_keyring_mod, "set_password", fail_promote_set)

    # MUST NOT raise.
    dropped = keystore.replace_data_key(new_dek)
    assert dropped == 0
    assert keystore.get_cached_dek() == new_dek

    # Mid-gap: live entry still old, staged entry present for the self-heal.
    assert _keyring_get(username) == old_kek_b64
    assert _keyring_get(staged_username) is not None

    # Stop blocking the live set so the self-heal can install the live KEK.
    block_live_set["on"] = False
    keystore.lock()
    assert keystore.unlock() == new_dek
    assert _keyring_get(username) != old_kek_b64
    assert _keyring_get(staged_username) is None


def test_rotate_primary_keyfile_promote_failure_does_not_raise(monkeypatch):
    """rotate_primary shares the raise-after-commit asymmetry — also best-effort.

    A rotate keeps the DEK, so a deferred promotion is never data-loss, but the
    contract must be uniform: once save_keystore commits, rotate_primary must not
    raise. The deferred promotion self-heals on the next unlock.
    """
    dek = keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()

    real_stage = keystore._stage_keyfile_slot

    def stage_then_raise_promote(slot_id, d):
        slot, staged = real_stage(slot_id, d)

        def raising_promote():
            raise OSError(13, "keyfile.new locked by AV during promote")

        return slot, keystore._StagedKEK(
            promote=raising_promote, rollback=staged.rollback
        )

    monkeypatch.setattr(keystore, "_stage_keyfile_slot", stage_then_raise_promote)

    # MUST NOT raise.
    keystore.rotate_primary("keyfile")

    # Mid-gap: live keyfile still old, sidecar present.
    assert keystore.keyfile_path().read_bytes() == old_kek
    assert keystore.keyfile_new_path().exists()

    # Self-heal on unlock returns the (unchanged) DEK and promotes the new KEK.
    keystore.lock()
    assert keystore.unlock() == dek
    assert keystore.keyfile_path().read_bytes() != old_kek
    assert not keystore.keyfile_new_path().exists()


def test_full_rekey_keeps_new_dek_when_promote_defers(monkeypatch):
    """migrate.full_rekey must NOT roll artifacts back when promote() defers.

    End-to-end linkage for the M2 finding: with a real keyfile keystore and a
    seeded envelope artifact, force replace_data_key's post-commit promote() to
    raise. The fixed raise/return contract is that replace_data_key returns
    success once the keystore is committed, so full_rekey does NOT run its
    artifact rollback — the artifact stays under the NEW DEK and the keystore
    (self-healed from the surviving sidecar on the next unlock) yields that same
    NEW DEK, so the artifact still decrypts. Before the fix, promote()'s raise
    propagated, full_rekey rolled the artifact back to OLD-DEK ciphertext, and
    the committed-new-DEK keystore could no longer decrypt it (the data-loss
    window this change closes).
    """
    import json

    from hermes_crypto import detect, envelope, migrate

    home = keystore.get_hermes_home()
    # auth.json is NOT env-framed, so it round-trips via plain envelope
    # encrypt/decrypt (no env framing to unwrap in the assertion below).
    auth = home / "auth.json"
    secret = json.dumps({"version": 1, "providers": {}, "credential_pool": {}})
    auth.write_text(secret, encoding="utf-8")

    # enable() creates the keyfile keystore and encrypts the seeded credential.
    migrate.enable("keyfile", force=True)
    assert detect.is_encrypted(auth.read_bytes())
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    # Force the post-commit promote to raise (AV/backup holding keyfile.new).
    real_stage = keystore._stage_keyfile_slot

    def stage_then_raise_promote(slot_id, d):
        slot, staged = real_stage(slot_id, d)

        def raising_promote():
            raise OSError(13, "keyfile.new locked by AV during promote")

        return slot, keystore._StagedKEK(
            promote=raising_promote, rollback=staged.rollback
        )

    monkeypatch.setattr(keystore, "_stage_keyfile_slot", stage_then_raise_promote)

    # full_rekey must report success (the promote deferral did NOT trigger the
    # artifact rollback that would have re-encrypted auth.json under old_dek).
    result = migrate.full_rekey(force=True)
    assert "auth.json" in result.rekeyed_files
    # The artifact is still encrypted (not rolled back / left plaintext).
    assert detect.is_encrypted(auth.read_bytes())

    # The keystore self-heals on unlock to the NEW dek, and the artifact (left
    # under the new dek by full_rekey) decrypts under it — no brick.
    keystore.lock()
    new_dek = keystore.unlock()
    assert new_dek != old_dek
    assert envelope.decrypt(auth.read_bytes(), new_dek).decode("utf-8") == secret


# ─── M3: back-to-back rekey must not clobber a load-bearing staged KEK ─────────
#
# The CLI flow `hermes encrypt rotate-key --full --key-source X` runs, in ONE
# process with NO unlock() between them: full_rekey → replace_data_key, then
# _finish_key_source_rotation → rotate_primary. If replace_data_key's
# post-commit promote DEFERS (swallowed failure), the committed keystore is
# openable ONLY by the staged KEK that survives in keyfile.new / ":new" — the
# live keyfile/entry still holds the OLD KEK. The load-time self-heal that would
# finish the promotion never runs (no unlock between the two ops). The naive
# second stage would overwrite keyfile.new / ":new" — the only key that opens
# the committed keystore — and a crash before the second commit would brick the
# DEK. The entry self-heal _reconcile_live_kek finishes the deferred promotion
# first, so the staged name is free, non-load-bearing scratch at staging time.


def _defer_keyfile_promote(monkeypatch):
    """Patch _stage_keyfile_slot so the post-commit promote is a no-op (defers).

    Models a swallowed promote failure: save_keystore commits the keystore under
    the freshly-staged KEK, but the live keyfile is never promoted, so the
    committed keystore is openable only via the surviving keyfile.new sidecar.

    Returns the captured real implementation so the SECOND op in a back-to-back
    test can restore it WITHOUT monkeypatch.undo() (undo() would also revert the
    autouse fixtures' patches — HERMES_HOME redirect, _live_system_guard, etc.;
    re-setattr to the captured real is surgical, see the M2 osreplace test).
    """
    real_stage = keystore._stage_keyfile_slot

    def stage_no_promote(slot_id, dek):
        slot, staged = real_stage(slot_id, dek)
        return slot, keystore._StagedKEK(promote=lambda: None, rollback=staged.rollback)

    monkeypatch.setattr(keystore, "_stage_keyfile_slot", stage_no_promote)
    return real_stage


def _defer_keyring_promote(monkeypatch):
    """Patch _stage_keyring_slot so the post-commit promote is a no-op (defers).

    Returns the captured real implementation for surgical restore (see
    :func:`_defer_keyfile_promote` for why we avoid monkeypatch.undo()).
    """
    real_stage = keystore._stage_keyring_slot

    def stage_no_promote(slot_id, dek):
        slot, staged = real_stage(slot_id, dek)
        return slot, keystore._StagedKEK(promote=lambda: None, rollback=staged.rollback)

    monkeypatch.setattr(keystore, "_stage_keyring_slot", stage_no_promote)
    return real_stage


def test_back_to_back_keyfile_rotate_reconciles_deferred_promote(monkeypatch):
    """replace_data_key defers, then rotate_primary (no unlock) heals first.

    Reproduces the back-to-back flow: a deferred keyfile promote leaves
    keyfile.new load-bearing, then rotate_primary runs in the same process
    without an intervening unlock(). The fix reconciles (promotes the staged
    sidecar to live) at rotate_primary's entry, so the committed keystore stays
    recoverable and rotate_primary's own fresh sidecar does not clobber the only
    key material.
    """
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    mid_dek = b"\x21" * 32

    # 1. replace_data_key with a deferred promote: keystore committed under
    #    KEK_A (in keyfile.new), live keyfile still old.
    real_stage = _defer_keyfile_promote(monkeypatch)
    keystore.replace_data_key(mid_dek)
    assert keystore.get_cached_dek() == mid_dek
    assert keystore.keyfile_path().read_bytes() == old_kek  # live still old
    assert keystore.keyfile_new_path().exists()  # KEK_A load-bearing in sidecar
    kek_a = keystore.keyfile_new_path().read_bytes()
    assert kek_a != old_kek

    # 2. rotate_primary in the SAME process, NO unlock() between them. Restore
    #    the real stage so the second op promotes normally; the entry reconcile
    #    must run first and finish the deferred KEK_A promotion.
    monkeypatch.setattr(keystore, "_stage_keyfile_slot", real_stage)
    keystore.rotate_primary("keyfile")

    # The committed keystore is recoverable, the live keyfile changed (a new
    # KEK_B from rotate_primary's promote), and the deferred sidecar was healed.
    keystore.lock()
    assert keystore.unlock() == mid_dek
    assert keystore.keyfile_path().read_bytes() != old_kek
    assert keystore.keyfile_path().read_bytes() != kek_a  # KEK_B now live


def test_back_to_back_keyfile_second_op_crash_before_commit_is_recoverable(monkeypatch):
    """Deferred promote, then a SECOND op that crashes before its commit: no brick.

    Step 1 leaves keyfile.new = KEK_A load-bearing (committed keystore openable
    only by KEK_A). Step 2 (rotate_primary) reconciles KEK_A → live, then its
    save_keystore raises (simulating a crash before the second commit). The
    pre-commit rollback discards rotate's fresh sidecar; the committed keystore
    is still under KEK_A, now held by the LIVE keyfile (promoted by reconcile),
    so a subsequent unlock() still recovers the correct DEK.
    """
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    mid_dek = b"\x22" * 32

    real_stage = _defer_keyfile_promote(monkeypatch)
    keystore.replace_data_key(mid_dek)
    monkeypatch.setattr(keystore, "_stage_keyfile_slot", real_stage)  # real for step 2
    kek_a = keystore.keyfile_new_path().read_bytes()
    assert kek_a != old_kek

    # Step 2: make the SECOND op crash right before it commits the new keystore.
    real_save = keystore.save_keystore

    def boom(_data):
        raise OSError("power loss before second commit")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.rotate_primary("keyfile")

    # rotate's fresh sidecar was rolled back; the committed keystore (still under
    # KEK_A) is now opened by the LIVE keyfile, which reconcile promoted to KEK_A.
    monkeypatch.setattr(keystore, "save_keystore", real_save)
    assert keystore.keyfile_path().read_bytes() == kek_a

    keystore.lock()
    assert keystore.unlock() == mid_dek  # no brick


def test_back_to_back_keyring_rotate_reconciles_deferred_promote(monkeypatch):
    """Keyring variant of the back-to-back reconcile."""
    keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    staged_username = keystore._keyring_username_new()
    old_kek_b64 = _keyring_get(username)
    mid_dek = b"\x23" * 32

    real_stage = _defer_keyring_promote(monkeypatch)
    keystore.replace_data_key(mid_dek)
    assert keystore.get_cached_dek() == mid_dek
    # Live entry still old; staged ":new" holds the load-bearing KEK_A.
    assert _keyring_get(username) == old_kek_b64
    kek_a_b64 = _keyring_get(staged_username)
    assert kek_a_b64 is not None and kek_a_b64 != old_kek_b64

    monkeypatch.setattr(keystore, "_stage_keyring_slot", real_stage)  # real for step 2
    keystore.rotate_primary("keyring")

    keystore.lock()
    assert keystore.unlock() == mid_dek
    # Live entry advanced (KEK_B from rotate); the deferred staged entry healed.
    assert _keyring_get(username) not in (old_kek_b64, None)


def test_back_to_back_keyring_second_op_crash_before_commit_is_recoverable(monkeypatch):
    """Keyring deferred promote, then a SECOND op crashing before commit: no brick."""
    keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    staged_username = keystore._keyring_username_new()
    old_kek_b64 = _keyring_get(username)
    mid_dek = b"\x24" * 32

    real_stage = _defer_keyring_promote(monkeypatch)
    keystore.replace_data_key(mid_dek)
    monkeypatch.setattr(keystore, "_stage_keyring_slot", real_stage)
    kek_a_b64 = _keyring_get(staged_username)
    assert kek_a_b64 is not None

    real_save = keystore.save_keystore

    def boom(_data):
        raise OSError("power loss before second commit")

    monkeypatch.setattr(keystore, "save_keystore", boom)
    with pytest.raises(OSError):
        keystore.rotate_primary("keyring")
    monkeypatch.setattr(keystore, "save_keystore", real_save)

    # Reconcile promoted KEK_A onto the live entry; rotate's staged entry was
    # rolled back. The committed keystore (under KEK_A) opens via the live entry.
    assert _keyring_get(username) == kek_a_b64
    keystore.lock()
    assert keystore.unlock() == mid_dek  # no brick


def test_back_to_back_via_replace_data_key_second_op_reconciles(monkeypatch):
    """The second op may also be replace_data_key (full_rekey twice) — also heals."""
    keystore.init_keystore("keyfile")
    old_kek = keystore.keyfile_path().read_bytes()
    mid_dek = b"\x25" * 32
    final_dek = b"\x26" * 32

    real_stage = _defer_keyfile_promote(monkeypatch)
    keystore.replace_data_key(mid_dek)
    monkeypatch.setattr(keystore, "_stage_keyfile_slot", real_stage)
    kek_a = keystore.keyfile_new_path().read_bytes()
    assert kek_a != old_kek

    # Second replace_data_key WITHOUT an intervening unlock(): the cached DEK is
    # still mid_dek (the first op cached it), so this is a valid in-process rekey.
    keystore.replace_data_key(final_dek)
    keystore.lock()
    assert keystore.unlock() == final_dek
    assert keystore.keyfile_path().read_bytes() not in (old_kek, kek_a)


def test_reconcile_raises_when_no_kek_on_disk_opens_keystore_keyfile(monkeypatch):
    """Entry reconcile RAISES (no clobber) when neither live nor staged KEK opens it.

    Models the catastrophic case: the committed keystore is wrapped under a KEK
    that exists in neither the live keyfile nor the staged sidecar (e.g. the
    keyfile was swapped for an unrelated 32-byte blob and no sidecar survives).
    Staging a new KEK here would overwrite the only remaining key material, so
    the entry reconcile must REFUSE and leave the committed keystore untouched.
    """
    keystore.init_keystore("keyfile")
    # Corrupt the live keyfile to a wrong (but well-formed) 32-byte KEK and
    # ensure no staged sidecar exists.
    keystore.keyfile_path().write_bytes(b"\x00" * 32)
    if keystore.keyfile_new_path().exists():
        keystore.keyfile_new_path().unlink()
    committed_before = keystore.keystore_path().read_bytes()

    with pytest.raises(KeystoreError, match="no key on disk"):
        keystore.rotate_primary("keyfile")

    # The committed keystore was NOT touched (no staging, no commit).
    assert keystore.keystore_path().read_bytes() == committed_before
    assert not keystore.keyfile_new_path().exists()


def test_reconcile_raises_when_no_kek_on_disk_opens_keystore_keyring(monkeypatch):
    """Keyring variant: reconcile refuses when no live/staged keyring KEK opens it."""
    import base64

    keystore.init_keystore("keyring")
    username = keystore._keyring_username()
    # Replace the live keyring KEK with an unrelated 32-byte value; no ":new".
    keystore._keyring_set_kek(username, b"\x00" * 32)
    keystore._keyring_delete_kek(keystore._keyring_username_new())
    committed_before = keystore.keystore_path().read_bytes()

    with pytest.raises(KeystoreError, match="no key on disk"):
        keystore.replace_data_key(b"\x27" * 32)

    assert keystore.keystore_path().read_bytes() == committed_before
    assert _keyring_get(keystore._keyring_username_new()) is None


def test_reconcile_is_noop_on_normal_path_keyfile(monkeypatch):
    """When the live KEK already opens the committed slot, reconcile is a no-op.

    Pins the guarantee that the common path is unchanged: with no deferred
    promotion pending, _reconcile_live_kek must NOT promote, NOT raise, and the
    rotation must proceed exactly as before.
    """
    keystore.init_keystore("keyfile")
    dek = keystore.get_cached_dek()
    old_kek = keystore.keyfile_path().read_bytes()
    assert not keystore.keyfile_new_path().exists()

    called = {"promote": 0}
    real_promote = keystore._promote_staged_for_slot

    def counting_promote(slot):
        called["promote"] += 1
        return real_promote(slot)

    monkeypatch.setattr(keystore, "_promote_staged_for_slot", counting_promote)

    keystore.rotate_primary("keyfile")
    # No deferred promotion existed → reconcile never called _promote_staged_for_slot.
    assert called["promote"] == 0
    new_kek = keystore.keyfile_path().read_bytes()
    assert new_kek != old_kek  # normal fresh-KEK rotation still happened
    keystore.lock()
    assert keystore.unlock() == dek


def test_reconcile_allows_rotate_when_recovery_present_despite_lost_keyfile():
    """Lost keyfile + recovery slot: rotate-to-passphrase via recovery is ALLOWED.

    Disaster-recovery path. The live keyfile is genuinely gone (detached USB /
    wiped profile) and no staging sidecar exists, but a recovery slot still wraps
    the DEK, so the keystore is provably healthy. The operator unlocks with the
    recovery code (DEK cached) and rotates onto a passphrase to abandon the lost
    keyfile. _reconcile_live_kek must NOT raise: neither the live nor the staged
    primary KEK opens the committed slot, but the recovery slot makes the DEK
    recoverable independently of the primary KEK and this rotation re-wraps the
    already-cached DEK, so proceeding is brick-safe. (Contrast the no-recovery
    case in test_reconcile_raises_when_no_kek_on_disk_opens_keystore_keyfile,
    which still refuses.)
    """
    dek = keystore.init_keystore("keyfile")
    code = keystore.add_recovery_slot(FAST_ARGON2)
    # Primary keyfile KEK is genuinely lost; no staged sidecar survives.
    keystore.keyfile_path().unlink()
    if keystore.keyfile_new_path().exists():
        keystore.keyfile_new_path().unlink()

    # Unlock via the recovery code → DEK cached, keystore provably recoverable.
    keystore.lock()
    assert keystore.unlock(recovery_code=code) == dek

    # Reconcile proceeds (recovery slot present); the rotation re-wraps the
    # cached DEK and rotate_primary preserves the recovery slot.
    keystore.rotate_primary("passphrase", new_passphrase="newpw", argon2_params=FAST_ARGON2)

    assert keystore.primary_slot_type() == "passphrase"
    keystore.lock()
    assert keystore.unlock(passphrase="newpw") == dek
    assert keystore.unlock(recovery_code=code) == dek  # recovery slot still works


def test_reconcile_allows_rotate_when_recovery_present_despite_lost_keyring():
    """Keyring variant: lost keyring KEK + recovery slot → rotate-to-passphrase allowed."""
    dek = keystore.init_keystore("keyring")
    code = keystore.add_recovery_slot(FAST_ARGON2)
    # Live keyring entry wiped; no ":new" staged entry survives.
    keystore._keyring_delete_kek(keystore._keyring_username())
    keystore._keyring_delete_kek(keystore._keyring_username_new())
    assert _keyring_get(keystore._keyring_username()) is None

    keystore.lock()
    assert keystore.unlock(recovery_code=code) == dek

    keystore.rotate_primary("passphrase", new_passphrase="newpw", argon2_params=FAST_ARGON2)

    assert keystore.primary_slot_type() == "passphrase"
    keystore.lock()
    assert keystore.unlock(passphrase="newpw") == dek
    assert keystore.unlock(recovery_code=code) == dek  # recovery slot still works
