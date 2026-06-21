"""Keystore: the wrapped Data Encryption Key and its key slots.

Design — envelope encryption with LUKS-style key slots:

* A single random 32-byte **Data Encryption Key (DEK)** encrypts every file
  and is the SQLCipher ``PRAGMA key``. It is generated once and never changes
  when a passphrase changes.
* The DEK is wrapped (AES-256-GCM) by a **Key Encryption Key (KEK)** and the
  wrapped copy is stored in ``~/.hermes/.encryption/keystore.json``.
* The keystore holds one or more **slots**, each an independent KEK-wrapped
  copy of the *same* DEK. A typical install has a ``primary`` slot plus an
  optional ``recovery`` slot. This makes changing a passphrase cheap (re-wrap,
  no data touched) and lets one install be unlocked by more than one method.

KEK sources:

* ``keyring``    — a random KEK held in the OS keyring (Windows Credential
  Manager / macOS Keychain / Linux Secret Service).
* ``passphrase`` — KEK derived from a user passphrase via Argon2id/scrypt.
* ``keyfile``    — a random KEK in ``~/.hermes/.encryption/keyfile`` (0600).
* ``recovery``   — KEK derived from a one-time base32 recovery code.

Every rekey (``replace_data_key``) and primary-slot rotation
(``rotate_primary``) generates a **fresh KEK** for ``keyfile`` and
``keyring`` slots — the old wrapping key must not survive either operation.

The decrypted DEK is cached in a module-level variable for the process
lifetime only — it is never written to disk in cleartext.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home

from utils import atomic_replace

from . import audit, kdf
from .errors import DecryptionError, DependencyError, KeystoreError, LockedError
from .fileio import atomic_write_private, harden_dir

KEYSTORE_VERSION = 1
KEYRING_SERVICE = "hermes-agent-encryption"

VALID_KEY_SOURCES = ("keyring", "passphrase", "keyfile")

_log = logging.getLogger(__name__)

# Process-lifetime cache of the decrypted DEK. Never persisted.
_cached_dek: Optional[bytes] = None


# ─── Paths ────────────────────────────────────────────────────────────────────


def encryption_dir() -> Path:
    """Return ``<HERMES_HOME>/.encryption`` (the keystore/keyfile/backup dir)."""
    return get_hermes_home() / ".encryption"


def keystore_path() -> Path:
    return encryption_dir() / "keystore.json"


def keyfile_path() -> Path:
    return encryption_dir() / "keyfile"


def keyfile_new_path() -> Path:
    """The staging sidecar for a crash-safe keyfile rekey (see ``_stage_keyfile_slot``).

    A new KEK is written here first; only after the keystore (with the DEK
    wrapped under that new KEK) is durably committed is it atomically promoted
    over :func:`keyfile_path`. A crash leaves the sidecar present so the
    load-time fallback in :func:`unlock` can finish the promotion.
    """
    return encryption_dir() / "keyfile.new"


def backup_dir() -> Path:
    return encryption_dir() / "backup"


def keystore_exists() -> bool:
    return keystore_path().exists()


# ─── DEK cache ────────────────────────────────────────────────────────────────


def cache_dek(dek: bytes) -> None:
    """Store the decrypted DEK in the process-lifetime cache."""
    global _cached_dek
    if len(dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    _cached_dek = dek


def get_cached_dek() -> Optional[bytes]:
    return _cached_dek


def is_unlocked() -> bool:
    return _cached_dek is not None


def lock() -> None:
    """Drop the cached DEK from memory."""
    global _cached_dek
    _cached_dek = None


# ─── Keyring helpers ──────────────────────────────────────────────────────────


def _keyring_username() -> str:
    """A keyring entry name namespaced by HERMES_HOME so profiles don't collide."""
    home = str(get_hermes_home().resolve())
    digest = hashlib.sha256(home.encode("utf-8")).hexdigest()[:16]
    return f"kek:{digest}"


def _keyring_username_new() -> str:
    """Staging entry name for a crash-safe keyring rekey.

    The OS keyring is single-valued per (service, username), so a new KEK is
    held under this distinct ``:new`` username until the keystore is committed
    and the new KEK is promoted to the live :func:`_keyring_username` entry. A
    crash leaves the staged entry present for the load-time fallback to finish.
    """
    return _keyring_username() + ":new"


def _import_keyring():
    try:
        import keyring
    except ImportError as exc:
        raise DependencyError(
            "The 'keyring' package is required for keyring-backed encryption. "
            "Install it with:  pip install 'hermes-agent[encryption]'"
        ) from exc
    return keyring


def keyring_is_secure() -> bool:
    """Return True only when the active keyring backend is a real OS keyring.

    On headless hosts ``keyring`` silently falls back to a plaintext file
    backend (``keyrings.alt``) or a no-op ``fail`` backend — either defeats
    the purpose, so keyring mode must be refused there in favour of a
    passphrase.
    """
    try:
        keyring = _import_keyring()
        backend = keyring.get_keyring()
    except Exception:
        return False
    module = type(backend).__module__ or ""
    insecure_prefixes = ("keyrings.alt", "keyring.backends.fail", "keyring.backends.null")
    return not any(module.startswith(prefix) for prefix in insecure_prefixes)


def _keyring_set_kek(username: str, kek: bytes) -> None:
    keyring = _import_keyring()
    keyring.set_password(KEYRING_SERVICE, username, base64.b64encode(kek).decode("ascii"))


def _keyring_get_kek(username: str) -> bytes:
    keyring = _import_keyring()
    stored = keyring.get_password(KEYRING_SERVICE, username)
    if not stored:
        raise KeystoreError(
            "No encryption key found in the OS keyring for this install. "
            "The keyring entry may have been deleted, or this is a different "
            "machine. Unlock with a recovery code instead."
        )
    try:
        kek = base64.b64decode(stored)
    except ValueError as exc:
        raise KeystoreError("corrupt keyring entry") from exc
    if len(kek) != 32:
        raise KeystoreError("corrupt keyring entry (wrong length)")
    return kek


def _keyring_delete_kek(username: str) -> None:
    try:
        keyring = _import_keyring()
        keyring.delete_password(KEYRING_SERVICE, username)
    except Exception:
        pass


# ─── Recovery codes ───────────────────────────────────────────────────────────


def generate_recovery_code() -> str:
    """Return a fresh 160-bit recovery code as dash-grouped base32 text."""
    raw = secrets.token_bytes(20)
    body = base64.b32encode(raw).decode("ascii").rstrip("=")
    return "-".join(body[i : i + 5] for i in range(0, len(body), 5))


def normalize_recovery_code(code: str) -> bytes:
    """Canonicalise a user-typed recovery code back to its raw bytes."""
    cleaned = "".join(ch for ch in (code or "").upper() if ch.isalnum())
    if not cleaned:
        raise ValueError("empty recovery code")
    padding = "=" * (-len(cleaned) % 8)
    try:
        return base64.b32decode(cleaned + padding)
    except ValueError as exc:
        raise ValueError("invalid recovery code") from exc


# ─── Slot construction / resolution ───────────────────────────────────────────


def _wrap_dek(dek: bytes, kek: bytes) -> str:
    from . import envelope

    return base64.b64encode(envelope.encrypt(dek, kek)).decode("ascii")


def _unwrap_dek(wrapped_b64: str, kek: bytes) -> bytes:
    from . import envelope

    try:
        wrapped = base64.b64decode(wrapped_b64)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise KeystoreError("keystore slot is malformed") from exc
    dek = envelope.decrypt(wrapped, kek)
    if len(dek) != 32:
        raise DecryptionError("unwrapped DEK has the wrong length")
    return dek


def _build_keyring_slot(slot_id: str, dek: bytes) -> Dict[str, Any]:
    if not keyring_is_secure():
        raise KeystoreError(
            "The OS keyring backend on this host is insecure or unavailable "
            "(no Secret Service / Credential Manager). Use 'passphrase' mode "
            "instead:  hermes encrypt enable --key-source passphrase"
        )
    username = _keyring_username()
    kek = secrets.token_bytes(32)
    _keyring_set_kek(username, kek)
    return {
        "id": slot_id,
        "type": "keyring",
        "keyring_id": username,
        "wrapped_dek": _wrap_dek(dek, kek),
    }


def _build_keyfile_slot(
    slot_id: str, dek: bytes, *, force_new_kek: bool = False
) -> Dict[str, Any]:
    """Build a keyfile slot wrapping *dek*.

    When *force_new_kek* is True the existing keyfile (if any) is overwritten
    in place with a freshly generated 32-byte KEK. ``atomic_write_private``
    handles the overwrite via os.replace, so this is race-free even when an
    AV/backup/concurrent-Hermes handle would block an unlink. Used by
    ``rotate_primary`` / ``replace_data_key`` to guarantee a new KEK on rekey
    even if a stale keyfile is still on disk.

    With the default ``force_new_kek=False`` an existing keyfile is honoured:
    callers like ``init_keystore`` must respect a pre-placed operator keyfile.
    """
    path = keyfile_path()
    if force_new_kek:
        kek = secrets.token_bytes(32)
        atomic_write_private(path, kek)
    elif path.exists():
        kek = path.read_bytes()
        if len(kek) != 32:
            raise KeystoreError("existing keyfile has the wrong length")
    else:
        kek = secrets.token_bytes(32)
        atomic_write_private(path, kek)
    return {
        "id": slot_id,
        "type": "keyfile",
        "wrapped_dek": _wrap_dek(dek, kek),
    }


# ─── Crash-safe rekey: stage → commit → promote → rollback ──────────────────────
#
# rotate_primary / replace_data_key used to overwrite the live keyfile / keyring
# KEK *before* save_keystore persisted the new KEK-wrapped DEK. A failure (or
# power loss) in that gap left the live KEK new but keystore.json still wrapped
# under the destroyed old KEK → DEK unrecoverable → every artifact bricked.
#
# The fix splits KEK generation from KEK *installation*: the new KEK is written
# to a sidecar (keyfile.new) or staged keyring entry (":new" username), the
# keystore is committed atomically, and only then is the new KEK promoted over
# the live one.
#
# Raise/return contract (relied on by migrate.full_rekey): rotate_primary /
# replace_data_key RAISE only on a PRE-commit failure — that branch rolls the
# sidecar back, leaving the OLD KEK + OLD keystore as the live, self-consistent
# pair, so a caller may treat "raised" as "keystore untouched". Once
# save_keystore returns, the post-commit promote is BEST-EFFORT and never
# propagates (see _promote_committed_kek): a promote failure — or a power loss
# in the commit→promote gap — leaves the keystore committed under the new KEK
# while that new KEK survives in the staged sidecar/":new" entry, so the
# load-time fallback in ``unlock`` (_try_staged_unlock) reads the staged KEK and
# self-heals the promotion on the next load. (Older revisions re-raised a
# post-commit promote failure; for replace_data_key that was a data-loss bug —
# migrate would roll artifacts back to OLD-DEK ciphertext while the committed
# keystore yields the NEW DEK, bricking every artifact.)


@dataclass
class _StagedKEK:
    """A new KEK held in a sidecar, with promote()/rollback() for the live slot.

    ``promote`` installs the staged KEK as the live KEK (only call *after* the
    keystore is durably committed). ``rollback`` discards the staged KEK,
    leaving the live KEK untouched (call when save_keystore failed). Both are
    idempotent and safe to call on the success-path cleanup too.
    """

    promote: Callable[[], None]
    rollback: Callable[[], None]


def _noop_staged_kek() -> _StagedKEK:
    """A no-op staged object for passphrase mode, so call sites stay uniform.

    Passphrase rekey derives the KEK in-memory and persists nothing outside
    save_keystore, so there is no separate KEK to stage, promote, or roll back.
    """
    return _StagedKEK(promote=lambda: None, rollback=lambda: None)


def _stage_keyfile_slot(slot_id: str, dek: bytes) -> tuple[Dict[str, Any], _StagedKEK]:
    """Generate a fresh keyfile KEK in a sidecar and wrap *dek* under it.

    The live keyfile is left UNTOUCHED. The returned slot dict is byte-identical
    to ``_build_keyfile_slot(..., force_new_kek=True)`` output (no schema
    change); only the timing of the KEK install differs.

    promote(): atomically replace the live keyfile with the sidecar (os.replace
    is atomic at the FS layer — the live keyfile is always fully-old or
    fully-new), then best-effort fsync the dir and drop the sidecar.

    rollback(): unlink the sidecar; the live keyfile was never touched.

    RESIDUAL POWER-LOSS WINDOW: the only window not closed is a crash *during*
    the os.replace inside promote(); but os.replace is atomic, and the sidecar
    is only unlinked after the replace lands, so the staged copy survives if the
    replace had not yet committed — the load-time fallback recovers either way.
    """
    new_path = keyfile_new_path()
    kek = secrets.token_bytes(32)
    atomic_write_private(new_path, kek)
    slot = {
        "id": slot_id,
        "type": "keyfile",
        "wrapped_dek": _wrap_dek(dek, kek),
    }

    def promote() -> None:
        live = keyfile_path()
        atomic_replace(new_path, live)
        try:
            dir_fd = os.open(str(live.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        # The replace consumed the sidecar; clear any stray leftover too.
        try:
            new_path.unlink()
        except OSError:
            pass

    def rollback() -> None:
        try:
            new_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    return slot, _StagedKEK(promote=promote, rollback=rollback)


def _stage_keyring_slot(slot_id: str, dek: bytes) -> tuple[Dict[str, Any], _StagedKEK]:
    """Generate a fresh keyring KEK under the staged username and wrap *dek*.

    The live keyring entry is left UNTOUCHED. ``keyring_id`` in the returned
    slot is the FINAL (live) username — after promotion the KEK lives there;
    the ``:new`` entry is only a temporary holding spot.

    promote(): read the staged KEK, set it on the live entry, delete the staged
    entry. The OS keyring has no atomic two-key swap, so a crash leaves the
    staged entry intact and the live entry either old (pre-set) or new
    (post-set); the load-time fallback tries both, so it is recoverable.

    rollback(): delete the staged entry; the live entry was never touched.
    """
    if not keyring_is_secure():
        raise KeystoreError(
            "The OS keyring backend on this host is insecure or unavailable "
            "(no Secret Service / Credential Manager). Use 'passphrase' mode "
            "instead:  hermes encrypt enable --key-source passphrase"
        )
    live_username = _keyring_username()
    staged_username = _keyring_username_new()
    kek = secrets.token_bytes(32)
    _keyring_set_kek(staged_username, kek)
    slot = {
        "id": slot_id,
        "type": "keyring",
        "keyring_id": live_username,
        "wrapped_dek": _wrap_dek(dek, kek),
    }

    def promote() -> None:
        staged_kek = _keyring_get_kek(staged_username)
        _keyring_set_kek(live_username, staged_kek)
        _keyring_delete_kek(staged_username)

    def rollback() -> None:
        _keyring_delete_kek(staged_username)

    return slot, _StagedKEK(promote=promote, rollback=rollback)


def _build_secret_slot(
    slot_id: str,
    slot_type: str,
    dek: bytes,
    secret: bytes,
    argon2_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a passphrase- or recovery-derived slot."""
    salt = secrets.token_bytes(16)
    kdf_id = kdf.preferred_kdf_id()
    if slot_type == "recovery":
        # High-entropy random secret — see kdf.RECOVERY_ARGON2_PARAMS.
        base_params = kdf.RECOVERY_ARGON2_PARAMS
    else:
        base_params = kdf.DEFAULT_ARGON2_PARAMS
    params = {**base_params, **(argon2_params or {})}
    kek = kdf.derive_kek(secret, salt, kdf_id, params)
    return {
        "id": slot_id,
        "type": slot_type,
        "kdf_id": kdf_id,
        "kdf_params": params if kdf_id == kdf.KDF_ARGON2ID else {},
        "salt": base64.b64encode(salt).decode("ascii"),
        "wrapped_dek": _wrap_dek(dek, kek),
    }


def _kek_for_slot(
    slot: Dict[str, Any],
    *,
    passphrase: Optional[str] = None,
    recovery_code: Optional[str] = None,
) -> bytes:
    slot_type = slot.get("type")
    if slot_type == "keyring":
        return _keyring_get_kek(slot.get("keyring_id") or _keyring_username())
    if slot_type == "keyfile":
        path = keyfile_path()
        if not path.exists():
            raise KeystoreError(f"keyfile missing: {path}")
        kek = path.read_bytes()
        if len(kek) != 32:
            raise KeystoreError("keyfile has the wrong length")
        return kek
    if slot_type in ("passphrase", "recovery"):
        if slot_type == "passphrase":
            if passphrase is None:
                raise LockedError("a passphrase is required to unlock this keystore")
            secret = passphrase.encode("utf-8")
        else:
            if recovery_code is None:
                raise LockedError("a recovery code is required")
            secret = normalize_recovery_code(recovery_code)
        try:
            salt = base64.b64decode(slot["salt"])
            kdf_id = int(slot["kdf_id"])
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise KeystoreError("keystore slot is malformed") from exc
        return kdf.derive_kek(
            secret, salt, kdf_id, slot.get("kdf_params") or None
        )
    raise KeystoreError(f"unknown slot type {slot_type!r}")


def _kek_for_slot_staged(slot: Dict[str, Any]) -> Optional[bytes]:
    """Return the STAGED KEK for a keyfile/keyring slot, or None if absent.

    Used by the load-time fallback in :func:`unlock` to close the power-loss
    window between save_keystore (commit) and the KEK promotion: if a crash
    landed in that gap, the keystore is already committed under the new KEK and
    that new KEK survives in the sidecar (keyfile.new) / staged keyring entry.
    """
    slot_type = slot.get("type")
    if slot_type == "keyfile":
        path = keyfile_new_path()
        if not path.exists():
            return None
        kek = path.read_bytes()
        return kek if len(kek) == 32 else None
    if slot_type == "keyring":
        try:
            return _keyring_get_kek(_keyring_username_new())
        except (KeystoreError, DependencyError):
            return None
    return None


def _promote_staged_for_slot(slot: Dict[str, Any]) -> None:
    """Finish a stalled rekey promotion from the staged KEK source (self-heal).

    Best-effort: a failure here (e.g. a read-only encryption dir) must NOT turn
    a successful staged-KEK unlock into a failure — the caller swallows errors.
    """
    slot_type = slot.get("type")
    if slot_type == "keyfile":
        atomic_replace(keyfile_new_path(), keyfile_path())
        try:
            keyfile_new_path().unlink()
        except OSError:
            pass
    elif slot_type == "keyring":
        staged_username = _keyring_username_new()
        kek = _keyring_get_kek(staged_username)
        _keyring_set_kek(slot.get("keyring_id") or _keyring_username(), kek)
        _keyring_delete_kek(staged_username)


def _slot_opens_with_kek(slot: Dict[str, Any], kek: Optional[bytes]) -> bool:
    """Return True iff *kek* unwraps the committed ``wrapped_dek`` in *slot*."""
    if kek is None:
        return False
    wrapped = slot.get("wrapped_dek")
    if not isinstance(wrapped, str):
        return False
    try:
        _unwrap_dek(wrapped, kek)
    except (DecryptionError, KeystoreError):
        return False
    return True


def _reconcile_live_kek(keystore: Dict[str, Any]) -> None:
    """Heal a deferred KEK promotion BEFORE staging a new KEK (entry self-heal).

    Closes the back-to-back-rekey BRICK window. After a deferred post-commit
    promote (see :func:`_promote_committed_kek`), the committed keystore is
    openable only by the STAGED KEK that still lives in the sidecar
    (``keyfile.new``) / staged keyring (``:new``) entry — the live keyfile /
    keyring entry still holds the OLD KEK. The load-time self-heal in
    :func:`unlock` normally finishes that promotion, but a second rekey in the
    SAME process (``rotate-key --full --key-source X``: replace_data_key →
    rotate_primary, with no intervening ``unlock``) never runs it. The next
    stage would then ``atomic_write_private(keyfile.new, KEK_new)`` /
    ``_keyring_set_kek(":new", KEK_new)`` and CLOBBER the only KEK that opens
    the committed keystore — a crash before the second commit bricks the DEK.

    This makes the staged sidecar name GUARANTEED to be free, non-load-bearing
    scratch space at the moment of staging:

    * If the LIVE KEK already opens the committed primary slot (the
      overwhelmingly common case — every non-deferred rekey), this is a NO-OP
      and behaviour is byte-for-byte as before.
    * Else, if the STAGED KEK opens the committed primary slot (a deferred
      promotion is pending), finish that promotion now — exactly as the
      load-time self-heal does (:func:`_promote_staged_for_slot`) — so the live
      KEK becomes load-bearing and the staged name is freed.
    * Else, neither KEK on disk opens the committed keystore. If a RECOVERY slot
      exists, the DEK is recoverable independently of the (lost) primary KEK and
      this rotation re-wraps the already-cached DEK under a fresh slot, so
      proceeding is brick-safe — this is the canonical "lost my keyfile, recover
      via recovery code, rotate onto a passphrase" path, so we do NOT raise.
      Only when there is ALSO no recovery slot — the committed keystore is then
      recoverable solely from the in-memory cached DEK — do we RAISE
      ``KeystoreError`` rather than stage: staging would overwrite the last
      surviving on-disk key material and brick the keystore; raising leaves the
      committed keystore + any surviving KEK untouched for a later ``unlock``/heal.

    Only ``keyfile`` / ``keyring`` primary slots have an external KEK store, so
    only those are reconciled. Passphrase/recovery primaries derive their KEK
    from the (salt, secret) pair with no sidecar, so reconcile is a no-op there.
    """
    primary = next(
        (s for s in _slots(keystore) if s.get("type") != "recovery"), None
    )
    if primary is None:
        return
    slot_type = primary.get("type")
    if slot_type not in ("keyfile", "keyring"):
        return  # passphrase/recovery: KEK is derived, no sidecar to reconcile

    # Does the LIVE KEK open the committed primary slot? (normal path → no-op)
    try:
        live_kek = _kek_for_slot(primary)
    except (KeystoreError, DependencyError, OSError):
        live_kek = None
    if _slot_opens_with_kek(primary, live_kek):
        return

    # Live KEK does not open it. A deferred promotion may have left the
    # load-bearing KEK in the staged sidecar/":new" entry — finish it now.
    try:
        staged_kek = _kek_for_slot_staged(primary)
    except (KeystoreError, DependencyError, OSError):
        staged_kek = None
    if _slot_opens_with_kek(primary, staged_kek):
        # Promote staged → live so the staged name is free, non-load-bearing
        # scratch before the caller stages a new KEK. Unlike the best-effort
        # load-time heal, this MUST land: if the promote fails the staged name
        # is still load-bearing, so re-raise (the caller must not proceed to
        # stage). The committed keystore + staged KEK are untouched on failure.
        try:
            _promote_staged_for_slot(primary)
        except (OSError, KeystoreError, DependencyError) as exc:
            raise KeystoreError(
                "A previous rekey left the data key recoverable only from a "
                "staged key sidecar, and completing that deferred promotion "
                f"failed ({type(exc).__name__}). Run 'hermes encrypt unlock' "
                "to self-heal, then retry the rotation."
            ) from exc
        return

    # Neither the live nor the staged KEK opens the committed keystore.
    #
    # If a recovery slot exists, the DEK is recoverable independently of the
    # (lost/corrupt) primary KEK: the caller already unlocked — rotate_primary /
    # replace_data_key require a cached DEK — and the committed recovery slot
    # still wraps that same DEK. The pending rotation re-wraps the cached DEK
    # under a fresh primary slot, and in THIS branch the staged sidecar does not
    # open the committed slot (by the branch precondition), so staging cannot
    # clobber the only key that opens it. Proceeding is therefore brick-safe and
    # is the canonical disaster-recovery path ("I lost my keyfile — unlock with
    # a recovery code and rotate onto a passphrase"). Refusing here would block
    # that legal, lossless rotation with a self-contradicting "unlock with a
    # recovery code first" message (the operator already did). The
    # refuse-to-clobber fail-safe below is preserved for the genuinely-precarious
    # no-recovery case, where the committed keystore is recoverable ONLY from the
    # in-memory cached DEK.
    if any(s.get("type") == "recovery" for s in _slots(keystore)):
        return

    # No recovery slot either: the committed keystore is recoverable ONLY from
    # the cached DEK in memory. Staging a new KEK now would overwrite the last
    # surviving on-disk key material, so a crash before the next commit would
    # brick the DEK. Refuse — the committed keystore is left intact for a later
    # heal (restore the keyfile / keyring entry, or add a recovery code first).
    raise KeystoreError(
        "Refusing to rotate: no key on disk (live keyfile/keyring entry nor "
        "the staged rekey sidecar) can open the committed keystore, and no "
        "recovery slot exists. Staging a new key here would destroy the only "
        "remaining key material. Restore the keyfile / keyring entry (or add a "
        "recovery code) first."
    )


def _promote_committed_kek(staged: "_StagedKEK", *, key_source: Optional[str]) -> None:
    """Install the live KEK AFTER the keystore is durably committed — best-effort.

    Contract: this is the post-commit half of the stage→commit→promote protocol
    and MUST NOT raise. The keystore is already committed under the NEW KEK, and
    that NEW KEK still survives in the staged sidecar (``keyfile.new``) / ``:new``
    keyring entry, so if ``staged.promote()`` fails (e.g. os.replace OSError
    because AV/backup grabbed ``keyfile.new``, or a transient keyring backend
    error in promote) the live KEK install is simply DEFERRED to the load-time
    self-heal in :func:`unlock` (via :func:`_try_staged_unlock`). Re-raising here
    would be a data-loss bug: callers like ``migrate.full_rekey`` treat ANY
    exception from ``replace_data_key`` as "keystore untouched" and roll every
    artifact back to its OLD-DEK ciphertext — but the keystore now yields the
    NEW DEK, so every artifact would become undecryptable.

    On promote failure we record a best-effort audit event (no secret material)
    and a logger.warning, then return normally so the caller proceeds to its
    success path.
    """
    try:
        staged.promote()
    except Exception as exc:  # noqa: BLE001 — must not propagate past a commit.
        # The keystore is committed under the new KEK; the staged sidecar/entry
        # survives, so unlock()'s _try_staged_unlock will finish the promotion
        # on the next load. Surface the deferral without leaking key material.
        _log.warning(
            "Keystore committed under the new KEK, but installing the live KEK "
            "failed (%s); promotion deferred to the load-time self-heal on next "
            "unlock.",
            type(exc).__name__,
        )
        audit.log_event(
            audit.KEYSTORE_ROTATED,
            audit.FAILURE,
            key_source=key_source,
            reason="kek_promote_deferred",
            error=type(exc).__name__,
        )


# ─── Keystore file I/O ────────────────────────────────────────────────────────


def load_keystore() -> Dict[str, Any]:
    path = keystore_path()
    if not path.exists():
        raise KeystoreError(
            "Encryption is enabled but no keystore exists. Run "
            "'hermes encrypt enable' to set one up."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise KeystoreError(f"keystore is unreadable or corrupt: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("slots"), list):
        raise KeystoreError("keystore is malformed")
    try:
        version = int(data.get("version", 0))
    except (TypeError, ValueError) as exc:
        raise KeystoreError("keystore is malformed (invalid version)") from exc
    if version > KEYSTORE_VERSION:
        raise KeystoreError(
            f"keystore version {version} is newer than this Hermes "
            f"understands ({KEYSTORE_VERSION}); upgrade Hermes."
        )
    return data


def save_keystore(data: Dict[str, Any]) -> None:
    harden_dir(encryption_dir())
    payload = json.dumps(data, indent=2, sort_keys=False).encode("utf-8")
    atomic_write_private(keystore_path(), payload)


def _slots(keystore: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(keystore.get("slots") or [])


def primary_slot_type() -> Optional[str]:
    """Return the type of the non-recovery slot, or None when no keystore exists."""
    if not keystore_exists():
        return None
    try:
        keystore = load_keystore()
    except KeystoreError:
        return None
    for slot in _slots(keystore):
        if slot.get("type") != "recovery":
            return slot.get("type")
    return None


def has_recovery_slot() -> bool:
    if not keystore_exists():
        return False
    try:
        keystore = load_keystore()
    except KeystoreError:
        return False
    return any(slot.get("type") == "recovery" for slot in _slots(keystore))


# ─── Public operations ────────────────────────────────────────────────────────


def init_keystore(
    key_source: str,
    *,
    passphrase: Optional[str] = None,
    argon2_params: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Create a brand-new keystore with one primary slot. Returns the DEK."""
    if key_source not in VALID_KEY_SOURCES:
        raise ValueError(f"key_source must be one of {VALID_KEY_SOURCES}")
    if keystore_exists():
        raise KeystoreError(
            "A keystore already exists. Use 'hermes encrypt rotate-key' to "
            "change keys, or 'hermes encrypt disable' first."
        )
    harden_dir(encryption_dir())
    dek = secrets.token_bytes(32)

    if key_source == "keyring":
        slot = _build_keyring_slot("primary", dek)
    elif key_source == "keyfile":
        slot = _build_keyfile_slot("primary", dek)
    else:  # passphrase
        if not passphrase:
            raise ValueError("passphrase mode requires a passphrase")
        slot = _build_secret_slot("primary", "passphrase", dek, passphrase.encode("utf-8"), argon2_params)

    keystore = {
        "version": KEYSTORE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "key_source": key_source,
        "slots": [slot],
    }
    save_keystore(keystore)
    cache_dek(dek)
    audit.log_event(audit.KEYSTORE_CREATED, audit.SUCCESS, key_source=key_source)
    return dek


def _try_staged_unlock(
    slot: Dict[str, Any], primary_error: Exception
) -> Optional[bytes]:
    """Attempt to unlock *slot* with its STAGED KEK after the live KEK failed.

    Returns the DEK and self-heals the promotion on success, or None if there
    is no staged source or it does not unwrap the committed slot. Only applies
    to keyfile/keyring slots; passphrase/recovery slots have no staged KEK.
    """
    if slot.get("type") not in ("keyfile", "keyring"):
        return None
    if not isinstance(primary_error, (DecryptionError, KeystoreError)):
        return None
    try:
        staged_kek = _kek_for_slot_staged(slot)
    except (KeystoreError, DependencyError, OSError):
        return None
    if staged_kek is None:
        return None
    try:
        wrapped_dek = slot["wrapped_dek"]
    except KeyError:
        return None
    try:
        dek = _unwrap_dek(wrapped_dek, staged_kek)
    except (DecryptionError, KeystoreError):
        return None
    # The staged KEK opens the committed slot → finish the stalled promotion so
    # the next load uses the live slot. Best-effort: never fail a recoverable
    # unlock if the heal write cannot land (e.g. read-only mount).
    try:
        _promote_staged_for_slot(slot)
    except (OSError, KeystoreError, DependencyError):
        pass
    return dek


def unlock(
    *,
    passphrase: Optional[str] = None,
    recovery_code: Optional[str] = None,
) -> bytes:
    """Unwrap the DEK from the keystore, cache it, and return it.

    With no arguments only ``keyring``/``keyfile`` slots can be opened. Pass
    *passphrase* or *recovery_code* to open the matching slot.

    For keyfile/keyring slots a crash-safe rekey fallback applies: if the live
    KEK fails to unwrap but a STAGED KEK (``keyfile.new`` / staged keyring
    entry) does, the keystore was committed under a new KEK that a crash left
    un-promoted — this finishes the promotion (self-heal) and returns the DEK.
    """
    keystore = load_keystore()
    slots = _slots(keystore)

    if recovery_code is not None:
        candidates = [s for s in slots if s.get("type") == "recovery"]
    elif passphrase is not None:
        candidates = [s for s in slots if s.get("type") == "passphrase"]
    else:
        candidates = [s for s in slots if s.get("type") in ("keyring", "keyfile")]

    if not candidates:
        raise KeystoreError(
            "No key slot matches the credentials provided. Available slot "
            f"types: {sorted({s.get('type') for s in slots})}."
        )

    last_error: Optional[Exception] = None
    for slot in candidates:
        try:
            kek = _kek_for_slot(slot, passphrase=passphrase, recovery_code=recovery_code)
            try:
                wrapped_dek = slot["wrapped_dek"]
            except KeyError as exc:
                raise KeystoreError("keystore slot is malformed") from exc
            dek = _unwrap_dek(wrapped_dek, kek)
        except (DecryptionError, KeystoreError, DependencyError, ValueError) as exc:
            # Load-time fallback (closes the rekey commit→promote power-loss
            # window): if this is a keyfile/keyring slot and a STAGED KEK exists,
            # the keystore may have been committed under the new KEK while a
            # crash prevented promoting it to the live slot. Retry with the
            # staged KEK; on success, self-heal the promotion so the next load
            # is clean. The heal is best-effort — a read-only encryption dir
            # must not turn a recoverable unlock into a failure.
            staged_dek = _try_staged_unlock(slot, exc)
            if staged_dek is not None:
                dek = staged_dek
            else:
                last_error = exc
                continue
        cache_dek(dek)
        audit.log_event(audit.KEYSTORE_UNLOCKED, audit.SUCCESS, slot=slot.get("type"))
        return dek

    if isinstance(last_error, DependencyError):
        audit.log_event(
            audit.KEYSTORE_UNLOCK_FAILED, audit.FAILURE, reason="missing_dependency"
        )
        raise last_error
    if isinstance(last_error, KeystoreError) and "slot is malformed" in str(last_error):
        audit.log_event(
            audit.KEYSTORE_UNLOCK_FAILED, audit.FAILURE, reason="malformed_slot"
        )
        raise last_error
    audit.log_event(audit.KEYSTORE_UNLOCK_FAILED, audit.FAILURE, reason="wrong_key_or_corrupt")
    raise DecryptionError(
        "Could not unlock the keystore — wrong passphrase/recovery code, or "
        "the keystore is corrupt."
    )


def add_recovery_slot(argon2_params: Optional[Dict[str, Any]] = None) -> str:
    """Add a recovery slot wrapping the current DEK. Returns the recovery code.

    Requires the keystore to be unlocked (the DEK must be in memory).

    Recovery slots use :data:`kdf.RECOVERY_ARGON2_PARAMS` by default (fast KDF
    for a 160-bit random secret). *argon2_params* overrides that base only for
    tests or reading back legacy slots created before the fast default existed.
    """
    dek = get_cached_dek()
    if dek is None:
        raise LockedError("unlock the keystore before adding a recovery code")
    keystore = load_keystore()
    slots = _slots(keystore)
    slots = [s for s in slots if s.get("type") != "recovery"]
    code = generate_recovery_code()
    slots.append(
        _build_secret_slot(
            "recovery", "recovery", dek, normalize_recovery_code(code), argon2_params
        )
    )
    keystore["slots"] = slots
    save_keystore(keystore)
    audit.log_event(audit.RECOVERY_CODE_ADDED, audit.SUCCESS)
    return code


def rotate_primary(
    new_key_source: Optional[str] = None,
    *,
    new_passphrase: Optional[str] = None,
    argon2_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Re-wrap the existing DEK under a fresh primary slot.

    This is the cheap rotation: the DEK (and therefore every encrypted file)
    is untouched — only the slot that wraps it changes. ``new_key_source`` may
    differ from the current one (e.g. move from keyring to passphrase).
    Requires the keystore to be unlocked.

    Crash-safe for keyfile/keyring sources: a fresh KEK is STAGED (sidecar /
    ``:new`` keyring entry), the keystore is committed atomically, then the new
    KEK is PROMOTED to the live slot.

    Raise/return contract: this function RAISES only if the keystore was NOT
    committed (a pre-commit staging/save failure rolls the staged KEK back,
    leaving the OLD KEK + OLD keystore live and consistent). Once
    ``save_keystore`` returns, the function MUST NOT raise — the post-commit
    promote is best-effort (see :func:`_promote_committed_kek`): a failure there
    (e.g. os.replace OSError, transient keyring error) is logged/audited and the
    live-KEK install is deferred to the load-time self-heal in :func:`unlock`,
    which reads the still-present staged KEK and finishes the promotion. This is
    a rotate (the DEK is unchanged), so a deferred promotion is never data-loss,
    but the contract is kept uniform with :func:`replace_data_key`, where it is.

    Passphrase mode derives the KEK in-memory and persists nothing outside
    save_keystore, so it stays exactly as before (no staging needed).
    """
    dek = get_cached_dek()
    if dek is None:
        raise LockedError("unlock the keystore before rotating the key")
    keystore = load_keystore()
    source = new_key_source or keystore.get("key_source")
    if source not in VALID_KEY_SOURCES:
        raise ValueError(f"key_source must be one of {VALID_KEY_SOURCES}")

    # Entry self-heal: finish any deferred KEK promotion on the CURRENTLY-
    # committed primary slot before staging a new KEK, so the staged sidecar /
    # ":new" entry is guaranteed to be free scratch space (not the only key
    # that opens the committed keystore). No-op on the normal path. See
    # _reconcile_live_kek. This runs before any staging, so on its raise the
    # committed keystore + surviving KEK material are untouched.
    _reconcile_live_kek(keystore)

    if source == "keyring":
        new_slot, staged = _stage_keyring_slot("primary", dek)
    elif source == "keyfile":
        # Fresh KEK on every rekey, written to a sidecar first (never the live
        # keyfile) so a save failure cannot brick the DEK — see _stage_keyfile_slot.
        new_slot, staged = _stage_keyfile_slot("primary", dek)
    else:
        if not new_passphrase:
            raise ValueError("passphrase rotation requires a new passphrase")
        new_slot = _build_secret_slot(
            "primary", "passphrase", dek, new_passphrase.encode("utf-8"), argon2_params
        )
        staged = _noop_staged_kek()

    recovery = [s for s in _slots(keystore) if s.get("type") == "recovery"]
    keystore["key_source"] = source
    keystore["slots"] = [new_slot] + recovery
    keystore["rotated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        save_keystore(keystore)  # ATOMIC commit of the new wrapped_dek
    except Exception:
        # Pre-commit failure: old KEK + old keystore remain live and
        # self-consistent. Roll the staged KEK back and re-raise — callers
        # rely on "raised => keystore untouched".
        staged.rollback()
        raise
    # Keystore is durably committed; install the new live KEK best-effort. Per
    # the raise/return contract this MUST NOT raise after the commit: the
    # keystore references the new KEK and the new KEK still survives in the
    # staged sidecar/entry, so on promote failure the load-time self-heal in
    # unlock() finishes the promotion (do NOT roll back — that would brick).
    _promote_committed_kek(staged, key_source=source)
    audit.log_event(audit.KEYSTORE_ROTATED, audit.SUCCESS, key_source=source)


def replace_data_key(
    new_dek: bytes,
    *,
    passphrase: Optional[str] = None,
) -> int:
    """Replace the DEK with a fresh random key, re-wrapping the primary slot.

    Every envelope-encrypted artifact must already have been re-encrypted with
    *new_dek* before this is called. Recovery slots are dropped because they
    wrapped the old DEK and cannot be updated without each recovery code.

    Returns the number of recovery slots removed. Requires the keystore to be
    unlocked with the *old* DEK still cached.

    Crash-safe for keyfile/keyring sources: a fresh KEK is STAGED (sidecar /
    ``:new`` keyring entry), the keystore is committed atomically, then the new
    KEK is PROMOTED to the live slot.

    Raise/return contract (load-bearing for ``migrate.full_rekey``): this
    function RAISES only if the keystore was NOT committed under *new_dek*. A
    pre-commit staging/save failure rolls the staged KEK back, leaving the OLD
    KEK + OLD keystore live and consistent, and the cached DEK is still the OLD
    one — so ``migrate.full_rekey`` can safely restore the OLD-DEK artifacts. But
    once ``save_keystore`` returns, this function MUST NOT raise: the keystore
    now yields *new_dek*, so a raise would make migrate roll every artifact back
    to OLD-DEK ciphertext while the keystore yields the new DEK — bricking every
    artifact. Therefore the post-commit promote is best-effort (see
    :func:`_promote_committed_kek`): on promote failure the still-present staged
    KEK lets the load-time self-heal in :func:`unlock` finish the promotion, and
    we proceed to the normal success path (cache *new_dek*, audit success).

    Passphrase mode stays exactly as before (in-memory re-wrap).
    """
    if len(new_dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    if get_cached_dek() is None:
        raise LockedError("unlock the keystore before replacing the data key")

    keystore = load_keystore()
    source = keystore.get("key_source")
    if source not in VALID_KEY_SOURCES:
        raise ValueError(f"key_source must be one of {VALID_KEY_SOURCES}")

    primary = next((s for s in _slots(keystore) if s.get("type") != "recovery"), None)
    if primary is None:
        raise KeystoreError("keystore has no primary slot")

    # Entry self-heal: finish any deferred KEK promotion on the CURRENTLY-
    # committed primary slot before staging a new KEK, so the staged sidecar /
    # ":new" entry is guaranteed to be free scratch space (not the only key
    # that opens the committed keystore). No-op on the normal path. See
    # _reconcile_live_kek. This runs before any staging and before new_dek is
    # committed, so on its raise the OLD KEK + OLD keystore stay live and
    # consistent — migrate.full_rekey safely rolls artifacts back to old_dek.
    _reconcile_live_kek(keystore)

    if source == "keyring":
        new_primary, staged = _stage_keyring_slot("primary", new_dek)
    elif source == "keyfile":
        # Fresh KEK on every rekey, written to a sidecar first (never the live
        # keyfile) so a save failure cannot brick the DEK — see _stage_keyfile_slot.
        new_primary, staged = _stage_keyfile_slot("primary", new_dek)
    else:
        if not passphrase:
            raise ValueError("passphrase is required to re-wrap a passphrase slot")
        new_primary = _rewrap_secret_slot(primary, new_dek, passphrase.encode("utf-8"))
        staged = _noop_staged_kek()

    recovery_count = sum(1 for s in _slots(keystore) if s.get("type") == "recovery")
    keystore["slots"] = [new_primary]
    keystore["rotated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        save_keystore(keystore)  # ATOMIC commit of the new wrapped_dek
    except Exception:
        # Pre-commit failure: old KEK + old keystore remain live and
        # self-consistent; the cached DEK is still the OLD one, so the caller's
        # (migrate.full_rekey) rollback to OLD-DEK artifacts is correct. This is
        # the ONLY branch that may raise — "raised => keystore not committed".
        staged.rollback()
        raise
    # Keystore is committed under the NEW KEK + new_dek. Install the live KEK
    # best-effort: per the raise/return contract above this MUST NOT raise after
    # the commit (a raise would make migrate roll artifacts back to OLD-DEK while
    # the keystore yields new_dek => every artifact undecryptable). On promote
    # failure the new KEK survives in the staged source and unlock()'s
    # _try_staged_unlock self-heals on the next load — do NOT roll back.
    _promote_committed_kek(staged, key_source=source)
    # cache_dek so the in-memory DEK matches what a fresh load now recovers
    # (from the live slot, or — if promote was deferred — the staged self-heal).
    cache_dek(new_dek)
    audit.log_event(
        audit.DATA_KEY_REKEYED,
        audit.SUCCESS,
        key_source=source,
        recovery_slots_dropped=recovery_count,
    )
    return recovery_count


def _rewrap_secret_slot(
    slot: Dict[str, Any],
    new_dek: bytes,
    secret: bytes,
) -> Dict[str, Any]:
    """Re-wrap *new_dek* under the same KDF parameters as an existing secret slot."""
    salt = base64.b64decode(slot["salt"])
    kek = kdf.derive_kek(
        secret, salt, int(slot["kdf_id"]), slot.get("kdf_params") or None
    )
    updated = dict(slot)
    updated["wrapped_dek"] = _wrap_dek(new_dek, kek)
    return updated


def destroy_keystore() -> None:
    """Remove the keystore, keyfile, and keyring entry. Used by ``disable``.

    The DEK must already be available to callers that still need to decrypt
    data — call this only after every file has been decrypted back to plaintext.
    """
    username = _keyring_username()
    keystore = None
    try:
        keystore = load_keystore()
    except KeystoreError:
        pass
    if keystore is not None:
        for slot in _slots(keystore):
            if slot.get("type") == "keyring":
                _keyring_delete_kek(slot.get("keyring_id") or username)
    # Best-effort: drop any staged-rekey leftovers so 'disable' leaves nothing
    # behind (a crashed rekey may have left a ":new" keyring entry / keyfile.new).
    _keyring_delete_kek(_keyring_username_new())
    remaining: list[tuple[Path, OSError]] = []
    for path in (keystore_path(), keyfile_path(), keyfile_new_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            remaining.append((path, exc))
    if remaining:
        audit.log_event(
            audit.KEYSTORE_DESTROYED,
            audit.FAILURE,
            paths=[str(p) for p, _ in remaining],
        )
        names = ", ".join(str(p) for p, _ in remaining)
        raise KeystoreError(
            f"could not remove keystore file(s): {names}. "
            "Retry 'hermes encrypt disable' once the file is unlocked."
        )
    lock()
    audit.log_event(audit.KEYSTORE_DESTROYED, audit.SUCCESS)
