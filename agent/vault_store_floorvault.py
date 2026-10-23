"""Opt-in FloorVault store for the local credential vault (``vault.store: floorvault``).

Drop-in replacement for :class:`agent.vault_store.VaultStore` backed by the
``floorvault`` package (the ``floorvault`` extra; PM installs it on first use):

- The master key lives in OS-backed custody (macOS Keychain, Windows DPAPI,
  Linux Secret Service), not in a ``vault.key`` file next to the ciphertext.
  Headless hosts supply a 64-hex key via ``FLOOR_VAULT_KEY`` (or
  ``VAULT_MASTER_KEY``); with neither, construction fails closed.
- Each field is sealed with AES-256-SIV whose associated data binds it to its
  app instance, table, record id and column, so a ciphertext copied onto
  another row fails verification instead of decrypting.
- Items live in an indexed SQLite file (``vault.db``); a lookup decrypts one
  row rather than the whole vault.

Constructing the store asks floorvault for hardened key memory; a deliberate
side effect of that hardening is ``RLIMIT_CORE=0`` — no core dumps for the rest
of the process lifetime, so a crash cannot write the master key to disk. The
cost lands outside the vault: Hermes crash diagnostics lose cores once the
store is first opened.

Handles keep the ``vault_`` prefix, so ``LocalLoginBackend`` and every caller
of ``get_vault_store()`` route here unchanged. Errors are re-raised as
:class:`agent.vault_store.VaultError` so existing ``except VaultError`` paths
keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlencode

from agent.vault_store import VAULT_KINDS, VaultError, VaultItemMeta, VaultStore

APP_INSTANCE_ID = "hermes-agent"
KEYRING_SERVICE = "hermes-agent-vault"

_STORES: dict[Path, FloorVaultStore] = {}
_STORES_LOCK = threading.Lock()


@contextmanager
def _as_vault_error() -> Iterator[None]:
    """Map floorvault failures onto Hermes's VaultError. Messages from floorvault never carry
    secret values (its own contract), but the decrypt failure is reworded to name the cause."""
    from floorvault import DecryptionVerificationError, FloorVaultError, KeyProviderError
    from floorvault.vaultkit.vault import VaultError as FloorVaultStoreError

    try:
        yield
    except DecryptionVerificationError as exc:
        raise VaultError("vault item failed authentication (tampered, moved, or wrong key)") from exc
    except KeyProviderError as exc:
        # str(exc) names the real cause (a malformed FLOOR_VAULT_KEY, a missing platform
        # custody extra); rewording it would point the user at the wrong fix.
        raise VaultError(f"FloorVault master key unavailable: {exc}") from exc
    except (FloorVaultStoreError, FloorVaultError) as exc:
        raise VaultError(str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise VaultError(f"vault write violated a constraint (duplicate id or missing field): {exc}") from exc
    except sqlite3.Error as exc:
        raise VaultError(f"vault database error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VaultError("vault item payload is corrupted (invalid JSON)") from exc
    except VaultError:
        raise
    # floorvault internals surface stdlib failures (a malformed OS-custody blob
    # raises ValueError/TypeError in the key container, fs failures raise
    # OSError); callers only handle VaultError, so every foreign failure must be
    # remapped, not passed through.
    except Exception as exc:
        raise VaultError(f"vault operation failed ({type(exc).__name__}): {exc}") from exc


def _require_floorvault() -> None:
    """PM installs the ``floorvault`` extra on first use (it is opt-in, not in ``[all]``)."""
    from pm import ensure_import, install_hint
    from pm.package import InstallError

    try:
        ensure_import("floorvault")
        import floorvault
    except (InstallError, ImportError) as exc:
        raise VaultError(f"the FloorVault vault store needs the floorvault package: {exc}. "
                         f"Run: {install_hint('floorvault')}") from exc


def _convert(meta: Any) -> VaultItemMeta:
    return VaultItemMeta(
        id=meta.id, kind=meta.kind, label=meta.label, origin=meta.origin, created_at=meta.created_at,
        identifier_type=meta.identifier_type, identifier=meta.identifier, has_otp=bool(meta.has_otp))


def _profile_identity(base_dir: Path) -> tuple[str, str]:
    """Return app-AAD and OS-custody identities bound to this vault directory."""
    canonical = str(Path(base_dir).expanduser().resolve())
    profile_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{APP_INSTANCE_ID}-{profile_id}", f"{KEYRING_SERVICE}-{profile_id}"


class FloorVaultStore:
    """Same public surface as ``VaultStore``: add_item, list_items, has_items, remove_item,
    get_meta, resolve_secret."""

    def __init__(self, base_dir: Path, *, crypto: Any = None):
        _require_floorvault()
        self._base = Path(base_dir).expanduser().resolve()
        with _as_vault_error():
            from floorvault import AdaptiveKeyProvider, FloorVault
            from floorvault.vaultkit.vault import VaultStore as _Store

            if crypto is None:
                app_instance_id, service_name = _profile_identity(self._base)
                key = AdaptiveKeyProvider(
                    service_name=service_name,
                    account_name="hermes-profile-v1",
                    fallback_dir=self._base,
                ).resolve_key()
                crypto = FloorVault(key, app_instance_id=app_instance_id)
            self._store = _Store(self._base, crypto=crypto)
        # sqlite3.connect creates vault.db under the process umask; tighten to the vault
        # convention (0600, like vault.key). The 0700 parent dir already shields it.
        try:
            os.chmod(self._base / "vault.db", 0o600)
        except OSError:
            pass

    def add_item(self, kind: str, label: str, secret: dict[str, Any], origin: Optional[str] = None,
                 *, item_id: Optional[str] = None) -> VaultItemMeta:
        # Gate on the Hermes kind set: floorvault also accepts "generic", which would create
        # an item the Fernet store could never write — a ghost if the user ever reverts.
        if kind not in VAULT_KINDS:
            raise VaultError(f"unknown vault kind {kind!r} (expected one of {VAULT_KINDS})")
        with _as_vault_error():
            return _convert(self._store.add_item(kind, label, dict(secret), origin, item_id=item_id))

    def list_items(self) -> list[VaultItemMeta]:
        with _as_vault_error():
            return [_convert(m) for m in self._store.list_items()]

    def has_items(self) -> bool:
        try:
            with _as_vault_error():
                return self._store.has_items()
        except Exception:  # health: allow BLE001 -- Fernet contract: any failure reads as "empty", never raises
            return False

    def remove_item(self, item_id: str) -> bool:
        with _as_vault_error():
            return self._store.remove_item(item_id)

    def get_meta(self, item_id: str) -> Optional[VaultItemMeta]:
        with _as_vault_error():
            meta = self._store.get_meta(item_id)
        return _convert(meta) if meta is not None else None

    def resolve_secret(self, item_id: str) -> dict[str, Any]:
        """Server-side only, same contract as ``VaultStore.resolve_secret``."""
        with _as_vault_error():
            return dict(self._store.resolve_secret(item_id))


def get_floorvault_store(base_dir: Path) -> FloorVaultStore:
    """One store per vault dir per process, so the OS key store is asked once, not on every fill."""
    base_dir = Path(base_dir)
    with _STORES_LOCK:
        store = _STORES.get(base_dir)
        if store is None:
            store = _STORES[base_dir] = FloorVaultStore(base_dir)
        return store


def _legacy_login_secret(rec: dict[str, Any]) -> dict[str, Any]:
    secret = dict(rec.get("secret") or {})
    if rec.get("kind") == "login":
        secret["identifier_type"] = rec.get("identifier_type")
        secret["identifier"] = rec.get("identifier")
        otp = str(secret.get("otp_secret") or "")
        if "|" in otp:
            # Stored Hermes seeds already include normalized non-default parameters. Feed
            # the dependency its supported URI input so it preserves that canonical value.
            parts = otp.split("|")
            if len(parts) != 4:
                raise VaultError("legacy authenticator parameters are malformed")
            seed, digits, period, algorithm = parts
            query = urlencode(dict(secret=seed, digits=digits, period=period, algorithm=algorithm))
            secret["otp_secret"] = f"otpauth://totp/Hermes?{query}"
    return secret


def migrate_from_fernet(legacy: VaultStore, target: FloorVaultStore, *,
                        on_verified: Optional[Callable[[], None]] = None) -> dict[str, Any]:
    """Copy every item of a Fernet vault into *target*, keeping each item's id so existing
    ``vault_…`` handles keep resolving. Idempotent: ids already in *target* are skipped. Every
    copied item is read back and compared before its authenticated retirement is recorded.
    Retired ids are skipped even if deleted from the target, so cleanup cannot resurrect them.
    ``on_verified`` runs under the legacy lock: the CLI flips config and optionally removes
    legacy files there. Fernet writers recheck that config under the same lock before writing.
    ``created_at`` is minted on insert and reports migration time."""
    with legacy._locked():
        records = legacy._read_all()
        with _as_vault_error():
            retired = target._store.list_legacy_retirements()
        present = {m.id for m in target.list_items()}
        copied = skipped = 0
        mismatched: list[str] = []
        for rec in records:
            item_id = str(rec.get("id"))
            if item_id in retired and item_id not in present:
                skipped += 1
                continue
            if item_id in present:
                skipped += 1
            else:
                target.add_item(str(rec.get("kind")), str(rec.get("label") or ""), _legacy_login_secret(rec),
                                rec.get("origin"), item_id=item_id)
                copied += 1
            if _matches(target, rec):
                with _as_vault_error():
                    target._store.retire_legacy_id(item_id, item_id)
            else:
                mismatched.append(item_id)
        verified = not mismatched
        if verified and on_verified is not None:
            try:
                on_verified()
            except VaultError:
                raise
            # Callback failures (config write, legacy removal) must reach the CLI
            # as VaultError, never a raw traceback.
            except Exception as exc:
                raise VaultError(f"post-verification step failed: {exc}") from exc
    return {"total": len(records), "copied": copied, "skipped": skipped,
            "verified": verified, "mismatched": mismatched}


def _matches(target: FloorVaultStore, rec: dict[str, Any]) -> bool:
    meta = target.get_meta(str(rec.get("id")))
    if meta is None or meta.kind != rec.get("kind") or meta.origin != rec.get("origin"):
        return False
    if meta.label != str(rec.get("label") or "").strip():
        return False
    if meta.identifier != (rec.get("identifier") or None):
        return False
    if meta.identifier_type != (rec.get("identifier_type") or None):
        return False
    return target.resolve_secret(meta.id) == {k: str(v) for k, v in (rec.get("secret") or {}).items()}
