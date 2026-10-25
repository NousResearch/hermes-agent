"""Local encrypted vault for browser autofill secrets.

Profile-scoped, model-blind credential store. Metadata (kind, label, origin,
timestamps) lives alongside an encrypted secret payload; the payload is
encrypted at rest with a locally generated Fernet key. The model only ever
sees opaque handles + metadata — secret values are resolved server-side by
the browser fill path and never enter tool results, logs, or the session DB.

Design notes:
- Follows the repo's "default frictionless, 0600 files OK" policy: the key
  file and vault file are created 0600 under ``<HERMES_HOME>/vault/``.
  ``vault.key_storage: keychain`` (opt-in) keeps the key in the OS keychain
  instead (agent/vault_keychain.py); ``hermes vault key migrate`` moves it.
- Ported design (opaque-handle vault fill) from Merit-Systems/OpenInstinct
  (MIT): lib/manager/server/secret-store.ts + vault services.
- Three item kinds: ``login`` (password-only secret), ``payment`` (card fields) and
  ``address``; ``PAYMENT_FIELDS`` / ``ADDRESS_FIELDS`` are the canonical payload names.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from utils import atomic_write_bytes

logger = logging.getLogger(__name__)

VAULT_KINDS = ("login", "payment", "address")

LOGIN_IDENTIFIER_TYPES = ("email", "phone", "username")

# Canonical secret-payload fields per non-login kind. Each maps to the WHATWG autocomplete token the
# browser fill targets (agent/vault_login_classifier.py); the Desktop Add dialog and `hermes vault add`
# both write these names, so the fill never has to guess a user's ad-hoc field naming.
PAYMENT_FIELDS = {
    "card_number": "cc-number", "cardholder_name": "cc-name", "exp_month": "cc-exp-month",
    "exp_year": "cc-exp-year", "cvc": "cc-csc", "billing_postal_code": "postal-code",
}
ADDRESS_FIELDS = {
    "address_line1": "address-line1", "address_line2": "address-line2", "city": "address-level2",
    "state": "address-level1", "postal_code": "postal-code", "country": "country-name",
}
REQUIRED_FIELDS = {"payment": ("card_number", "exp_month", "exp_year", "cvc"),
                   "address": ("address_line1", "city", "postal_code", "country")}

_DEFAULT_PORTS = {"http": 80, "https": 443}

_LOCK = threading.Lock()

# fcntl is Unix-only; Windows locks a byte range with msvcrt (same shape as tools/skill_usage.py).
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    with suppress(ImportError):
        import msvcrt


def configured_key_storage() -> str:
    """``vault.key_storage``: where a NEW vault's key goes (``file`` default, or ``keychain``)."""
    from hermes_cli.config import load_config_readonly

    section = load_config_readonly().get("vault") or {}
    value = section.get("key_storage") if isinstance(section, dict) else None
    return str(value or "file").strip().lower()


class VaultError(Exception):
    """Vault failure that is safe to surface (never contains secret values)."""


_OTP_ALGOS = {"SHA1": "sha1", "SHA256": "sha256", "SHA512": "sha512"}


def normalize_otp_secret(value: str) -> str:
    """Accept a raw base32 seed or an ``otpauth://totp/...`` URI. Returns the canonical stored form:
    the bare uppercase base32 seed, followed by ``|digits|period|algo`` ONLY when the URI departs from
    the RFC 6238 defaults (6 / 30 / SHA1), so a plain seed stays a plain seed. Non-default parameters
    are honoured, not dropped: an 8-digit or 60-second authenticator would otherwise get wrong codes."""
    value = (value or "").strip()
    if not value:
        return ""
    digits, period, algo = 6, 30, "SHA1"
    if value.lower().startswith("otpauth://"):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(value)
        if parsed.netloc.lower() != "totp":
            raise VaultError("only otpauth://totp links are supported (counter-based HOTP is not)")
        qs = {k.lower(): v[0] for k, v in parse_qs(parsed.query).items()}
        value = qs.get("secret", "")
        try:
            digits = int(qs.get("digits", digits))
            period = int(qs.get("period", period))
        except ValueError:
            raise VaultError("otpauth:// digits/period must be integers")
        algo = qs.get("algorithm", algo).upper().replace("-", "")
        if digits not in (6, 7, 8) or period <= 0 or algo not in _OTP_ALGOS:
            raise VaultError("unsupported otpauth:// parameters (digits 6-8, period > 0, SHA1/SHA256/SHA512)")
    seed = re.sub(r"[\s-]", "", value).upper().rstrip("=")
    if not seed or re.search(r"[^A-Z2-7]", seed):
        raise VaultError("authenticator key must be a base32 secret or an otpauth:// URI")
    if (digits, period, algo) == (6, 30, "SHA1"):
        return seed
    return f"{seed}|{digits}|{period}|{algo}"


def totp_now(seed: str, *, digits: int = 6, period: int = 30, at: Optional[float] = None) -> str:
    """RFC 6238 TOTP for a stored seed (see normalize_otp_secret for the ``seed|digits|period|algo``
    form). Stdlib only."""
    import base64
    import hashlib
    import hmac
    import struct
    import time as _time

    algo = "sha1"
    if "|" in seed:
        seed, d, p, a = seed.split("|", 3)
        digits, period, algo = int(d), int(p), _OTP_ALGOS.get(a.upper(), "sha1")
    key = base64.b32decode(seed + "=" * (-len(seed) % 8), casefold=True)
    counter = int((at if at is not None else _time.time()) // period)
    digest = hmac.new(key, struct.pack(">Q", counter), getattr(hashlib, algo)).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def normalize_origin(url_or_origin: str) -> str:
    """Normalize a URL or origin to ``scheme://host[:port]``.

    Default ports (80 for http, 443 for https) are stripped so that
    ``https://example.com`` and ``https://example.com:443`` compare equal.
    Raises :class:`VaultError` for values without a scheme + host.
    """
    value = (url_or_origin or "").strip()
    if not value:
        raise VaultError("origin is required")
    if "://" not in value:
        raise VaultError(f"origin must include a scheme (got {value!r})")
    parts = urlsplit(value)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host:
        raise VaultError(f"could not parse origin from {value!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise VaultError(f"invalid port in origin {value!r}") from exc
    if port is None or port == _DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


@dataclass(frozen=True)
class VaultItemMeta:
    """Metadata-only view of a vault item. Never contains secret values.

    For ``kind='login'`` the identifier (email/username/phone) is metadata,
    not a secret: the agent may see it and type it itself. Only the password
    is vault-secret.
    """

    id: str
    kind: str
    label: str
    origin: Optional[str]
    created_at: str
    identifier_type: Optional[str] = None
    identifier: Optional[str] = None
    has_otp: bool = False  # a TOTP seed is stored: 2FA codes can be minted without asking the user
    # Every origin the password manager bound to this item (manager backends only;
    # ``origin`` is the first/primary one). Fill matching stays exact-origin against
    # this list — no wildcard or subdomain inference is ever derived from it.
    allowed_origins: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "origin": self.origin,
            "created_at": self.created_at,
        }
        if self.identifier is not None:
            out["identifier"] = self.identifier
            out["identifier_type"] = self.identifier_type
        if self.has_otp:
            out["has_otp"] = True
        if len(self.allowed_origins) > 1:
            out["allowed_origins"] = list(self.allowed_origins)
        return out


class VaultStore:
    """Encrypted, profile-scoped vault under ``<HERMES_HOME>/vault/``."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._base = Path(base_dir) if base_dir is not None else (
            Path(get_hermes_home()) / "vault"
        )
        self._vault_path = self._base / "vault.json.enc"
        self._key_path = self._base / "vault.key"
        # Keychain mode (vault.key_storage: keychain) keeps only this non-secret pointer beside the
        # ciphertext; agent/vault_keychain.py owns the entry it names.
        self._keyref_path = self._base / "vault.keyref"

    # -- key / crypto ------------------------------------------------------

    def _ensure_dir(self) -> None:
        make_user_dir(self._base, mode=0o700, parents=True, exist_ok=True)
        # Route through the canonical securer (honors managed/NixOS
        # group-share mode and HERMES_UID/GID ownership) rather than a
        # bespoke chmod — same requirement as the browser-profile snapshot
        # dir (f1d05c review).
        try:
            from hermes_cli.config import _secure_dir

            _secure_dir(self._base)
        except Exception:
            try:
                os.chmod(self._base, 0o700)
            except OSError:
                pass

    @contextmanager
    def _locked(self):
        """Serialize read-modify-write cycles across threads AND processes: the Desktop gateway, a CLI
        `hermes vault add` and a TUI slash worker all write the same ``vault.json.enc``; two unlocked
        writers would drop each other's items."""
        with _LOCK:
            self._ensure_dir()
            lock_path = self._base / ".vault.lock"
            if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
                lock_path.write_text(" ", encoding="utf-8")  # msvcrt needs a non-empty byte range to lock
            with open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8") as fd:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                elif msvcrt:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    with suppress(OSError):
                        if fcntl:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                        elif msvcrt:
                            fd.seek(0)
                            msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)

    def _fernet(self):
        from cryptography.fernet import Fernet

        return Fernet(self._key())

    def _has_ciphertext(self) -> bool:
        return self._vault_path.exists() and self._vault_path.stat().st_size > 0

    def _key(self) -> bytes:
        """``vault.key`` when present (the legacy layout wins), else the keychain entry ``vault.keyref``
        names, else a new key for a vault that has no items yet. Never mints a key over ciphertext:
        that would make the saved items unrecoverable for good."""
        from agent import vault_keychain as kc

        self._ensure_dir()
        if self._key_path.exists():
            with suppress(OSError):
                os.chmod(self._key_path, 0o600)
            return self._key_path.read_bytes().strip()
        ref = kc.read_ref(self._keyref_path)
        if ref is not None:
            try:
                ref = self._adopt(ref)
                key = kc.load_key(ref)
            except kc.KeychainUnavailable as exc:
                key, why = None, f"the OS keychain is not reachable here: {exc}"
            else:
                why = f"the keychain entry {ref.get('service')!r} / {ref['account']!r} is gone"
            if key:
                return key
        else:
            why = f"{self._key_path.name} is missing"
        if self._has_ciphertext():
            raise VaultError(self._missing_key_message(why))
        return self._create_key()

    def _missing_key_message(self, why: str) -> str:
        return (f"The vault key could not be found ({why}). vault.json.enc in {self._base} was left untouched. "
                "If this profile came from another machine, run `hermes vault key migrate --to file` there and "
                "copy its vault.key next to vault.json.enc; if the keychain entry was deleted, restore it from a "
                "keychain backup. `hermes vault key` shows where the key is expected. To start over instead, "
                "delete vault.json.enc (its saved items are lost).")

    def _conflicting_key_message(self) -> str:
        # Never overwrite or delete the other copy here: it may be the only key that opens the vault.
        return (f"{self._key_path} does not decrypt vault.json.enc, so the key was not moved and nothing "
                "was deleted. If the vault's key is in the OS keychain, move this vault.key aside and run "
                "`hermes vault key` again.")

    def _create_key(self) -> bytes:
        from cryptography.fernet import Fernet

        from agent import vault_keychain as kc

        key = Fernet.generate_key()
        if configured_key_storage() == "keychain":
            ref = kc.new_ref(self._base)
            try:
                kc.store_key(ref, key)
            except kc.KeychainUnavailable as exc:
                # Headless boxes (no Secret Service), and installs without the keychain extra, keep
                # working: the key goes to the 0600 file exactly as it always has.
                logger.warning("vault.key_storage is keychain, but %s; using %s instead", exc, self._key_path)
            else:
                kc.write_ref(self._keyref_path, ref)
                return key
        kc.remove_file(self._keyref_path)
        self._write_key_file(key)
        return key

    def _write_key_file(self, key: bytes) -> None:
        # Atomic + directory fsync: an interrupted write must never leave a partial vault.key, which
        # reads would prefer over a working keychain entry. Callers hold the vault lock.
        atomic_write_bytes(self._key_path, key, mode=0o600, fsync_dir=True)

    def _adopt(self, ref: dict) -> dict:
        """Give a copied/moved vault (profile clone or rename) its own keychain entry. Never deletes the
        entry it came from: another copy may still depend on it. Raises KeychainUnavailable only when the
        shared entry cannot be read; if the new entry cannot be written, the shared one keeps serving."""
        from agent import vault_keychain as kc

        if kc.owns(ref, self._base):
            return ref
        key = kc.load_key(ref)
        if not self._decrypts(key):
            return ref
        mine = kc.new_ref(self._base)
        try:
            kc.store_key(mine, key)
            kc.write_ref(self._keyref_path, mine)
        except (kc.KeychainUnavailable, OSError):
            kc.delete_key(mine)  # no orphan per retry; the shared entry keeps serving reads
            logger.warning("could not give the copied vault in %s its own keychain entry", self._base, exc_info=True)
            return ref
        return mine

    def _decrypts(self, key: Optional[bytes]) -> bool:
        from cryptography.fernet import Fernet, InvalidToken

        if not key:
            return False
        if not self._has_ciphertext():
            return True
        try:
            Fernet(key).decrypt(self._vault_path.read_bytes())
        except (InvalidToken, ValueError):
            return False
        return True

    # -- key storage ---------------------------------------------------------

    def key_location(self) -> dict[str, Any]:
        """Where this vault's key lives (never the key itself), for ``hermes vault key``."""
        from agent import vault_keychain as kc

        ref = kc.read_ref(self._keyref_path)
        if self._key_path.exists():
            where = {"storage": "file", "path": str(self._key_path)}
            if ref is not None:  # a keychain copy that could not be removed yet
                where["keychain_entry_left"] = ref["account"]
            return where
        return dict(ref) if ref else {"storage": "none", "path": str(self._key_path)}

    def migrate_key(self, target: str) -> str:
        """Move the key between ``vault.key`` and the OS keychain. The destination is written and proven
        to decrypt the vault before the source is removed, so an interrupted move leaves a working vault
        and re-running it finishes the job."""
        if target not in ("file", "keychain"):
            raise VaultError("key storage must be 'file' or 'keychain'")
        with self._locked():
            self._ensure_dir()
            return self._migrate_to_keychain() if target == "keychain" else self._migrate_to_file()

    def _migrate_to_keychain(self) -> str:
        from agent import vault_keychain as kc

        existing = kc.read_ref(self._keyref_path)
        if not self._key_path.exists():
            if existing is not None:  # already moved: prove the entry still opens the vault
                try:
                    key = kc.load_key(self._adopt(existing))
                except kc.KeychainUnavailable as exc:
                    raise VaultError(f"cannot read the vault key from the OS keychain: {exc}") from exc
                if not self._decrypts(key):
                    raise VaultError(self._missing_key_message(f"the keychain entry {existing['account']!r} does not open the vault"))
                return "keychain"
            if self._has_ciphertext():
                raise VaultError(self._missing_key_message(f"{self._key_path.name} is missing"))
            try:
                kc.os_keyring()  # nothing to move yet, but the new vault's key must have somewhere to go
            except kc.KeychainUnavailable as exc:
                raise VaultError(f"cannot use the OS keychain: {exc}") from exc
            return "keychain"
        key = self._key_path.read_bytes().strip()
        if not self._decrypts(key):
            raise VaultError(self._conflicting_key_message())
        # Resume into our own entry from an interrupted move; a copied pointer's entry is not ours to overwrite.
        ref = existing if existing is not None and kc.owns(existing, self._base) else kc.new_ref(self._base)
        try:
            kc.store_key(ref, key)
            if not self._decrypts(kc.load_key(ref)):
                raise kc.KeychainUnavailable("the key read back from the keychain does not decrypt the vault")
        except kc.KeychainUnavailable as exc:
            if ref is not existing:
                kc.delete_key(ref)
            raise VaultError(f"could not move the vault key to the OS keychain ({exc}); "
                             f"it stays in {self._key_path}") from exc
        kc.write_ref(self._keyref_path, ref)
        kc.remove_file(self._key_path)
        return "keychain"

    def _migrate_to_file(self) -> str:
        from agent import vault_keychain as kc

        ref = kc.read_ref(self._keyref_path)
        if self._key_path.exists():
            if ref is not None:  # a move stopped with both copies present; the file is the one in use
                if not self._decrypts(self._key_path.read_bytes().strip()):
                    raise VaultError(self._conflicting_key_message())
                self._release_ref(ref)
            return "file"
        if ref is None:
            if self._has_ciphertext():
                raise VaultError(self._missing_key_message(f"{self._key_path.name} is missing"))
            return "file"
        try:
            key = kc.load_key(ref)
        except kc.KeychainUnavailable as exc:
            raise VaultError(f"cannot read the vault key from the OS keychain: {exc}") from exc
        if not key or not self._decrypts(key):
            raise VaultError(self._missing_key_message(f"the keychain entry {ref['account']!r} does not hold this vault's key"))
        self._write_key_file(key)
        if self._key_path.read_bytes().strip() != key:
            kc.remove_file(self._key_path)
            raise VaultError(f"could not write {self._key_path}; the key stays in the OS keychain")
        self._release_ref(ref)
        return "file"

    def _release_ref(self, ref: dict) -> None:
        """The file holds the key now: drop the keychain copy. An entry this vault does not own (copied
        pointer) is left alone; if deleting ours fails, the pointer stays so a rerun can finish."""
        from agent import vault_keychain as kc

        if kc.owns(ref, self._base) and not kc.delete_key(ref):
            return
        kc.remove_file(self._keyref_path)

    # -- persistence -------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._vault_path.exists():
            return []
        blob = self._vault_path.read_bytes()
        if not blob:
            return []
        from cryptography.fernet import InvalidToken

        try:
            raw = self._fernet().decrypt(blob)
        except InvalidToken as exc:
            raise VaultError(
                "vault file could not be decrypted (key mismatch or corruption)"
            ) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VaultError(
                "vault file is corrupted (invalid JSON)"
            ) from exc
        if not isinstance(data, dict):
            raise VaultError("vault file is corrupted (unexpected shape)")
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    def _write_all(self, items: list[dict[str, Any]]) -> None:
        self._ensure_dir()
        payload = json.dumps({"version": 1, "items": items}).encode("utf-8")
        blob = self._fernet().encrypt(payload)
        # fsync_dir: the directory entry must be durable too (power loss between rename and next sync).
        atomic_write_bytes(self._vault_path, blob, mode=0o600, fsync_dir=True)

    # -- public API ----------------------------------------------------------

    def add_item(
        self,
        kind: str,
        label: str,
        secret: dict[str, Any],
        origin: Optional[str] = None,
    ) -> VaultItemMeta:
        """Add an item. ``secret`` is the sensitive payload (encrypted at rest).

        For ``kind='login'``, ``origin`` is required and the payload must
        contain ``identifier_type``, ``identifier`` and ``password``. The
        identifier fields are NOT secret — they are moved into item metadata
        (the agent may see and type the identifier itself); only
        ``password`` stays in the encrypted secret payload. ``payment`` and
        ``address`` payloads remain fully secret.
        """
        if kind not in VAULT_KINDS:
            raise VaultError(f"unknown vault kind {kind!r} (expected one of {VAULT_KINDS})")
        label = (label or "").strip()
        if not label:
            raise VaultError("label is required")
        norm_origin: Optional[str] = None
        identifier: Optional[str] = None
        identifier_type: Optional[str] = None
        secret = dict(secret)
        if kind == "login":
            if not origin:
                raise VaultError("origin is required for login items")
            norm_origin = normalize_origin(origin)
            id_type = secret.pop("identifier_type", None)
            if id_type not in LOGIN_IDENTIFIER_TYPES:
                raise VaultError(
                    f"identifier_type must be one of {LOGIN_IDENTIFIER_TYPES}"
                )
            identifier = str(secret.pop("identifier", "") or "").strip()
            if not identifier or not secret.get("password"):
                raise VaultError("login items require identifier and password")
            identifier_type = str(id_type)
            # Login secret payload is password (+ optional TOTP seed); identifier lives in
            # metadata and any stray origin echo is dropped.
            otp_secret = normalize_otp_secret(str(secret.get("otp_secret") or ""))
            secret = {"password": secret["password"], **({"otp_secret": otp_secret} if otp_secret else {})}
        else:
            allowed = PAYMENT_FIELDS if kind == "payment" else ADDRESS_FIELDS
            secret = {k: str(v) for k, v in secret.items() if k in allowed and str(v or "").strip()}
            missing = [f for f in REQUIRED_FIELDS[kind] if f not in secret]
            if missing:
                raise VaultError(f"{kind} items require {', '.join(missing)}")
            if origin:
                norm_origin = normalize_origin(origin)

        item_id = f"vault_{uuid.uuid4().hex[:12]}"
        record = {
            "id": item_id,
            "kind": kind,
            "label": label,
            "origin": norm_origin,
            "created_at": datetime.now(UTC).isoformat(),
            "identifier_type": identifier_type,
            "identifier": identifier,
            "secret": dict(secret),
        }
        with self._locked():
            items = self._read_all()
            items.append(record)
            self._write_all(items)
        return self._meta(record)

    def list_items(self) -> list[VaultItemMeta]:
        """Metadata-only listing. Secret payloads are never included."""
        with self._locked():
            return [self._meta(rec) for rec in self._read_all()]

    def has_items(self) -> bool:
        try:
            with self._locked():
                return bool(self._read_all())
        except Exception:
            return False

    def remove_item(self, item_id: str) -> bool:
        with self._locked():
            items = self._read_all()
            remaining = [rec for rec in items if rec.get("id") != item_id]
            if len(remaining) == len(items):
                return False
            self._write_all(remaining)
            return True

    def get_meta(self, item_id: str) -> Optional[VaultItemMeta]:
        with self._locked():
            for rec in self._read_all():
                if rec.get("id") == item_id:
                    return self._meta(rec)
        return None

    def resolve_secret(self, item_id: str) -> dict[str, Any]:
        """Resolve the decrypted secret payload for server-side use ONLY.

        Callers must never place the returned values into tool results,
        logs, exceptions, or any string that reaches the session DB.
        """
        with self._locked():
            for rec in self._read_all():
                if rec.get("id") == item_id:
                    return dict(rec.get("secret") or {})
        raise VaultError(f"no vault item with id {item_id!r}")

    @staticmethod
    def _meta(rec: dict[str, Any]) -> VaultItemMeta:
        identifier = rec.get("identifier")
        return VaultItemMeta(
            id=str(rec.get("id", "")),
            kind=str(rec.get("kind", "")),
            label=str(rec.get("label", "")),
            origin=rec.get("origin"),
            created_at=str(rec.get("created_at", "")),
            identifier_type=rec.get("identifier_type") if identifier else None,
            identifier=identifier or None,
            has_otp=bool((rec.get("secret") or {}).get("otp_secret")),
        )


def get_vault_store() -> VaultStore:
    """Default profile-scoped vault store."""
    return VaultStore()


def scrub_secret_from_text(text: str, secret: dict[str, Any]) -> str:
    """Defensively strip any secret values from a string (e.g. an exception
    message) before it can be surfaced. Case-sensitive exact substring scrub."""
    scrubbed = text
    for value in secret.values():
        if isinstance(value, str) and len(value) >= 3 and value in scrubbed:
            scrubbed = scrubbed.replace(value, "[REDACTED]")
    # Also collapse anything that looks like a leaked password-ish token in
    # common key=value echoes.
    scrubbed = re.sub(r"(password['\"]?\s*[:=]\s*)\S+", r"\1[REDACTED]", scrubbed)
    return scrubbed
