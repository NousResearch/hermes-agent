"""KeePass / KeePassXC (``.kdbx``) logins as a vault backend, read in-process with pykeepass.

Unlock: the master password typed into the surface's masked prompt opens the configured
database (``vault.keepass.database``, plus ``keyfile`` when set; a keyfile-only database opens
with no password). The decrypted database stays in process memory only, scoped to the profile,
and is dropped after ``relock_after_minutes`` without use, an explicit lock, or the owning
session ending. No child process is spawned, so the master password never reaches argv, a
child env, a log line or a file.

List: every entry outside the recycle bin with a non-empty password; an entry without a usable
URL is listed with ``origin=None`` and the fill refuses it. Resolve: the entry's password by UUID.

TOTP is not implemented yet: ``resolve_otp`` keeps the base default (None, the user is asked).
pykeepass is the optional ``keepass`` extra and is imported only when a database is opened.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultError, VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_DEFAULT_RELOCK_MINUTES = 15.0
# The shared unlock registry drops a manager session after 30 idle minutes, and that is the real
# ceiling for how long a decrypted database can stay resident without use. A larger configured
# value is clamped here instead of silently disappearing at the registry's TTL.
_REGISTRY_IDLE_TTL_MINUTES = 30.0

_lock = threading.RLock()  # reentrant: a registry lock listener runs while lock() holds it
# (profile home, database path, key file path) → (open PyKeePass, last used). Keyed by the file,
# not just the profile: pointing vault.keepass.database at another .kdbx must not keep serving the
# previous database's entries from the cache. Liveness is also registered with ``unlock``, so a
# lock / session release / idle expiry there retires the database here on its next access.
_open: dict[tuple[str, str, str], tuple[Any, float]] = {}
_timers: dict[tuple[str, str, str], threading.Timer] = {}
_timer_tokens: dict[tuple[str, str, str], object] = {}


def _on_registry_lock(profile_home: str, backend: str) -> None:
    """A Lock, a session end or a shutdown retired the token: drop decrypted databases now.

    Without this, a lock that is never followed by another backend call (the usual case when the
    source is disabled) would leave the decrypted file resident for the life of the process.
    """
    if backend != KeepassLoginBackend.name:
        return
    with _lock:
        for slot in [s for s in _open if s[0] == profile_home]:
            del _open[slot]
            timer = _timers.pop(slot, None)
            _timer_tokens.pop(slot, None)
            if timer is not None:
                timer.cancel()


def _home() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def _identifier_type(identifier: Optional[str]) -> Optional[str]:
    # Same inference as browser_vault_save_login, so a KeePass login reads like a locally saved one.
    if not identifier:
        return None
    return "email" if "@" in identifier else ("phone" if identifier.lstrip("+").isdigit() else "username")


class KeepassLoginBackend(LoginBackend):
    name = "keepass"
    display_name = "KeePass"
    prefix = "kp:"
    needs_unlock = True

    @property
    def allows_empty_master(self) -> bool:
        # Only a keyfile-only database opens with no master password. Advertising this for a
        # password-protected database would turn the surface's cancel (an empty answer) into a
        # spurious "wrong master password".
        return bool(str(self.cfg.get("keyfile") or "").strip())

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def _relock_after_s(self) -> float:
        minutes = self.cfg.get("relock_after_minutes")
        if minutes is None:
            return _DEFAULT_RELOCK_MINUTES * 60
        try:
            value = float(minutes)
        except (TypeError, ValueError):
            raise RuntimeError(f"vault.keepass.relock_after_minutes must be a number of minutes (got {minutes!r})") from None
        if value <= 0:
            raise RuntimeError(f"vault.keepass.relock_after_minutes must be greater than 0 (got {minutes!r})")
        return min(value, _REGISTRY_IDLE_TTL_MINUTES) * 60

    def _paths(self) -> tuple[Path, Optional[Path]]:
        database = str(self.cfg.get("database") or "").strip()
        if not database:
            raise RuntimeError("no KeePass database configured — set vault.keepass.database to your .kdbx file")
        keyfile = str(self.cfg.get("keyfile") or "").strip()
        return Path(os.path.expanduser(database)), (Path(os.path.expanduser(keyfile)) if keyfile else None)

    def _slot(self) -> tuple[str, str, str]:
        database, keyfile = self._paths()
        return (_home(), str(database), str(keyfile) if keyfile is not None else "")

    def _db(self, *, touch: bool = True) -> Optional[Any]:
        """The open database for this profile and file, or None when locked. Every public method
        goes through here, so the relock window is enforced on each call rather than by a timer."""
        live = _unlock.get_session_token(self.name) is not None if touch else _unlock.is_unlocked(self.name)
        slot = self._slot()
        with _lock:
            entry = _open.get(slot)
            if entry is None:
                return None
            # Read the knob only once a database is actually open: a typo in
            # vault.keepass.relock_after_minutes must not break list/fill for the other backends.
            try:
                relock_after_s = self._relock_after_s()
            except RuntimeError:
                if not touch:
                    # Status probes are called outside browser_vault_list's per-backend guard;
                    # never let a malformed knob hide every other backend's items.
                    return entry[0]
                raise
            kp, last_used = entry
            now = time.monotonic()
            expired = now - last_used > relock_after_s
            if expired or not live:
                del _open[slot]
                kp = None
                if expired:
                    # Retire the token under the same lock, so an unlock committing in this gap
                    # cannot be silently retired after reporting success.
                    _unlock.lock(self.name)
            elif touch:
                _open[slot] = (kp, now)
                self._schedule_relock(slot)
        return kp

    def is_unlocked(self) -> bool:
        return self._db(touch=False) is not None

    def unlock(self, master_password: str) -> None:
        try:
            from pykeepass import PyKeePass
            from pykeepass.exceptions import CredentialsError
        except ImportError:
            raise RuntimeError("KeePass support is not installed — install the Hermes `keepass` extra (pykeepass)") from None
        database, keyfile = self._paths()
        if not database.is_file():
            raise RuntimeError(f"KeePass database not found: {database} — check vault.keepass.database")
        if keyfile is not None and not keyfile.is_file():
            raise RuntimeError(f"KeePass key file not found: {keyfile} — check vault.keepass.keyfile")
        generation = _unlock.begin_unlock(self.name)
        try:
            kp = PyKeePass(str(database), password=master_password or None,
                           keyfile=str(keyfile) if keyfile is not None else None)
        except CredentialsError:
            raise RuntimeError("KeePass unlock failed: wrong master password or key file") from None
        # The unlock registry holds an opaque marker, not a credential: it carries the session
        # ownership, lock generation and idle expiry the other managers share.
        if not _unlock.store_session_token(self.name, secrets.token_hex(16), generation):
            raise RuntimeError("KeePass was locked while unlocking; try again")
        # Commit under the same lock the explicit lock() takes, and re-read the token inside it:
        # a lock that landed after store_session_token must not leave a decrypted database resident.
        with _lock:
            if _unlock.get_session_token(self.name) is None:
                raise RuntimeError("KeePass was locked while unlocking; try again")
            slot = self._slot()
            # A database switched in the config must not keep the old file decrypted alongside it.
            for other in [s for s in _open if s[0] == slot[0] and s != slot]:
                del _open[other]
            _open[slot] = (kp, time.monotonic())
            self._schedule_relock(slot)

    # A background timer makes the configured window real: without it the database would stay
    # resident until the next call, which may never come.
    def _schedule_relock(self, slot: tuple[str, str, str]) -> None:
        timer = _timers.pop(slot, None)
        if timer is not None:
            timer.cancel()
        try:
            delay = self._relock_after_s()
        except RuntimeError:
            # A malformed knob is reported by the next real access; unlocking itself must not fail
            # because of it, and without a delay there is simply no timer.
            return
        token = object()
        _timer_tokens[slot] = token
        timer = threading.Timer(delay, self._expire_now, args=(slot, token))
        timer.daemon = True
        _timers[slot] = timer
        timer.start()

    def _expire_now(self, slot: tuple[str, str, str], token: object) -> None:
        """Timer callback: drop the database only if it was not used or rescheduled since."""
        with _lock:
            if _timer_tokens.get(slot) is not token:
                return  # a touch rescheduled the window; this timer is stale
            entry = _open.get(slot)
            if entry is None:
                _timers.pop(slot, None)
                _timer_tokens.pop(slot, None)
                return
            _kp, last_used = entry
            try:
                window = self._relock_after_s()
            except RuntimeError:
                window = _DEFAULT_RELOCK_MINUTES * 60
            if time.monotonic() - last_used < window:
                # A touch refreshed the slot between the timer firing and this callback.
                self._schedule_relock(slot)
                return
            del _open[slot]
            _timers.pop(slot, None)
            _timer_tokens.pop(slot, None)
            # Target the slot's own profile: this thread carries no HERMES_HOME of its own.
            _unlock.lock_profile(slot[0], self.name)

    def lock(self) -> None:
        # Dropping the database and retiring the session token happen under one lock, so an
        # unlock in flight cannot commit a database after the lock has been acknowledged.
        with _lock:
            slot = self._slot()
            _open.pop(slot, None)
            timer = _timers.pop(slot, None)
            _timer_tokens.pop(slot, None)
            if timer is not None:
                timer.cancel()
            _unlock.lock(self.name)

    def _entries(self, kp: Any) -> list:
        recycle_bin = kp.recyclebin_group
        out = []
        for entry in kp.entries:
            if not entry.password:
                continue
            if recycle_bin is not None and _in_group(entry, recycle_bin):
                continue
            out.append(entry)
        return out

    def _meta(self, entry: Any) -> VaultItemMeta:
        origin: Optional[str] = None
        if entry.url:
            try:
                origin = normalize_origin(entry.url)
            except VaultError:
                origin = None  # e.g. a scheme-less "example.com": listed, but never a fill target
        username = (entry.username or "").strip() or None
        return VaultItemMeta(
            id=f"{self.prefix}{entry.uuid}", kind="login", label=entry.title or "",
            origin=origin, created_at=entry.ctime.isoformat() if entry.ctime else "",
            identifier_type=_identifier_type(username), identifier=username,
            allowed_origins=(origin,) if origin else ())

    def list_items(self) -> list[VaultItemMeta]:
        kp = self._db()
        if kp is None:
            return []
        # Fail loud: a database that cannot be read raises to the caller (browser_vault_list reports
        # it under ``errors``) instead of degrading to an empty or partial listing that reads as
        # "no saved logins". Only a per-entry URL that is not an origin is tolerated.
        return [self._meta(entry) for entry in self._entries(kp)]

    def _find(self, kp: Any, handle: str) -> Any:
        try:
            wanted = uuid.UUID(handle[len(self.prefix):]) if self.owns(handle) else None
        except ValueError:
            wanted = None
        entry = next((e for e in self._entries(kp) if e.uuid == wanted), None) if wanted else None
        if entry is None:
            raise RuntimeError("KeePass login not found; list logins again")
        return entry

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        kp = self._db()
        if kp is None:
            return None
        try:
            return self._meta(self._find(kp, handle))
        except RuntimeError:
            return None

    def resolve_password(self, handle: str) -> str:
        kp = self._db()
        if kp is None:
            raise UnlockRequired(self)
        return self._find(kp, handle).password


def _in_group(entry: Any, group: Any) -> bool:
    parent = entry.group
    while parent is not None:
        if parent.uuid == group.uuid:
            return True
        parent = parent.parentgroup
    return False


# Keep the registry's Lock / session end / shutdown honest for the decrypted database.
_unlock.register_lock_listener(_on_registry_lock)
