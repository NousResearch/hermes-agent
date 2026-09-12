"""KeePass / KeePassXC logins as a vault backend (``keepassxc-cli``).

Unlock: KeePassXC mints no session token, so the database key IS this backend's
session credential. ``unlock`` proves it with one ``db-info`` open and keeps it
in the same in-process, profile-scoped store ``op``/``bw`` keep their tokens in
(``unlock.py``: idle TTL, explicit lock, per-session release), and every call
hands it to the CLI on stdin — never argv, never the environment, never disk. A
database opened by ``password_file`` / ``password_env`` (or a key file alone)
needs no prompt at all.

List: one ``export --format csv``; the Password column is dropped as rows are
parsed, so listing never materializes credentials. Resolve: ``show -a Password
-- <db> <entry>`` for the one entry being filled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.keepass import (
    DEFAULT_PASSWORD_ENV,
    export_entries,
    resolve_password,
    show_attribute,
    verify_database,
)
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)


class KeePassLoginBackend(LoginBackend):
    name = "keepass"
    display_name = "KeePassXC"
    prefix = "keepass:"
    needs_unlock = True

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    # ── config ─────────────────────────────────────────────────────────────

    def _db(self) -> str:
        raw = str(self.cfg.get("db") or "").strip()
        if not raw:
            raise RuntimeError("KeePass database is not configured — set vault.keepass.db to the .kdbx path")
        path = Path(raw).expanduser()
        if not path.is_file():
            raise RuntimeError(f"KeePass database {path} was not found — check vault.keepass.db")
        return str(path)

    def _keyfile(self) -> str:
        return str(self.cfg.get("keyfile") or "").strip()

    def _binary_path(self) -> str:
        return str(self.cfg.get("binary_path") or "")

    def _static_key(self) -> Optional[str]:
        """The key the user configured out of band (``password_file``, or the ``password_env``
        secret read through the profile's scope — never the raw process env)."""
        from agent.secret_scope import get_secret

        password_env = str(self.cfg.get("password_env") or "").strip() or DEFAULT_PASSWORD_ENV
        try:
            value = str(get_secret(password_env, "") or "")
        except Exception:  # noqa: BLE001 — unscoped multiplex read: fall through to the prompt
            value = ""
        return resolve_password(self.cfg, password_env, env={password_env: value})

    def _key(self) -> Optional[str]:
        """Key for one CLI call: the configured one, else the session key from the unlock prompt.
        Raises ``UnlockRequired`` when neither exists (the surface then prompts)."""
        static = self._static_key()
        if static is not None:
            return static
        session = _unlock.get_session_token(self.name)
        if session is None:
            raise UnlockRequired(self)
        return session

    def is_unlocked(self) -> bool:
        if _unlock.is_unlocked(self.name):
            return True
        try:
            # A configured-but-unreadable password_file still counts as "not locked": that is a
            # configuration error the read call reports, not a reason to ask for a password.
            return self._static_key() is not None
        except RuntimeError:
            return True

    # ── auth ───────────────────────────────────────────────────────────────

    def unlock(self, master_password: str) -> None:
        generation = _unlock.begin_unlock(self.name)
        verify_database(db=self._db(), key=master_password, keyfile=self._keyfile(),
                        binary_path=self._binary_path())
        if not _unlock.store_session_token(self.name, master_password, generation):
            raise RuntimeError("KeePassXC was locked while unlocking; try again")

    # ── backend contract ───────────────────────────────────────────────────

    def _entries(self):
        return export_entries(db=self._db(), key=self._key(), keyfile=self._keyfile(),
                              binary_path=self._binary_path())

    def list_items(self) -> List[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        out: List[VaultItemMeta] = []
        for entry in self._entries():
            origin = _first_origin(entry.url)
            if not origin:
                continue  # a login without a URL has nothing to bind a fill to
            username = entry.username or None
            out.append(VaultItemMeta(
                id=f"{self.prefix}{entry.path}", kind="login", label=entry.title, origin=origin,
                created_at=entry.created_at, identifier_type="username" if username else None,
                identifier=username))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        return show_attribute(db=self._db(), entry=handle[len(self.prefix):], key=self._key(),
                              keyfile=self._keyfile(), binary_path=self._binary_path())


def _first_origin(urls: str) -> Optional[str]:
    """First usable origin from an entry's URL field (KeePassXC stores one URL per line)."""
    for line in (urls or "").splitlines():
        try:
            return normalize_origin(line)
        except Exception:  # noqa: BLE001 — a junk URL just disqualifies the entry for filling
            continue
    return None


def keepass_installed() -> bool:
    """Detection helper for ``hermes vault sources`` (same shape as the other managers)."""
    return find_keepassxc() is not None
