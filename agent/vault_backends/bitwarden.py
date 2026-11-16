"""Bitwarden Password Manager logins as a vault backend (``bw`` CLI).

This is the personal/org *password* vault (``bw``), distinct from the
Bitwarden Secrets Manager (``bws``) source that hydrates API keys at startup.
Unlock: ``bw unlock --raw --passwordenv VAR`` (the CLI rejects a piped password) mints a
``BW_SESSION`` token. List: ``bw list items`` filtered to type=1 (login) with
a URI. Resolve: ``bw get password <id>``.

Freshness: ``bw`` reads a local cache that only ``bw sync`` refreshes, so an item
shared into the vault after ``bw login`` stays invisible. The backend syncs on
demand instead of on a timer: right after unlock, on a listing when the last
sync is older than ``vault.bitwarden.sync_on_list_seconds`` (default 60, 0 turns
on-demand sync off), and once when a handle is not found. A failed sync is
logged and the cached items are used.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_secret_env
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_DEFAULT_SYNC_ON_LIST_SECONDS = 60.0
_MISS_RESYNC_SECONDS = 5.0  # an unknown handle re-syncs unless the listing just did
# Backends are built fresh per tool call (enabled_backends()), so the sync clock lives at module
# level, keyed by the bw binary: one process shares one bw data directory per binary.
_SYNC_LOCK = threading.Lock()
_LAST_SYNC: dict[str, float] = {}
_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "BITWARDENCLI_APPDATA_DIR")


class BitwardenLoginBackend(LoginBackend):
    name = "bitwarden"
    display_name = "Bitwarden"
    prefix = "bw:"
    needs_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def _sync_on_list_seconds(self) -> float:
        raw = self.cfg.get("sync_on_list_seconds", _DEFAULT_SYNC_ON_LIST_SECONDS)
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return _DEFAULT_SYNC_ON_LIST_SECONDS

    def _sync(self, *, max_age: Optional[float] = None) -> bool:
        """Best-effort ``bw sync``. Skipped when a sync ran within ``max_age`` seconds.

        Never raises for a failed sync (offline server, timeout): the cached vault still
        works, so the failure is only logged. A locked vault still raises UnlockRequired
        through ``_run``.
        """
        key = str(self.cfg.get("binary_path") or shutil.which("bw") or "bw")
        with _SYNC_LOCK:
            last = _LAST_SYNC.get(key)
            if max_age is not None and last is not None and time.monotonic() - last < max_age:
                return False
            try:
                self._run("sync")
            except UnlockRequired:
                raise
            except Exception as exc:
                logger.warning("Bitwarden sync failed; using cached vault items (%s)", type(exc).__name__)
                return False
            _LAST_SYNC[key] = time.monotonic()
            return True

    def _bw(self) -> Path:
        explicit = str(self.cfg.get("binary_path") or "")
        found = explicit or shutil.which("bw")
        if not found:
            raise RuntimeError("Bitwarden CLI (bw) not found — install it or set vault.bitwarden.binary_path")
        return Path(found)

    def _env(self, session_token: Optional[str]) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["NO_COLOR"] = "1"
        if session_token:
            env["BW_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return _unlock.is_unlocked(self.name)

    def unlock(self, master_password: str) -> None:
        # bw refuses a piped password ("Master password is required"); its non-interactive contract is
        # --passwordenv: the variable exists only in the child's environment, never in argv or ours.
        generation = _unlock.begin_unlock(self.name)
        proc = run_with_secret_env([str(self._bw()), "unlock", "--raw", "--nointeraction", "--passwordenv", "HERMES_BW_MASTER"],
                                   env=self._env(None), secret_env="HERMES_BW_MASTER", secret=master_password,
                                   timeout=_TIMEOUT, label="bw")
        token = (proc.stdout or "").strip()
        if proc.returncode != 0 or not token:
            err = scrub_ansi(proc.stderr or "").strip()[:200]
            if "not logged in" in err.lower():
                err = "not logged in — run `bw login` once in a terminal first"
            raise RuntimeError(f"Bitwarden unlock failed: {err or 'no session key'}")
        if not _unlock.store_session_token(self.name, token, generation):
            raise RuntimeError("Bitwarden was locked while unlocking; try again")
        if self._sync_on_list_seconds() <= 0:
            return
        try:
            self._sync()
        except UnlockRequired:
            pass  # locked again straight away; the next list reports it

    def _run(self, *args: str) -> str:
        token = _unlock.get_session_token(self.name)
        if not token:
            raise UnlockRequired(self)
        proc = run_cli([str(self._bw()), *args, "--nointeraction"], env=self._env(token), timeout=_TIMEOUT,
                       label="bw", timeout_message="bw timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or "")
            if "locked" in err.lower() or "session" in err.lower():
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            raise RuntimeError(f"bw failed: {err[:200]}")
        return proc.stdout or ""

    def list_items(self) -> list[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        interval = self._sync_on_list_seconds()
        if interval > 0:
            self._sync(max_age=interval)
        return self._list_cached()

    def _list_cached(self) -> list[VaultItemMeta]:
        raw = json.loads(self._run("list", "items") or "[]")
        out: list[VaultItemMeta] = []
        for item in raw if isinstance(raw, list) else []:
            if item.get("type") != 1 or not isinstance(item.get("login"), dict):
                continue
            login = item["login"]
            origins: list[str] = []
            for uri in login.get("uris") or []:
                if uri.get("match") == 5:  # Bitwarden URI match "Never": not a fill target
                    continue
                try:
                    origin = normalize_origin(str(uri.get("uri") or ""))
                except Exception:
                    continue
                if origin and origin not in origins:
                    origins.append(origin)
            if not origins:
                continue
            username = str(login.get("username") or "").strip() or None
            # Fill targets are browser pages, so app URIs (androidapp:// etc.) never widen
            # the fill set; an app-URI-only item keeps its single origin exactly as before.
            web_origins = tuple(o for o in origins if o.startswith(("http://", "https://"))) or (origins[0],)
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item.get('id')}", kind="login", label=str(item.get("name") or origins[0]),
                origin=origins[0], created_at=str(item.get("creationDate") or ""),
                identifier_type="username" if username else None, identifier=username,
                allowed_origins=web_origins))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        meta = next((m for m in self.list_items() if m.id == handle), None)
        if meta is None and self.is_unlocked() and self._sync_on_list_seconds() > 0 \
                and self._sync(max_age=_MISS_RESYNC_SECONDS):
            meta = next((m for m in self._list_cached() if m.id == handle), None)
        return meta

    def resolve_password(self, handle: str) -> str:
        return self._run("get", "password", handle[len(self.prefix):]).rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `bw get totp <id>` mints the current code from the item's TOTP seed; "No TOTP available" otherwise.
        try:
            code = self._run("get", "totp", handle[len(self.prefix):]).strip()
        except Exception:
            return None
        return code if code.isdigit() else None
