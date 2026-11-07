"""Bitwarden Password Manager logins as a vault backend (``bw`` CLI).

This is the personal/org *password* vault (``bw``), distinct from the
Bitwarden Secrets Manager (``bws``) source that hydrates API keys at startup.
Unlock: ``bw unlock --raw --passwordenv VAR`` (the CLI rejects a piped password) mints a
``BW_SESSION`` token. List: ``bw list items`` filtered to type=1 (login) with
a URI. Resolve: ``bw get password <id>``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends import session_file as _session_file
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import (
    EXTERNAL_SESSION_REJECTED,
    EXTERNAL_SESSION_UNAVAILABLE,
    LOCKED_BY_USER,
    LoginBackend,
    UnlockRequired,
    run_with_secret_env,
)
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "BITWARDENCLI_APPDATA_DIR")


class BitwardenLoginBackend(LoginBackend):
    name = "bitwarden"
    display_name = "Bitwarden"
    prefix = "bw:"
    needs_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

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
        self.lock_reason = None
        if _unlock.is_unlocked(self.name):
            return True
        session_key, self.lock_reason = self._external_session()
        return session_key is not None

    def _external_session(self) -> tuple[Optional[str], Optional[str]]:
        """(key, None) or (None, public lock reason). Returned rather than stored, so callers sharing this
        instance cannot swap each other's reason between the check and the UnlockRequired."""
        path = str(self.cfg.get("session_file") or "")
        if not path:
            return None, None
        if _unlock.is_fenced(self.name):
            return None, LOCKED_BY_USER
        try:
            session = _session_file.read_session_file(path)
        except _session_file.SessionFileUnavailable as exc:
            _session_file.forget(path)
            logger.debug("Bitwarden %s: %s", EXTERNAL_SESSION_UNAVAILABLE, exc.detail)
            return None, EXTERNAL_SESSION_UNAVAILABLE
        reason = None if _session_file.is_verified(path, session.fingerprint) else self._verify_session(path, session)
        return (None, reason) if reason else (session.token, None)

    def _verify_session(self, path: str, session: _session_file.SessionFileRead) -> Optional[str]:
        """None when ``bw status`` confirms the key unlocks the vault, else the public lock reason."""
        try:
            proc = run_cli([str(self._bw()), "status", "--nointeraction"], env=self._env(session.token),
                           timeout=_TIMEOUT, label="bw", timeout_message="bw timed out", stdin=subprocess.DEVNULL)
        except (RuntimeError, ValueError):  # missing / unspawnable / timed out; ValueError: NUL in binary_path
            reason = EXTERNAL_SESSION_UNAVAILABLE
        else:
            try:
                status = json.loads(proc.stdout or "")
            except ValueError:
                status = None
            if proc.returncode == 0 and isinstance(status, dict) and status.get("status") == "unlocked":
                _session_file.mark_verified(path, session.fingerprint)
                return None
            reason = EXTERNAL_SESSION_REJECTED
        _session_file.forget(path)
        logger.debug("Bitwarden %s", reason)
        return reason

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

    def _run(self, *args: str) -> str:
        token = _unlock.get_session_token(self.name)
        interactive = token is not None
        reason = None
        if not interactive:
            token, reason = self._external_session()
        self.lock_reason = reason
        if not token:
            raise UnlockRequired(self, reason)
        proc = run_cli([str(self._bw()), *args, "--nointeraction"], env=self._env(token), timeout=_TIMEOUT,
                       label="bw", timeout_message="bw timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or "")
            if "locked" in err.lower() or "session" in err.lower():
                if interactive:
                    _unlock.invalidate_session_token(self.name, token)
                    raise UnlockRequired(self)
                _session_file.forget(str(self.cfg["session_file"]))
                self.lock_reason = EXTERNAL_SESSION_REJECTED
                raise UnlockRequired(self, EXTERNAL_SESSION_REJECTED)
            err = err.replace(token, "[REDACTED]")
            raise RuntimeError(f"bw failed: {err[:200]}")
        return proc.stdout or ""

    def list_items(self) -> list[VaultItemMeta]:
        if not self.is_unlocked():
            return []
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
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        return self._run("get", "password", handle[len(self.prefix):]).rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `bw get totp <id>` mints the current code from the item's TOTP seed; "No TOTP available" otherwise.
        try:
            code = self._run("get", "totp", handle[len(self.prefix):]).strip()
        except Exception:
            return None
        return code if code.isdigit() else None
