"""Bitwarden Password Manager logins as a vault backend (``bw`` CLI).

This is the personal/org *password* vault (``bw``), distinct from the
Bitwarden Secrets Manager (``bws``) source that hydrates API keys at startup.
Unlock: ``bw unlock --raw --passwordenv VAR`` (the CLI rejects a piped password) mints a
``BW_SESSION`` token. When the surface has a human, a bare ``bw unlock --raw`` is tried first, so the
Bitwarden desktop app can approve with Touch ID / Windows Hello; the master-password prompt stays the
fallback. List: ``bw list items`` filtered to type=1 (login) with
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
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import (LoginBackend, UnlockRequired, run_for_secretless_unlock,
                                       run_with_secret_env)
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
# A desktop approval (Touch ID / Windows Hello) answered in time, or the CLI is already falling back
# to its own password prompt, which we cannot answer. Bounds the extra latency of the attempt.
_DESKTOP_TIMEOUT = 10.0
_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "BITWARDENCLI_APPDATA_DIR",
             # Documented override for a desktop app installed outside the standard location.
             "BITWARDEN_DESKTOP_PROXY_PATH")


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
        return _unlock.is_unlocked(self.name)

    def try_secretless_unlock(self) -> bool:
        """Let the running Bitwarden desktop app approve the unlock instead of typing the master
        password: a bare ``bw unlock --raw`` (no password, no ``--nointeraction``) makes the CLI try
        its desktop biometric path first, which is Touch ID / Windows Hello (Desktop 2026.9.0+, with
        biometric unlock enabled for this account, and the server flag ``biometrics-sdk-ipc`` on).

        Returns False whenever the CLI cannot approve it — an older CLI answering "Master password is
        required", a denied or timed-out prompt, no desktop app — and the caller then prompts for the
        master password exactly as before. Nothing is written unless a session key comes back."""
        if self.cfg.get("desktop_biometric") is False or not _unlock.can_prompt_here():
            # The interactive gate is enforced here as well as in the callers: this method is public on
            # a backend, and a direct call must never raise a desktop approval prompt where nobody can
            # answer it (cron, webhooks, api_server, single-query).
            return False
        generation = _unlock.begin_unlock(self.name)
        proc = run_for_secretless_unlock([str(self._bw()), "unlock", "--raw"], env=self._env(None),
                                         timeout=_DESKTOP_TIMEOUT, label="bw")
        token = (proc.stdout or "").strip() if proc is not None and proc.returncode == 0 else ""
        if not token:
            return False
        if not _unlock.store_session_token(self.name, token, generation):
            return False
        logger.info("Bitwarden unlocked with the desktop app's approval (no master password typed)")
        return True

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
