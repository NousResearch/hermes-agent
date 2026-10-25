"""Bitwarden Password Manager logins as a vault backend (``bw`` CLI).

This is the personal/org *password* vault (``bw``), distinct from the
Bitwarden Secrets Manager (``bws``) source that hydrates API keys at startup.

Unlock model: the ``bw`` CLI is zero-knowledge and per-process — ``bw unlock``
mints a session key that does NOT persist to its state file (verified against
vaultTimeout=never: fresh processes still report ``locked``). In-process, the
key is held in the profile-scoped unlock table (memory only, 30-min idle TTL).

Opt-in persistence (``vault.bitwarden.persist_session_key: true``): the session
key is a stable local decryption token, valid until the master password
changes. Stored 0600 at ``<hermes_home>/vault/bw_session.key`` it lets every
fresh Hermes process (bridge workers, cron, new sessions) auto-unlock without
prompting — matching the browser extension's "never ask again" posture. The
masked-prompt path re-mints and rewrites the file, so a master-password change
self-heals on next unlock; a rejected key is deleted so the next use re-prompts.
Default is off, keeping the upstream memory-only posture.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_secret_env
from agent.vault_store import VaultItemMeta, normalize_origin
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "BITWARDENCLI_APPDATA_DIR")

_PERSISTED_KEY_NAME = "bw_session.key"


def _persisted_key_path() -> Path:
    return Path(get_hermes_home()) / "vault" / _PERSISTED_KEY_NAME


class BitwardenLoginBackend(LoginBackend):
    name = "bitwarden"
    display_name = "Bitwarden"
    prefix = "bw:"
    needs_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        # Opt-in (default off = upstream posture: token in memory only). With
        # ``persist_session_key: true`` the session key is stored 0600 at
        # <hermes_home>/vault/bw_session.key so fresh processes (bridge workers,
        # cron, new sessions) auto-unlock without prompting — the user's
        # browser-extension "never ask again" behaviour.
        self._persist = bool(self.cfg.get("persist_session_key"))
        self._persisted_key: Optional[str] = None
        if self._persist:
            path = _persisted_key_path()
            if path.is_file():
                try:
                    key = path.read_text().strip()
                    if key:
                        # 0600 gate: a world-readable key file is a security incident, refuse it.
                        mode = stat.S_IMODE(path.stat().st_mode)
                        if mode & 0o077:
                            logger.warning("bitwarden: %s is group/other-readable (%o); ignoring it", path, mode)
                        else:
                            self._persisted_key = key
                except OSError as exc:
                    logger.warning("bitwarden: cannot read persisted session key: %s", exc)

    def _bw(self) -> Path:
        explicit = str(self.cfg.get("binary_path") or "")
        found = explicit or shutil.which("bw")
        if not found:
            raise RuntimeError("Bitwarden CLI (bw) not found — install it or set vault.bitwarden.binary_path")
        return Path(found)

    def _token(self) -> Optional[str]:
        return _unlock.get_session_token(self.name) or self._persisted_key

    def _env(self, session_token: Optional[str]) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["NO_COLOR"] = "1"
        if session_token:
            env["BW_SESSION"] = session_token
        return env

    def _forget_persisted_key(self) -> None:
        """Session key rejected by bw (master password changed / revoked): drop both
        in-memory and on-disk copies so the next use re-prompts instead of looping."""
        self._persisted_key = None
        if not self._persist:
            return
        path = _persisted_key_path()
        try:
            if path.is_file():
                path.unlink()
                logger.info("bitwarden: removed stale persisted session key")
        except OSError:
            pass

    def is_unlocked(self) -> bool:
        if _unlock.is_unlocked(self.name):
            return True
        if not self._persist or self._persisted_key is None:
            return False
        # Re-check the file against the CURRENT hermes home: a profile switch must not
        # let another profile's key keep this backend unlocked.
        if not _persisted_key_path().is_file():
            self._persisted_key = None
            return False
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
        # Persist so bridge workers / cron / future processes skip the prompt entirely.
        if not self._persist:
            return
        try:
            path = _persisted_key_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(token + "\n")
            self._persisted_key = token
        except OSError as exc:
            logger.warning("bitwarden: in-process unlock OK but persist failed (%s) — will re-prompt after restart", exc)

    def _run(self, *args: str) -> str:
        proc = run_cli([str(self._bw()), *args, "--nointeraction"], env=self._env(self._token()), timeout=_TIMEOUT,
                       label="bw", timeout_message="bw timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or "")
            if "locked" in err.lower() or "session" in err.lower():
                _unlock.lock(self.name)
                self._forget_persisted_key()
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
