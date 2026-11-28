"""KeePass (KeePassXC) logins as a vault backend (``keepassxc-cli``).

Config (``vault.keepass`` in config.yaml):
  database: ~/path/to/file.kdbx      (required)
  key_file: ~/path/to/file.keyx      (optional)
  keychain_service: hermes-keepass   (optional: master password kept in the macOS Keychain,
  keychain_account: hermes            so the person never has to type it again)

Without a Keychain item the backend is locked until the surface prompts for the master
password (same as 1Password/Bitwarden). The master password is fed on stdin only, held in
process memory for the idle TTL, never written anywhere. Listing exports the database as
CSV in-process and keeps metadata only; a password is read one entry at a time at fill time.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultError, VaultItemMeta, normalize_origin

_TIMEOUT = 60.0
_APP_CLI = "/Applications/KeePassXC.app/Contents/MacOS/keepassxc-cli"


def find_cli(cfg: Optional[dict] = None) -> Optional[str]:
    explicit = str((cfg or {}).get("binary_path") or "")
    for c in (explicit, _APP_CLI, shutil.which("keepassxc-cli") or ""):
        if c and Path(c).is_file():
            return c
    return None


class KeePassLoginBackend(LoginBackend):
    name = "keepass"
    display_name = "KeePass"
    prefix = "kp:"
    needs_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self._cache: Optional[list[VaultItemMeta]] = None

    # -- paths / secrets -------------------------------------------------
    def _db(self) -> str:
        return os.path.expanduser(str(self.cfg.get("database") or ""))

    def _key_args(self) -> list[str]:
        k = os.path.expanduser(str(self.cfg.get("key_file") or ""))
        return ["-k", k] if k else []

    def _keychain_master(self) -> Optional[str]:
        svc = str(self.cfg.get("keychain_service") or "")
        if not svc:
            return None
        argv = ["security", "find-generic-password", "-s", svc, "-w"]
        acct = str(self.cfg.get("keychain_account") or "")
        if acct:
            argv[2:2] = ["-a", acct]
        try:  # macOS-only helper; a missing `security` or a slow Keychain just means "not stored"
            r = subprocess.run(argv, capture_output=True, timeout=15, check=False,
                           text=True, encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.rstrip("\n") if r.returncode == 0 and r.stdout else None

    def _master(self) -> str:
        m = _unlock.get_session_token(self.name)
        if m:
            return m
        m = self._keychain_master()
        if m:
            _unlock.store_session_token(self.name, m, _unlock.begin_unlock(self.name))
            return m
        raise UnlockRequired(self)

    def is_unlocked(self) -> bool:
        if _unlock.is_unlocked(self.name):
            return True
        try:
            self._master()
            return True
        except UnlockRequired:
            return False

    def unlock(self, master_password: str) -> None:
        generation = _unlock.begin_unlock(self.name)
        r = self._cli_raw(["db-info"], master_password)
        if r.returncode != 0:
            raise RuntimeError("KeePass unlock failed: wrong master password or key file")
        if not _unlock.store_session_token(self.name, master_password, generation):
            raise RuntimeError("KeePass was locked while unlocking; try again")

    # -- cli -------------------------------------------------------------
    def _cli_raw(self, args: list[str], master: str) -> subprocess.CompletedProcess:
        cli = find_cli(self.cfg)
        if not cli:
            raise RuntimeError("KeePassXC isn't installed")
        argv = [cli, args[0], "-q", *self._key_args(), self._db(), *args[1:]]
        try:
            return subprocess.run(argv, input=master + "\n", timeout=_TIMEOUT, check=False, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("keepassxc-cli timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to invoke keepassxc-cli: {exc}") from exc

    def _cli(self, *args: str) -> str:
        r = self._cli_raw(list(args), self._master())
        if r.returncode != 0:
            raise RuntimeError(f"keepassxc-cli {args[0]} failed: {(r.stderr or '').strip()[:200]}")
        return r.stdout or ""

    # -- listing ---------------------------------------------------------
    def list_items(self) -> list[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        if self._cache is not None:
            return self._cache
        raw = self._cli("export", "-f", "csv")
        out: list[VaultItemMeta] = []
        for row in csv.DictReader(io.StringIO(raw)):
            row.pop("Password", None)            # metadata only from here on
            row.pop("Notes", None)
            group = (row.get("Group") or "").split("/", 1)
            path = ((group[1] + "/") if len(group) > 1 and group[1] else "") + (row.get("Title") or "")
            if not path or path.startswith("Recycle Bin"):
                continue
            url = (row.get("URL") or "").strip()
            if url and "://" not in url:
                url = "https://" + url
            try:
                origin = normalize_origin(url) if url else None
            except VaultError:
                origin = None
            if not origin:
                continue
            user = (row.get("Username") or "").strip() or None
            out.append(VaultItemMeta(
                id=f"{self.prefix}{path}", kind="login", label=row.get("Title") or path, origin=origin,
                created_at=row.get("Created") or "", identifier_type="username" if user else None,
                identifier=user, has_otp=bool((row.get("TOTP") or "").strip()), allowed_origins=(origin,)))
            row.clear()
        del raw
        self._cache = out
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        return self._cli("show", "-s", "-a", "Password", handle[len(self.prefix):]).rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        try:
            code = self._cli("show", "-t", handle[len(self.prefix):]).strip()
        except (RuntimeError, UnlockRequired):
            return None
        return code if code.isdigit() else None
