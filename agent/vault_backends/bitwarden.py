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
import stat
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_secret_env
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

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def _bw(self) -> Path:
        explicit = str(self.cfg.get("binary_path") or "")
        found = explicit or shutil.which("bw")
        if not found:
            raise RuntimeError("Bitwarden CLI (bw) not found — install it or set vault.bitwarden.binary_path")
        return Path(found)

    def _env(self, session_token: Optional[str]) -> Dict[str, str]:
        from hermes_constants import get_hermes_home
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        home = get_hermes_home()
        launch_home = Path(os.environ.get("HERMES_HOME") or home)
        default_home = (launch_home.parent.parent if launch_home.parent.name == "profiles"
                        else launch_home)
        # A multiplex gateway may be launched from a named profile while serving the default.
        # Never carry that launch account's CLI paths into the default profile.
        if home == default_home and launch_home != home:
            env["HOME"] = str(home.parent)
            env["XDG_CONFIG_HOME"] = str(home.parent / ".config")
            env.pop("BITWARDENCLI_APPDATA_DIR", None)
        elif home != default_home:
            env["HOME"] = str(home)
            env["XDG_CONFIG_HOME"] = str(home / ".config")
            env["BITWARDENCLI_APPDATA_DIR"] = str(home / "bitwarden-cli")
        env["NO_COLOR"] = "1"
        if session_token:
            env["BW_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return _unlock.is_unlocked(self.name)

    def unlock_unattended(self) -> bool:
        """Opt-in helper mints a session token without handing bootstrap credentials to the model."""
        helper = str(self.cfg.get("unattended_helper") or "")
        if not helper:
            return False
        path = Path(helper)
        if not path.is_absolute() or os.name != "posix" or not Path("/proc/self/fd").is_dir():
            raise RuntimeError("Bitwarden unattended helper requires an absolute path and procfs")
        from hermes_constants import get_hermes_home
        env = self._env(None)
        env["HERMES_HOME"] = str(get_hermes_home())
        generation = _unlock.begin_unlock(self.name)
        try:
            # Open every directory without following links, then execute the already-checked
            # inode through its inherited fd. A path swap cannot substitute a new executable.
            with ExitStack() as stack:
                fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
                stack.callback(os.close, fd)
                for part in path.parts[1:-1]:
                    directory = os.fstat(fd)
                    if (directory.st_uid not in (0, os.geteuid()) or
                        (directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and
                         not (directory.st_uid == 0 and directory.st_mode & stat.S_ISVTX))):
                        raise RuntimeError("Bitwarden unattended helper has an untrusted directory")
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    stack.callback(os.close, next_fd)
                    fd = next_fd
                directory = os.fstat(fd)
                if (directory.st_uid not in (0, os.geteuid()) or
                    (directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and
                     not (directory.st_uid == 0 and directory.st_mode & stat.S_ISVTX))):
                    raise RuntimeError("Bitwarden unattended helper has an untrusted directory")
                executable = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                stack.callback(os.close, executable)
                info = os.fstat(executable)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.geteuid()) or
                    info.st_mode & (stat.S_IWGRP | stat.S_IWOTH) or
                    not info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                    raise RuntimeError("Bitwarden unattended helper has unsafe ownership or permissions")
                proc = subprocess.run([f"/proc/self/fd/{executable}"], pass_fds=(executable,),
                                      env=env, stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Bitwarden unattended helper unavailable or timed out") from exc
        token = proc.stdout.strip()
        if proc.returncode != 0 or not token or "\n" in token or "\r" in token:
            raise RuntimeError("Bitwarden unattended helper failed; check account binding and bootstrap")
        if not _unlock.store_session_token(self.name, token, generation):
            raise RuntimeError("Bitwarden was locked while unlocking; try again")
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

    def list_items(self) -> List[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        raw = json.loads(self._run("list", "items") or "[]")
        out: List[VaultItemMeta] = []
        for item in raw if isinstance(raw, list) else []:
            if item.get("type") != 1 or not isinstance(item.get("login"), dict):
                continue
            login = item["login"]
            origins: List[str] = []
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
