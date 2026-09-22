"""Bitwarden Password Manager logins as a vault backend (``bw`` CLI).

This is the personal/org *password* vault (``bw``), distinct from the
Bitwarden Secrets Manager (``bws``) source that hydrates API keys at startup.
Unlock: ``bw unlock --raw --passwordenv VAR`` (the CLI rejects a piped password) mints a
``BW_SESSION`` token. List: ``bw list items`` filtered to type=1 (login) with
a URI. Resolve: ``bw get password <id>``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends import unlock as _unlock
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_secret_env
from agent.vault_store import VaultItemMeta, normalize_origin, scrub_secret_from_text

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
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["NO_COLOR"] = "1"
        if session_token:
            env["BW_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return _unlock.is_unlocked(self.name)

    def capabilities(self):
        return super().capabilities() | {"create_login", "update_login", "remove"}

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
            err = scrub_secret_from_text(err, {"master_password": master_password})
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

    def _run_encoded(self, payload: Dict, *args: str) -> str:
        """Send Bitwarden's required base64 JSON through stdin, never argv or a temporary file."""
        token = _unlock.get_session_token(self.name)
        if not token:
            raise UnlockRequired(self)
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        try:
            proc = subprocess.run(  # noqa: S603 — argv list, no shell
                [str(self._bw()), *args, "--nointeraction"], env=self._env(token), input=encoded,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("bw timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"failed to invoke bw: {exc}") from exc
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or "")
            login = payload.get("login")
            secrets = {"encoded_payload": encoded}
            if isinstance(login, dict):
                for key in ("password", "totp"):
                    value = login.get(key)
                    if isinstance(value, str):
                        secrets[key] = value
            err = scrub_secret_from_text(err, secrets)
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
            meta = self._meta_from_item(item)
            if meta is not None:
                out.append(meta)
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

    def create_login(self, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        item = json.loads(self._run("get", "template", "item") or "{}")
        login = json.loads(self._run("get", "template", "item.login") or "{}")
        item.update({"type": 1, "name": label})
        login.update({
            "username": identifier,
            "password": password,
            "uris": [{"match": None, "uri": origin}],
        })
        if otp_secret is not None:
            login["totp"] = otp_secret
        item["login"] = login
        created = json.loads(self._run_encoded(item, "create", "item") or "{}")
        meta = self._meta_from_item(created)
        if meta is None:
            raise RuntimeError("bw created a login but returned no usable item metadata")
        return VaultItemMeta(meta.id, meta.kind, meta.label, meta.origin, meta.created_at,
                             identifier_type, meta.identifier)

    def update_login(self, handle: str, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        item_id = handle[len(self.prefix):]
        item = json.loads(self._run("get", "item", item_id) or "{}")
        if item.get("type") != 1:
            raise RuntimeError("Bitwarden item is not a login")
        login = item.get("login")
        if not isinstance(login, dict):
            raise RuntimeError("Bitwarden login payload is invalid")
        item["name"] = item.get("name") or label
        login["username"] = identifier
        login["password"] = password
        if otp_secret is not None:
            login["totp"] = otp_secret
        uris = login.setdefault("uris", [])
        if not isinstance(uris, list):
            uris = login["uris"] = []
        if not any(isinstance(uri, dict) and uri.get("uri") == origin for uri in uris):
            uris.append({"match": None, "uri": origin})
        updated = json.loads(self._run_encoded(item, "edit", "item", item_id) or "{}")
        meta = self._meta_from_item(updated)
        if meta is None:
            raise RuntimeError("bw updated a login but returned no usable item metadata")
        return VaultItemMeta(meta.id, meta.kind, meta.label, meta.origin, meta.created_at,
                             identifier_type, meta.identifier)

    def remove_item(self, handle: str) -> bool:
        if not self.is_unlocked():
            raise UnlockRequired(self)
        if self.get_meta(handle) is None:
            return False
        self._run("delete", "item", handle[len(self.prefix):])
        return True

    def _meta_from_item(self, item: Dict) -> Optional[VaultItemMeta]:
        if item.get("type") != 1 or not isinstance(item.get("login"), dict):
            return None
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            return None
        login = item["login"]
        origins: List[str] = []
        for uri in login.get("uris") or []:
            if not isinstance(uri, dict) or uri.get("match") == 5:
                continue
            try:
                origin = normalize_origin(str(uri.get("uri") or ""))
            except Exception:
                continue
            if origin and origin not in origins:
                origins.append(origin)
        if not origins:
            return None
        origin = origins[0]
        web_origins = tuple(o for o in origins if o.startswith(("http://", "https://"))) or (origin,)
        username = str(login.get("username") or "").strip() or None
        return VaultItemMeta(
            id=f"{self.prefix}{item_id}", kind="login", label=str(item.get("name") or origin),
            origin=origin, created_at=str(item.get("creationDate") or ""),
            identifier_type="username" if username else None, identifier=username,
            allowed_origins=web_origins)
