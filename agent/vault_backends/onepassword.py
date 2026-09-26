"""1Password Login items as a vault backend (``op`` CLI).

Unlock: ``op signin --raw`` with the master password on stdin (desktop-app
integration or account-level auth) mints an ``OP_SESSION_<account>`` token.
A configured service-account token skips the prompt entirely (headless).
List: ``op item list --categories Login --format json`` → title, urls,
username. Resolve: ``op item get <id> --fields label=password --reveal``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import secrets
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli
from agent.secret_sources.onepassword import _OP_ENV_ALLOWLIST, _scrub, find_op
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_stdin_secret
from agent.vault_backends import unlock as _unlock
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_MAX_GENERATED_PENDING = 128


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True
    supports_generated_logins = True

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self._generated = {}
        self._generated_lock = threading.RLock()
        from agent.secret_scope import get_secret
        env_name = str(self.cfg.get("service_account_token_env") or "OP_SERVICE_ACCOUNT_TOKEN")
        self._service_token = get_secret(env_name, "") or ""

    # ── auth ────────────────────────────────────────────────────────────────

    def _op(self) -> Path:
        op = find_op(str(self.cfg.get("binary_path") or ""))
        if op is None:
            raise RuntimeError("1Password CLI (op) not found — install it or set vault.onepassword.binary_path")
        return op

    def _env(self, session_token: Optional[str]) -> Dict[str, str]:
        from agent.secret_scope import get_secret
        env = {k: os.environ[k] for k in _OP_ENV_ALLOWLIST if k in os.environ and not k.startswith("OP_CONNECT_")}
        # Connect credentials outrank OP_SERVICE_ACCOUNT_TOKEN inside op, so they must come from the
        # profile's own secret scope like the service token does — never from the launch environment.
        for k in ("OP_CONNECT_HOST", "OP_CONNECT_TOKEN"):
            if v := get_secret(k, ""):
                env[k] = v
        env["NO_COLOR"] = "1"
        account = str(self.cfg.get("account") or "")
        if account:
            env["OP_ACCOUNT"] = account
        if self._service_token:
            env["OP_SERVICE_ACCOUNT_TOKEN"] = self._service_token
        elif session_token:
            # op signin --raw prints the bare token; the env var name carries the account shorthand,
            # which op also accepts as plain OP_SESSION for the default account.
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return bool(self._service_token) or _unlock.is_unlocked(self.name)

    def unlock(self, master_password: str) -> None:
        """Mint a session token from the master password (consumed on stdin, never argv)."""
        generation = _unlock.begin_unlock(self.name)
        cmd = [str(self._op()), "signin", "--raw"]
        if account := str(self.cfg.get("account") or ""):
            cmd += ["--account", account]
        proc = run_with_stdin_secret(cmd, env=self._env(None), secret=master_password, timeout=_TIMEOUT, label="op")
        token = (proc.stdout or "").strip()
        if proc.returncode != 0 or not token:
            raise RuntimeError(f"1Password unlock failed: {_scrub(proc.stderr or '')[:200] or 'no session token'}")
        if not _unlock.store_session_token(self.name, token, generation):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _run(self, *args: str) -> str:
        token = None if self._service_token else _unlock.get_session_token(self.name)
        if not self._service_token and not token:
            raise UnlockRequired(self)
        try:
            proc = run_cli([str(self._op()), *args], env=self._env(token), timeout=_TIMEOUT, label="op",
                           timeout_message="op timed out", stdin=subprocess.DEVNULL)
        except Exception:
            # CLI failures/timeouts may retain secret-bearing output in their exception.
            raise RuntimeError("1Password command failed") from None
        if proc.returncode != 0:
            err = _scrub(proc.stderr or "")
            if "session" in err.lower() or "sign in" in err.lower() or "not signed in" in err.lower():
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            raise RuntimeError("1Password command failed")
        return proc.stdout or ""

    # ── backend contract ───────────────────────────────────────────────────
    def list_items(self) -> List[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        raw = json.loads(self._run("item", "list", "--categories", "Login", "--format", "json") or "[]")
        out: List[VaultItemMeta] = []
        for item in raw if isinstance(raw, list) else []:
            urls = [str(u["href"]) for u in item.get("urls") or [] if isinstance(u, dict) and u.get("href")]
            origins = _all_origins(urls)
            if not origins:
                continue
            username = str(item.get("additional_information") or "").strip() or None
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item.get('id')}", kind="login", label=str(item.get("title") or origins[0]),
                origin=origins[0], created_at=str(item.get("created_at") or ""),
                identifier_type="username" if username else None, identifier=username,
                allowed_origins=_web_origins(origins)))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def _service_vault_args(self) -> List[str]:
        """Scope item reads for 1Password service accounts.

        The CLI permits listing login metadata across the service account's vaults,
        but requires an explicit vault for `item get`. When there is precisely one
        accessible vault, discover it without reading any secret values.
        """
        if not self._service_token:
            return []
        raw = json.loads(self._run("vault", "list", "--format", "json") or "[]")
        vaults = raw if isinstance(raw, list) else []
        if len(vaults) != 1 or not vaults[0].get("id"):
            raise RuntimeError("1Password service account must expose exactly one vault for browser-login fills")
        return ["--vault", str(vaults[0]["id"])]

    def _run_generated(self, *args: str) -> str:
        # CLI errors may echo generated fields or include captured stdout in a timeout.
        try:
            return self._run(*args)
        except Exception:
            raise RuntimeError("1Password generated login operation failed") from None

    def create_generated_login(self, origin: str, label: str) -> str:
        from agent.vault_signup import validate_label, validate_origin
        validate_label(label)
        validate_origin(origin)
        title = "[Pending signup " + secrets.token_hex(12) + "] " + label
        with self._generated_lock:
            if len(self._generated) >= _MAX_GENERATED_PENDING:
                raise ValueError("Too many pending generated logins")
            try:
                scope = self._service_vault_args()
                raw = json.loads(self._run_generated("item", "create", "--category=login", "--title", title,
                                           "--url", origin, "--generate-password=letters,digits,symbols,32",
                                           "--format=json", *scope))
                item_id = raw.get("id")
                if not isinstance(item_id, str) or not re.fullmatch(r"[a-z0-9]{26}", item_id):
                    raise ValueError("Invalid item ID")
            except Exception:
                raise RuntimeError("1Password generated login creation failed") from None
            handle = self.prefix + item_id
            self._generated[handle] = (title, origin, scope)
            return handle

    def _finalize_generated(self, handle: str, label: Optional[str]) -> None:
        with self._generated_lock:
            if handle not in self._generated:
                raise ValueError("Unknown or nonpending generated login")
            title, origin, scope = self._generated[handle]
            try:
                # Metadata-only lookup also rejects an item renamed outside this process.
                meta = self.get_meta(handle)
                if meta is None or meta.label != title or meta.origin != origin:
                    raise ValueError("Generated login is no longer pending")
                if self._service_vault_args() != scope:
                    raise ValueError("Generated login vault changed")
                args = (["edit", handle[len(self.prefix):], "--title", label, "--format=json"]
                        if label is not None else ["delete", handle[len(self.prefix):]])
                self._run_generated("item", *args, *scope)
            except Exception:
                raise ValueError("Cannot finalize pending generated login") from None
            del self._generated[handle]

    def commit_generated_login(self, handle: str, label: str) -> None:
        from agent.vault_signup import validate_label
        self._finalize_generated(handle, validate_label(label))

    def discard_generated_login(self, handle: str) -> None:
        self._finalize_generated(handle, None)

    def resolve_password(self, handle: str) -> str:
        item_id = handle[len(self.prefix):]
        if handle not in self._generated:
            return self._run("item", "get", item_id, *self._service_vault_args(),
                             "--fields", "label=password", "--reveal").rstrip("\r\n")
        try:
            scope = self._service_vault_args()
            if scope != self._generated[handle][2]:
                raise ValueError("Generated login vault changed")
            return self._run_generated("item", "get", item_id, *scope,
                                       "--fields", "label=password", "--reveal").rstrip("\r\n")
        except Exception:
            raise RuntimeError("1Password password resolution failed") from None

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            code = self._run("item", "get", handle[len(self.prefix):], *self._service_vault_args(), "--otp").strip()
        except Exception:
            return None
        return code if code.isdigit() else None


def _web_origins(origins: List[str]) -> tuple:
    """Fill targets are browser pages, so app URIs (``androidapp://`` etc.) never
    widen the fill set; an item whose only URI is an app URI keeps its single
    (unfillable-from-a-page) origin exactly as before."""
    web = tuple(o for o in origins if o.startswith(("http://", "https://")))
    return web or (origins[0],)


def _all_origins(urls: List[str]) -> List[str]:
    """Every normalized origin saved on the item, deduped, order preserved.

    A 1Password Login item can carry several websites; each of them is a place the
    user told 1Password the credential belongs, so all of them are valid fill targets.
    """
    out: List[str] = []
    for u in urls:
        try:
            origin = normalize_origin(u)
        except Exception:
            continue
        if origin not in out:
            out.append(origin)
    return out
