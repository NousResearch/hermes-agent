"""1Password Login items as a vault backend (``op`` CLI).

Unlock: ``op signin`` (no ``--raw``) so 1Password.app can do Touch ID, then
``op account get`` to confirm (``op whoami`` lies under app integration).
Hermes never collects the master password. A service-account token skips unlock.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli
from agent.secret_sources.onepassword import _OP_ENV_ALLOWLIST, _scrub, find_op
from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_backends import unlock as _unlock
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
# In-memory marker: this process may call `op` with desktop-app integration (no OP_SESSION).
# Never passed to the child as an env value.
_APP_SESSION = "__hermes_op_app__"
_UNLOCK_HINT = (
    "Unlock 1Password with Touch ID (Settings → Security), and turn on "
    "Settings → Developer → Integrate with 1Password CLI. Hermes does not take your master password."
)


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True
    app_unlock = True

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
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
        elif session_token and session_token != _APP_SESSION:
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return bool(self._service_token) or _unlock.is_unlocked(self.name)

    def unlock(self, master_password: str = "") -> None:
        """Unlock via the 1Password app (Touch ID). Never collect the master password."""
        _ = master_password  # old Settings UI may still send one; never hand it to `op`
        generation = _unlock.begin_unlock(self.name)
        op = str(self._op())
        env = self._env(None)
        # Documented app-integration path: `op signin` with no --raw. 1Password.app
        # does Touch ID. stdin is /dev/null so a classic password prompt cannot run.
        try:
            proc = run_cli(
                [op, "signin"], env=env, timeout=_TIMEOUT, label="op",
                timeout_message="op timed out waiting for 1Password Touch ID", stdin=subprocess.DEVNULL)
        except RuntimeError as exc:
            raise RuntimeError(_UNLOCK_HINT) from exc
        # `op whoami` reports "not signed in" even when app-integration sessions work
        # (`op account get` / item list succeed). Do not use whoami as the probe.
        probe = run_cli(
            [op, "account", "get"], env=env, timeout=_TIMEOUT, label="op",
            timeout_message="op timed out waiting for 1Password", stdin=subprocess.DEVNULL)
        if probe.returncode != 0:
            err = _scrub((proc.stderr or "") + "\n" + (probe.stderr or ""))[:160]
            raise RuntimeError(f"{_UNLOCK_HINT} ({err or 'not signed in'})")
        if not _unlock.store_session_token(self.name, _APP_SESSION, generation):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _run(self, *args: str) -> str:
        token = None if self._service_token else _unlock.get_session_token(self.name)
        if not self._service_token and not token:
            raise UnlockRequired(self)
        proc = run_cli([str(self._op()), *args], env=self._env(token), timeout=_TIMEOUT, label="op",
                       timeout_message="op timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = _scrub(proc.stderr or "")
            if "session" in err.lower() or "sign in" in err.lower() or "not signed in" in err.lower():
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            raise RuntimeError(f"op failed: {err[:200]}")
        return proc.stdout or ""

    # ── backend contract ───────────────────────────────────────────────────
    def list_items(self) -> List[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        raw = json.loads(self._run("item", "list", "--categories", "Login", "--format", "json") or "[]")
        out: List[VaultItemMeta] = []
        for item in raw if isinstance(raw, list) else []:
            urls = [str(u["href"]) for u in item.get("urls") or [] if isinstance(u, dict) and u.get("href")]
            origin = _first_origin(urls)
            if not origin:
                continue
            username = str(item.get("additional_information") or "").strip() or None
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item.get('id')}", kind="login", label=str(item.get("title") or origin),
                origin=origin, created_at=str(item.get("created_at") or ""),
                identifier_type="username" if username else None, identifier=username))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        item_id = handle[len(self.prefix):]
        return self._run("item", "get", item_id, "--fields", "label=password", "--reveal").rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            code = self._run("item", "get", handle[len(self.prefix):], "--otp").strip()
        except Exception:
            return None
        return code if code.isdigit() else None


def _first_origin(urls: List[str]) -> Optional[str]:
    for u in urls:
        try:
            return normalize_origin(u)
        except Exception:
            continue
    return None
