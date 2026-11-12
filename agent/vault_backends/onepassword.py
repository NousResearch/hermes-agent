"""1Password Login items as a vault backend (``op`` CLI).

Unlock: ``op signin --raw`` mints a session token for manual account auth.
Desktop-app integration instead authorizes through 1Password without a token;
a metadata-only command verifies access before Hermes grants its unlock lease.
A configured service-account token skips the prompt entirely (headless).
List: ``op item list --categories Login --format json`` → title, urls,
username, vault. Resolve: ``op item get <id> --vault <vault-id> ...`` (service
accounts reject an item read without a vault query).
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
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_stdin_secret
from agent.vault_backends import unlock as _unlock
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_APP_UNLOCK_HINT = (
    "Unlock the 1Password app on the computer running the Hermes backend, enable "
    "Settings > Developer > Integrate with 1Password CLI, and approve its request "
    "(Windows Hello on Windows)."
)


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True
    supports_app_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self._account = str(self.cfg.get("account") or os.environ.get("OP_ACCOUNT") or "")
        from agent.secret_scope import get_secret
        env_name = str(self.cfg.get("service_account_token_env") or "OP_SERVICE_ACCOUNT_TOKEN")
        self._service_token = get_secret(env_name, "") or ""
        self._item_vaults: dict[str, str] = {}  # item id -> vault id from the latest ``item list``

    # ── auth ────────────────────────────────────────────────────────────────

    def _op(self) -> Path:
        op = find_op(str(self.cfg.get("binary_path") or ""))
        if op is None:
            raise RuntimeError("1Password CLI (op) not found — install it or set vault.onepassword.binary_path")
        return op

    def _env(self, session_token: Optional[str]) -> dict[str, str]:
        from agent.secret_scope import get_secret
        env = {k: os.environ[k] for k in _OP_ENV_ALLOWLIST if k in os.environ and not k.startswith("OP_CONNECT_")}
        # Connect credentials outrank OP_SERVICE_ACCOUNT_TOKEN inside op, so they must come from the
        # profile's own secret scope like the service token does — never from the launch environment.
        if session_token != "":  # native authorization must not be proven by Connect
            for k in ("OP_CONNECT_HOST", "OP_CONNECT_TOKEN"):
                if v := get_secret(k, ""):
                    env[k] = v
        if session_token == "":
            # These locate the user's desktop approval surface; no ambient auth is inherited.
            for k in ("DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "XDG_SESSION_TYPE",
                      "OP_BIOMETRIC_UNLOCK_ENABLED"):
                if k in os.environ:
                    env[k] = os.environ[k]
        env["NO_COLOR"] = "1"
        account = self._account
        if account:
            env["OP_ACCOUNT"] = account
        if self._service_token and session_token != "":
            env["OP_SERVICE_ACCOUNT_TOKEN"] = self._service_token
        elif session_token:
            # op signin --raw prints the bare token; the env var name carries the account shorthand,
            # which op also accepts as plain OP_SESSION for the default account.
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return bool(self._service_token) or _unlock.is_unlocked(self.name, context=self._account)

    def unlock(self, master_password: str) -> None:
        """Authorize explicitly, using the app or a password consumed only on stdin."""
        generation = _unlock.begin_unlock(self.name)
        token = ""
        if master_password:
            cmd = [str(self._op()), "signin", "--raw"]
            if self._account:
                cmd += ["--account", self._account]
            proc = run_with_stdin_secret(cmd, env=self._env(None), secret=master_password, timeout=_TIMEOUT, label="op")
            token = (proc.stdout or "").strip()
            if proc.returncode != 0:
                error = _scrub(proc.stderr or "").replace(master_password, "[REDACTED]")
                raise RuntimeError(f"1Password unlock failed: {error[:200] or 'sign-in was not authorized'}. {_APP_UNLOCK_HINT}")
        if not token:
            _require_native_viewer()
            # Desktop integration deliberately exports no OP_SESSION token. Do not
            # mistake account registration (or `whoami`) for usable authorization.
            probe = run_cli([str(self._op()), "vault", "list", "--format", "json"],
                            env=self._env(""), timeout=_TIMEOUT, label="op",
                            timeout_message=f"1Password authorization timed out. {_APP_UNLOCK_HINT}",
                            stdin=subprocess.DEVNULL)
            try:
                verified = probe.returncode == 0 and isinstance(json.loads(probe.stdout or ""), list)
            except ValueError:
                verified = False
            if not verified:
                raise RuntimeError(f"1Password desktop authorization could not be verified. {_APP_UNLOCK_HINT}")
        # Empty string is a verified desktop lease, not a credential. The existing
        # profile/owner/TTL/generation store remains the sole unlock authority.
        if not _unlock.store_session_token(self.name, token, generation, context=self._account):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _run(self, *args: str) -> str:
        token = None if self._service_token else _unlock.get_session_token(self.name, context=self._account, touch=False)
        if not self._service_token:
            if token is None:
                raise UnlockRequired(self)
            if token == "":
                _require_native_viewer()
            token = _unlock.get_session_token(self.name, context=self._account)
            if token is None:
                raise UnlockRequired(self)
        proc = run_cli([str(self._op()), *args], env=self._env(token), timeout=_TIMEOUT, label="op",
                       timeout_message="op timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = _scrub(proc.stderr or "")
            # A service account never holds a session, so an auth failure there is not a lock to undo:
            # surface op's own error instead of an unlock prompt that reports "already unlocked".
            if not self._service_token and any(w in err.lower() for w in (
                "session", "sign in", "not signed in", "authorization prompt dismissed",
            )):
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            raise RuntimeError(f"op failed: {err[:200]}")
        return proc.stdout or ""

    # ── backend contract ───────────────────────────────────────────────────
    def list_items(self) -> list[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        raw = json.loads(self._run("item", "list", "--categories", "Login", "--format", "json") or "[]")
        out: list[VaultItemMeta] = []
        self._item_vaults = {}
        for item in raw if isinstance(raw, list) else []:
            vault = item.get("vault") or {}
            if item.get("id"):
                self._item_vaults[str(item["id"])] = str(vault.get("id") or "") if isinstance(vault, dict) else ""
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

    def _item_selector(self, handle: str) -> list[str]:
        """``op item get`` target for a listed login. A service account must name the item's vault
        (``--vault``) and ``item list`` reports it, so the vault comes from the same listing that
        proved the handle is a Login item; a handle nothing listed is refused before any secret read."""
        item_id = handle[len(self.prefix):]
        if item_id not in self._item_vaults:
            self.list_items()
        if item_id not in self._item_vaults:
            raise RuntimeError("1Password login not found; list logins again")
        vault_id = self._item_vaults[item_id]
        return [item_id, "--vault", vault_id] if vault_id else [item_id]

    def resolve_password(self, handle: str) -> str:
        return self._run("item", "get", *self._item_selector(handle),
                         "--fields", "label=password", "--reveal").rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            code = self._run("item", "get", *self._item_selector(handle), "--otp").strip()
        except Exception:
            return None
        return code if code.isdigit() else None


def _require_native_viewer() -> None:
    from tools.approval_context import _no_user_can_answer
    if _no_user_can_answer():
        raise RuntimeError("Native 1Password access requires an interactive session; use a service account for unattended jobs")


def _web_origins(origins: list[str]) -> tuple:
    """Fill targets are browser pages, so app URIs (``androidapp://`` etc.) never
    widen the fill set; an item whose only URI is an app URI keeps its single
    (unfillable-from-a-page) origin exactly as before."""
    web = tuple(o for o in origins if o.startswith(("http://", "https://")))
    return web or (origins[0],)


def _all_origins(urls: list[str]) -> list[str]:
    """Every normalized origin saved on the item, deduped, order preserved.

    A 1Password Login item can carry several websites; each of them is a place the
    user told 1Password the credential belongs, so all of them are valid fill targets.
    """
    out: list[str] = []
    for u in urls:
        try:
            origin = normalize_origin(u)
        except Exception:
            continue
        if origin not in out:
            out.append(origin)
    return out
