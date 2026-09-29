"""1Password Login items as a vault backend (``op`` CLI).

Unlock: ``op signin --raw`` mints a session token for manual account auth.
Desktop-app integration instead authorizes through 1Password without a token;
a metadata-only command verifies access before Hermes grants its unlock lease.
A configured service-account token skips the prompt entirely (headless).
List: ``op item list --categories Login --format json`` → title, urls,
username. Resolve: ``op item get <id> --fields label=password --reveal``.
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
    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}

    def _auth_sources(self) -> tuple[str, str, str]:
        """Read this call's credentials from the active profile scope, never an instance cache."""
        from agent.secret_scope import get_secret

        host = (get_secret("OP_CONNECT_HOST", "") or "").strip()
        token = (get_secret("OP_CONNECT_TOKEN", "") or "").strip()
        env_name = str(self.cfg.get("service_account_token_env") or "OP_SERVICE_ACCOUNT_TOKEN")
        service_token = (get_secret(env_name, "") or "").strip()
        if host or token:
            return ("connect" if host and token else "unavailable", host, token)
        if service_token:
            return ("service_account", "", service_token)
        return ("interactive", "", "")

    @staticmethod
    def _native_app_eligible(binary_path: str) -> bool:
        """Eligibility is based on the process host and its resolved CLI, not the requesting client."""
        from hermes_platform.host import facts
        from hermes_platform.resolver import LookupContext, locate_command

        if facts.os_family() not in {"win32", "darwin", "linux"} or not facts.interactive_session():
            return False
        result = locate_command(binary_path or "op", LookupContext(path=os.environ.get("PATH")))
        return result.found

    @property
    def supports_app_unlock(self) -> bool:
        return self._native_app_eligible(str(self.cfg.get("binary_path") or ""))

    def auth_capabilities(self) -> Dict[str, object]:
        mode, _, _ = self._auth_sources()
        if mode == "unavailable":
            return {"mode": "unavailable", "methods": [], "native_app_eligible": False,
                    "reason": "connect_incomplete"}
        if mode in {"connect", "service_account"}:
            reason = "service_account_configured" if mode == "service_account" else None
            return {"mode": mode, "methods": [], "native_app_eligible": False, "reason": reason}

        from hermes_platform.host import facts

        native = self.supports_app_unlock
        methods = (["app"] if native else [])
        if facts.interactive_session():
            methods.append("password")
        if not methods:
            return {"mode": "unavailable", "methods": [], "native_app_eligible": False,
                    "reason": "interactive_host_unavailable"}
        return {"mode": "interactive", "methods": methods, "native_app_eligible": native,
                "reason": None if native else "native_app_ineligible"}

    # ── auth ────────────────────────────────────────────────────────────────

    def _op(self) -> Path:
        op = find_op(str(self.cfg.get("binary_path") or ""))
        if op is None:
            raise RuntimeError("1Password CLI (op) not found — install it or set vault.onepassword.binary_path")
        return op

    def _env(self, session_token: Optional[str], *, app_auth: Optional[bool] = None) -> Dict[str, str]:
        mode, connect_host, auth_token = self._auth_sources()
        if mode == "unavailable":
            raise UnlockRequired(self)
        env = {k: os.environ[k] for k in _OP_ENV_ALLOWLIST if k in os.environ and not k.startswith("OP_CONNECT_")}
        env["NO_COLOR"] = "1"
        account = str(self.cfg.get("account") or "")
        if account:
            env["OP_ACCOUNT"] = account
        if mode == "connect":
            env["OP_CONNECT_HOST"] = connect_host
            env["OP_CONNECT_TOKEN"] = auth_token
        elif mode == "service_account":
            env["OP_SERVICE_ACCOUNT_TOKEN"] = auth_token
        elif session_token:
            # op signin --raw prints the bare token; the env var name carries the account shorthand,
            # which op also accepts as plain OP_SESSION for the default account.
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        if app_auth is not None:
            # Explicit app mode enables integration, explicit password mode bypasses it, and
            # manager calls pass through the setting that matches the stored lease type.
            env["OP_LOAD_DESKTOP_APP_SETTINGS"] = "true" if app_auth else "false"
        return env

    def is_unlocked(self) -> bool:
        mode, _, _ = self._auth_sources()
        return mode in {"connect", "service_account"} or (mode == "interactive" and _unlock.is_unlocked(self.name))

    def unlock(self, master_password: Optional[str] = None, *, method: Optional[str] = None) -> None:
        """Authorize explicitly, using the app or a password consumed only on stdin."""
        mode, _, _ = self._auth_sources()
        if mode in {"connect", "service_account"}:
            return
        if mode == "unavailable":
            raise RuntimeError("1Password Connect credentials are incomplete; configure both Connect host and token.")
        selected = method or ("app" if not master_password else "password")
        if selected not in {"app", "password"}:
            raise RuntimeError("1Password unlock method is unsupported.")
        if selected == "app" and not self.supports_app_unlock:
            raise RuntimeError("1Password app unlock is unavailable on this backend host.")
        if selected == "password":
            from hermes_platform.host import facts
            if not facts.interactive_session():
                raise RuntimeError("1Password password unlock is unavailable on this backend host.")
            if not master_password:
                raise RuntimeError("1Password master password was not provided.")
        generation = _unlock.begin_unlock(self.name)
        cmd = [str(self._op()), "signin", "--raw"]
        if account := str(self.cfg.get("account") or ""):
            cmd += ["--account", account]
        proc = run_with_stdin_secret(
            cmd, env=self._env(None, app_auth=selected == "app"),
            secret=master_password or "", timeout=_TIMEOUT, label="op")
        token = (proc.stdout or "").strip()
        if proc.returncode != 0:
            error = _scrub(proc.stderr or "")
            if master_password:
                error = error.replace(master_password, "[REDACTED]")
            raise RuntimeError(f"1Password unlock failed: {error[:200] or 'sign-in was not authorized'}. {_APP_UNLOCK_HINT}")
        if not token:
            if selected == "password":
                raise RuntimeError("1Password password unlock did not return a session token.")
            # Desktop integration deliberately exports no OP_SESSION token. Do not
            # mistake account registration (or `whoami`) for usable authorization.
            probe = run_cli([str(self._op()), "vault", "list", "--format", "json"],
                            env=self._env(None, app_auth=True), timeout=_TIMEOUT, label="op",
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
        if not _unlock.store_session_token(self.name, token, generation):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _run(self, *args: str) -> str:
        mode, _, _ = self._auth_sources()
        token = _unlock.get_session_token(self.name) if mode == "interactive" else None
        if mode not in {"connect", "service_account"} and token is None:
            raise UnlockRequired(self)
        proc = run_cli([str(self._op()), *args],
                       env=self._env(token, app_auth=(token == "") if mode == "interactive" else None),
                       timeout=_TIMEOUT, label="op",
                       timeout_message="op timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = _scrub(proc.stderr or "")
            if any(message in err.lower() for message in (
                "session", "sign in", "not signed in", "authorization prompt dismissed",
            )):
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            _, connect_host, auth_secret = self._auth_sources()
            for secret in (connect_host, auth_secret, token or ""):
                if secret:
                    err = err.replace(secret, "[REDACTED]")
            raise RuntimeError(f"op failed: {err[:200]}")
        return proc.stdout or ""

    # ── backend contract ───────────────────────────────────────────────────
    def list_items(self) -> List[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        vault_args = ["--vault", str(self.cfg["vault"])] if str(self.cfg.get("vault") or "").strip() else []
        raw = json.loads(self._run("item", "list", "--categories", "Login", *vault_args, "--format", "json") or "[]")
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

    def resolve_password(self, handle: str) -> str:
        item_id = handle[len(self.prefix):]
        vault_args = ["--vault", str(self.cfg["vault"])] if str(self.cfg.get("vault") or "").strip() else []
        return self._run("item", "get", item_id, *vault_args, "--fields", "label=password", "--reveal").rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            vault_args = ["--vault", str(self.cfg["vault"])] if str(self.cfg.get("vault") or "").strip() else []
            code = self._run("item", "get", handle[len(self.prefix):], *vault_args, "--otp").strip()
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
