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
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli
from agent.secret_sources.onepassword import _OP_ENV_ALLOWLIST, _scrub, find_op
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_stdin_secret
from agent.vault_backends import unlock as _unlock
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self._item_vaults: Dict[str, str] = {}
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
            # A configured service account is the complete auth context. Do
            # not consult or fall back to the desktop app for vault actions.
            env["OP_LOAD_DESKTOP_APP_SETTINGS"] = "false"
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
        vaults = self._selected_vaults() if self._service_token or self._configured_vault_refs() else []
        batches = []
        if vaults:
            for vault in vaults:
                raw = json.loads(self._run(
                    "item", "list", "--categories", "Login", "--vault", vault,
                    "--format", "json") or "[]")
                batches.append((vault, raw))
        else:
            raw = json.loads(self._run("item", "list", "--categories", "Login", "--format", "json") or "[]")
            batches.append((None, raw))

        out: List[VaultItemMeta] = []
        for vault, raw in batches:
            for item in raw if isinstance(raw, list) else []:
                urls = [str(u["href"]) for u in item.get("urls") or [] if isinstance(u, dict) and u.get("href")]
                origins = _all_origins(urls)
                if not origins:
                    continue
                username = str(item.get("additional_information") or "").strip() or None
                item_id = str(item.get("id") or "")
                if not item_id:
                    continue
                if vault:
                    self._item_vaults[item_id] = vault
                out.append(VaultItemMeta(
                    id=f"{self.prefix}{item_id}", kind="login", label=str(item.get("title") or origins[0]),
                    origin=origins[0], created_at=str(item.get("created_at") or ""),
                    identifier_type="username" if username else None, identifier=username,
                    allowed_origins=_web_origins(origins)))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        wanted_vault, wanted_item = self._parse_handle(handle)
        for meta in self.list_items():
            vault, item_id = self._parse_handle(meta.id)
            if meta.id == handle or (item_id == wanted_item and (wanted_vault is None or vault == wanted_vault)):
                return meta
        return None

    def resolve_password(self, handle: str) -> str:
        vault, item_id = self._resolved_handle(handle)
        args = ["item", "get", item_id]
        if vault:
            args += ["--vault", vault]
        args += ["--fields", "label=password", "--reveal"]
        return self._run(*args).rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        vault, item_id = self._resolved_handle(handle)
        args = ["item", "get", item_id]
        if vault:
            args += ["--vault", vault]
        args.append("--otp")
        try:
            code = self._run(*args).strip()
        except Exception:
            return None
        return code if code.isdigit() else None

    def _configured_vault_refs(self) -> List[str]:
        refs = self.cfg.get("vaults")
        if refs is None:
            refs = self.cfg.get("vault")
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            return []
        return [str(ref).strip() for ref in refs if str(ref).strip()]

    def _selected_vaults(self) -> List[str]:
        raw = json.loads(self._run("vault", "list", "--format", "json") or "[]")
        available = [entry for entry in raw if isinstance(entry, dict) and entry.get("id")]
        configured = self._configured_vault_refs()
        if not configured:
            return [str(entry["id"]) for entry in available]

        selected: List[str] = []
        for ref in configured:
            match = next((entry for entry in available
                          if ref in (str(entry.get("id") or ""), str(entry.get("name") or ""))), None)
            if match is None:
                raise RuntimeError(f"configured 1Password vault is unavailable: {ref}")
            vault_id = str(match["id"])
            if vault_id not in selected:
                selected.append(vault_id)
        return selected

    def _parse_handle(self, handle: str) -> tuple[Optional[str], str]:
        payload = handle[len(self.prefix):] if handle.startswith(self.prefix) else handle
        if ":" in payload:
            vault, item_id = payload.split(":", 1)
            return vault or None, item_id
        return None, payload

    def _resolved_handle(self, handle: str) -> tuple[Optional[str], str]:
        vault, item_id = self._parse_handle(handle)
        if vault or not self._service_token:
            return vault, item_id
        vault = self._item_vaults.get(item_id)
        if vault is None:
            self.list_items()
            vault = self._item_vaults.get(item_id)
        if vault is None:
            raise RuntimeError(f"1Password item not found: {handle}")
        return vault, item_id


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
            candidate = u if "://" in u else f"https://{u.lstrip('/')}"
            origin = normalize_origin(candidate)
        except Exception:
            continue
        if origin not in out:
            out.append(origin)
    return out
