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
from agent.vault_store import VaultItemMeta, normalize_origin, normalize_otp_secret, totp_now

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True

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
        elif session_token:
            # op signin --raw prints the bare token; the env var name carries the account shorthand,
            # which op also accepts as plain OP_SESSION for the default account.
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        return env

    def _connect_credentials(self):
        from agent.secret_scope import get_secret
        host, token = get_secret("OP_CONNECT_HOST", ""), get_secret("OP_CONNECT_TOKEN", "")
        if bool(host) != bool(token):
            raise RuntimeError("1Password Connect requires both host and token")
        return host, token

    def _connect_get(self, path: str):
        # op item list does not support Connect. Use the official read-only API;
        # never leak response bodies, redirected credentials or request exceptions.
        import requests
        from urllib.parse import urlsplit
        host, token = self._connect_credentials()
        if not host or not token:
            raise UnlockRequired(self)
        url = urlsplit(host)
        if (url.username or url.password or url.query or url.fragment or
                not (url.scheme == "https" or (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")))):
            raise RuntimeError("Connect requires HTTPS or loopback HTTP")
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(host.rstrip("/") + path,
                    headers={"Authorization": "Bearer " + token}, timeout=_TIMEOUT, allow_redirects=False)
                if response.status_code != 200:
                    raise RuntimeError("Connect read refused")
                return response.json()
        except Exception:
            raise RuntimeError("1Password Connect read failed") from None

    @staticmethod
    def _connect_ids(handle: str):
        import re
        match = re.fullmatch(r"op:connect:([a-z0-9]{26}):([a-z0-9]{26})", handle)
        if not match:
            raise ValueError("Invalid Connect login handle")
        return match.groups()

    def _connect_item(self, handle: str):
        vault_id, item_id = self._connect_ids(handle)
        item = self._connect_get(f"/v1/vaults/{vault_id}/items/{item_id}")
        if (not isinstance(item, dict) or item.get("id") != item_id or
                (item.get("vault") or {}).get("id") != vault_id or
                item.get("category") != "LOGIN" or item.get("state", "ACTIVE") != "ACTIVE"):
            raise RuntimeError("Connect login identity mismatch")
        return item

    @staticmethod
    def _connect_otp_seed(item) -> Optional[str]:
        """Canonical RFC 6238 seed from a Connect item's one-time-password field, else None.

        The same helper backs both ``has_otp`` (the agent's "codes are minted automatically"
        hint) and ``resolve_otp``, so a stored item is never announced as automatic unless a
        code can actually be minted from it. Every eligible field is examined: an unusable
        candidate (blank, unnormalisable, or one the minter rejects) must not hide a later
        usable seed — the base32 alphabet check alone would accept an alphabet-valid but
        undecodable secret (e.g. "A"), or a period too large for the runtime division.
        """
        for field in item.get("fields", []):
            if field.get("type") != "OTP" and field.get("purpose") != "ONE_TIME_PASSWORD":
                continue
            raw = field.get("value")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                seed = normalize_otp_secret(raw)
                if seed and totp_now(seed):  # fail closed: no mintable seed, no claim
                    return seed
            except Exception:
                continue
        return None

    @staticmethod
    def _connect_meta(item, vault_id):
        handle = f"op:connect:{vault_id}:{item.get('id')}"
        OnePasswordLoginBackend._connect_ids(handle)
        origins = _all_origins([u.get("href", "") for u in item.get("urls", [])])
        if not origins:
            return None
        origin = origins[0]
        username = next((f.get("value") for f in item.get("fields", []) if f.get("purpose") == "USERNAME"), None)
        return VaultItemMeta(id=handle, kind="login", label=str(item.get("title") or origin),
                             has_otp=OnePasswordLoginBackend._connect_otp_seed(item) is not None,
                             origin=origin, created_at=str(item.get("createdAt") or ""),
                             identifier_type="username" if username else None, identifier=username,
                             allowed_origins=_web_origins(origins))

    def is_unlocked(self) -> bool:
        try:
            _, connect_token = self._connect_credentials()
        except RuntimeError:
            return False
        return bool(connect_token) or bool(self._service_token) or _unlock.is_unlocked(self.name)

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
        self._connect_credentials()  # Misconfigured Connect must never fall back to another route.
        if not self.is_unlocked():
            return []
        if self._connect_credentials()[1]:
            out = []
            for vault in self._connect_get("/v1/vaults"):
                vault_id = vault["id"]
                self._connect_ids(f"op:connect:{vault_id}:{vault_id}")
                for item in self._connect_get(f"/v1/vaults/{vault_id}/items"):
                    if item.get("category") == "LOGIN" and item.get("state", "ACTIVE") == "ACTIVE":
                        meta = self._connect_meta(item, vault_id)
                        if meta:
                            out.append(meta)
            return out
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
        if self._connect_credentials()[1]:
            item = self._connect_item(handle)
            return self._connect_meta(item, item["vault"]["id"])
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        if self._connect_credentials()[1]:
            fields = self._connect_item(handle).get("fields", [])
            passwords = [f.get("value") for f in fields if f.get("purpose") == "PASSWORD"]
            if len(passwords) != 1 or not isinstance(passwords[0], str) or not passwords[0]:
                raise RuntimeError("Connect login requires one password field")
            return passwords[0]
        item_id = handle[len(self.prefix):]
        return self._run("item", "get", item_id, "--fields", "label=password", "--reveal").rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        if self._connect_credentials()[1]:
            # Connect exposes the item's OTP field (an otpauth:// URI); mint the code locally,
            # never over the CLI. No stored seed → None → the user is asked.
            seed = self._connect_otp_seed(self._connect_item(handle))
            return totp_now(seed) if seed else None
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            code = self._run("item", "get", handle[len(self.prefix):], "--otp").strip()
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
