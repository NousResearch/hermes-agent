"""Proton Pass Login items as a vault backend (``pass-cli``).

Unlock is PAT-based and headless: ``PROTON_PASS_PERSONAL_ACCESS_TOKEN``
(``pass-cli login --pat``) mints an agent session scoped to its own
``PROTON_PASS_SESSION_DIR`` under Hermes' temp dir, so the agent's reads
never collide with the user's desktop-app session. There is no
master-password CLI unlock in Proton Pass, so without a PAT the backend is
"locked" — the browser vault reports it under ``locked`` with
``unavailable_in_this_session`` (the surface has nothing to prompt).

List: ``pass-cli vault list --output json`` then
``item list --vault-name <name> --filter-type login --filter-state active
--output json`` (ids/titles only — Proton keeps URLs/usernames behind a
per-item view). Metadata reads use field-level ``item view --field …``
only: a full agent-session ``item view --output json`` masks the password
but NOT ``totp_uri``/passkeys, so it must never enter the metadata path.
Resolve: ``item view --field password`` (audited; ``--field urls`` /
``--field email`` / ``--field username`` feed list metadata). OTP:
``totp generate <otpauth-uri>``.

Items are pinned to one **named vault** (default ``Personal``) so the agent
only ever sees a single vault's logins; ``vault.protonpass.vault`` chooses
another. Reads carry ``PROTON_PASS_AGENT_REASON`` (mandatory for
agent-session reads; audited client-side).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_scope import get_secret
from agent.secret_sources.base import run_cli, scrub_ansi
from agent.vault_backends.base import LoginBackend, UnlockRequired
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_REASON = "Hermes browser credential vault: listing/resolving a login for autofill"

_ENV_KEEP = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
             "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME")
# Agent sessions get their own PROTON_PASS_SESSION_DIR so they can never
# collide with (or evict) the user's own pass-cli / desktop-app session.
_SESSION_ROOT = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "hermes-pass-cli"


class ProtonPassLoginBackend(LoginBackend):
    name = "protonpass"
    display_name = "Proton Pass"
    prefix = "pp:"
    # No master-password CLI unlock exists; without a PAT there is nothing to
    # prompt, so the surface must show the backend as unreachable, not ask.
    needs_unlock = True

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self._pat_env = str(self.cfg.get("pat_env") or "PROTON_PASS_PERSONAL_ACCESS_TOKEN")
        self._pat = ""
        self._pat_read = False
        # One backend instance covers one named vault (handle namespace is
        # <prefix><id>; a second vault would need its own prefix — out of scope).
        self.vault = str(self.cfg.get("vault") or "Personal").strip() or "Personal"
        self._logged_in = False
        self._list_cache: Optional[List[Dict]] = None

    @property
    def pat(self) -> str:
        """The Personal Access Token, resolved lazily from the profile secret scope."""
        if not self._pat_read:
            self._pat = get_secret(self._pat_env, "") or ""
            self._pat_read = True
        return self._pat

    # ── auth / session ─────────────────────────────────────────────────────

    @property
    def _session_dir(self) -> Path:
        explicit = str(self.cfg.get("session_dir") or "")
        return Path(explicit) if explicit else _SESSION_ROOT

    def _bin(self) -> Path:
        explicit = str(self.cfg.get("binary_path") or "")
        found = explicit or shutil.which("pass-cli")
        if not found:
            raise RuntimeError("Proton Pass CLI (pass-cli) not found — install it or set vault.protonpass.binary_path")
        return Path(found)

    def _env(self) -> Dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["NO_COLOR"] = "1"
        env["PROTON_PASS_SESSION_DIR"] = str(self._session_dir)
        env["PROTON_PASS_AGENT_REASON"] = _REASON
        # Never inherit (or leak) a PAT that is not the profile-scoped one.
        if self.pat:
            env["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] = self.pat
        return env

    def is_unlocked(self) -> bool:
        return bool(self.pat)

    def unlock(self, master_password: str) -> None:  # type: ignore[override]  # no master-password path
        raise RuntimeError(
            "Proton Pass has no master-password unlock. Set the agent PAT in "
            "~/.hermes/.env as PROTON_PASS_PERSONAL_ACCESS_TOKEN (or point "
            "vault.protonpass.pat_env at the env var that holds it), then re-run."
        )

    def _login_if_needed(self) -> None:
        """Mint the agent session once; ``login --pat`` is idempotent and cheap."""
        if self._logged_in:
            return
        proc = run_cli([str(self._bin()), "login", "--pat", self.pat], env=self._env(),
                       timeout=_TIMEOUT, label="pass-cli", timeout_message="pass-cli login timed out")
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or proc.stdout or "")
            # A warm session in our isolated dir makes pass-cli exit 0 with
            # "Error: Already authenticated" on stderr — that IS the logged-in state.
            if "already authenticated" not in err.lower():
                raise RuntimeError(f"Proton Pass login failed: {err[:200] or 'no session token'}")
        self._logged_in = True

    def _run(self, *args: str) -> str:
        if not self.pat:
            raise UnlockRequired(self)
        self._login_if_needed()
        proc = run_cli([str(self._bin()), *args], env=self._env(), timeout=_TIMEOUT,
                       label="pass-cli", timeout_message="pass-cli timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = scrub_ansi(proc.stderr or proc.stdout or "")
            low = err.lower()
            if "session" in low or "authenticated" in low or "log in" in low or "not logged" in low:
                # Agent session expired (PAT sessions are short-lived): re-login once, retry once.
                self._logged_in = False
                self._login_if_needed()
                proc = run_cli([str(self._bin()), *args], env=self._env(), timeout=_TIMEOUT,
                               label="pass-cli", timeout_message="pass-cli timed out", stdin=subprocess.DEVNULL)
                if proc.returncode == 0:
                    return proc.stdout or ""
                err = scrub_ansi(proc.stderr or proc.stdout or "")
            raise RuntimeError(f"pass-cli failed: {err[:200]}")
        return proc.stdout or ""

    @staticmethod
    def _json(out: str) -> Dict:
        try:
            raw = json.loads(out or "{}")
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}

    # ── backend contract ───────────────────────────────────────────────────

    def _list_items(self) -> List[Dict]:
        if self._list_cache is not None:
            return self._list_cache
        raw = self._json(self._run("vault", "list", "--output", "json"))
        vaults = [v for v in raw.get("vaults") or []
                  if isinstance(v, dict) and v.get("name") == self.vault]
        out: List[Dict] = []
        for v in vaults:
            raw = self._json(self._run("item", "list", "--vault-name", str(v.get("name")),
                                       "--filter-type", "login", "--filter-state", "active",
                                       "--output", "json"))
            for item in raw.get("items") or []:
                if isinstance(item, dict):
                    out.append(item)
        self._list_cache = out
        return out

    def _read_field(self, item_id: str, field: str) -> str:
        """One revealed field of one login item (via the audited agent path)."""
        return self._run("item", "view", "--vault-name", self.vault, "--item-id", item_id,
                         "--field", field).rstrip("\r\n")

    def list_items(self) -> List[VaultItemMeta]:
        """Instant metadata from `item list` (titles only — Proton keeps URLs/usernames
        behind a slow per-item view). Origin/identifier are resolved on demand by
        `get_meta` (fill path), so a 150-item vault lists in one call instead of 300
        × ~3s CLI round trips. Items without a resolved URL are excluded from fill by
        the base contract (no origin → not fillable)."""
        if not self.is_unlocked():
            return []
        out: List[VaultItemMeta] = []
        for item in self._list_items():
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item_id}", kind="login",
                label=str(item.get("title") or item_id), origin=None,
                created_at=str(item.get("create_time") or "")))
        return out

    def _read_meta_fields(self, item_id: str) -> Dict[str, str]:
        """Metadata-only reads (identifier + urls), never the full item view.

        An agent-session full ``item view --output json`` masks ``password``
        but returns ``totp_uri``/passkeys in clear, so it must never feed
        list metadata. The field-level reads are the audited narrow channel.
        """
        urls = ""
        try:
            urls = self._read_field(item_id, "urls")
        except Exception:
            return {}
        if not urls.strip():
            return {}
        email = ""
        try:
            email = self._read_field(item_id, "email")
        except Exception:
            email = ""
        username = ""
        if not email.strip():
            try:
                username = self._read_field(item_id, "username")
            except Exception:
                username = ""
        return {"urls": urls, "email": email.strip(), "username": username.strip()}

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        """Fill-path metadata: resolves the item's URLs + identifier (2 field reads,
        ~7s on real pass-cli) — only for the one item being filled."""
        item_id = handle[len(self.prefix):] if handle.startswith(self.prefix) else ""
        if not item_id:
            return None
        try:
            fields = self._read_meta_fields(item_id)
        except Exception:
            return None
        origins = _all_origins(fields.get("urls", ""))
        if not origins:
            # The item exists but has no web URL: present it with the title so the
            # origin-mismatch path (rather than a missing-item error) explains why it
            # cannot be filled.
            label = next((str(i.get("title") or item_id) for i in self._list_items()
                          if str(i.get("id") or "") == item_id), item_id)
            return VaultItemMeta(id=handle, kind="login", label=label, origin=origins[0] if origins else None,
                                 created_at="")
        identifier = fields.get("email") or fields.get("username") or ""
        identifier = identifier.strip() or None
        title = next((str(i.get("title") or "") for i in self._list_items()
                      if str(i.get("id") or "") == item_id), "")
        return VaultItemMeta(
            id=handle, kind="login", label=title or origins[0], origin=origins[0],
            created_at="",
            identifier_type=("email" if identifier and "@" in identifier else "username")
            if identifier else None,
            identifier=identifier,
            allowed_origins=_web_origins(origins))

    def resolve_password(self, handle: str) -> str:
        return self._read_field(handle[len(self.prefix):], "password")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `totp generate <otpauth-uri>` mints the current code from the item's seed.
        try:
            seed = self._read_field(handle[len(self.prefix):], "totp_uri")
        except Exception:
            return None
        if not seed.strip() or "***" in seed:  # masked by the agent-session policy
            return None
        try:
            proc = run_cli([str(self._bin()), "totp", "generate", seed.strip()], env=self._env(),
                           timeout=_TIMEOUT, label="pass-cli", timeout_message="pass-cli totp timed out")
            code = (proc.stdout or "").strip() if proc.returncode == 0 else ""
        except Exception:
            return None
        return code if code.isdigit() else None


def _web_origins(origins: List[str]) -> tuple:
    """Fill targets are browser pages: app URIs never widen the fill set."""
    web = tuple(o for o in origins if o.startswith(("http://", "https://")))
    return web or (origins[0],)


def _all_origins(urls: str) -> List[str]:
    """URLs come back comma-separated from pass-cli; normalize + dedupe, order preserved."""
    out: List[str] = []
    for part in urls.split(","):
        u = part.strip()
        if not u:
            continue
        try:
            origin = normalize_origin(u)
        except Exception:
            continue
        if origin not in out:
            out.append(origin)
    return out