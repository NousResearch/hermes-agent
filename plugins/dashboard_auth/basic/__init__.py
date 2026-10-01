"""BasicAuthProvider — username/password dashboard auth (no OAuth IDP).

Login is a credential form (``supports_password`` + ``complete_password_login``); cookies,
verify, refresh, ws-tickets and logout are the shared framework. Sessions are stateless
HMAC-signed tokens (no IDP, no database); passwords use stdlib scrypt and login always hashes
even for an unknown username (no username-enumeration timing oracle). Config: ``dashboard.
basic_auth.{username,password_hash|password,secret,session_ttl_seconds}`` or the
``HERMES_DASHBOARD_BASIC_AUTH_*`` env vars (env wins when non-empty; see ``_settings``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Optional

from hermes_cli.dashboard_auth import DashboardAuthProvider, InvalidCredentialsError, RefreshExpiredError, Session
from plugins.dashboard_auth._shared import (
    NonInteractiveMixin, SkipRegistration, load_config_section, register_provider, resolve_env_or_cfg)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-basic"

# The middleware transparently refreshes via the 30-day refresh token when the
# access token lapses, so the TTL controls refresh frequency, not login length.
_DEFAULT_TTL_SECONDS = 12 * 60 * 60
_REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60

# Interactive-login scrypt parameters (~16 MiB, a few ms); n must be a power of two.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_SALT_BYTES = 16

# HMAC-SHA256 digest is appended to signed tokens as a fixed-length suffix
# (no separator — binary HMAC bytes can't be confused with a delimiter).
_SIG_LEN = hashlib.sha256().digest_size

LAST_SKIP_REASON: str = ""


# ---- Password hashing (stdlib scrypt) ----

def hash_password(password: str) -> str:
    """Return a ``scrypt$n$r$p$<salt_b64>$<dk_b64>`` hash string. Public so operators can
    precompute ``password_hash`` for config.yaml (the plaintext then never sits at rest):
    ``python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('pw'))"``."""
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN, maxmem=0)
    salt_b64, dk_b64 = base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt_b64}${dk_b64}"


def _verify_password(password: str, encoded: str) -> bool:
    """Constant-time scrypt verify. False on any malformed hash string."""
    try:
        scheme, n_s, r_s, p_s, salt_b64, dk_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except (ValueError, TypeError):
        return False
    try:
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected), maxmem=0)
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


# Verified against when the username is unknown so "no such user" and "wrong
# password" take comparable time.
_DUMMY_HASH = hash_password("dummy-password-for-constant-time-verify")


# ---- Token signing (stateless HMAC-signed blobs) ----

def _sign(payload: dict, secret: bytes) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + sig).decode()


def _unsign(token: str, secret: bytes, kind: str) -> Optional[dict]:
    """Return the payload if the signature is valid, ``kind`` matches and it
    is unexpired; ``None`` otherwise (including on any decode error)."""
    try:
        blob = base64.urlsafe_b64decode(token.encode())
        if len(blob) <= _SIG_LEN:
            return None
        raw, sig = blob[:-_SIG_LEN], blob[-_SIG_LEN:]
        expected = hmac.new(secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(raw)
    except Exception:
        return None
    if payload.get("kind") != kind or payload.get("exp", 0) <= int(time.time()):
        return None
    return payload


# ---- Provider ----

class BasicAuthProvider(NonInteractiveMixin, DashboardAuthProvider):
    """Username/password provider with stateless HMAC-signed sessions."""

    name = "basic"
    display_name = "Username & Password"
    supports_password = True
    _NOT_INTERACTIVE = "BasicAuthProvider is password-only; use complete_password_login."
    _NO_START_LOGIN = (
        "BasicAuthProvider is password-only; there is no OAuth redirect flow. "
        "The login page POSTs to /auth/password-login instead.")

    def __init__(self, *, username: str, password_hash: str, secret: bytes, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        if not username:
            raise ValueError("username must be non-empty")
        if not password_hash:
            raise ValueError("password_hash must be non-empty")
        if len(secret) < 16:
            raise ValueError("secret must be at least 16 bytes")
        self._username = username
        self._password_hash = password_hash
        self._secret = secret
        self._ttl = max(60, int(ttl_seconds))

    # ---- password login ----------------------------------------------------

    def complete_password_login(self, *, username: str, password: str) -> Session:
        # Always run a scrypt verify (real hash if the username matches, else the dummy)
        # and compare the username with compare_digest too, so neither the username nor
        # its length leaks via timing.
        username_ok = hmac.compare_digest(username.encode("utf-8"), self._username.encode("utf-8"))
        password_ok = _verify_password(password, self._password_hash if username_ok else _DUMMY_HASH)
        if not (username_ok and password_ok):
            raise InvalidCredentialsError("invalid username or password")
        return self._mint_session(self._username)

    # ---- session lifecycle -------------------------------------------------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        payload = _unsign(access_token, self._secret, "access")
        if payload is None:
            return None
        return self._session(str(payload.get("sub", "")), int(payload["exp"]), access_token, "")

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        payload = _unsign(refresh_token, self._secret, "refresh")
        if payload is None:
            raise RefreshExpiredError("refresh token expired or invalid")
        return self._mint_session(str(payload.get("sub", self._username)))

    def revoke_session(self, *, refresh_token: str) -> None:
        # Stateless tokens — nothing to revoke server-side; the session expires within its TTL. Must not raise.
        return None

    # ---- internals ---------------------------------------------------------

    def _mint_session(self, user_id: str) -> Session:
        now = int(time.time())
        exp = now + self._ttl
        return self._session(
            user_id, exp,
            _sign({"sub": user_id, "kind": "access", "exp": exp}, self._secret),
            _sign({"sub": user_id, "kind": "refresh", "exp": now + _REFRESH_TTL_SECONDS}, self._secret))

    def _session(self, user_id: str, exp: int, access_token: str, refresh_token: str) -> Session:
        return Session(
            user_id=user_id, email="", display_name=user_id, org_id="", provider=self.name,
            expires_at=exp, access_token=access_token, refresh_token=refresh_token)


# ---- Plugin entry point ----

def _load_config_basic_auth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "basic_auth")


def _resolve_secret(cfg_section: dict) -> bytes:
    """Resolve the token-signing secret (base64, hex, or raw text). When unset, generates
    a random per-process secret (sessions then don't survive a restart or span multiple
    workers — logged at INFO)."""
    raw = resolve_env_or_cfg("HERMES_DASHBOARD_BASIC_AUTH_SECRET", cfg_section.get("secret"))
    if not raw:
        logger.info(
            "dashboard-auth-basic: no 'secret' configured; generating a random "
            "per-process signing key. Sessions will not survive a restart or span "
            "multiple workers. Set dashboard.basic_auth.secret (or "
            "HERMES_DASHBOARD_BASIC_AUTH_SECRET) for stable sessions.")
        return secrets.token_bytes(32)
    for decoder in (base64.b64decode, bytes.fromhex):
        try:
            decoded = decoder(raw)
            if len(decoded) >= 16:
                return decoded
        except (ValueError, TypeError):
            pass
    return raw.encode("utf-8")


def _settings() -> dict:
    """Resolve BasicAuthProvider kwargs from env/config; raises ``SkipRegistration``."""
    section = _load_config_basic_auth_section()

    def setting(env_name: str, cfg_key: str) -> str:
        return resolve_env_or_cfg(env_name, section.get(cfg_key, ""))

    username = setting("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "username")
    password_hash = setting("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH", "password_hash")
    plaintext = setting("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "password")
    ttl_raw = setting("HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS", "session_ttl_seconds")
    if not username:
        raise SkipRegistration(
            "dashboard.basic_auth.username is not set (and HERMES_DASHBOARD_BASIC_AUTH_USERNAME "
            "is empty). Set a username and a password (or password_hash) under "
            "dashboard.basic_auth in config.yaml to enable username/password dashboard "
            "login, or use the OAuth provider, or pass --insecure to skip the auth gate.")
    if not password_hash and not plaintext:
        raise SkipRegistration(
            "dashboard.basic_auth.username is set but neither password_hash nor password "
            "is configured. Provide one of them (password_hash is preferred — compute it "
            "with plugins.dashboard_auth.basic.hash_password).",
            level="warning")
    # Precedence: env password (hashed in-memory) overrides any config password_hash so
    # operators can rotate without editing config; a config password_hash wins over a
    # config-only plaintext password (preferred at-rest form).
    plaintext_from_env = os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "").strip()
    if plaintext_from_env:
        password_hash = hash_password(plaintext_from_env)
        logger.info("dashboard-auth-basic: hashed env-supplied password in-memory (overrides any config password_hash).")
    elif not password_hash:
        password_hash = hash_password(plaintext)
        logger.info(
            "dashboard-auth-basic: hashed plaintext password in-memory. "
            "For production, precompute dashboard.basic_auth.password_hash "
            "and remove the plaintext password from config.")
    try:
        ttl = int(ttl_raw) if ttl_raw else _DEFAULT_TTL_SECONDS
    except ValueError:
        ttl = _DEFAULT_TTL_SECONDS
    return {"username": username, "password_hash": password_hash, "secret": _resolve_secret(section), "ttl_seconds": ttl}


def register(ctx) -> None:
    """Register ``BasicAuthProvider`` when username + (password or
    password_hash) are configured; a no-op for OAuth / ``--insecure`` setups."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, BasicAuthProvider, _settings)
    if kwargs is not None:
        logger.info("dashboard-auth-basic: registered password provider (username=%s)", kwargs["username"])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'LoginStart': ('hermes_cli.dashboard_auth', 'LoginStart'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
