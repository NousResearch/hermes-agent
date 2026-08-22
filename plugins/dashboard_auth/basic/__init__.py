"""BasicAuthProvider — username/password dashboard auth (no OAuth IDP).

A self-hosted "just put a password on my dashboard" provider. It plugs
into the same ``DashboardAuthProvider`` framework as the Nous OAuth
provider, but authenticates with a username + password instead of an
OAuth redirect: it sets ``supports_password = True`` and implements
``complete_password_login``. The login page renders a credential form for
it; everything downstream of login (session cookies, verify, refresh,
ws-tickets, logout) is identical to the OAuth path because a password
session is just a :class:`Session` with provider-minted opaque tokens.

This provider has **no external IDP and no database**. Credentials are
configured up front; sessions are stateless HMAC-signed tokens this
provider mints and verifies itself. That keeps it zero-infrastructure —
appropriate for a single-box self-hosted dashboard.

Configuration surfaces (env wins over config.yaml when set non-empty),
mirroring the Nous provider's precedence convention:

  ``config.yaml`` — canonical surface::

      dashboard:
        basic_auth:
          username: admin               # required
          # Provide EITHER a precomputed scrypt hash (preferred — no
          # plaintext at rest) ...
          password_hash: "scrypt$..."   # see hash_password()
          # ... OR a plaintext password (hashed in-memory at load).
          password: "s3cret"
          secret: "<32+ random bytes, base64 or hex>"  # optional; token-signing key
          session_ttl_seconds: 43200    # optional; access-token lifetime (default 12h)

  Environment overrides::

      HERMES_DASHBOARD_BASIC_AUTH_USERNAME
      HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH   # preferred
      HERMES_DASHBOARD_BASIC_AUTH_PASSWORD        # plaintext fallback
      HERMES_DASHBOARD_BASIC_AUTH_SECRET
      HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS

If ``secret`` is not configured, a random per-process secret is generated
at startup. That's fine for a single-process dashboard, but means all
sessions are invalidated on restart and sessions don't survive across
multiple worker processes — set an explicit ``secret`` for stable
multi-worker / restart-surviving sessions.

Password hashing uses stdlib :func:`hashlib.scrypt` (memory-hard, no
third-party dependency). ``complete_password_login`` runs a constant-time
comparison and always performs a hash even for an unknown username, so
the endpoint is not a username-enumeration timing oracle.

Skip reasons:
  Like the Nous provider, this exposes a module-level ``LAST_SKIP_REASON``
  the gate's fail-closed branch can surface when the plugin loads but
  declines to register (no username/password configured).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCredentialsError,
    LoginStart,
    RefreshExpiredError,
    Session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Access-token lifetime. The middleware transparently refreshes via the
# refresh token (30-day) when the access token lapses, so this controls
# how often a refresh round trip happens, not how long the user stays
# logged in.
_DEFAULT_TTL_SECONDS = 12 * 60 * 60  # 12h
_REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60  # 30d

# scrypt parameters (RFC 7914 / stdlib hashlib.scrypt). n must be a power
# of two; these are the widely-recommended interactive-login parameters
# (~16 MiB, a few ms on commodity hardware).
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_SALT_BYTES = 16

# Length of the HMAC-SHA256 digest appended as a fixed-length suffix to
# signed tokens (no separator — binary HMAC bytes can't be confused with
# a delimiter).
_SIG_LEN = hashlib.sha256().digest_size


LAST_SKIP_REASON: str = ""


# ---------------------------------------------------------------------------
# Password hashing (stdlib scrypt)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a ``scrypt$n$r$p$<salt_b64>$<dk_b64>`` hash string.

    Use this to precompute ``password_hash`` for config.yaml so plaintext
    never sits at rest. Exposed as a module function so operators can run
    ``python -c "from plugins.dashboard_auth.basic import hash_password;
    print(hash_password('pw'))"``.
    """
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=0,
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}$"
        f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"
    )


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
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=0,
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


# A fixed dummy hash used to spend ~equal time when the username is
# unknown, so an attacker can't distinguish "no such user" (fast) from
# "wrong password" (slow scrypt) by timing. Computed once at import.
_DUMMY_HASH = hash_password("dummy-password-for-constant-time-verify")


# ---------------------------------------------------------------------------
# Token signing (stateless HMAC-signed blobs)
# ---------------------------------------------------------------------------


def _sign(payload: dict, secret: bytes) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + sig).decode()


def _unsign(token: str, secret: bytes) -> Optional[dict]:
    try:
        blob = base64.urlsafe_b64decode(token.encode())
        if len(blob) <= _SIG_LEN:
            return None
        raw, sig = blob[:-_SIG_LEN], blob[-_SIG_LEN:]
        expected = hmac.new(secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Revocation set (jti -> exp)
# ---------------------------------------------------------------------------

# Default location for the persisted revocation set. Kept under the Hermes
# state dir so it survives restarts and is covered by the same backup story
# as the rest of the deployment's state.
_REVOCATIONS_REL_PATH = Path("state") / "dashboard-auth-basic-revocations.json"


class _RevocationStore:
    """Persisted set of revoked session ids (``jti`` -> token ``exp``).

    Tokens carry a random ``jti`` (see ``_mint_session``). Revoking a
    session records its ``jti`` here; ``verify_session`` and
    ``refresh_session`` reject any token whose ``jti`` is present, so a
    revoked session dies immediately instead of living out its ~30-day
    refresh TTL.

    The set is bounded by construction: an entry is only meaningful until
    the revoked token's own ``exp`` passes, and ``_flush`` drops expired
    entries, so the file never holds more than the number of sessions
    revoked within one refresh TTL. A JSON file is sufficient for the
    single-box deployment this provider targets — no database needed.

    Thread-safe. Writes are atomic (temp file + ``os.replace``). All
    filesystem failures are swallowed and degrade to in-memory-only
    revocation for the life of the process — auth must never fail because
    the revocation store broke.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._revoked: dict[str, int] = {}  # jti -> exp (unix seconds)
        self._mtime: Optional[float] = None
        self._load()

    # ---- public API ------------------------------------------------------

    def revoke(self, jti: str, exp: int) -> None:
        """Record ``jti`` as revoked until ``exp``. Never raises."""
        with self._lock:
            self._revoked[jti] = exp
            self._flush()

    def is_revoked(self, jti: str) -> bool:
        """True if ``jti`` was revoked. Never raises."""
        with self._lock:
            self._reload_if_changed()
            return jti in self._revoked

    # ---- internals -------------------------------------------------------

    def _load(self) -> None:
        """Load the persisted set, dropping entries whose exp has passed."""
        self._revoked = {}
        if self._path is None:
            return
        try:
            raw = self._path.read_text()
            data = json.loads(raw)
            now = int(time.time())
            self._revoked = {
                str(jti): int(exp)
                for jti, exp in data.items()
                if isinstance(exp, int) and exp > now
            }
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 — corrupt file must not break auth
            logger.warning(
                "dashboard-auth-basic: failed to load revocation set from "
                "%s (%s); starting empty",
                self._path,
                exc_info=True,
            )
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            self._mtime = None

    def _reload_if_changed(self) -> None:
        """Re-read the file when another process (or worker) updated it.

        Enables multi-worker deployments with an explicit secret: a
        revocation written by one worker is picked up by the others on
        their next check. A failed stat just keeps the in-memory view.
        """
        if self._path is None:
            return
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    def _flush(self) -> None:
        """Persist the set atomically, dropping expired entries."""
        if self._path is None:
            return
        now = int(time.time())
        self._revoked = {
            jti: exp for jti, exp in self._revoked.items() if exp > now
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(self._revoked, sort_keys=True))
            os.replace(tmp, self._path)
            self._mtime = self._path.stat().st_mtime
        except Exception:  # noqa: BLE001 — write failure degrades gracefully
            logger.warning(
                "dashboard-auth-basic: failed to persist revocation set to "
                "%s; keeping in-memory set for this process",
                self._path,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class BasicAuthProvider(DashboardAuthProvider):
    """Username/password provider with stateless HMAC-signed sessions."""

    name = "basic"
    display_name = "Username & Password"
    supports_password = True

    def __init__(
        self,
        *,
        username: str,
        password_hash: str,
        secret: bytes,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        revocations_path: Optional[Path] = None,
    ) -> None:
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
        # Path to the persisted revocation set. ``None`` (tests, ad-hoc
        # construction) degrades to in-memory-only revocation for the life
        # of the process; ``register()`` passes the real state path so
        # revocations survive restarts.
        self._revocations = _RevocationStore(path=revocations_path)

    # ---- OAuth methods: not used (pure-password provider) ------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(
            "BasicAuthProvider is password-only; there is no OAuth redirect "
            "flow. The login page POSTs to /auth/password-login instead."
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise NotImplementedError(
            "BasicAuthProvider is password-only; use complete_password_login."
        )

    # ---- password login ----------------------------------------------------

    def complete_password_login(
        self, *, username: str, password: str
    ) -> Session:
        # Constant-time-ish: always run a scrypt verify (against the real
        # hash if the username matches, else a dummy hash) so an unknown
        # username and a wrong password take comparable time. Compare the
        # username with compare_digest too, to avoid a length/byte timing
        # leak on the username itself.
        username_ok = hmac.compare_digest(
            username.encode("utf-8"), self._username.encode("utf-8")
        )
        target_hash = self._password_hash if username_ok else _DUMMY_HASH
        password_ok = _verify_password(password, target_hash)
        if not (username_ok and password_ok):
            raise InvalidCredentialsError("invalid username or password")
        return self._mint_session(self._username)

    # ---- session lifecycle -------------------------------------------------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        payload = _unsign(access_token, self._secret)
        if (
            payload is None
            or payload.get("kind") != "access"
            or payload.get("exp", 0) <= int(time.time())
        ):
            return None
        # A token whose jti is in the revocation set is dead even if its
        # signature and exp are fine — the session was logged out.
        jti = payload.get("jti")
        if jti and self._revocations.is_revoked(str(jti)):
            return None
        return self._session_from_payload(access_token, "", payload)

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        payload = _unsign(refresh_token, self._secret)
        if (
            payload is None
            or payload.get("kind") != "refresh"
            or payload.get("exp", 0) <= int(time.time())
        ):
            raise RefreshExpiredError("refresh token expired or invalid")
        # Revoked refresh tokens are rejected like expired ones, so the
        # middleware forces a re-login instead of silently re-minting.
        jti = payload.get("jti")
        if jti and self._revocations.is_revoked(str(jti)):
            raise RefreshExpiredError("refresh token revoked")
        # Rotation carries the session's jti forward: revoking one jti
        # kills the whole session lineage, not just the presented token.
        return self._mint_session(str(payload.get("sub", self._username)), jti=jti)

    def revoke_session(self, *, refresh_token: str) -> None:
        # Record the session's jti in the revocation set so verify_session
        # and refresh_session reject it immediately, instead of waiting out
        # the ~30-day refresh TTL. Best-effort and must not raise — the
        # logout route still clears the client cookies regardless.
        if not refresh_token:
            return None
        payload = _unsign(refresh_token, self._secret)
        if payload is None or payload.get("kind") != "refresh":
            # Not one of our tokens (foreign provider, malformed, or an
            # access token) — nothing to revoke.
            return None
        jti = payload.get("jti")
        if not jti:
            # Token minted before jti existed; nothing to key a revocation
            # on. Falls back to TTL expiry.
            return None
        self._revocations.revoke(str(jti), int(payload.get("exp", 0)))
        return None

    # ---- internals ---------------------------------------------------------

    def _mint_session(
        self, user_id: str, *, jti: Optional[str] = None
    ) -> Session:
        now = int(time.time())
        exp = now + self._ttl
        # jti = random 128-bit session identity shared by the access and
        # refresh tokens of one session lineage. Refresh rotation passes
        # the existing jti forward, so a single revocation entry kills the
        # whole lineage (old + rotated tokens). Absent jti (legacy tokens)
        # just means "not revocable".
        session_jti = jti or secrets.token_urlsafe(16)
        access_token = _sign(
            {
                "sub": user_id,
                "kind": "access",
                "exp": exp,
                "jti": session_jti,
            },
            self._secret,
        )
        refresh_token = _sign(
            {
                "sub": user_id,
                "kind": "refresh",
                "exp": now + _REFRESH_TTL_SECONDS,
                "jti": session_jti,
            },
            self._secret,
        )
        return Session(
            user_id=user_id,
            email="",
            display_name=user_id,
            org_id="",
            provider=self.name,
            expires_at=exp,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    def _session_from_payload(
        self, access_token: str, refresh_token: str, payload: dict
    ) -> Session:
        user_id = str(payload.get("sub", ""))
        return Session(
            user_id=user_id,
            email="",
            display_name=user_id,
            org_id="",
            provider=self.name,
            expires_at=int(payload["exp"]),
            access_token=access_token,
            refresh_token=refresh_token,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_basic_auth_section() -> dict:
    """Return ``dashboard.basic_auth`` from config.yaml, or ``{}``.

    Robust to load_config() raising, the keys being absent, or the value
    not being a dict — every shape falls through to ``{}``.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-basic: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "basic_auth", default=None)
    return section if isinstance(section, dict) else {}


def _resolve(env_name: str, cfg_section: dict, cfg_key: str) -> str:
    """Env-wins-over-config resolution; empty env treated as unset."""
    env = os.environ.get(env_name, "").strip()
    if env:
        return env
    return str(cfg_section.get(cfg_key, "") or "").strip()


def _resolve_secret(cfg_section: dict) -> bytes:
    """Resolve the token-signing secret.

    Accepts base64 or hex or raw text from config/env. When unset,
    generates a random per-process secret (sessions then don't survive a
    restart or span multiple workers — logged at INFO).
    """
    raw = _resolve(
        "HERMES_DASHBOARD_BASIC_AUTH_SECRET", cfg_section, "secret"
    )
    if not raw:
        logger.info(
            "dashboard-auth-basic: no 'secret' configured; generating a "
            "random per-process signing key. Sessions will not survive a "
            "restart or span multiple workers. Set dashboard.basic_auth."
            "secret (or HERMES_DASHBOARD_BASIC_AUTH_SECRET) for stable "
            "sessions."
        )
        return secrets.token_bytes(32)
    # Try base64, then hex, then fall back to the raw UTF-8 bytes.
    for decoder in (base64.b64decode, bytes.fromhex):
        try:
            decoded = decoder(raw)
            if len(decoded) >= 16:
                return decoded
        except (ValueError, TypeError):
            pass
    return raw.encode("utf-8")


def _default_revocations_path() -> Path:
    """``$HERMES_HOME/state/dashboard-auth-basic-revocations.json``.

    Uses ``hermes_constants.get_hermes_home()`` (a leaf module — no import
    cycle) so profile overrides and the native-Windows fallback are
    honored, mirroring ``hermes_cli.dashboard_auth.audit``.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _REVOCATIONS_REL_PATH


def register(ctx) -> None:
    """Plugin entry — registers BasicAuthProvider when credentials exist.

    Loopback / ``--insecure`` operators and anyone using the OAuth
    provider leave ``dashboard.basic_auth`` unset, so this plugin is a
    no-op for them. When username + (password or password_hash) are
    configured, it registers a password provider that the login page
    renders as a credential form.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    section = _load_config_basic_auth_section()
    username = _resolve(
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME", section, "username"
    )
    password_hash = _resolve(
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH", section, "password_hash"
    )
    plaintext = _resolve(
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", section, "password"
    )
    ttl_raw = _resolve(
        "HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS", section, "session_ttl_seconds"
    )

    if not username:
        LAST_SKIP_REASON = (
            "dashboard.basic_auth.username is not set (and "
            "HERMES_DASHBOARD_BASIC_AUTH_USERNAME is empty). Set a username "
            "and a password (or password_hash) under dashboard.basic_auth in "
            "config.yaml to enable username/password dashboard login, or use "
            "the OAuth provider, or pass --insecure to skip the auth gate."
        )
        logger.debug("dashboard-auth-basic: %s", LAST_SKIP_REASON)
        return

    if not password_hash and not plaintext:
        LAST_SKIP_REASON = (
            "dashboard.basic_auth.username is set but neither password_hash "
            "nor password is configured. Provide one of them (password_hash "
            "is preferred — compute it with "
            "plugins.dashboard_auth.basic.hash_password)."
        )
        logger.warning("dashboard-auth-basic: %s", LAST_SKIP_REASON)
        return

    # Precedence (env-wins convention): a password supplied via the
    # HERMES_DASHBOARD_BASIC_AUTH_PASSWORD env var overrides a config.yaml
    # password_hash, so an operator can rotate the password by setting an
    # env var without editing config. A password_hash (precomputed) wins
    # over a config-only plaintext password at the same tier — it's the
    # preferred at-rest form. Concretely:
    #   * env password set        → hash it (overrides any config hash)
    #   * else config password_hash set → use it
    #   * else config plaintext password → hash it in-memory
    plaintext_from_env = os.environ.get(
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", ""
    ).strip()
    if plaintext_from_env:
        password_hash = hash_password(plaintext_from_env)
        logger.info(
            "dashboard-auth-basic: hashed env-supplied password in-memory "
            "(overrides any config password_hash)."
        )
    elif not password_hash:
        # config-only plaintext password.
        password_hash = hash_password(plaintext)
        logger.info(
            "dashboard-auth-basic: hashed plaintext password in-memory. "
            "For production, precompute dashboard.basic_auth.password_hash "
            "and remove the plaintext password from config."
        )

    secret = _resolve_secret(section)

    try:
        ttl = int(ttl_raw) if ttl_raw else _DEFAULT_TTL_SECONDS
    except ValueError:
        ttl = _DEFAULT_TTL_SECONDS

    try:
        provider = BasicAuthProvider(
            username=username,
            password_hash=password_hash,
            secret=secret,
            ttl_seconds=ttl,
            revocations_path=_default_revocations_path(),
        )
    except ValueError as exc:
        LAST_SKIP_REASON = f"BasicAuthProvider construction failed: {exc}"
        logger.warning("dashboard-auth-basic: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-basic: registered password provider (username=%s)",
        username,
    )
