"""LdapAuthProvider — LDAP / Active Directory dashboard auth (password login).

Plugs into the ``DashboardAuthProvider`` framework as a pure-password
provider (``supports_password = True``): the login page renders a
credential form, ``/auth/password-login`` calls
``complete_password_login``, and everything downstream (session cookies,
verify, refresh, WS tickets, logout, audit) is the shared framework path.

Credentials are verified with an **LDAP bind** — never stored, hashed, or
compared locally. Two mutually exclusive bind modes:

  * **Direct bind** — ``user_dn_template`` like
    ``uid={username},ou=people,dc=example,dc=com``. The username is
    RDN-escaped and substituted, then the provider binds as that DN with
    the supplied password. Simple; no service account.
  * **Search-then-bind** — a service account (``bind_dn`` +
    ``bind_password``; empty ``bind_dn`` = anonymous) searches
    ``user_search_base`` with ``user_search_filter`` (default
    ``(uid={username})``; use ``(sAMAccountName={username})`` for AD) for
    exactly one entry, then re-binds as the found DN. Email / display
    name come from the entry, and ``refresh_session`` re-checks the DN
    still exists, so a *deleted or moved* account is cut off at the
    next access-token expiry. A merely **disabled** account still
    exists at its DN, so the probe cannot see it: that session keeps
    refreshing until the refresh token expires.

Sessions are stateless HMAC-signed tokens minted by this provider (same
scheme as ``plugins/dashboard_auth/basic``): ``verify_session`` — called
on every request — never touches the directory. Only login and (search
mode) refresh do LDAP I/O, and both carry connect/receive timeouts.

Security invariants:
  * An **empty or whitespace password is rejected before any bind** —
    LDAP servers treat an empty password as a successful *anonymous*
    bind, so skipping this check would be a full auth bypass.
  * Usernames are escaped per RFC 4515 (search filters) / RFC 4514
    (DN templates) before interpolation — no LDAP injection.
  * ``ldaps://`` or StartTLS is required unless ``allow_insecure`` is
    set explicitly; certificate validation is on by default.
  * Unknown-user vs wrong-password is never distinguished; on unknown
    user (search mode) a dummy bind equalises timing.
  * **Referral chasing is off** (``auto_referrals=False``): ldap3's
    default follows a directory-supplied referral and re-binds with the
    SAME credentials — the service account, or the end user's own DN and
    password — to a host named by directory data, and a plain ``ldap://``
    referral from an ``ldaps://`` deployment would bind in the clear.

Configuration surfaces (env wins over config.yaml when set non-empty),
mirroring the ``basic`` provider's precedence convention:

  ``config.yaml`` — canonical surface::

      dashboard:
        ldap_auth:
          server_url: ldaps://ldap.example.com        # required
          # EITHER (direct bind):
          user_dn_template: "uid={username},ou=people,dc=example,dc=com"
          # OR (search-then-bind):
          bind_dn: "cn=hermes,ou=svc,dc=example,dc=com"   # empty = anonymous search
          bind_password: "..."
          user_search_base: "ou=people,dc=example,dc=com"
          user_search_filter: "(uid={username})"      # optional (default shown)
          # Optional hardening / shaping:
          require_group: "cn=hermes-users,ou=groups,dc=example,dc=com"
          start_tls: false
          allow_insecure: false                       # permit plain ldap://
          ca_certs_file: /etc/ssl/private-ca.pem
          email_attribute: mail
          display_name_attribute: cn
          display_name: "LDAP"                        # login-form label
          secret: "<32+ random bytes, base64 or hex>" # token-signing key
          session_ttl_seconds: 43200                  # 12h access tokens
          refresh_ttl_seconds: 2592000                # 30d refresh tokens
          timeout_seconds: 5
          verify_user_on_refresh: true                # search mode only

  Environment overrides::

      HERMES_DASHBOARD_LDAP_SERVER_URL
      HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE
      HERMES_DASHBOARD_LDAP_BIND_DN
      HERMES_DASHBOARD_LDAP_BIND_PASSWORD
      HERMES_DASHBOARD_LDAP_USER_SEARCH_BASE
      HERMES_DASHBOARD_LDAP_USER_SEARCH_FILTER
      HERMES_DASHBOARD_LDAP_REQUIRE_GROUP
      HERMES_DASHBOARD_LDAP_START_TLS          # "1"/"true" to enable
      HERMES_DASHBOARD_LDAP_ALLOW_INSECURE     # "1"/"true" to enable
      HERMES_DASHBOARD_LDAP_CA_CERTS_FILE
      HERMES_DASHBOARD_LDAP_SECRET
      HERMES_DASHBOARD_LDAP_TTL_SECONDS

The ``ldap3`` dependency is lazy-installed via ``tools/lazy_deps.py``
("auth.ldap") only when the plugin is actually configured; this module
never imports ldap3 at import time.

Skip reasons:
  Like the other bundled providers, ``register()`` resolves its kwargs in
  ``_settings()`` (declining with ``SkipRegistration``) and delegates the
  construct/register/skip bookkeeping to
  ``plugins.dashboard_auth._shared.register_provider``, which stores the
  operator-facing text in module-level ``LAST_SKIP_REASON`` for the auth
  gate's fail-closed diagnostics.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Callable, Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCredentialsError,
    ProviderError,
    RefreshExpiredError,
    Session,
)
from plugins.dashboard_auth._shared import (
    NonInteractiveMixin,
    SkipRegistration,
    load_config_section,
    register_provider,
    resolve_env_or_cfg,
)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-ldap"


_DEFAULT_TTL_SECONDS = 12 * 60 * 60  # 12h access tokens
_DEFAULT_REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60  # 30d refresh tokens
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_USER_SEARCH_FILTER = "(uid={username})"

# Fixed-length HMAC-SHA256 suffix on signed tokens (same scheme as the
# ``basic`` provider — binary HMAC bytes can't collide with a delimiter).
_SIG_LEN = hashlib.sha256().digest_size

# Nonexistent DN used to equalise timing when a search finds no user:
# we still attempt one bind so "unknown user" and "wrong password" cost
# roughly the same wall-clock at this endpoint.
_DUMMY_BIND_DN = "uid=hermes-nonexistent-timing-pad,dc=invalid"

LAST_SKIP_REASON: str = ""


# ---------------------------------------------------------------------------
# Token signing (stateless HMAC-signed blobs — same scheme as `basic`)
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
# Provider
# ---------------------------------------------------------------------------


class LdapAuthProvider(NonInteractiveMixin, DashboardAuthProvider):
    """LDAP-bind password provider with stateless HMAC-signed sessions."""

    name = "ldap"
    display_name = "LDAP"
    supports_password = True
    _NOT_INTERACTIVE = (
        "LdapAuthProvider is password-only; use complete_password_login."
    )
    _NO_START_LOGIN = (
        "LdapAuthProvider is password-only; there is no OAuth redirect "
        "flow. The login page POSTs to /auth/password-login instead."
    )

    def __init__(
        self,
        *,
        server_url: str,
        secret: bytes,
        user_dn_template: str = "",
        bind_dn: str = "",
        bind_password: str = "",
        user_search_base: str = "",
        user_search_filter: str = _DEFAULT_USER_SEARCH_FILTER,
        require_group: str = "",
        email_attribute: str = "mail",
        display_name_attribute: str = "cn",
        display_name: str = "LDAP",
        start_tls: bool = False,
        allow_insecure: bool = False,
        ca_certs_file: str = "",
        session_ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        refresh_ttl_seconds: int = _DEFAULT_REFRESH_TTL_SECONDS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        verify_user_on_refresh: bool = True,
        connection_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not (
            server_url.startswith("ldap://")
            or server_url.startswith("ldaps://")
        ):
            raise ValueError(
                "server_url must start with ldap:// or ldaps:// "
                f"(got {server_url!r})"
            )
        if (
            server_url.startswith("ldap://")
            and not start_tls
            and not allow_insecure
        ):
            raise ValueError(
                "plain ldap:// without start_tls sends passwords in "
                "cleartext; set start_tls: true, use ldaps://, or set "
                "allow_insecure: true to accept the risk explicitly"
            )
        if len(secret) < 16:
            raise ValueError("secret must be at least 16 bytes")
        if user_dn_template and user_search_base:
            raise ValueError(
                "user_dn_template (direct bind) and user_search_base "
                "(search-then-bind) are mutually exclusive — configure "
                "exactly one"
            )
        if not user_dn_template and not user_search_base:
            raise ValueError(
                "no bind mode configured: set user_dn_template (direct "
                "bind) or user_search_base (search-then-bind)"
            )
        if user_dn_template and "{username}" not in user_dn_template:
            raise ValueError(
                "user_dn_template must contain the {username} placeholder"
            )
        if user_search_base and "{username}" not in user_search_filter:
            raise ValueError(
                "user_search_filter must contain the {username} placeholder"
            )
        # Containing {username} isn't enough: a stray second placeholder
        # ("uid={username},ou={dept},...") passes the checks above and
        # then raises KeyError from .format() at the FIRST login — a 500
        # on the login endpoint, long after startup. Trial-format both
        # templates now so a bad one is a construction error (and hence a
        # register() skip reason) instead. Literal braces are vanishingly
        # rare in DNs and filters; treating a format failure as a config
        # error is the intent.
        for label, template in (
            ("user_dn_template", user_dn_template),
            # Only meaningful in search mode; the direct-bind path never
            # formats the (defaulted) filter.
            ("user_search_filter", user_search_filter if user_search_base else ""),
        ):
            if not template:
                continue
            try:
                template.format(username="hermes-placeholder-probe")
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"{label} contains unsupported placeholder(s); only "
                    f"{{username}} is allowed: {exc}"
                ) from exc

        self._server_url = server_url
        self._secret = secret
        self._user_dn_template = user_dn_template
        self._bind_dn = bind_dn
        self._bind_password = bind_password
        self._user_search_base = user_search_base
        self._user_search_filter = user_search_filter
        self._require_group = require_group
        self._email_attr = email_attribute
        self._display_attr = display_name_attribute
        self.display_name = display_name or "LDAP"
        self._start_tls = start_tls
        self._ca_certs_file = ca_certs_file
        self._ttl = max(60, int(session_ttl_seconds))
        self._refresh_ttl = max(300, int(refresh_ttl_seconds))
        self._timeout = float(timeout_seconds)
        self._verify_user_on_refresh = bool(verify_user_on_refresh)
        self._factory = connection_factory or self._default_factory

    # ---- OAuth methods: not used (pure-password provider) ------------------
    # start_login / complete_login come from NonInteractiveMixin and raise
    # NotImplementedError with the messages above.

    # ---- password login ------------------------------------------------------

    def complete_password_login(
        self, *, username: str, password: str
    ) -> Session:
        username = (username or "").strip()
        # SECURITY: an empty password is an ANONYMOUS bind on LDAP servers
        # (RFC 4513 §5.1.2) and would "succeed" — reject before any bind.
        if not username or not password or not password.strip():
            raise InvalidCredentialsError("invalid username or password")

        if self._user_dn_template:
            from ldap3.utils.dn import escape_rdn

            user_dn = self._user_dn_template.format(
                username=escape_rdn(username)
            )
            attrs: dict = {}
        else:
            user_dn, attrs = self._search_user(username)
            if user_dn is None:
                # Timing pad: unknown-user should cost about the same as
                # wrong-password, so attempt one bind against a fixed
                # nonexistent DN before rejecting.
                pad = self._bind(user=_DUMMY_BIND_DN, password=password)
                if pad is not None:  # pragma: no cover — defensive
                    pad.unbind()
                raise InvalidCredentialsError("invalid username or password")

        conn = self._bind(user=user_dn, password=password)
        if conn is None:
            raise InvalidCredentialsError("invalid username or password")
        try:
            if self._require_group and not self._user_in_group(
                conn, user_dn, username
            ):
                raise InvalidCredentialsError(
                    "invalid username or password"
                )
        finally:
            try:
                conn.unbind()
            except Exception:  # noqa: BLE001 — teardown must not mask errors
                pass
        return self._mint_session(username, user_dn, attrs)

    # ---- internals: LDAP I/O -------------------------------------------------

    def _default_factory(
        self, *, user: Optional[str], password: Optional[str]
    ):
        """Build a real ldap3 Connection (unbound) with TLS + timeouts.

        Imported lazily — ldap3 is only installed once the plugin is
        configured (tools/lazy_deps.py "auth.ldap").
        """
        import ssl

        import ldap3

        tls = None
        if self._server_url.startswith("ldaps://") or self._start_tls:
            tls = ldap3.Tls(
                validate=ssl.CERT_REQUIRED,
                ca_certs_file=self._ca_certs_file or None,
            )
        server = ldap3.Server(
            self._server_url,
            connect_timeout=self._timeout,
            tls=tls,
            get_info=ldap3.NONE,
        )
        conn = ldap3.Connection(
            server,
            user=user or None,
            password=password or None,
            client_strategy=ldap3.SYNC,
            # Must be an int: on POSIX ldap3 2.9.1 packs this with
            # struct.pack('LL', ...) for SO_RCVTIMEO, and a float raises
            # struct.error before the connection ever opens.
            receive_timeout=max(1, int(self._timeout)),
            raise_exceptions=False,
            auto_bind=False,
            # never replay credentials to a host named by directory data
            auto_referrals=False,
        )
        if self._start_tls:
            # open() connects the socket; if the StartTLS upgrade then
            # fails, ldap3 leaves that socket open (strategy/base.py
            # raises without closing). Close it ourselves or every failed
            # login against a TLS-broken directory leaks an fd.
            try:
                conn.open()
                conn.start_tls()
            except Exception:
                try:
                    conn.unbind()
                except Exception:  # noqa: BLE001 — cleanup must not mask
                    pass
                raise
        return conn

    def _bind(
        self, *, user: Optional[str], password: Optional[str]
    ):
        """Create a connection and bind. Bound connection, or None if the
        server rejected the credentials. ProviderError on transport
        failure (unreachable, TLS failure, timeout)."""
        from ldap3.core.exceptions import LDAPException

        conn = None
        try:
            conn = self._factory(user=user, password=password)
            ok = conn.bind()
        except LDAPException as exc:
            # The connection may already own a connected (or half-open)
            # socket: ldap3 2.9.1 raises LDAPSocketOpenError from a failed
            # connect() / TLS wrap without closing it. Close it here, or a
            # down directory leaks one fd per hit on this unauthenticated
            # endpoint.
            if conn is not None:
                try:
                    conn.unbind()
                except Exception:  # noqa: BLE001 — cleanup must not mask
                    pass
            raise ProviderError(f"LDAP server unreachable: {exc}") from exc
        if not ok:
            try:
                conn.unbind()
            except Exception:  # noqa: BLE001
                pass
            return None
        return conn

    # ---- session lifecycle (stateless tokens — no LDAP I/O) ----------------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        payload = _unsign(access_token, self._secret)
        if (
            payload is None
            or payload.get("kind") != "access"
            or payload.get("exp", 0) <= int(time.time())
        ):
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
        # Search mode: re-check the account still exists in the directory
        # so a deleted-or-moved user is cut off at access-token expiry
        # instead of riding the 30-day refresh horizon. A merely disabled
        # account still exists at its DN, so this probe cannot see it.
        # Direct mode has no service credentials to search with, so
        # refresh is token-only there (documented tradeoff).
        if self._verify_user_on_refresh and self._user_search_base:
            if not self._user_still_present(str(payload.get("dn", ""))):
                raise RefreshExpiredError(
                    "user no longer present in directory"
                )
        return self._mint_session(
            str(payload.get("sub", "")),
            str(payload.get("dn", "")),
            {
                "email": str(payload.get("em", "")),
                "display": str(payload.get("nm", "")),
            },
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        # Stateless tokens — nothing to revoke server-side. Best-effort
        # no-op, must not raise.
        _ = refresh_token
        return None

    # ---- internals: token minting ------------------------------------------

    def _mint_session(
        self, username: str, user_dn: str, attrs: dict
    ) -> Session:
        now = int(time.time())
        exp = now + self._ttl
        email = str(attrs.get("email", "") or "")
        display = str(attrs.get("display", "") or "") or username
        access_token = _sign(
            {
                "sub": username, "dn": user_dn, "em": email, "nm": display,
                "kind": "access", "exp": exp,
            },
            self._secret,
        )
        refresh_token = _sign(
            {
                "sub": username, "dn": user_dn, "em": email, "nm": display,
                "kind": "refresh", "exp": now + self._refresh_ttl,
            },
            self._secret,
        )
        return Session(
            user_id=username,
            email=email,
            display_name=display,
            org_id="",
            provider=self.name,
            expires_at=exp,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    def _session_from_payload(
        self, access_token: str, refresh_token: str, payload: dict
    ) -> Session:
        username = str(payload.get("sub", ""))
        return Session(
            user_id=username,
            email=str(payload.get("em", "")),
            display_name=str(payload.get("nm", "")) or username,
            org_id="",
            provider=self.name,
            expires_at=int(payload["exp"]),
            access_token=access_token,
            refresh_token=refresh_token,
        )

    # ---- internals: LDAP I/O (implemented in later tasks) -------------------

    def _search_user(self, username: str):
        """Search-then-bind leg 1: find the user's DN + profile attributes.

        Returns ``(user_dn, {"email": ..., "display": ...})`` on a unique
        match, ``(None, {})`` when no entry matches. Zero matches and
        multiple matches are both treated as "not found" — a filter that
        matches several entries must never let a bind against ANY of them
        succeed. Raises ``ProviderError`` when the service bind is
        rejected or the directory is unreachable — both mean *we* cannot
        verify anyone, which is an operator problem (503), not a
        credentials problem (401).
        """
        import ldap3
        from ldap3.core.exceptions import LDAPException
        from ldap3.utils.conv import escape_filter_chars

        conn = self._bind(
            user=self._bind_dn or None,
            password=self._bind_password or None,
        )
        if conn is None:
            raise ProviderError(
                "LDAP service-account bind was rejected — check "
                "dashboard.ldap_auth.bind_dn / bind_password"
            )
        try:
            flt = self._user_search_filter.format(
                username=escape_filter_chars(username)
            )
            ok = conn.search(
                search_base=self._user_search_base,
                search_filter=flt,
                search_scope=ldap3.SUBTREE,
                attributes=[self._email_attr, self._display_attr],
            )
            if ok:
                entries = list(conn.entries)
            else:
                # A falsy search is an LDAP *result* code, not a
                # transport error (raise_exceptions=False) — typically
                # insufficientAccessRights on the service account or a
                # user_search_base that doesn't exist. Every login then
                # gets a generic 401, so say so once here or the
                # misconfiguration is invisible. No username, no
                # password: this line goes to ordinary server logs.
                entries = []
                logger.warning(
                    "dashboard-auth-ldap: user search under %r failed "
                    "(result=%r) — every login will be rejected. Check "
                    "dashboard.ldap_auth.user_search_base and the service "
                    "account's read access.",
                    self._user_search_base,
                    getattr(conn, "result", None),
                )
        except LDAPException as exc:
            raise ProviderError(f"LDAP search failed: {exc}") from exc
        finally:
            try:
                conn.unbind()
            except Exception:  # noqa: BLE001
                pass

        if len(entries) != 1:
            if len(entries) > 1:
                logger.warning(
                    "dashboard-auth-ldap: user_search_filter matched %d "
                    "entries for a single username — rejecting login. "
                    "Tighten dashboard.ldap_auth.user_search_filter.",
                    len(entries),
                )
            return None, {}

        entry = entries[0]

        def _first(attr_name: str) -> str:
            try:
                val = entry[attr_name].value
            except Exception:  # noqa: BLE001 — attribute absent
                return ""
            if isinstance(val, (list, tuple)):
                val = val[0] if val else ""
            return str(val or "")

        return entry.entry_dn, {
            "email": _first(self._email_attr),
            "display": _first(self._display_attr),
        }

    def _user_in_group(self, conn, user_dn: str, username: str) -> bool:
        """BASE-scope membership probe on the require_group entry.

        Covers the three common group schemas in one filter:
        groupOfNames (member), groupOfUniqueNames (uniqueMember), and
        posixGroup (memberUid — matched by username, not DN). Runs on the
        user's own freshly-bound connection in both modes, so the
        directory ACLs must let authenticated users read the group entry
        (the default on OpenLDAP and AD).
        """
        import ldap3
        from ldap3.core.exceptions import LDAPException
        from ldap3.utils.conv import escape_filter_chars

        dn_esc = escape_filter_chars(user_dn)
        uid_esc = escape_filter_chars(username)
        flt = (
            f"(|(member={dn_esc})"
            f"(uniqueMember={dn_esc})"
            f"(memberUid={uid_esc}))"
        )
        try:
            ok = conn.search(
                search_base=self._require_group,
                search_filter=flt,
                search_scope=ldap3.BASE,
                attributes=[],
            )
        except LDAPException as exc:
            raise ProviderError(
                f"LDAP group check failed: {exc}"
            ) from exc
        if not ok:
            # Same trap as the user search: an ACL that hides the group
            # entry from ordinary users makes this probe fail for
            # EVERYONE, and the verdict is the generic 401 of a wrong
            # password. Log the group DN and the result code only — no
            # username, no user DN, no password.
            logger.warning(
                "dashboard-auth-ldap: require_group membership probe on "
                "%r failed (result=%r) — every login will be rejected. "
                "Check the DN exists and authenticated users may read it.",
                self._require_group,
                getattr(conn, "result", None),
            )
        return bool(ok and conn.entries)

    def _user_still_present(self, user_dn: str) -> bool:
        """Refresh-time existence probe (search mode only).

        BASE-scope search on the user's DN with the service account.
        False → the account was deleted/moved (caller raises
        RefreshExpiredError). ProviderError propagates when the directory
        is unreachable — per the framework contract the middleware then
        503s without clearing cookies.
        """
        if not user_dn:
            return False
        import ldap3
        from ldap3.core.exceptions import (
            LDAPException,
            LDAPNoSuchObjectResult,
        )

        conn = self._bind(
            user=self._bind_dn or None,
            password=self._bind_password or None,
        )
        if conn is None:
            raise ProviderError(
                "LDAP service-account bind was rejected during refresh"
            )
        try:
            ok = conn.search(
                search_base=user_dn,
                search_filter="(objectClass=*)",
                search_scope=ldap3.BASE,
                attributes=[],
            )
            # The normal "user gone" path: connections are built with
            # raise_exceptions=False, so an LDAP *result* code such as
            # noSuchObject comes back as a falsy search, not an exception.
            return bool(ok and conn.entries)
        except LDAPNoSuchObjectResult:
            # Same verdict for a connection_factory that opted into
            # raise_exceptions=True, where the result code is raised.
            return False
        except LDAPException as exc:
            # Everything else that raises here is transport-level — the
            # socket died mid-refresh, the receive timed out, the session
            # was terminated. That is an outage, NOT a deleted account:
            # returning False would make the caller raise
            # RefreshExpiredError and log the user out. The contract says
            # an unreachable directory must 503 with cookies intact.
            raise ProviderError(
                f"LDAP refresh check failed: {exc}"
            ) from exc
        finally:
            try:
                conn.unbind()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_ldap_section() -> dict:
    """Return ``dashboard.ldap_auth`` from config.yaml, or ``{}``.

    Robust to load_config() raising, the keys being absent, or the value
    not being a dict — every shape falls through to ``{}``. Kept as a
    module-level function so tests can monkeypatch the config surface.
    """
    return load_config_section(logger, _TAG, "dashboard", "ldap_auth")


def _resolve_bool(env_name: str, cfg_value: Any) -> bool:
    """Truthy-string resolution on top of ``resolve_env_or_cfg``."""
    return resolve_env_or_cfg(env_name, cfg_value).lower() in (
        "1", "true", "yes", "on"
    )


def _resolve_secret(cfg_section: dict) -> bytes:
    """Resolve the token-signing secret (base64, hex, or raw text).

    When unset, generates a random per-process secret — sessions then
    don't survive a restart or span multiple workers (logged at INFO).
    """
    raw = resolve_env_or_cfg(
        "HERMES_DASHBOARD_LDAP_SECRET", cfg_section.get("secret")
    )
    if not raw:
        logger.info(
            "dashboard-auth-ldap: no 'secret' configured; generating a "
            "random per-process signing key. Sessions will not survive a "
            "restart or span multiple workers. Set dashboard.ldap_auth."
            "secret (or HERMES_DASHBOARD_LDAP_SECRET) for stable sessions."
        )
        return secrets.token_bytes(32)
    for decoder in (base64.b64decode, bytes.fromhex):
        try:
            decoded = decoder(raw)
            if len(decoded) >= 16:
                return decoded
        except (ValueError, TypeError):
            pass
    return raw.encode("utf-8")


def _ensure_ldap3() -> None:
    """Lazy-install ldap3 (tools/lazy_deps.py 'auth.ldap').

    Raises on failure — ``_settings()`` converts that into a
    ``SkipRegistration``. Split out as a module function so tests can
    stub it.
    """
    from tools.lazy_deps import ensure

    ensure("auth.ldap", prompt=False)


def _settings() -> dict:
    """Resolve LdapAuthProvider kwargs from env/config; raises ``SkipRegistration``."""
    section = _load_config_ldap_section()

    def setting(env_name: str, cfg_key: str) -> str:
        return resolve_env_or_cfg(env_name, section.get(cfg_key, ""))

    server_url = setting("HERMES_DASHBOARD_LDAP_SERVER_URL", "server_url")
    if not server_url:
        raise SkipRegistration(
            "dashboard.ldap_auth.server_url is not set (and "
            "HERMES_DASHBOARD_LDAP_SERVER_URL is empty). Set it plus a "
            "bind mode (user_dn_template, or bind_dn/bind_password + "
            "user_search_base) under dashboard.ldap_auth in config.yaml "
            "to enable LDAP dashboard login."
        )

    user_dn_template = setting(
        "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE", "user_dn_template"
    )
    user_search_base = setting(
        "HERMES_DASHBOARD_LDAP_USER_SEARCH_BASE", "user_search_base"
    )
    if not user_dn_template and not user_search_base:
        raise SkipRegistration(
            "dashboard.ldap_auth.server_url is set but no bind mode is "
            "configured. Set user_dn_template (direct bind) OR "
            "user_search_base [+ bind_dn/bind_password] (search-then-bind).",
            level="warning",
        )

    # After the two config gates on purpose: an unconfigured install must
    # never import or lazy-install ldap3.
    try:
        _ensure_ldap3()
    except Exception as exc:  # noqa: BLE001 — FeatureUnavailable et al.
        raise SkipRegistration(
            f"the ldap3 dependency is not available and could not be "
            f"lazy-installed: {exc}. Install it manually: "
            f"uv pip install 'ldap3==2.9.1' (or: pip install ldap3==2.9.1).",
            level="warning",
        ) from exc

    ttl_raw = setting(
        "HERMES_DASHBOARD_LDAP_TTL_SECONDS", "session_ttl_seconds"
    )
    try:
        ttl = int(ttl_raw) if ttl_raw else _DEFAULT_TTL_SECONDS
    except ValueError:
        ttl = _DEFAULT_TTL_SECONDS
    try:
        refresh_ttl = int(
            str(section.get("refresh_ttl_seconds", "") or "").strip()
            or _DEFAULT_REFRESH_TTL_SECONDS
        )
    except ValueError:
        refresh_ttl = _DEFAULT_REFRESH_TTL_SECONDS
    try:
        timeout = float(
            str(section.get("timeout_seconds", "") or "").strip()
            or _DEFAULT_TIMEOUT_SECONDS
        )
    except ValueError:
        timeout = _DEFAULT_TIMEOUT_SECONDS
    verify_on_refresh_raw = section.get("verify_user_on_refresh", True)
    verify_on_refresh = (
        str(verify_on_refresh_raw).lower() not in ("0", "false", "no", "off")
    )

    return {
        "server_url": server_url,
        "secret": _resolve_secret(section),
        "user_dn_template": user_dn_template,
        "bind_dn": setting("HERMES_DASHBOARD_LDAP_BIND_DN", "bind_dn"),
        "bind_password": setting(
            "HERMES_DASHBOARD_LDAP_BIND_PASSWORD", "bind_password"
        ),
        "user_search_base": user_search_base,
        "user_search_filter": setting(
            "HERMES_DASHBOARD_LDAP_USER_SEARCH_FILTER", "user_search_filter"
        ) or _DEFAULT_USER_SEARCH_FILTER,
        "require_group": setting(
            "HERMES_DASHBOARD_LDAP_REQUIRE_GROUP", "require_group"
        ),
        "email_attribute": str(
            section.get("email_attribute", "") or ""
        ).strip() or "mail",
        "display_name_attribute": str(
            section.get("display_name_attribute", "") or ""
        ).strip() or "cn",
        "display_name": str(
            section.get("display_name", "") or ""
        ).strip() or "LDAP",
        "start_tls": _resolve_bool(
            "HERMES_DASHBOARD_LDAP_START_TLS", section.get("start_tls", "")
        ),
        "allow_insecure": _resolve_bool(
            "HERMES_DASHBOARD_LDAP_ALLOW_INSECURE",
            section.get("allow_insecure", ""),
        ),
        "ca_certs_file": setting(
            "HERMES_DASHBOARD_LDAP_CA_CERTS_FILE", "ca_certs_file"
        ),
        "session_ttl_seconds": ttl,
        "refresh_ttl_seconds": refresh_ttl,
        "timeout_seconds": timeout,
        "verify_user_on_refresh": verify_on_refresh,
    }


def register(ctx) -> None:
    """Plugin entry — registers LdapAuthProvider when configured.

    A no-op (with a diagnostic skip reason) unless dashboard.ldap_auth
    provides a server_url plus exactly one bind mode. ldap3 is only
    installed once that configuration exists, so unconfigured installs
    never pay for the dependency.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(
        ctx, logger, _TAG, LdapAuthProvider, _settings
    )
    if kwargs is not None:
        logger.info(
            "dashboard-auth-ldap: registered LDAP password provider "
            "(server=%s, mode=%s%s)",
            kwargs["server_url"],
            "direct-bind" if kwargs["user_dn_template"] else "search-then-bind",
            f", require_group={kwargs['require_group']}"
            if kwargs["require_group"]
            else "",
        )
