"""WebauthnDashboardAuthProvider — passkey (WebAuthn) dashboard auth.

A credential provider with no password form and no OAuth redirect:
``GET /auth/webauthn/options`` -> browser ``navigator.credentials.get()`` ->
``POST /auth/webauthn/verify`` -> session. Sessions ride the SHARED stateless
HMAC seam (``plugins/dashboard_auth/_shared::sign_session_token``) — bit-identical
``{sub, kind, exp}`` tokens to the basic provider's password login, so cookies,
middleware verify/refresh/rotation and the native broker tail are unchanged
(``cookies.py`` untouched).

Credential scope: enrolled passkeys live in ``dashboard.webauthn.credential_store``
(default ``~/.hermes/dashboard_webauthn.json``, 0600, public-key material only,
atomic writes, fail-closed load — integrity, not secrecy). Ceremony challenges
are an in-memory TTL store (``hermes_cli/dashboard_auth/webauthn_state.py``,
native_flow-shaped; documented single-process envelope).

Offering (providers list / passkey button / ceremonies) is ORIGIN-GATED:
``webauthn_offered_for_request`` matches the request origin against the
configured origins, so plain ``http://<LAN-IP>`` binds keep the exact
password-only status quo and the button never half-appears (design doc §11).

Config (config.yaml only — AGENTS.md ``.env``-is-secrets-only; ``secret`` may
ride ``.env`` because it is a credential): ``dashboard.webauthn.{rp_id, rp_name,
origins, allowed_credential_ids, user_verification, session_ttl_seconds,
credential_store, challenge_ttl_seconds, secret}``.

Dependency: Yubico ``fido2`` via the optional ``webauthn`` extra
(``pip install 'hermes-agent[webauthn]'``); the import stays lazy so plugin
discovery is cheap for installs that don't use it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider, InvalidCredentialsError, ProviderError, RefreshExpiredError, Session)
from hermes_cli.dashboard_auth import webauthn_state
from plugins.dashboard_auth._shared import (
    NonInteractiveMixin, SkipRegistration, load_config_section, register_provider,
    resolve_session_secret, sign_session_token, unsign_session_token)
from plugins.dashboard_auth.webauthn import crypto, store

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-webauthn"

# Access-token TTL default: mirrors basic's 12h (the middleware transparently
# refreshes via the 30-day RT; the TTL controls refresh cadence, not session
# length). Config ``dashboard.webauthn.session_ttl_seconds`` overrides.
_DEFAULT_TTL_SECONDS = 12 * 60 * 60
_REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60
_USER_ID = "hermes-operator"
_CHALLENGE_TTL_DEFAULT = 120
_CHALLENGE_TTL_MIN = 30
_CHALLENGE_TTL_MAX = 300

LAST_SKIP_REASON: str = ""


def _normalize_origin(raw: str) -> Optional[str]:
    """Canonical ``scheme://host[:nondefault-port]`` or None. Default ports are
    elided so ``https://host:443`` (or a Host header carrying the port) compares
    equal to the bare origin."""
    try:
        parts = urlsplit(str(raw).strip())
    except ValueError:
        return None
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None
    port = parts.port
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        port = None
    host = parts.hostname
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{scheme}://{netloc}"


def request_origin(request) -> str:
    """The origin a request came in as (scheme + authority, default-port elided).

    ``request.url`` already reflects ``X-Forwarded-Proto/Host`` because the
    dashboard mounts uvicorn's ``proxy_headers`` middleware, so a TLS-terminating
    edge presents the public origin here — the origin the passkey ceremony is
    exercised from.
    """
    return _normalize_origin(f"{request.url.scheme}://{request.url.netloc}") or ""


def _fido2_or_skip() -> None:
    """Import ``fido2`` or raise a SkipRegistration with the exact install fix."""
    try:
        import fido2
    except ImportError as exc:
        raise SkipRegistration(
            "dashboard.webauthn is configured but the 'fido2' library is not "
            "installed. Install the optional extra: pip install 'hermes-agent[webauthn]' "
            "(fido2>=2.2.1,<3), or restart the dashboard once — its start-up "
            "provisions the extra from this block automatically.", level="warning") from exc


def _resolve_origins(section: dict, rp_id: str) -> tuple:
    """Validated, canonicalized origins; fail-loud (``SkipRegistration`` warning)
    when anything could never satisfy the W3C RP-ID↔origin pairing via fido2's
    ``rpid.verify_rp_id`` (design doc §4/§9.2). A config the browser can never
    present must not half-enable the provider — RP ID is frozen per credential,
    so a bad pairing bricks keys AFTER enrolment and fails before any key is
    minted instead."""
    raw_origins = section.get("origins")
    if isinstance(raw_origins, str):
        raw_origins = [raw_origins]
    if not isinstance(raw_origins, list) or not raw_origins:
        raise SkipRegistration(
            "dashboard.webauthn.origins must be a non-empty list of the exact origins "
            "the browser will present (scheme + host[:port], e.g. "
            "'https://dashboard.example.com'). The SSH-tunnel path is "
            "'http://localhost:9119' — localhost is a secure context even over plain http.",
            level="warning")
    origins: list[str] = []
    for raw in raw_origins:
        canonical = _normalize_origin(raw)
        if canonical is None:
            raise SkipRegistration(
                f"dashboard.webauthn.origins entry {raw!r} is not a valid origin "
                "(use scheme://host[:port]).", level="warning")
        host = urlsplit(canonical).hostname or ""
        if canonical.startswith("http://") and host != "localhost" and not host.endswith(".localhost"):
            raise SkipRegistration(
                f"dashboard.webauthn.origins entry {raw!r} is plain http on a "
                "non-localhost host. WebAuthn requires a secure context: use "
                "https:// (reverse proxy or 'tailscale serve'), or the SSH-tunnel "
                "path 'http://localhost:<port>'.", level="warning")
        if not crypto.verify_rp_id(rp_id, canonical):
            raise SkipRegistration(
                f"dashboard.webauthn.origins entry {raw!r} does not pair with rp_id "
                f"{rp_id!r}: the RP ID must equal the origin's host or be a "
                "registrable suffix of it. Fix rp_id or the origin; a mismatched "
                "pairing cannot work after enrolment.", level="warning")
        if canonical not in origins:
            origins.append(canonical)
    return tuple(origins)


def _resolve_challenge_ttl(section: dict) -> int:
    raw = section.get("challenge_ttl_seconds", 0)
    try:
        ttl = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "dashboard-auth-webauthn: unreadable challenge_ttl_seconds %r; using %ds",
            raw, _CHALLENGE_TTL_DEFAULT)
        return _CHALLENGE_TTL_DEFAULT
    if ttl == 0:  # unset -> plugin default (config_defaults documents 0 = default)
        return _CHALLENGE_TTL_DEFAULT
    return max(_CHALLENGE_TTL_MIN, min(_CHALLENGE_TTL_MAX, ttl))


def _resolve_ttl(section: dict) -> int:
    try:
        ttl = int(section.get("session_ttl_seconds", 0))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_SECONDS
    return max(60, ttl)


def _resolve_uv(section: dict) -> str:
    uv = str(section.get("user_verification", "preferred")).strip().lower() or "preferred"
    if uv not in ("preferred", "required"):
        raise SkipRegistration(
            f"dashboard.webauthn.user_verification {uv!r} is not supported; use "
            "'preferred' or 'required'.", level="warning")
    return uv


def _resolve_allowed_ids(section: dict) -> tuple:
    raw = section.get("allowed_credential_ids")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise SkipRegistration(
            "dashboard.webauthn.allowed_credential_ids must be a list of base64url "
            "credential ids (the browser's rawId).", level="warning")
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _load_records(store_path, rp_id: str) -> list:
    """Fail-closed store read; raises SkipRegistration (warning) rather than
    half-trusting a corrupt/tampered file."""
    try:
        doc = store.load(store_path)
    except store.StoreError as exc:
        raise SkipRegistration(str(exc), level="warning") from exc
    if doc.get("rp_id") and doc.get("rp_id") != rp_id:
        raise SkipRegistration(
            f"credential store {store_path} was written for rp_id {doc.get('rp_id')!r} "
            f"but the configured rp_id is {rp_id!r}. RP ID is frozen per credential — "
            "restore the original rp_id, or reset enrolment (delete the store) and "
            "re-enrol.", level="warning")
    for record in doc.get("credentials") or []:
        if not isinstance(record, dict) or not record.get("credential_id") or not record.get("public_key"):
            raise SkipRegistration(
                f"credential store {store_path} holds a malformed credential entry; "
                "failing closed rather than half-trusting it.", level="warning")
    return [record for record in doc.get("credentials") or []]


def _resolve_secret(section: dict) -> bytes:
    """Session-signing secret: rides the basic-auth deployment secret first
    (design doc §7 — the passkey provider mints basic-shaped tokens), then the
    webauthn credential/config twin, then a random per-process key (logged)."""
    basic_section = load_config_section(logger, _TAG, "dashboard", "basic_auth")
    return resolve_session_secret(
        tag=_TAG, logger=logger, candidates=[
            ("HERMES_DASHBOARD_BASIC_AUTH_SECRET", basic_section.get("secret")),
            ("HERMES_DASHBOARD_WEBAUTHN_SECRET", section.get("secret")),
        ])


def _settings() -> dict:
    """Resolve WebauthnDashboardAuthProvider kwargs from config; raises
    ``SkipRegistration`` (default-off) with actionable reasons."""
    section = load_config_section(logger, _TAG, "dashboard", "webauthn")
    if not section:
        raise SkipRegistration(
            "dashboard.webauthn is not configured (rp_id/rp_name/origins); the "
            "passkey provider stays unregistered. See the web-dashboard docs for "
            "the applicability matrix (TLS origins, or the localhost SSH tunnel).")
    _fido2_or_skip()
    rp_id = str(section.get("rp_id", "")).strip().strip(".")
    rp_name = str(section.get("rp_name", "")).strip()
    if not rp_id or not rp_name:
        raise SkipRegistration(
            "dashboard.webauthn.rp_id and rp_name are both required (rp_id = the "
            "registrable domain the origin belongs to; rp_name = the label shown in "
            "the browser's passkey UI). Passkeys bind to rp_id FOREVER — decide it "
            "before enrolling.", level="warning")
    origins = _resolve_origins(section, rp_id)
    uv = _resolve_uv(section)
    store_path = store.resolve_store_path(str(section.get("credential_store", "")).strip())
    return {
        "rp_id": rp_id, "rp_name": rp_name, "origins": origins,
        "credential_store": store_path, "records": _load_records(store_path, rp_id),
        "allowed_credential_ids": _resolve_allowed_ids(section),
        "user_verification": uv, "ttl_seconds": _resolve_ttl(section),
        "challenge_ttl_seconds": _resolve_challenge_ttl(section),
        "secret": _resolve_secret(section),
    }


def credential_id_b64(credential_id_hex: str) -> str:
    """The config ``allowed_credential_ids`` spelling (base64url rawId) for a
    matched credential's hex id."""
    return crypto.websafe_encode(bytes.fromhex(credential_id_hex))


def _log_ceremony_failure(phase: str, exc: BaseException) -> None:
    """Operator-visible ceremony rejection detail (library text only — no
    credential material): 'Invalid origin', 'Wrong challenge', 'Invalid
    signature', an unparseable response body, a store error, …."""
    logger.warning(
        "dashboard-auth-webauthn: %s completed with %s: %s",
        phase, type(exc).__name__, exc)


# ---- Provider ----


class WebauthnDashboardAuthProvider(NonInteractiveMixin, DashboardAuthProvider):
    """Passkey provider minting stateless basic-shaped HMAC sessions."""

    name = "webauthn"
    display_name = "Passkey (WebAuthn)"
    supports_webauthn = True
    _NOT_INTERACTIVE = "The webauthn provider signs in via the browser WebAuthn ceremony."
    _NO_START_LOGIN = (
        "The webauthn provider has no OAuth redirect flow; the login page runs the "
        "ceremony via /auth/webauthn/options + /auth/webauthn/verify.")

    def __init__(
        self, *, rp_id: str, rp_name: str, origins: tuple, secret: bytes, credential_store,
        records: Optional[list] = None, allowed_credential_ids: tuple = (),
        user_verification: str = "preferred", ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        challenge_ttl_seconds: int = _CHALLENGE_TTL_DEFAULT) -> None:
        if not rp_id or not rp_name or not origins:
            raise ValueError("rp_id, rp_name and origins are required")
        if len(secret) < 16:
            raise ValueError("secret must be at least 16 bytes")
        self._rp_id = rp_id
        self._rp_name = rp_name
        self._origins = tuple(origins)
        self._secret = secret
        self._ttl = max(60, int(ttl_seconds))
        self._challenge_ttl = max(
            _CHALLENGE_TTL_MIN, min(_CHALLENGE_TTL_MAX, int(challenge_ttl_seconds)))
        self._uv = user_verification
        self._store_path = credential_store
        self._allowed = tuple(allowed_credential_ids)
        self._records: list[dict] = list(records or [])
        self._store_lock = threading.Lock()
        # Adapter + credential hydration raise ValueError on malformed records;
        # register_provider turns that into a fail-closed skip.
        self._adapter = crypto.Fido2Adapter(rp_id=rp_id, rp_name=rp_name)
        self._credentials: list = []
        for record in self._records:
            credential_id = crypto.websafe_decode(record["credential_id"])
            public_key_cbor = crypto.websafe_decode(record["public_key"])
            if not credential_id or not public_key_cbor:
                raise ValueError(
                    f"credential store record {record['credential_id'][:8]!r}… holds an "
                    "unreadable credential_id/public_key")
            self._credentials.append(
                self._adapter.rebuild_credential(credential_id, public_key_cbor))

    @property
    def challenge_ttl_seconds(self) -> int:
        return self._challenge_ttl

    # ---- HTTP-surface identity ----------------------------------------------

    def webauthn_offered_for_request(self, request) -> bool:
        """Origin-gated activation: the provider is offered only where the
        browser can present a configured origin. A LAN ``http://<IP>`` request
        (not a secure context, wrong RP ID) sees no passkey button and its
        ceremonies answer 404 — the password-only status quo, by design."""
        return request_origin(request) in self._origins

    # ---- login ceremony -------------------------------------------------------

    def webauthn_login_options(self, *, client_ip: str) -> tuple[dict, str]:
        # allowCredentials hints only credentials the config allowlist scopes to;
        # with none (and none enrolled) the login can never complete — say so.
        usable = self._filter_credentials()
        if self._allowed and not usable:
            raise ProviderError(
                "every enrolled passkey is excluded by "
                "dashboard.webauthn.allowed_credential_ids")
        if not self._credentials:
            raise ProviderError(
                "no passkey enrolled for this dashboard yet; enrol one with "
                "'hermes dashboard webauthn enrol' or from a signed-in session")
        options, state = self._adapter.authenticate_options(
            user_verification=self._uv, credentials=usable)
        handle = webauthn_state.register_challenge(
            challenge=state["challenge"], user_verification=state["user_verification"],
            kind="authentication", client_ip=client_ip, ttl_seconds=self._challenge_ttl)
        return options, handle

    def complete_webauthn_login(self, *, handle: str, credential: Mapping) -> Session:
        parked = webauthn_state.consume_challenge(handle, kind="authentication")
        credential_id_hex = ""
        try:
            credential_id_hex, new_count, _uv_ok = self._adapter.complete_authentication(
                crypto.parked_to_state(parked.challenge, parked.user_verification),
                self._credentials, credential)
        except InvalidCredentialsError:
            raise
        except Exception as exc:  # health: allow BLE001 -- any rejection is the generic 401
            _log_ceremony_failure("authentication", exc)
            raise InvalidCredentialsError("passkey rejected") from exc
        if self._allowed and credential_id_b64(credential_id_hex) not in self._allowed:
            raise InvalidCredentialsError("passkey not allowed by configuration")
        self._update_sign_count(credential_id_hex, new_count)
        return self._mint_session()

    # ---- enrolment ceremony -----------------------------------------------------

    def webauthn_register_options(self, *, client_ip: str) -> tuple[dict, str]:
        options, state = self._adapter.register_options(
            user_verification=self._uv, credentials=self._credentials)
        handle = webauthn_state.register_challenge(
            challenge=state["challenge"], user_verification=state["user_verification"],
            kind="registration", client_ip=client_ip, user_id=_USER_ID,
            ttl_seconds=self._challenge_ttl)
        return options, handle

    def complete_webauthn_registration(
        self, *, handle: str, credential: Mapping, label: str) -> dict:
        parked = webauthn_state.consume_challenge(handle, kind="registration")
        try:
            attested, user_present, _uv_ok, raw_auth_data = self._adapter.complete_registration(
                crypto.parked_to_state(parked.challenge, parked.user_verification), credential)
        except Exception as exc:  # any ceremony rejection is a generic credential failure
            raise InvalidCredentialsError("passkey enrolment rejected") from exc
        if not user_present:
            raise InvalidCredentialsError("passkey enrolment rejected")
        credential_id = crypto.websafe_encode(attested.credential_id)
        if self._allowed and credential_id not in self._allowed:
            raise InvalidCredentialsError(
                "credential id is not on dashboard.webauthn.allowed_credential_ids "
                f"({credential_id!r}); add it to the allowlist or clear the list")
        with self._store_lock:
            doc = store.load(self._store_path)
            if any(record.get("credential_id") == credential_id
                   for record in doc.get("credentials") or []):
                raise InvalidCredentialsError("passkey already enrolled")
            fresh = {
                "credential_id": credential_id,
                "public_key": crypto.websafe_encode(crypto.cose_key_bytes(attested.public_key)),
                "transports": _client_transports(credential),
                "sign_count": crypto.sign_count_from_bytes(raw_auth_data),
                "label": (label or "").strip()[:80] or "passkey",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            doc.setdefault("credentials", []).append(fresh)
            doc["rp_id"] = self._rp_id
            store.prune_pending_enrolments(doc)
            store.save(self._store_path, doc)
            self._records.append(fresh)
            self._credentials.append(attested)
        logger.info(
            "dashboard-auth-webauthn: enrolled passkey %s… (label %r); %d credential(s) stored",
            credential_id[:16], fresh["label"], len(self._records))
        return {"credential_id": credential_id, "label": fresh["label"],
                "created_at": fresh["created_at"]}

    # ---- session lifecycle (shared basic seam) ----------------------------------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        payload = unsign_session_token(access_token, self._secret, "access")
        if payload is None:
            return None
        return self._session(str(payload.get("sub", _USER_ID)), int(payload["exp"]),
                             access_token, "")

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        payload = unsign_session_token(refresh_token, self._secret, "refresh")
        if payload is None:
            raise RefreshExpiredError("refresh token expired or invalid")
        return self._mint_session(str(payload.get("sub", _USER_ID)))

    def revoke_session(self, *, refresh_token: str) -> None:
        # Stateless tokens — nothing to revoke server-side; must not raise.
        return None

    # ---- one-shot CLI enrolment tokens (headless bootstrap, design §2.2) ----

    def issue_enrolment_token(self, *, now: Optional[int] = None) -> str:
        """Write a fresh single-use enrolment token into the store's pending set
        (the CLI prints an ``<origin>/auth/webauthn/enrol?token=…`` URL)."""
        with self._store_lock:
            doc = store.load(self._store_path)
            store.prune_pending_enrolments(doc, now=now)
            token = store.add_pending_enrolment(doc, now=now)
            store.save(self._store_path, doc)
        return token

    def peek_enrolment_token(self, token: str, *, now: Optional[int] = None) -> bool:
        """Whether ``token`` names an unconsumed, unexpired pending enrolment."""
        with self._store_lock:
            doc = store.load(self._store_path)
            return store.check_pending_enrolment(doc, token, now=now)

    def consume_enrolment_token(self, token: str) -> bool:
        """Single-use: drop the token on enrolment success. Best-effort by callers
        (the enrolment has already completed when this runs)."""
        with self._store_lock:
            doc = store.load(self._store_path)
            consumed = store.consume_pending_enrolment(doc, token)
            if consumed:
                store.save(self._store_path, doc)
        return consumed

    # ---- internals -----------------------------------------------------------

    def _mint_session(self, user_id: str = _USER_ID) -> Session:
        now = int(time.time())
        exp = now + self._ttl
        return self._session(
            user_id, exp,
            sign_session_token({"sub": user_id, "kind": "access", "exp": exp}, self._secret),
            sign_session_token({"sub": user_id, "kind": "refresh",
                                "exp": now + _REFRESH_TTL_SECONDS}, self._secret))

    def _session(self, user_id: str, exp: int, access_token: str, refresh_token: str) -> Session:
        return Session(
            user_id=user_id, email="", display_name=self._rp_name, org_id="",
            provider=self.name, expires_at=exp, access_token=access_token,
            refresh_token=refresh_token)

    def _filter_credentials(self) -> list:
        """Credentials the config allowlist scopes to (empty allowlist = all)."""
        if not self._allowed:
            return list(self._credentials)
        return [credential for credential, record in zip(self._credentials, self._records)
                if record.get("credential_id") in self._allowed]

    def _update_sign_count(self, credential_id_hex: str, new_count: int) -> None:
        """Warn-not-fail on a regressed/zero counter (many authenticators report
        0; design doc §9.3). Best-effort persist: a store write failure logs and
        continues — authentication already succeeded."""
        for record in self._records:
            if record.get("credential_id") != credential_id_b64(credential_id_hex):
                continue
            previous = int(record.get("sign_count", 0) or 0)
            if new_count and previous and new_count <= previous:
                logger.warning(
                    "dashboard-auth-webauthn: sign count did not increase for %s… "
                    "(%s -> %s); counter untrustworthy (synced passkeys and 0-only "
                    "authenticators are exempt by design) — accepted, not enforced",
                    credential_id_hex[:16], previous, new_count)
            if new_count == previous:
                return
            record["sign_count"] = new_count
            try:
                with self._store_lock:
                    doc = store.load(self._store_path)
                    for stored in doc.get("credentials") or []:
                        if stored.get("credential_id") == record.get("credential_id"):
                            stored["sign_count"] = new_count
                    store.save(self._store_path, doc)
            except Exception as exc:  # health: allow BLE001 -- best-effort persistence
                logger.warning(
                    "dashboard-auth-webauthn: could not persist sign count: %s", exc)
            return


def _client_transports(credential: Mapping) -> list:
    """The browser-reported transports (echoed into the stored record), or
    ``internal`` as the honest default."""
    response = credential.get("response")
    raw = None
    if isinstance(response, Mapping):
        raw = response.get("transports")
    if raw is None and isinstance(credential.get("transports"), list):
        raw = credential["transports"]
    transports = [str(t) for t in raw or []][:8] or ["internal"]
    return transports


# ---- Plugin entry point ----


def register(ctx) -> None:
    """Register ``WebauthnDashboardAuthProvider`` when a valid, WebAuthn-capable
    config resolves; a no-op otherwise (default-off — ``/login`` is byte-identical)."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(
        ctx, logger, _TAG, WebauthnDashboardAuthProvider, _settings)
    if kwargs is not None:
        logger.info(
            "dashboard-auth-webauthn: registered passkey provider (rp_id=%s, origins=%s)",
            kwargs["rp_id"], ", ".join(kwargs["origins"]))