"""Kilo Gateway device authorization.

Kilo Gateway authenticates users via a browser-based device authorization
flow. Unlike the Nous Portal flow (RFC 8628 with refresh tokens), Kilo's
flow is custom:

- ``POST /api/device-auth/codes`` (no body, no client_id) returns a short
  verification code and URL.
- ``GET  /api/device-auth/codes/{code}`` is polled until the user approves
  in the browser. Status is conveyed via HTTP status codes:
  ``202`` pending, ``200`` approved (JSON body carries the token),
  ``403`` denied, ``410`` expired.

The resulting token is long-lived (~1 year) and has **no refresh_token**, so
at runtime it behaves like an API key. The provider profile keeps
``auth_type="api_key"``; this module only handles *acquisition* — it does
not touch the conversation loop, toolsets, or runtime credential
resolution.

Reference: ``packages/kilo-gateway/src/auth/device-auth-tui.ts`` and
``api/profile.ts`` in the Kilo client.
"""

from __future__ import annotations

import logging
import os
import time
import webbrowser
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from hermes_cli.auth import (
    _can_open_graphical_browser,
    _is_remote_session,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Defensive last-resort fallback only. The authoritative Kilo host lives in
# PROVIDER_REGISTRY["kilocode"].inference_base_url; _kilo_registry_base()
# derives the API root from it so there is a single source of truth. This
# constant is only used if the registry lookup itself fails.
DEFAULT_KILO_API_BASE = "https://api.kilo.ai"

# The Kilo device-auth bearer is long-lived (~1 year, no refresh), so it is
# pinned to the Kilo origin the same way the xAI OAuth bearer is pinned to
# *.x.ai (see _xai_validate_inference_base_url).
_KILO_HOST = "kilo.ai"

# Device-auth endpoints live under the API root, siblings of /api/gateway.
_DEVICE_AUTH_INITIATE_PATH = "/api/device-auth/codes"
_DEVICE_AUTH_POLL_PATH = "/api/device-auth/codes/{code}"

# Polling cadence — mirrors the upstream Kilo client (POLL_INTERVAL_MS = 3000).
_KILO_POLL_INTERVAL_SECONDS = 3

# Header sent on inference requests when an organization is selected.
# Matches HEADER_ORGANIZATIONID in the Kilo client (api/constants.ts).
KILO_ORG_HEADER = "X-KILOCODE-ORGANIZATIONID"


# ── Base URL resolution ─────────────────────────────────────────────────────

def _strip_api_segment(base_url: str) -> str:
    """Return the API root by dropping the trailing ``/api/<segment>`` path.

    Given ``https://api.kilo.ai/api/gateway`` → ``https://api.kilo.ai``.
    Given ``https://api.kilo.ai/api/openrouter/v1`` → ``https://api.kilo.ai``.
    Given ``https://api.kilo.ai`` → ``https://api.kilo.ai``.
    Mirrors the ``route()`` helper in the Kilo client (api/url.ts), which
    locates the last ``api`` path segment and treats everything before it as
    the host root.
    """
    try:
        parsed = urlparse(base_url)
    except Exception:
        return base_url.rstrip("/")
    parts = [p for p in parsed.path.split("/") if p]
    api_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "api":
            api_idx = i
            break
    if api_idx is not None:
        parts = parts[:api_idx]
    reconstructed = parsed._replace(path="/" + "/".join(parts) if parts else "")
    # Normalize: strip trailing slash, drop empty query/fragment.
    url = reconstructed.geturl()
    return url.rstrip("/")


def _kilo_registry_base() -> str:
    """Derive the Kilo API root from the registered kilocode inference base.

    Single source of truth: the host comes from
    ``PROVIDER_REGISTRY["kilocode"].inference_base_url`` (e.g.
    ``https://api.kilo.ai/api/gateway``), stripped of its ``/api/<segment>``
    suffix. Falls back to ``DEFAULT_KILO_API_BASE`` only if the provider is
    not registered (should not happen in normal operation).
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        return _strip_api_segment(PROVIDER_REGISTRY["kilocode"].inference_base_url)
    except Exception as exc:
        logger.debug(
            "kilo registry base lookup failed (%s); using hardcoded fallback.", exc
        )
        return DEFAULT_KILO_API_BASE


def _validate_kilo_base_url(value: str, *, fallback: str, env_name: str) -> str:
    """Refuse a non-Kilo base URL for bearer-bearing device-auth/profile calls.

    The Kilo device-auth token is long-lived (~1 year) with no refresh_token,
    so a tampered ``KILO_API_URL`` / ``KILOCODE_BASE_URL`` (or a hostile shell
    init) pointing at a non-HTTPS or non-Kilo host would ship the bearer to a
    third party on every profile/defaults request, silently. Mirror
    ``_xai_validate_inference_base_url``: require HTTPS and pin the origin to
    ``kilo.ai`` / ``*.kilo.ai``. On rejection, fall back to the default and
    log a warning rather than raise — a bad env var should not deadlock
    authentication, but it should also never leak the bearer.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(
            "Ignoring malformed Kilo base_url override %r (from %s); using %s instead.",
            candidate, env_name, fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS Kilo base_url override %r from %s (kilo bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, env_name, fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring Kilo base_url override %r from %s with no hostname; using %s instead.",
            candidate, env_name, fallback,
        )
        return fallback
    if host != _KILO_HOST and not host.endswith("." + _KILO_HOST):
        logger.warning(
            "Refusing Kilo base_url override %r from %s — host %r is not on the Kilo "
            "origin (expected kilo.ai or a *.kilo.ai subdomain). The kilo bearer is only "
            "valid against Kilo's API; sending it elsewhere would leak the credential. "
            "Falling back to %s.",
            candidate, env_name, host, fallback,
        )
        return fallback
    return candidate


def _validate_kilo_verification_url(url: str) -> Optional[str]:
    """Return the URL only if it is an HTTPS URL on the Kilo origin.

    The ``verificationUrl`` is network-sourced (from the device-auth initiate
    response). With an attacker-controlled base override it could point at a
    phishing page that mimics Kilo's approval screen to capture the user's
    interactive login credentials. Refuse anything that is not HTTPS on the
    Kilo origin; the caller aborts the login on refusal rather than directing
    the user's browser at an untrusted URL.
    """
    candidate = (url or "").strip()
    if not candidate:
        return None
    try:
        parsed = urlparse(candidate)
    except Exception:
        return None
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if host != _KILO_HOST and not host.endswith("." + _KILO_HOST):
        return None
    return candidate


def kilo_api_base() -> str:
    """Resolve the Kilo API root used for device-auth and profile calls.

    Precedence:
      1. ``KILO_API_URL`` — the upstream Kilo client's own override env var
         (kept for fidelity with the TS client's ``ENV_KILO_API_URL``).
      2. Derived from ``KILOCODE_BASE_URL`` (the inference override already
         declared in ``PROVIDER_REGISTRY``) by stripping ``/api/<segment>``.
      3. Derived from ``PROVIDER_REGISTRY["kilocode"].inference_base_url``
         (the single source of truth for the Kilo host).

    All overrides are validated with ``_validate_kilo_base_url`` (HTTPS +
    Kilo origin) so the long-lived device-auth bearer is never shipped to a
    non-Kilo or cleartext endpoint. No new ``HERMES_*`` behavioral env var is
    introduced — only Kilo's own conventions are honored.
    """
    fallback = _kilo_registry_base()
    override = (os.getenv("KILO_API_URL") or "").strip()
    if override:
        return _validate_kilo_base_url(override, fallback=fallback, env_name="KILO_API_URL")
    gateway_override = (os.getenv("KILOCODE_BASE_URL") or "").strip()
    if gateway_override:
        return _validate_kilo_base_url(
            _strip_api_segment(gateway_override), fallback=fallback, env_name="KILOCODE_BASE_URL"
        )
    return fallback


# ── Device-auth flow ────────────────────────────────────────────────────────

def _initiate_device_auth(client: httpx.Client, api_base: str) -> Dict[str, Any]:
    """POST to the device-code endpoint and return the verification details."""
    response = client.post(
        f"{api_base}{_DEVICE_AUTH_INITIATE_PATH}",
        headers={"Content-Type": "application/json"},
    )
    if response.status_code == 429:
        raise RuntimeError(
            "Too many pending Kilo authorization requests. Please try again later."
        )
    if not response.is_success:
        raise RuntimeError(
            f"Failed to initiate Kilo device authorization: {response.status_code}"
        )
    data = response.json()
    missing = [f for f in ("code", "verificationUrl", "expiresIn") if f not in data]
    if missing:
        raise RuntimeError(
            f"Kilo device-auth response missing fields: {', '.join(missing)}"
        )
    return data


def _poll_device_auth(
    client: httpx.Client, api_base: str, code: str, expires_in: int
) -> Dict[str, Any]:
    """Poll the device-auth endpoint until approval, denial, or expiry.

    Returns the approved payload (``{status, token, userEmail}``). Raises on
    denial, expiry, or transport errors.
    """
    deadline = time.monotonic() + max(1, expires_in)
    # Kilo's cadence is a fixed 3s (mirrors the upstream client's
    # POLL_INTERVAL_MS = 3000); do NOT clamp it through the Nous cap, which
    # would poll ~3x faster and risk a 429 aborting the whole login.
    interval = max(1, _KILO_POLL_INTERVAL_SECONDS)
    url = f"{api_base}{_DEVICE_AUTH_POLL_PATH.format(code=code)}"

    while time.monotonic() < deadline:
        response = client.get(url)
        if response.status_code == 202:
            time.sleep(interval)
            continue
        if response.status_code == 403:
            raise RuntimeError("Kilo authorization denied by user")
        if response.status_code == 410:
            raise RuntimeError("Kilo authorization code expired")
        if response.status_code == 429:
            # Rate-limited mid-poll: back off and keep waiting rather than
            # aborting the login. A transient 429 is not a terminal state.
            time.sleep(interval * 2)
            continue
        if not response.is_success:
            raise RuntimeError(
                f"Failed to poll Kilo device authorization: {response.status_code}"
            )
        data = response.json()
        if data.get("status") != "approved" or not data.get("token"):
            # Defensive: a 200 without an approved status is unexpected; keep polling.
            time.sleep(interval)
            continue
        return data

    raise TimeoutError("Timed out waiting for Kilo device authorization")


def kilo_device_auth_login(
    *,
    api_base: Optional[str] = None,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    on_verification: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Run the Kilo device-auth flow and return credentials without persisting.

    Returns ``{token, user_email, obtained_at}``. The caller is responsible
    for storing the token (typically as a credential-pool entry).
    """
    api_base = (api_base or kilo_api_base()).rstrip("/")
    if _is_remote_session():
        open_browser = False

    print("Starting Kilo Gateway login via device authorization...")
    print(f"  API: {api_base}")

    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        auth_data = _initiate_device_auth(client, api_base)
        code = str(auth_data["code"])
        verification_url = str(auth_data["verificationUrl"])
        expires_in = int(auth_data["expiresIn"])

        # The verificationUrl is network-sourced; refuse anything not on the
        # Kilo origin so a tampered base override can't redirect the user's
        # browser at a phishing page that mimics the Kilo approval screen.
        trusted_url = _validate_kilo_verification_url(verification_url)
        if trusted_url is None:
            raise RuntimeError(
                "Kilo device-auth returned an untrusted verification URL "
                f"(expected an HTTPS URL on *.kilo.ai, got {verification_url!r}); "
                "aborting login to avoid redirecting your browser at an untrusted page."
            )
        verification_url = trusted_url

        print()
        print("To continue:")
        print(f"  1. Open: {verification_url}")
        print(f"  2. If prompted, enter code: {code}")

        if open_browser and _can_open_graphical_browser():
            opened = webbrowser.open(verification_url)
            if opened:
                print("  (Opened browser for verification)")
            else:
                print("  Could not open browser automatically — use the URL above.")

        # Surface the verification URL/code to an out-of-band consumer (e.g.
        # the TUI gateway, whose stdout is a JSON-RPC pipe). Best-effort.
        if on_verification is not None:
            try:
                on_verification(verification_url, code)
            except Exception:
                pass

        effective_interval = max(1, _KILO_POLL_INTERVAL_SECONDS)
        print(f"Waiting for approval (polling every {effective_interval}s)...")

        result = _poll_device_auth(client, api_base, code, expires_in)

    token = str(result["token"])
    user_email = str(result.get("userEmail") or "")
    print()
    if user_email:
        print(f"Authenticated as {user_email}")
    else:
        print("Authentication successful.")

    return {
        "token": token,
        "user_email": user_email,
        "obtained_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Profile / defaults ──────────────────────────────────────────────────────

def fetch_kilo_profile(
    token: str, *, api_base: Optional[str] = None, timeout_seconds: float = 15.0
) -> Optional[Dict[str, Any]]:
    """Fetch the user profile (email, name, organizations).

    Returns ``None`` on auth/network failure so callers can proceed with a
    personal-account default rather than aborting the whole login.
    """
    api_base = (api_base or kilo_api_base()).rstrip("/")
    try:
        response = httpx.get(
            f"{api_base}/api/profile",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=timeout_seconds,
        )
        if not response.is_success:
            logger.debug("Kilo profile fetch failed: %s", response.status_code)
            return None
        data = response.json()
    except Exception as exc:
        logger.debug("Kilo profile fetch error: %s", exc)
        return None

    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    return {
        "email": user.get("email") or data.get("email") or "",
        "name": user.get("name") or data.get("name"),
        "organizations": data.get("organizations") or [],
    }


def fetch_kilo_default_model(
    token: Optional[str] = None,
    organization_id: Optional[str] = None,
    *,
    api_base: Optional[str] = None,
    timeout_seconds: float = 15.0,
) -> Optional[str]:
    """Fetch the default model for the given account context.

    Returns the model id string, or ``None`` if the endpoint is unreachable
    or returns no default. Callers fall back to the curated catalog.
    """
    api_base = (api_base or kilo_api_base()).rstrip("/")
    path = (
        f"/api/organizations/{organization_id}/defaults"
        if organization_id
        else "/api/defaults"
    )
    headers: Dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.get(f"{api_base}{path}", headers=headers, timeout=timeout_seconds)
        if not response.is_success:
            return None
        data = response.json()
    except Exception as exc:
        logger.debug("Kilo defaults fetch error: %s", exc)
        return None

    if token:
        model = data.get("defaultModel")
    else:
        model = data.get("defaultFreeModel")
    return model or None


def prompt_kilo_organization(organizations: List[Dict[str, Any]]) -> Optional[str]:
    """Prompt the user to pick a personal account or an organization.

    Returns the selected organization id, or ``None`` for a personal account.
    Falls back to a numbered prompt when curses is unavailable (piped stdin).
    """
    if not organizations:
        return None

    choices = ["Personal Account"]
    for org in organizations:
        choices.append(str(org.get("name") or org.get("id") or "Organization"))

    try:
        from hermes_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice("Select Kilo account:", choices, 0)
        if idx >= 0:
            print()
            if idx == 0:
                return None
            org = organizations[idx - 1]
            return str(org.get("id") or "")
    except Exception:
        pass

    print("Select Kilo account:")
    for i, label in enumerate(choices, 1):
        marker = "→" if i == 1 else " "
        print(f"  {marker} {i}. {label}")
    print()
    try:
        raw = input(f"  Choice [1-{len(choices)}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
    except ValueError:
        return None
    if idx <= 0:
        return None
    if idx - 1 < len(organizations):
        return str(organizations[idx - 1].get("id") or "")
    return None


# ── Shared acquisition helper ───────────────────────────────────────────────

def acquire_and_store_kilo_credential(
    pool,
    *,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    label: Optional[str] = None,
) -> "PooledCredential":  # type: ignore[name-defined]
    """Run device-auth, fetch profile, prompt org, create + store a pool entry.

    Shared by ``hermes auth add kilocode`` and ``_model_flow_kilo`` so the two
    paths can't drift. Stores the token as ``auth_type=api_key`` (long-lived,
    no refresh) and the selected org in ``entry.extra["organization_id"]``.

    The runtime derives the org header from the selected pool entry so token
    rotation cannot pair one credential with another credential's organization.
    """
    import uuid

    from hermes_cli.auth import PROVIDER_REGISTRY
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        PooledCredential,
        SOURCE_MANUAL_DEVICE_CODE,
        label_from_token,
    )

    gateway_base = PROVIDER_REGISTRY["kilocode"].inference_base_url

    creds = kilo_device_auth_login(
        open_browser=open_browser,
        timeout_seconds=timeout_seconds,
    )
    token = creds["token"]

    org_id = None
    profile = fetch_kilo_profile(token)
    if profile and profile.get("organizations"):
        org_id = prompt_kilo_organization(profile["organizations"])

    resolved_label = (label or "").strip() or label_from_token(
        token, f"kilocode-device-{len(pool.entries()) + 1}",
    )
    entry = PooledCredential(
        provider="kilocode",
        id=uuid.uuid4().hex[:6],
        label=resolved_label,
        auth_type=AUTH_TYPE_API_KEY,
        priority=0,
        source=SOURCE_MANUAL_DEVICE_CODE,
        access_token=token,
        base_url=gateway_base,
    )
    entry.extra["organization_id"] = org_id
    pool.add_entry(entry)
    return entry


# ── Organization header ─────────────────────────────────────────────────────

def is_kilo_gateway_origin(base_url: str) -> bool:
    """True when ``base_url`` is an https origin on the Kilo Gateway host."""
    try:
        parsed = urlparse(str(base_url or "").strip())
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == _KILO_HOST or host.endswith("." + _KILO_HOST)
    )


def kilo_organization_header(
    base_url: str,
    organization_id: Optional[str],
) -> Dict[str, str]:
    """Return the credential-bound org header for a validated Kilo origin."""
    org_id = str(organization_id or "").strip()
    if not org_id or not is_kilo_gateway_origin(base_url):
        return {}
    return {KILO_ORG_HEADER: org_id}
