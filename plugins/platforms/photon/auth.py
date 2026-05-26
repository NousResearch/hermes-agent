"""
Photon Dashboard + Spectrum API client and device-code login flow.

This module is pure Python — it intentionally does not depend on
``spectrum-ts``.  All management-plane operations (login, create
project, create user, register webhook) talk to Photon's HTTP API
directly:

    Dashboard API   https://app.photon.codes/api/...
                    OAuth bearer token from device flow

    Spectrum API    https://spectrum.photon.codes/projects/{id}/...
                    HTTP Basic with (projectId, projectSecret)

The webhook receiver + Node sidecar in ``adapter.py`` consume the
credentials this module persists to ``~/.hermes/auth.json``.

Reference docs (read at integration time):
  https://photon.codes/docs/api-reference/introduction
  https://photon.codes/docs/api-reference/device-login/request-device-+-user-code
  https://photon.codes/docs/api-reference/device-login/exchange-device-code-for-token
  https://photon.codes/docs/api-reference/projects/create-project
  https://photon.codes/docs/api-reference/users/create-user
  https://photon.codes/docs/webhooks/overview
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a hermes dependency
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants

# Photon's published OAuth device-client identifier for first-party CLIs.
# We use a fixed "hermes-agent" client_id string — Photon's device endpoint
# accepts any opaque client_id and ties the bearer token to the approving
# user, not to the client.  If Photon later requires registered clients,
# this is the one knob to update.
DEFAULT_CLIENT_ID = "hermes-agent"

DEFAULT_DASHBOARD_HOST = "https://app.photon.codes"
DEFAULT_SPECTRUM_HOST = "https://spectrum.photon.codes"

# Polling defaults per RFC 8628.  Photon may override via `interval` /
# `expires_in` fields in the device-code response — those win.
DEFAULT_POLL_INTERVAL = 5
DEFAULT_POLL_TIMEOUT = 900  # 15 minutes is conservative; Photon returns expires_in

E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


# ---------------------------------------------------------------------------
# auth.json helpers — share the file with the rest of hermes-agent.

def _auth_json_path() -> Path:
    """Resolve ``~/.hermes/auth.json`` honouring the active Hermes profile."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return Path(get_hermes_home()) / "auth.json"
    except Exception:
        return Path(os.path.expanduser("~/.hermes")) / "auth.json"


def _load_auth() -> Dict[str, Any]:
    path = _auth_json_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("photon: could not read %s: %s", path, e)
        return {}


def _save_auth(data: Dict[str, Any]) -> None:
    path = _auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def load_photon_token() -> Optional[str]:
    """Return the bearer token stored by ``login()`` or ``None``."""
    auth = _load_auth()
    pool = auth.get("credential_pool", {}).get("photon") or []
    if isinstance(pool, list) and pool:
        token = pool[0].get("access_token") or pool[0].get("token")
        if token:
            return str(token)
    # Backwards-compat shape: providers.photon.access_token
    legacy = auth.get("providers", {}).get("photon", {})
    if legacy.get("access_token"):
        return str(legacy["access_token"])
    return None


def store_photon_token(token: str) -> None:
    """Persist a dashboard bearer token under ``credential_pool.photon``."""
    auth = _load_auth()
    auth.setdefault("credential_pool", {})["photon"] = [
        {"access_token": token, "issued_at": int(time.time())}
    ]
    _save_auth(auth)


def load_project_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(project_id, project_secret)`` from auth.json + env override."""
    env_id = os.getenv("PHOTON_PROJECT_ID")
    env_sec = os.getenv("PHOTON_PROJECT_SECRET")
    if env_id and env_sec:
        return env_id, env_sec
    auth = _load_auth()
    proj = auth.get("credential_pool", {}).get("photon_project") or []
    if isinstance(proj, list) and proj:
        entry = proj[0]
        return (
            env_id or entry.get("project_id"),
            env_sec or entry.get("project_secret"),
        )
    return env_id, env_sec


def store_project_credentials(project_id: str, project_secret: str, **extra: Any) -> None:
    """Persist the Spectrum project's id+secret under ``credential_pool.photon_project``."""
    auth = _load_auth()
    record = {
        "project_id": project_id,
        "project_secret": project_secret,
        "issued_at": int(time.time()),
    }
    record.update(extra)
    auth.setdefault("credential_pool", {})["photon_project"] = [record]
    _save_auth(auth)


# ---------------------------------------------------------------------------
# Device login flow (RFC 8628)

@dataclass
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str]
    expires_in: int
    interval: int


def _dashboard_host() -> str:
    return (os.getenv("PHOTON_DASHBOARD_HOST") or DEFAULT_DASHBOARD_HOST).rstrip("/")


def _spectrum_host() -> str:
    return (os.getenv("PHOTON_API_HOST") or DEFAULT_SPECTRUM_HOST).rstrip("/")


def request_device_code(
    *, client_id: str = DEFAULT_CLIENT_ID, scope: Optional[str] = None,
) -> DeviceCode:
    """POST ``/api/auth/device/code`` and return the device + user codes."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon device login")
    url = f"{_dashboard_host()}/api/auth/device/code"
    body: Dict[str, Any] = {"client_id": client_id}
    if scope:
        body["scope"] = scope
    resp = httpx.post(url, json=body, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return DeviceCode(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        verification_uri_complete=data.get("verification_uri_complete"),
        expires_in=int(data.get("expires_in") or DEFAULT_POLL_TIMEOUT),
        interval=int(data.get("interval") or DEFAULT_POLL_INTERVAL),
    )


def poll_for_token(
    code: DeviceCode,
    *,
    client_id: str = DEFAULT_CLIENT_ID,
    timeout: Optional[int] = None,
    interval: Optional[int] = None,
    on_pending: Optional[callable] = None,
) -> str:
    """Poll ``/api/auth/device/token`` until the user approves.

    Returns the bearer token from the ``set-auth-token`` response header
    (Photon's documented mechanism).  Falls back to ``session.access_token``
    in the JSON body if the header is absent — see the API spec.
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon device login")
    url = f"{_dashboard_host()}/api/auth/device/token"
    deadline = time.time() + (timeout or code.expires_in or DEFAULT_POLL_TIMEOUT)
    sleep = interval or code.interval or DEFAULT_POLL_INTERVAL
    while time.time() < deadline:
        try:
            resp = httpx.post(
                url,
                json={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": code.device_code,
                    "client_id": client_id,
                },
                timeout=30.0,
            )
        except httpx.RequestError as e:
            logger.warning("photon: device-token poll failed: %s", e)
            time.sleep(sleep)
            continue
        if resp.status_code == 200:
            token = resp.headers.get("set-auth-token")
            if not token:
                body = resp.json() or {}
                session = body.get("session") or {}
                token = session.get("access_token") or body.get("access_token")
            if not token:
                raise RuntimeError(
                    "Photon returned 200 but no token in headers or body"
                )
            return token
        if resp.status_code == 400:
            # RFC 8628 §3.5 — error codes are returned with 400.
            body: Dict[str, Any] = {}
            try:
                body = resp.json() or {}
            except json.JSONDecodeError:
                pass
            err = body.get("error") or body.get("message") or ""
            if err in ("authorization_pending", "slow_down"):
                if on_pending:
                    try:
                        on_pending()
                    except Exception:
                        pass
                if err == "slow_down":
                    sleep += 5
                time.sleep(sleep)
                continue
            if err in ("expired_token", "access_denied"):
                raise RuntimeError(f"Photon login failed: {err}")
            # Unknown error — surface it
            raise RuntimeError(f"Photon device token error: {err or resp.text}")
        # Unexpected status; log and retry
        logger.warning(
            "photon: device-token unexpected status %s: %s",
            resp.status_code, resp.text[:200],
        )
        time.sleep(sleep)
    raise TimeoutError("Photon device login timed out")


def login_device_flow(
    *,
    client_id: str = DEFAULT_CLIENT_ID,
    open_browser: bool = True,
    on_user_code: Optional[callable] = None,
) -> str:
    """Run the full device-code login flow and persist the token.

    Returns the bearer token.  ``on_user_code`` is a callback receiving the
    :class:`DeviceCode` so callers can print + optionally open the browser.
    """
    code = request_device_code(client_id=client_id)
    if on_user_code:
        try:
            on_user_code(code)
        except Exception:
            pass
    if open_browser:
        try:
            import webbrowser
            target = code.verification_uri_complete or code.verification_uri
            webbrowser.open(target, new=2)
        except Exception:
            pass
    token = poll_for_token(code, client_id=client_id)
    store_photon_token(token)
    return token


# ---------------------------------------------------------------------------
# Dashboard API: create project

def create_project(
    token: str,
    *,
    name: str,
    location: str = "United States",
    platforms: Optional[list] = None,
) -> Dict[str, Any]:
    """POST ``/api/projects/`` with ``spectrum: true`` and return the response.

    The response includes ``spectrumProjectId`` and ``projectSecret`` — those
    are the HTTP Basic credentials for the Spectrum API.  Photon only
    returns ``projectSecret`` to project owners at creation time.
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon project creation")
    url = f"{_dashboard_host()}/api/projects/"
    body: Dict[str, Any] = {
        "name": name,
        "location": location,
        "spectrum": True,
        "platforms": platforms or ["imessage"],
    }
    resp = httpx.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Spectrum API: create user

def create_user(
    project_id: str,
    project_secret: str,
    *,
    phone_number: str,
    user_type: str = "shared",
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
    assigned_phone_number: Optional[str] = None,
) -> Dict[str, Any]:
    """POST ``/projects/{id}/users/`` on the Spectrum API.

    For free users we always pass ``type=shared``; Photon's Cosmos pool
    assigns the iMessage line.  ``assigned_phone_number`` is only valid
    for the paid ``dedicated`` mode.
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon user creation")
    if not E164_RE.match(phone_number):
        raise ValueError(
            f"phone_number must be E.164 (e.g. +15551234567); got {phone_number!r}"
        )
    url = f"{_spectrum_host()}/projects/{project_id}/users/"
    body: Dict[str, Any] = {"type": user_type, "phoneNumber": phone_number}
    if first_name:
        body["firstName"] = first_name
    if last_name:
        body["lastName"] = last_name
    if email:
        body["email"] = email
    if assigned_phone_number:
        body["assignedPhoneNumber"] = assigned_phone_number
    resp = httpx.post(
        url,
        json=body,
        auth=(project_id, project_secret),
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json() or {}
    if not data.get("succeed"):
        raise RuntimeError(
            f"Photon create-user failed: {data.get('message') or data}"
        )
    return data.get("data") or {}


# ---------------------------------------------------------------------------
# Spectrum API: webhook registration
#
# Endpoints from https://photon.codes/docs/webhooks/overview:
#   POST   /projects/{id}/webhooks/          register, returns signing secret ONCE
#   GET    /projects/{id}/webhooks/          list
#   DELETE /projects/{id}/webhooks/{wid}     remove

def register_webhook(
    project_id: str, project_secret: str, *, webhook_url: str,
) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is required for Photon webhook registration")
    url = f"{_spectrum_host()}/projects/{project_id}/webhooks/"
    resp = httpx.post(
        url,
        json={"webhookUrl": webhook_url},
        auth=(project_id, project_secret),
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json() or {}
    if not data.get("succeed"):
        raise RuntimeError(
            f"Photon register-webhook failed: {data.get('message') or data}"
        )
    return data.get("data") or {}


def list_webhooks(project_id: str, project_secret: str) -> list:
    if httpx is None:
        raise RuntimeError("httpx is required for Photon webhook listing")
    url = f"{_spectrum_host()}/projects/{project_id}/webhooks/"
    resp = httpx.get(url, auth=(project_id, project_secret), timeout=30.0)
    resp.raise_for_status()
    data = resp.json() or {}
    return data.get("data") or []


def delete_webhook(
    project_id: str, project_secret: str, *, webhook_id: str,
) -> None:
    if httpx is None:
        raise RuntimeError("httpx is required for Photon webhook deletion")
    url = f"{_spectrum_host()}/projects/{project_id}/webhooks/{webhook_id}"
    resp = httpx.delete(url, auth=(project_id, project_secret), timeout=30.0)
    if resp.status_code not in (200, 204, 404):
        resp.raise_for_status()
