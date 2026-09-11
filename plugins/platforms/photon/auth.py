"""Photon Dashboard API client + device-code login flow (pure Python, no spectrum-ts).

Management calls hit ``https://app.photon.codes/api/...`` (OAuth 2.0 device flow, Bearer)
like the official CLI. The dashboard project ``id`` *is* the Spectrum Cloud project id and
Spectrum is always provisioned at create-time; the sidecar authenticates with
``(id, projectSecret)``. Storage: runtime SDK creds -> ``~/.hermes/.env``; management
metadata -> ``auth.json`` under ``credential_pool.photon`` (device token), ``photon_project``
(ids + secret for offline status) and ``photon_user`` (numbers).
"""
from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
import uuid
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a hermes dependency
    httpx = None  # type: ignore[assignment]

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
import contextlib

logger = logging.getLogger(__name__)


class PhotonDashboardAuthError(RuntimeError):
    """Raised when Photon rejects a device-flow token for the dashboard API."""


# Hosted Photon allowlists device clients (unregistered → 400 invalid_client); use Photon's
# published CLI client until Hermes gets its own client_id.
DEFAULT_CLIENT_ID = "photon-cli"
DEFAULT_SCOPE = "openid profile email"
DEFAULT_DASHBOARD_HOST = "https://app.photon.codes"
DEFAULT_SPECTRUM_HOST = "https://spectrum.photon.codes"
DEFAULT_PROJECT_NAME = "Hermes Agent"
DEFAULT_POLL_INTERVAL = 5  # RFC 8628 polling defaults; Photon's `interval` / `expires_in` win
DEFAULT_POLL_TIMEOUT = 1800
E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


# -- auth.json helpers (shares the file with the rest of hermes-agent) ------------

def _auth_json_path() -> Path:
    """``~/.hermes/auth.json`` honouring the active Hermes profile."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "auth.json"
    except Exception:
        return Path(os.path.expanduser("~/.hermes")) / "auth.json"


def _load_auth() -> Dict[str, Any]:
    path = _auth_json_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("photon: could not read %s: %s", path, e)
        return {}


def _save_auth(data: Dict[str, Any]) -> None:
    path = _auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Random per-process temp name (no collisions / pre-planted symlinks), created 0o600
    # atomically via O_EXCL so the bearer token is never world-readable at umask.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:  # fdopen failed before owning the descriptor — nothing else will close it
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    try:
        with fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _pool_first(auth: Dict[str, Any], key: str) -> Any:
    """First entry of ``credential_pool.<key>`` (a list), or None."""
    pool = auth.get("credential_pool", {}).get(key) or []
    return pool[0] if isinstance(pool, list) and pool else None


def _store_pool_record(key: str, record: Dict[str, Any]) -> None:
    """Replace ``credential_pool.<key>`` with ``[record]`` under the cross-process lock."""
    from hermes_cli.auth import _auth_store_lock
    with _auth_store_lock():
        auth = _load_auth()
        auth.setdefault("credential_pool", {})[key] = [record]
        _save_auth(auth)


def load_photon_token() -> Optional[str]:
    """Return the device-flow bearer token stored by ``login()`` or ``None``."""
    auth = _load_auth()
    entry = _pool_first(auth, "photon") or {}
    legacy = auth.get("providers", {}).get("photon", {})  # backwards-compat shape
    token = entry.get("access_token") or entry.get("token") or legacy.get("access_token")
    return str(token) if token else None


def store_photon_token(token: str) -> None:
    """Persist a dashboard bearer token under ``credential_pool.photon``."""
    _store_pool_record("photon", {"access_token": token, "issued_at": int(time.time())})


def clear_photon_token() -> None:
    """Remove any stored Photon dashboard token (before re-authentication)."""
    auth = _load_auth()
    pool = auth.get("credential_pool", {})
    photon = pool.get("photon", [])
    if isinstance(photon, list) and photon:
        pool["photon"] = []
        _save_auth(auth)
    providers = auth.get("providers", {})  # legacy shape
    if "photon" in providers:
        providers["photon"] = {}
        _save_auth(auth)


def check_photon_token_valid(token: str) -> bool:
    """True if the dashboard API accepts the token; a definitive rejection is stale,
    transient failures (network, 5xx) count as valid so they don't force a re-login."""
    if not token:
        return False
    try:
        validate_photon_token(token)
    except PhotonDashboardAuthError:
        return False
    except Exception:
        pass
    return True


def load_project_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Runtime SDK creds ``(spectrum_project_id, project_secret)``: process env wins
    (``.env`` is loaded at gateway startup), then ``auth.json`` for offline/status."""
    env_id = _get_scoped_secret("PHOTON_PROJECT_ID")
    env_sec = _get_scoped_secret("PHOTON_PROJECT_SECRET")
    if env_id and env_sec:
        return env_id, env_sec
    entry = _pool_first(_load_auth(), "photon_project")
    if entry is None:
        return env_id, env_sec
    # back-compat: old records used "project_id" for the spectrum id
    return env_id or entry.get("spectrum_project_id") or entry.get("project_id"), env_sec or entry.get("project_secret")


def load_dashboard_project_id() -> Optional[str]:
    """Project id for management API calls — prefers ``spectrum_project_id`` (on
    pre-backfill installs the old ``dashboard_project_id`` is diverged and 404s)."""
    env_id = _get_scoped_secret("PHOTON_DASHBOARD_PROJECT_ID")
    if env_id:
        return env_id
    entry = _pool_first(_load_auth(), "photon_project") or {}
    return entry.get("spectrum_project_id") or entry.get("dashboard_project_id") or entry.get("project_id")


def store_project_credentials(
    *, spectrum_project_id: str, project_secret: str,
    dashboard_project_id: Optional[str] = None, name: Optional[str] = None) -> None:
    """Persist project credentials to both .env (runtime) and auth.json (mgmt/offline status)."""
    record: Dict[str, Any] = {
        "spectrum_project_id": spectrum_project_id, "project_secret": project_secret, "issued_at": int(time.time())}
    record.update({k: v for k, v in (("dashboard_project_id", dashboard_project_id), ("name", name)) if v})
    _store_pool_record("photon_project", record)
    _persist_runtime_env(spectrum_project_id, project_secret)


def store_user_numbers(
    *, phone_number: Optional[str] = None, assigned_phone_number: Optional[str] = None,
    user_id: Optional[str] = None, dashboard_project_id: Optional[str] = None) -> None:
    """Persist non-secret Photon user numbers for offline ``status`` output."""
    if not phone_number and not assigned_phone_number:
        return
    record: Dict[str, Any] = {"issued_at": int(time.time())}
    record.update({k: v for k, v in (
        ("phone_number", phone_number), ("assigned_phone_number", assigned_phone_number),
        ("user_id", user_id), ("dashboard_project_id", dashboard_project_id)) if v})
    _store_pool_record("photon_user", record)


def _persist_runtime_env(spectrum_project_id: str, project_secret: str) -> None:
    """Write the SDK creds to ``~/.hermes/.env`` (secret never bound to a printable local
    in a caller — CodeQL clean flow)."""
    try:
        from hermes_cli.config import save_env_value
    except ImportError:
        logger.warning("photon: hermes_cli.config unavailable — skipping .env write")
        return
    try:
        save_env_value("PHOTON_PROJECT_ID", spectrum_project_id)
        save_env_value("PHOTON_PROJECT_SECRET", project_secret)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("photon: could not write project creds to .env: %s", e)


# -- HTTP plumbing ----------------------------------------------------------------

def _dashboard_host() -> str:
    return (os.getenv("PHOTON_DASHBOARD_HOST") or DEFAULT_DASHBOARD_HOST).rstrip("/")


def _spectrum_host() -> str:
    return (os.getenv("PHOTON_SPECTRUM_HOST") or DEFAULT_SPECTRUM_HOST).rstrip("/")


def _bearer(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _basic(project_id: str, project_secret: str) -> Dict[str, str]:
    token = b64encode(f"{project_id}:{project_secret}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _require_httpx(what: str = "") -> None:
    if httpx is None:
        raise RuntimeError(f"httpx is required for Photon{what}")


def _dashboard_get(path: str, token: str, *, what: str = " device login") -> Any:
    _require_httpx(what)
    return httpx.get(f"{_dashboard_host()}{path}", headers=_bearer(token), timeout=30.0)


def _dashboard_post(path: str, body: Dict[str, Any], token: str, *, what: str = "") -> Any:
    """POST to the dashboard, raise for HTTP errors, return the decoded body."""
    _require_httpx(what)
    resp = httpx.post(f"{_dashboard_host()}{path}", json=body, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    return resp.json() or {}


def _response_error_detail(resp: Any) -> str:
    data = None
    with contextlib.suppress(Exception):
        data = resp.json()
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            if data.get(key):
                return str(data[key])
        return json.dumps(data, sort_keys=True)[:500]
    text = getattr(resp, "text", "") or ""
    return text[:500] if text else "no response body"


def _raise_for_status(resp: Any, action: str) -> None:
    status = getattr(resp, "status_code", 200)
    if status >= 400:
        raise RuntimeError(f"Photon {action} failed: HTTP {status}: {_response_error_detail(resp)}")


def _safe(fn: Callable[[], None]) -> None:
    with contextlib.suppress(Exception):
        fn()


# -- Device login flow (RFC 8628) ----------------------------------------------------

@dataclass
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str]
    expires_in: int
    interval: int


@dataclass(frozen=True)
class _DeviceTokenCandidate:
    """A token-like value extracted from the device-token response."""
    source: str
    token: str


def request_device_code(
    *, client_id: str = DEFAULT_CLIENT_ID, scope: Optional[str] = DEFAULT_SCOPE) -> DeviceCode:
    """POST ``/api/auth/device/code`` and return the device + user codes."""
    _require_httpx(" device login")
    body: Dict[str, Any] = {"client_id": client_id}
    if scope:
        body["scope"] = scope
    resp = httpx.post(f"{_dashboard_host()}/api/auth/device/code", json=body, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return DeviceCode(
        device_code=data["device_code"], user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        verification_uri_complete=data.get("verification_uri_complete"),
        expires_in=int(data.get("expires_in") or DEFAULT_POLL_TIMEOUT),
        interval=int(data.get("interval") or DEFAULT_POLL_INTERVAL))


def poll_for_token(
    code: DeviceCode, *, client_id: str = DEFAULT_CLIENT_ID, timeout: Optional[int] = None,
    interval: Optional[int] = None, on_pending: Optional[Callable[[], None]] = None) -> str:
    """Poll ``/api/auth/device/token`` until approved (official-CLI semantics: sleep first;
    ``authorization_pending`` keeps the interval, ``slow_down`` +5s, HTTP 429 +10s,
    ``access_denied``/``expired_token`` abort)."""
    _require_httpx(" device login")
    url = f"{_dashboard_host()}/api/auth/device/token"
    deadline = time.time() + (timeout or code.expires_in or DEFAULT_POLL_TIMEOUT)
    sleep = interval if interval is not None else (code.interval or DEFAULT_POLL_INTERVAL)
    grant = {"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": code.device_code,
             "client_id": client_id}

    def _pending() -> None:
        if on_pending:
            _safe(on_pending)
    while time.time() < deadline:
        time.sleep(sleep)
        try:
            resp = httpx.post(url, json=grant, timeout=30.0)
        except httpx.RequestError as e:
            logger.warning("photon: device-token poll failed: %s", e)
            continue
        if resp.status_code == 200:
            body: Any = {}
            with contextlib.suppress(TypeError, ValueError):  # json.JSONDecodeError is a ValueError
                body = resp.json() or {}
            body = body if isinstance(body, dict) else {}
            candidates = _device_response_token_candidates(body, headers=getattr(resp, "headers", {}))
            if not candidates:
                raise RuntimeError(
                    "Photon returned 200 but no token candidate in the device-token response "
                    "(expected access_token, data.access_token, accessToken, or set-auth-token).")
            return candidates[0].token
        if resp.status_code == 429:  # RFC 8628 §3.5 — treat as slow_down
            sleep += 10
            _pending()
            continue
        if resp.status_code == 400:
            body = {}
            with contextlib.suppress(json.JSONDecodeError):
                body = resp.json() or {}
            err = body.get("error") or body.get("message") or ""
            if err in ("authorization_pending", "slow_down"):
                if err == "slow_down":
                    sleep += 5
                _pending()
                continue
            if err in ("expired_token", "access_denied"):
                raise RuntimeError(f"Photon login failed: {err}")
            raise RuntimeError(f"Photon device token error: {err or resp.text}")
        logger.warning("photon: device-token unexpected status %s: %s", resp.status_code, resp.text[:200])
    raise TimeoutError("Photon device login timed out")


def _device_response_token_candidates(body: Dict[str, Any], *, headers: Optional[Any] = None) -> list:
    """De-duplicated token candidates from a device-token response — Photon has returned
    tokens under several keys across versions plus the ``set-auth-token`` header, so
    collect every shape for validation."""
    session = body.get("session") if isinstance(body.get("session"), dict) else {}
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    raw = (
        ("access_token", body.get("access_token")), ("accessToken", body.get("accessToken")),
        ("session.access_token", session.get("access_token")),
        ("data.access_token", data.get("access_token")), ("data.accessToken", data.get("accessToken")),
        ("set-auth-token", _header_value(headers, "set-auth-token")))
    candidates: list = []
    seen: set = set()
    for source, value in raw:
        token = _clean_bearer_token(value)
        if token and token not in seen:
            seen.add(token)
            candidates.append(_DeviceTokenCandidate(source=source, token=token))
    return candidates


def _clean_bearer_token(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    return (token[7:].strip() if token.lower().startswith("bearer ") else token) or None


def _header_value(headers: Optional[Any], name: str) -> Optional[str]:
    if not headers:
        return None
    with contextlib.suppress(AttributeError):
        value = headers.get(name)
        if value:
            return str(value)
    with contextlib.suppress(TypeError, ValueError):
        for key, value in dict(headers).items():
            if str(key).lower() == name.lower() and value:
                return str(value)
    return None


def validate_photon_token(token: str) -> Dict[str, Any]:
    """Verify a device-flow token against ``/api/auth/get-session`` AND ``/api/projects/`` —
    the device flow can mint tokens that pass the session lookup but are rejected by the
    project APIs setup depends on."""
    def _get(path: str, rejected: str) -> Any:
        resp = _dashboard_get(path, token)
        if resp.status_code in (401, 403):
            raise PhotonDashboardAuthError(rejected)
        resp.raise_for_status()
        return resp
    data = _get("/api/auth/get-session",
                "Photon issued a device token, but the dashboard session lookup rejected it.").json()
    user = data.get("user") if isinstance(data, dict) else None
    if not isinstance(user, dict) or not user:
        raise PhotonDashboardAuthError(
            "Photon issued a device token, but the dashboard session lookup did not recognize it.")
    _get("/api/projects/", "Photon device token was accepted for the session lookup but rejected by the project API.")
    return user


def _validated_dashboard_token(candidates: list) -> str:
    """Return the first candidate token that passes dashboard validation."""
    if not candidates:
        raise RuntimeError("Photon returned 200 but no token candidate in the device-token response.")
    dashboard_error: Optional[PhotonDashboardAuthError] = None
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            validate_photon_token(candidate.token)
            return candidate.token
        except Exception as exc:
            last_error = exc
            if isinstance(exc, PhotonDashboardAuthError):
                dashboard_error = exc
    if dashboard_error is not None:
        sources = ", ".join(c.source for c in candidates) or "none"
        raise PhotonDashboardAuthError(
            f"{dashboard_error} Device login returned no project-valid dashboard token (tried: {sources})."
        ) from dashboard_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Photon did not return a usable dashboard token")


def login_device_flow(
    *, client_id: str = DEFAULT_CLIENT_ID, open_browser: bool = True,
    on_user_code: Optional[Callable[["DeviceCode"], None]] = None) -> str:
    """Run the full device-code login flow, validate the token against the dashboard API
    before persisting it, and return it. ``on_user_code`` receives the :class:`DeviceCode`."""
    code = request_device_code(client_id=client_id)
    if on_user_code:
        _safe(lambda: on_user_code(code))
    if open_browser:
        with contextlib.suppress(Exception):
            import webbrowser
            webbrowser.open(code.verification_uri_complete or code.verification_uri, new=2)
    first_token = poll_for_token(code, client_id=client_id)
    token = _validated_dashboard_token([_DeviceTokenCandidate(source="poll", token=first_token)])
    store_photon_token(token)
    return token


# -- Dashboard API: projects --------------------------------------------------------

def _unwrap_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "projects", "users", "lines", "items"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict):
                nested = next((inner[k] for k in ("projects", "users", "lines", "items")
                               if isinstance(inner.get(k), list)), None)
                if nested is not None:
                    return nested
    return []


def _dashboard_list(path: str, token: str) -> List[Dict[str, Any]]:
    resp = _dashboard_get(path, token, what="")
    resp.raise_for_status()
    return _unwrap_list(resp.json())


def _raise_on_error_key(data: Dict[str, Any], action: str) -> None:
    if data.get("error"):
        raise RuntimeError(f"Photon {action} failed: {data['error']}")


def list_projects(token: str) -> List[Dict[str, Any]]:
    """GET ``/api/projects`` — return the caller's projects."""
    return _dashboard_list("/api/projects", token)


def find_project_by_name(token: str, name: str) -> Optional[Dict[str, Any]]:
    """First project whose name matches (case-insensitive)."""
    target = (name or "").strip().lower()
    for proj in list_projects(token):
        if (proj.get("name") or "").strip().lower() == target:
            return proj
    return None


def create_project(
    token: str, *, name: str = DEFAULT_PROJECT_NAME, location: str = "United States") -> Dict[str, Any]:
    """POST ``/api/projects`` and return the project (Spectrum is always provisioned;
    the request carries no ``spectrum`` flag)."""
    body: Dict[str, Any] = {"name": name, "location": location, "template": False, "observability": False}
    data = _dashboard_post("/api/projects", body, token, what=" project creation")
    if not isinstance(data, dict):
        raise RuntimeError("Photon create-project returned an unexpected response")
    _raise_on_error_key(data, "create-project")
    if data.get("succeed") is False:
        raise RuntimeError(f"Photon create-project failed: {data.get('message') or data}")
    project: Dict[str, Any] = data["data"] if isinstance(data.get("data"), dict) else data
    if not project.get("id"):
        raise RuntimeError("Photon create-project did not return a project id")
    return project


def regenerate_project_secret(token: str, project_id: str) -> str:
    """POST ``/api/projects/{id}/regenerate-secret`` → the new secret (the only way to
    read one — persist it immediately)."""
    data = _dashboard_post(f"/api/projects/{project_id}/regenerate-secret", {}, token)
    _raise_on_error_key(data, "regenerate-secret")
    secret = data.get("projectSecret")
    if not secret:
        raise RuntimeError("Photon regenerate-secret returned no projectSecret")
    return str(secret)


# -- Spectrum API: users -------------------------------------------------------------

def _normalize_phone(phone: str) -> str:
    """Reduce a phone string to ``+`` and digits for dedup comparison."""
    return re.sub(r"[^\d+]", "", phone or "")


def list_users(project_id: str, project_secret: str) -> List[Dict[str, Any]]:
    """GET Spectrum Cloud ``/projects/{id}/users/`` → ``SpectrumUser[]``."""
    _require_httpx()
    url = f"{_spectrum_host()}/projects/{project_id}/users/"
    resp = httpx.get(url, headers=_basic(project_id, project_secret), timeout=30.0)
    _raise_for_status(resp, "list-users")
    return _unwrap_list(resp.json())


def find_user_by_phone(project_id: str, project_secret: str, phone_number: str) -> Optional[Dict[str, Any]]:
    """Existing Spectrum user with the given phone number, or None."""
    target = _normalize_phone(phone_number)
    for user in list_users(project_id, project_secret):
        if _normalize_phone(user.get("phoneNumber") or "") == target:
            return user
    return None


def create_user(
    project_id: str, project_secret: str, *, phone_number: str, first_name: Optional[str] = None,
    last_name: Optional[str] = None, email: Optional[str] = None, send_invite: bool = False) -> Dict[str, Any]:
    """POST Spectrum Cloud ``/projects/{id}/users/`` and return the user."""
    _require_httpx(" user creation")
    if not E164_RE.match(phone_number):
        raise ValueError(f"phone_number must be E.164 (e.g. +15551234567); got {phone_number!r}")
    url = f"{_spectrum_host()}/projects/{project_id}/users/"
    body: Dict[str, Any] = {"type": "shared", "phoneNumber": phone_number}
    if send_invite:
        logger.debug("photon: send_invite is ignored by Spectrum shared-user creation")
    body.update({k: v for k, v in (("firstName", first_name), ("lastName", last_name), ("email", email)) if v})
    resp = httpx.post(url, json=body, headers=_basic(project_id, project_secret), timeout=30.0)
    _raise_for_status(resp, "create-user")
    data = resp.json() or {}
    _raise_on_error_key(data, "create-user")
    user = data.get("user") or data.get("data") or data
    if not isinstance(user, dict):
        raise RuntimeError("Photon create-user returned an unexpected response")
    return user


def register_user_if_absent(
    project_id: str, project_secret: str, *, phone_number: str, first_name: Optional[str] = None,
    last_name: Optional[str] = None, email: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
    """Idempotently register a Spectrum user → ``(user, created)``; the official CLI does
    no dedup, so we add it to keep ``setup`` re-runnable."""
    existing = find_user_by_phone(project_id, project_secret, phone_number)
    if existing is not None:
        return existing, False
    return create_user(project_id, project_secret, phone_number=phone_number, first_name=first_name,
                       last_name=last_name, email=email), True


def user_assigned_line(user: Optional[Dict[str, Any]]) -> Optional[str]:
    """The iMessage number a user texts to reach the agent (``assignedPhoneNumber``, the
    dashboard's "TEXTS ON" column). None when unset (freshly created user)."""
    val = user.get("assignedPhoneNumber") if user else None
    return str(val) if val else None


def load_user_numbers() -> Tuple[Optional[str], Optional[str]]:
    """``(operator_phone_number, assigned_phone_number)`` for status."""
    entry = _pool_first(_load_auth(), "photon_user")
    entry = entry if isinstance(entry, dict) else {}
    phone = entry.get("phone_number") or entry.get("phoneNumber")
    assigned = entry.get("assigned_phone_number") or entry.get("assignedPhoneNumber")
    return (str(phone) if phone else _configured_operator_phone(), str(assigned) if assigned else None)


def refresh_user_numbers(project_id: str, project_secret: str) -> Tuple[Optional[str], Optional[str]]:
    """Refresh cached user numbers from Photon without provisioning anything."""
    phone, assigned = load_user_numbers()
    if phone:
        user = find_user_by_phone(project_id, project_secret, phone)
    else:
        users = list_users(project_id, project_secret)
        user = users[0] if len(users) == 1 else None
    user_id = None
    if user:
        user_id = user.get("id")
        dashboard_phone = _normalize_phone(str(user.get("phoneNumber") or ""))
        if E164_RE.match(dashboard_phone):
            phone = dashboard_phone
        assigned = user_assigned_line(user)
    dashboard_id = load_dashboard_project_id()
    dashboard_token = load_photon_token() if not assigned else None
    if dashboard_token and dashboard_id:
        try:
            line = get_imessage_line(dashboard_token, dashboard_id, create_if_missing=False)
            if line and line.get("phoneNumber"):
                assigned = str(line["phoneNumber"])
        except Exception as e:
            logger.debug("photon: could not refresh iMessage line for status: %s", e)
    store_user_numbers(phone_number=phone, assigned_phone_number=assigned,
                       user_id=str(user_id) if user_id else None, dashboard_project_id=dashboard_id)
    return phone, assigned


def _configured_operator_phone() -> Optional[str]:
    """Infer the operator's E.164 number from existing Photon env settings."""
    home = _normalize_phone(_get_config_env_value("PHOTON_HOME_CHANNEL") or "")
    if home and E164_RE.match(home):
        return home
    allowed = _get_config_env_value("PHOTON_ALLOWED_USERS") or ""
    candidates = [n for n in map(_normalize_phone, re.split(r"[,\s]+", allowed)) if E164_RE.match(n)]
    return candidates[0] if len(candidates) == 1 else None


def _get_config_env_value(key: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value
    except Exception:
        return os.getenv(key)
    return get_env_value(key)


# -- Dashboard API: iMessage lines (the assigned number inventory) --------------------

def list_lines(token: str, project_id: str) -> List[Dict[str, Any]]:
    """GET ``/api/projects/{id}/lines`` → ``[{id, platform, phoneNumber, status}]``."""
    return _dashboard_list(f"/api/projects/{project_id}/lines", token)


def add_line(token: str, project_id: str, *, platform: str = "imessage") -> Dict[str, Any]:
    """POST ``/api/projects/{id}/lines`` to provision a new line."""
    data = _dashboard_post(f"/api/projects/{project_id}/lines", {"platform": platform}, token)
    _raise_on_error_key(data, "add-line")
    return data.get("line") or data


def get_imessage_line(
    token: str, project_id: str, *, create_if_missing: bool = True) -> Optional[Dict[str, Any]]:
    """The project's iMessage line, provisioning one if absent and ``create_if_missing``;
    None if there is none and provisioning failed."""
    line = next((ln for ln in list_lines(token, project_id) if (ln.get("platform") or "").lower() == "imessage"), None)
    if line is not None or not create_if_missing:
        return line
    try:
        return add_line(token, project_id, platform="imessage")
    except Exception as e:
        logger.warning("photon: could not auto-provision iMessage line: %s", e)
        return None


# -- Credential status (display-only — never emits raw secret material) ---------------

def print_credential_summary(emit: Any = print) -> None:
    """Pretty-print the credential status table via *emit*. Every secret-bearing read is
    reduced to a display literal here; the callback only ever receives the assembled
    banner, so no tainted value escapes to the caller."""
    sid, sec = load_project_credentials()
    phone, assigned = load_user_numbers()
    rows = [
        "Photon iMessage status",
        "──────────────────────",
        "  device token        : " + (
            "✓ stored" if load_photon_token() else "✗ missing (run `hermes photon setup`)"),
        "  project id          : " + (sid if sid else "✗ missing"),
        "  project secret      : " + ("✓ stored" if sec else "✗ missing"),
        "  my number           : " + (phone if phone else "✗ missing (run `hermes photon setup --phone ...`)"),
        "  assigned number     : " + (assigned if assigned else "✗ missing (run `hermes photon setup`)")]
    emit("\n".join(rows))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def credential_summary() -> Dict[str, str]:
    """Return a fully pre-formatted credential status dict (no raw secrets)."""
    def _present_token() -> str:
        return (
            "✓ stored" if load_photon_token()
            else "✗ missing (run `hermes photon setup`)"
        )

    def _present_project_id() -> str:
        sid, _sec = load_project_credentials()
        return sid or "✗ missing"

    def _present_secret() -> str:
        _sid, sec = load_project_credentials()
        return "✓ stored" if sec else "✗ missing"

    def _present_phone() -> str:
        phone, _assigned = load_user_numbers()
        return phone or "✗ missing (run `hermes photon setup --phone ...`)"

    def _present_assigned_phone() -> str:
        _phone, assigned = load_user_numbers()
        return assigned or "✗ missing (run `hermes photon setup`)"

    return {
        "device_token": _present_token(),
        "project_id": _present_project_id(),
        "project_key": _present_secret(),
        "phone_number": _present_phone(),
        "assigned_phone_number": _present_assigned_phone(),
    }

def get_session(token: str) -> Dict[str, Any]:
    """GET ``/api/auth/get-session`` — confirm the token + fetch the user."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/auth/get-session"
    resp = httpx.get(url, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    return resp.json() or {}
# ---- END PLUGIN-COMPAT ----
