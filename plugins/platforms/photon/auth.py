"""
Photon Dashboard API client + device-code login flow.

This module is pure Python — it intentionally does not depend on
``spectrum-ts``.  Every management-plane operation (login, find/create
project, rotate the project secret, register a user, list the assigned
iMessage line) talks to Photon's **Dashboard API** on a single host,
exactly like the official Photon CLI (``photon-hq/cli``):

    Dashboard API   https://app.photon.codes/api/...
                    OAuth 2.0 device flow, Bearer access token

A Photon project has a single identifier: the dashboard ``id`` *is* the
Spectrum Cloud project id. They used to diverge (a separate
``spectrumProjectId`` field), but the dashboard unified them — every
project is created with matching ids and the pre-existing diverged rows
were backfilled so ``project.id == spectrumProjectId`` everywhere
(dashboard ENG-1582). Spectrum is always enabled and provisioned at
create-time, so there is no enable/toggle step anymore.

The ``spectrum-ts`` SDK (run by the Node sidecar) authenticates to Spectrum
Cloud with ``(id, projectSecret)`` — the same ``id`` used in Dashboard API
paths — which we persist as ``PHOTON_PROJECT_ID`` for the runtime.

Credential storage mirrors every other Hermes channel:

    * runtime SDK creds  -> ``~/.hermes/.env``  (``PHOTON_PROJECT_ID`` =
      project id, ``PHOTON_PROJECT_SECRET``) via ``save_env_value``
    * management metadata -> ``~/.hermes/auth.json`` under
      ``credential_pool.photon`` (device token),
      ``credential_pool.photon_project`` (dashboard id, spectrum id, name), and
      ``credential_pool.photon_user`` (operator number + assigned text line)

Reference: https://github.com/photon-hq/cli and
https://photon.codes/docs/api-reference/device-login/request-device-+-user-code
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

logger = logging.getLogger(__name__)


class PhotonDashboardAuthError(RuntimeError):
    """Raised when Photon rejects a device-flow token for the dashboard API."""

# ---------------------------------------------------------------------------
# Constants

# Hosted Photon allowlists registered device clients on the device-code
# endpoint — an unregistered client_id is rejected with
# `400 {"error":"invalid_client"}`.  Use Photon's published CLI device
# client (matches `CLI_CLIENT_ID` in photon-hq/cli) until the dashboard API
# registers Hermes as its own client_id.
DEFAULT_CLIENT_ID = "photon-cli"
DEFAULT_SCOPE = "openid profile email"

DEFAULT_DASHBOARD_HOST = "https://app.photon.codes"
DEFAULT_SPECTRUM_HOST = "https://spectrum.photon.codes"

# Default name of the project Hermes provisions for the operator.
DEFAULT_PROJECT_NAME = "Hermes Agent"

# Polling defaults per RFC 8628.  Photon overrides via `interval` /
# `expires_in` in the device-code response — those win.
DEFAULT_POLL_INTERVAL = 5
DEFAULT_POLL_TIMEOUT = 1800  # 30 min, matching the CLI's fallback

E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


# ---------------------------------------------------------------------------
# auth.json helpers — share the file with the rest of hermes-agent.

def _auth_json_path() -> Path:
    """Resolve ``~/.hermes/auth.json`` honouring the active Hermes profile."""
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
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("photon: could not read %s: %s", path, e)
        return {}


def _save_auth(data: Dict[str, Any]) -> None:
    path = _auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process random temp suffix avoids collisions between concurrent
    # writers and stale leftovers from a crashed prior write.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    # Create with 0o600 atomically via os.open(O_EXCL) + fdopen: the old
    # open() → write → chmod() sequence left a window where the bearer
    # token sat world-readable at process umask (typically 0o644), and the
    # predictable temp name could be pre-planted (symlink attack). Mirrors
    # hermes_cli/auth.py:_save_auth_store (#19673, #21148).
    fd = os.open(
        str(tmp),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def load_photon_token() -> Optional[str]:
    """Return the device-flow bearer token stored by ``login()`` or ``None``."""
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
    """Return the runtime SDK creds ``(spectrum_project_id, project_secret)``.

    Precedence: process env (``~/.hermes/.env`` is loaded into the gateway's
    environment at startup) wins, then ``auth.json`` for offline / status
    use.  This is the pair the Node sidecar feeds to ``spectrum-ts``; the id
    is the unified project id (dashboard id == spectrumProjectId).
    """
    env_id = os.getenv("PHOTON_PROJECT_ID")
    env_sec = os.getenv("PHOTON_PROJECT_SECRET")
    if env_id and env_sec:
        return env_id, env_sec
    auth = _load_auth()
    proj = auth.get("credential_pool", {}).get("photon_project") or []
    if isinstance(proj, list) and proj:
        entry = proj[0]
        # back-compat: old records used "project_id" for the spectrum id
        sid = entry.get("spectrum_project_id") or entry.get("project_id")
        return (env_id or sid, env_sec or entry.get("project_secret"))
    return env_id, env_sec


def load_dashboard_project_id() -> Optional[str]:
    """Return the project id used for management API calls.

    Post-unification the dashboard id and the Spectrum id are the same value,
    so we prefer the stored ``spectrum_project_id``: for pre-backfill installs
    the old ``dashboard_project_id`` is the diverged id that the unification
    rewrote (it now 404s), while the Spectrum id always matches the live row.
    Falls back to the legacy keys for older records.
    """
    env_id = os.getenv("PHOTON_DASHBOARD_PROJECT_ID")
    if env_id:
        return env_id
    auth = _load_auth()
    proj = auth.get("credential_pool", {}).get("photon_project") or []
    if isinstance(proj, list) and proj:
        entry = proj[0]
        return (
            entry.get("spectrum_project_id")
            or entry.get("dashboard_project_id")
            or entry.get("project_id")
        )
    return None


def store_project_credentials(
    *,
    spectrum_project_id: str,
    project_secret: str,
    dashboard_project_id: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Persist project credentials to both .env (runtime) and auth.json (mgmt).

    The runtime SDK creds land in ``~/.hermes/.env`` via the same
    ``save_env_value`` helper every other channel uses, so the gateway picks
    them up from the environment with zero adapter changes.  A copy of the
    non-secret ids (plus the secret, for offline ``status``) is written to
    ``auth.json`` so management commands work even when ``.env`` hasn't been
    loaded into the current process.
    """
    auth = _load_auth()
    record: Dict[str, Any] = {
        "spectrum_project_id": spectrum_project_id,
        "project_secret": project_secret,
        "issued_at": int(time.time()),
    }
    if dashboard_project_id:
        record["dashboard_project_id"] = dashboard_project_id
    if name:
        record["name"] = name
    auth.setdefault("credential_pool", {})["photon_project"] = [record]
    _save_auth(auth)
    _persist_runtime_env(spectrum_project_id, project_secret)


def store_user_numbers(
    *,
    phone_number: Optional[str] = None,
    assigned_phone_number: Optional[str] = None,
    user_id: Optional[str] = None,
    dashboard_project_id: Optional[str] = None,
) -> None:
    """Persist non-secret Photon user numbers for offline ``status`` output."""
    if not phone_number and not assigned_phone_number:
        return
    auth = _load_auth()
    record: Dict[str, Any] = {"issued_at": int(time.time())}
    if phone_number:
        record["phone_number"] = phone_number
    if assigned_phone_number:
        record["assigned_phone_number"] = assigned_phone_number
    if user_id:
        record["user_id"] = user_id
    if dashboard_project_id:
        record["dashboard_project_id"] = dashboard_project_id
    auth.setdefault("credential_pool", {})["photon_user"] = [record]
    _save_auth(auth)


def _persist_runtime_env(spectrum_project_id: str, project_secret: str) -> None:
    """Write the SDK creds to ``~/.hermes/.env`` (canonical runtime store).

    Isolated in its own helper so the secret value flows straight into
    ``save_env_value`` without ever being bound to a printable local in a
    caller — same CodeQL-clean-flow rationale as the rest of this module.
    """
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


@dataclass(frozen=True)
class _DeviceTokenCandidate:
    """A token-like value extracted from the device-token response."""
    source: str
    token: str


def _dashboard_host() -> str:
    return (os.getenv("PHOTON_DASHBOARD_HOST") or DEFAULT_DASHBOARD_HOST).rstrip("/")


def _spectrum_host() -> str:
    return (os.getenv("PHOTON_SPECTRUM_HOST") or DEFAULT_SPECTRUM_HOST).rstrip("/")


def _bearer(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _basic(project_id: str, project_secret: str) -> Dict[str, str]:
    token = b64encode(f"{project_id}:{project_secret}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _response_error_detail(resp: Any) -> str:
    try:
        data = resp.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            val = data.get(key)
            if val:
                return str(val)
        return json.dumps(data, sort_keys=True)[:500]
    text = getattr(resp, "text", "") or ""
    return text[:500] if text else "no response body"


def _raise_for_status(resp: Any, action: str) -> None:
    status = getattr(resp, "status_code", 200)
    if status < 400:
        return
    raise RuntimeError(
        f"Photon {action} failed: HTTP {status}: {_response_error_detail(resp)}"
    )


def request_device_code(
    *, client_id: str = DEFAULT_CLIENT_ID, scope: Optional[str] = DEFAULT_SCOPE,
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
    on_pending: Optional[Callable[[], None]] = None,
) -> str:
    """Poll ``/api/auth/device/token`` until the user approves.

    Mirrors the official CLI's polling loop: sleep first, then poll;
    ``authorization_pending`` keeps the interval, ``slow_down`` adds 5s,
    HTTP 429 adds 10s, and ``access_denied`` / ``expired_token`` abort.

    The bearer token comes from the response body's top-level
    ``access_token`` (better-auth device-grant shape), with
    ``session.access_token`` and the ``set-auth-token`` header kept as
    fallbacks for API drift.
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon device login")
    url = f"{_dashboard_host()}/api/auth/device/token"
    deadline = time.time() + (timeout or code.expires_in or DEFAULT_POLL_TIMEOUT)
    sleep = interval if interval is not None else (code.interval or DEFAULT_POLL_INTERVAL)
    while time.time() < deadline:
        time.sleep(sleep)
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
            continue
        if resp.status_code == 200:
            body: Dict[str, Any] = {}
            try:
                decoded = resp.json() or {}
                body = decoded if isinstance(decoded, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                body = {}
            candidates = _device_response_token_candidates(
                body, headers=getattr(resp, "headers", {}),
            )
            if not candidates:
                raise RuntimeError(
                    "Photon returned 200 but no token candidate in the "
                    "device-token response (expected access_token, "
                    "data.access_token, accessToken, or set-auth-token)."
                )
            return candidates[0].token
        if resp.status_code == 429:
            # RFC 8628 §3.5 — treat 429 as slow_down.
            sleep += 10
            if on_pending:
                _safe(on_pending)
            continue
        if resp.status_code == 400:
            body = {}
            try:
                body = resp.json() or {}
            except json.JSONDecodeError:
                pass
            err = body.get("error") or body.get("message") or ""
            if err == "authorization_pending":
                if on_pending:
                    _safe(on_pending)
                continue
            if err == "slow_down":
                sleep += 5
                if on_pending:
                    _safe(on_pending)
                continue
            if err in ("expired_token", "access_denied"):
                raise RuntimeError(f"Photon login failed: {err}")
            raise RuntimeError(f"Photon device token error: {err or resp.text}")
        logger.warning(
            "photon: device-token unexpected status %s: %s",
            resp.status_code, resp.text[:200],
        )
    raise TimeoutError("Photon device login timed out")


def _device_response_token_candidates(
    body: Dict[str, Any],
    *,
    headers: Optional[Any] = None,
) -> list:
    """Extract de-duplicated token candidates from a device-token response.

    Photon's device-token endpoint has returned tokens under several keys
    across versions (``access_token``, ``accessToken``, ``data.*``) and the
    documented ``set-auth-token`` response header.  We collect every shape so
    the caller can validate each against the dashboard API before trusting it.
    """
    candidates: list = []
    seen: set = set()

    def add(source: str, value: Any) -> None:
        token = _clean_bearer_token(value)
        if not token or token in seen:
            return
        seen.add(token)
        candidates.append(_DeviceTokenCandidate(source=source, token=token))

    add("access_token", body.get("access_token"))
    add("accessToken", body.get("accessToken"))
    session = body.get("session")
    if isinstance(session, dict):
        add("session.access_token", session.get("access_token"))
    data = body.get("data")
    if isinstance(data, dict):
        add("data.access_token", data.get("access_token"))
        add("data.accessToken", data.get("accessToken"))
    add("set-auth-token", _header_value(headers, "set-auth-token"))
    return candidates


def _clean_bearer_token(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def _header_value(headers: Optional[Any], name: str) -> Optional[str]:
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value:
            return str(value)
    except AttributeError:
        pass
    try:
        for key, value in dict(headers).items():
            if str(key).lower() == name.lower() and value:
                return str(value)
    except (TypeError, ValueError):
        return None
    return None


def _dashboard_get(path: str, token: str) -> Any:
    if httpx is None:
        raise RuntimeError("httpx is required for Photon device login")
    url = f"{_dashboard_host()}{path}"
    return httpx.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )


def validate_photon_token(token: str) -> Dict[str, Any]:
    """Verify a device-flow token is usable for dashboard project APIs.

    The device flow can return a token that authenticates the Better Auth
    session lookup but is rejected by the project APIs.  Validate against
    ``/api/auth/get-session`` and ``/api/projects/`` so we fail loudly at
    login instead of saving a token that 404s/401s downstream.
    """
    resp = _dashboard_get("/api/auth/get-session", token)
    if resp.status_code in (401, 403):
        raise PhotonDashboardAuthError(
            "Photon issued a device token, but the dashboard session lookup "
            "rejected it."
        )
    resp.raise_for_status()
    data = resp.json()
    user = data.get("user") if isinstance(data, dict) else None
    if not isinstance(user, dict) or not user:
        raise PhotonDashboardAuthError(
            "Photon issued a device token, but the dashboard session lookup "
            "did not recognize it."
        )
    projects_resp = _dashboard_get("/api/projects/", token)
    if projects_resp.status_code in (401, 403):
        raise PhotonDashboardAuthError(
            "Photon device token was accepted for the session lookup but "
            "rejected by the project API."
        )
    projects_resp.raise_for_status()
    return user


def _validated_dashboard_token(candidates: list) -> str:
    """Return the first candidate token that passes dashboard validation."""
    if not candidates:
        raise RuntimeError(
            "Photon returned 200 but no token candidate in the device-token "
            "response."
        )
    dashboard_error: Optional[PhotonDashboardAuthError] = None
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            validate_photon_token(candidate.token)
            return candidate.token
        except PhotonDashboardAuthError as exc:
            dashboard_error = exc
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue
    if dashboard_error is not None:
        sources = ", ".join(c.source for c in candidates) or "none"
        raise PhotonDashboardAuthError(
            f"{dashboard_error} Device login returned no project-valid "
            f"dashboard token (tried: {sources})."
        ) from dashboard_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Photon did not return a usable dashboard token")


def _safe(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        pass


def login_device_flow(
    *,
    client_id: str = DEFAULT_CLIENT_ID,
    open_browser: bool = True,
    on_user_code: Optional[Callable[["DeviceCode"], None]] = None,
) -> str:
    """Run the full device-code login flow and persist the token.

    Returns the bearer token.  ``on_user_code`` receives the
    :class:`DeviceCode` so callers can print it + optionally open a browser.
    """
    code = request_device_code(client_id=client_id)
    if on_user_code:
        _safe(lambda: on_user_code(code))
    if open_browser:
        try:
            import webbrowser
            target = code.verification_uri_complete or code.verification_uri
            webbrowser.open(target, new=2)
        except Exception:
            pass
    # Poll once for the approved token, then collect every candidate shape so
    # we can validate against the dashboard API before persisting (avoids
    # saving a token that authenticates the session lookup but 404s on the
    # project APIs).
    first_token = poll_for_token(code, client_id=client_id)
    candidates = [_DeviceTokenCandidate(source="poll", token=first_token)]
    token = _validated_dashboard_token(candidates)
    store_photon_token(token)
    return token


def get_session(token: str) -> Dict[str, Any]:
    """GET ``/api/auth/get-session`` — confirm the token + fetch the user."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/auth/get-session"
    resp = httpx.get(url, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    return resp.json() or {}


# ---------------------------------------------------------------------------
# Dashboard API: projects

def _unwrap_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "projects", "users", "lines", "items"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict):
                for nested_key in ("projects", "users", "lines", "items"):
                    nested = inner.get(nested_key)
                    if isinstance(nested, list):
                        return nested
    return []


def list_projects(token: str) -> List[Dict[str, Any]]:
    """GET ``/api/projects`` — return the caller's projects."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/projects"
    resp = httpx.get(url, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    return _unwrap_list(resp.json())


def find_project_by_name(token: str, name: str) -> Optional[Dict[str, Any]]:
    """Return the first project whose name matches (case-insensitive)."""
    target = (name or "").strip().lower()
    for proj in list_projects(token):
        if (proj.get("name") or "").strip().lower() == target:
            return proj
    return None


def create_project(
    token: str,
    *,
    name: str = DEFAULT_PROJECT_NAME,
    location: str = "United States",
) -> Dict[str, Any]:
    """POST ``/api/projects`` and return ``{success, id}``.

    Spectrum is always provisioned at create-time, so the request body no
    longer carries a ``spectrum`` flag (the field was dropped from the API).
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon project creation")
    url = f"{_dashboard_host()}/api/projects"
    body: Dict[str, Any] = {
        "name": name,
        "location": location,
        "template": False,
        "observability": False,
    }
    resp = httpx.post(url, json=body, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    data = resp.json() or {}
    if data.get("error"):
        raise RuntimeError(f"Photon create-project failed: {data['error']}")
    if not data.get("id"):
        raise RuntimeError("Photon create-project did not return a project id")
    return data


def regenerate_project_secret(token: str, project_id: str) -> str:
    """POST ``/api/projects/{id}/regenerate-secret`` → the new project secret.

    This is the only way to read a project secret (the dashboard shows it
    exactly once), so callers should persist the returned value immediately.
    """
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/projects/{project_id}/regenerate-secret"
    resp = httpx.post(url, json={}, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    data = resp.json() or {}
    if data.get("error"):
        raise RuntimeError(f"Photon regenerate-secret failed: {data['error']}")
    secret = data.get("projectSecret")
    if not secret:
        raise RuntimeError("Photon regenerate-secret returned no projectSecret")
    return str(secret)


# ---------------------------------------------------------------------------
# Spectrum API: users

def _normalize_phone(phone: str) -> str:
    """Reduce a phone string to ``+`` and digits for dedup comparison."""
    return re.sub(r"[^\d+]", "", phone or "")


def list_users(project_id: str, project_secret: str) -> List[Dict[str, Any]]:
    """GET Spectrum Cloud ``/projects/{id}/users/`` → ``SpectrumUser[]``."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_spectrum_host()}/projects/{project_id}/users/"
    resp = httpx.get(url, headers=_basic(project_id, project_secret), timeout=30.0)
    _raise_for_status(resp, "list-users")
    return _unwrap_list(resp.json())


def find_user_by_phone(
    project_id: str, project_secret: str, phone_number: str,
) -> Optional[Dict[str, Any]]:
    """Return an existing Spectrum user with the given phone number, or None."""
    target = _normalize_phone(phone_number)
    for user in list_users(project_id, project_secret):
        if _normalize_phone(user.get("phoneNumber") or "") == target:
            return user
    return None


def create_user(
    project_id: str,
    project_secret: str,
    *,
    phone_number: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
    send_invite: bool = False,
) -> Dict[str, Any]:
    """POST Spectrum Cloud ``/projects/{id}/users/`` and return the user."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon user creation")
    if not E164_RE.match(phone_number):
        raise ValueError(
            f"phone_number must be E.164 (e.g. +15551234567); got {phone_number!r}"
        )
    url = f"{_spectrum_host()}/projects/{project_id}/users/"
    body: Dict[str, Any] = {"type": "shared", "phoneNumber": phone_number}
    if send_invite:
        logger.debug("photon: send_invite is ignored by Spectrum shared-user creation")
    if first_name:
        body["firstName"] = first_name
    if last_name:
        body["lastName"] = last_name
    if email:
        body["email"] = email
    resp = httpx.post(
        url,
        json=body,
        headers=_basic(project_id, project_secret),
        timeout=30.0,
    )
    _raise_for_status(resp, "create-user")
    data = resp.json() or {}
    if data.get("error"):
        raise RuntimeError(f"Photon create-user failed: {data['error']}")
    user = data.get("user") or data.get("data") or data
    if isinstance(user, dict):
        return user
    raise RuntimeError("Photon create-user returned an unexpected response")


def register_user_if_absent(
    project_id: str,
    project_secret: str,
    *,
    phone_number: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Idempotently register a Spectrum user.

    Returns ``(user, created)`` — ``created`` is False when a user with the
    same phone number already exists (the official CLI does no dedup, so we
    add it here to make ``setup`` safely re-runnable).
    """
    existing = find_user_by_phone(project_id, project_secret, phone_number)
    if existing is not None:
        return existing, False
    user = create_user(
        project_id,
        project_secret,
        phone_number=phone_number,
        first_name=first_name,
        last_name=last_name,
        email=email,
    )
    return user, True


def user_assigned_line(user: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the iMessage number a Spectrum user is assigned to text on.

    This is the user's ``assignedPhoneNumber`` (the dashboard's "TEXTS ON"
    column) — i.e. the number to text to reach the agent, as opposed to the
    user's own ``phoneNumber``. On shared-number plans there is no dedicated
    entry in ``/lines``, so this per-user field is the source of truth.
    Returns ``None`` when unset (e.g. a freshly created, not-yet-assigned user).
    """
    if not user:
        return None
    val = user.get("assignedPhoneNumber")
    return str(val) if val else None


def load_user_numbers() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(operator_phone_number, assigned_phone_number)`` for status."""
    auth = _load_auth()
    user_entries = auth.get("credential_pool", {}).get("photon_user") or []
    if isinstance(user_entries, list) and user_entries:
        entry = user_entries[0] or {}
        if isinstance(entry, dict):
            phone = entry.get("phone_number") or entry.get("phoneNumber")
            assigned = (
                entry.get("assigned_phone_number")
                or entry.get("assignedPhoneNumber")
            )
            if phone or assigned:
                return (
                    str(phone) if phone else _configured_operator_phone(),
                    str(assigned) if assigned else None,
                )
    return _configured_operator_phone(), None


def refresh_user_numbers(
    project_id: str, project_secret: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Refresh cached user numbers from Photon without provisioning anything."""
    phone, cached_assigned = load_user_numbers()
    user: Optional[Dict[str, Any]] = None
    if phone:
        user = find_user_by_phone(project_id, project_secret, phone)
    else:
        users = list_users(project_id, project_secret)
        if len(users) == 1:
            user = users[0]

    user_id = None
    assigned: Optional[str] = cached_assigned
    if user:
        user_id = user.get("id")
        dashboard_phone = _normalize_phone(str(user.get("phoneNumber") or ""))
        if E164_RE.match(dashboard_phone):
            phone = dashboard_phone
        assigned = user_assigned_line(user)

    dashboard_id = load_dashboard_project_id()
    if not assigned:
        dashboard_token = load_photon_token()
        if dashboard_token and dashboard_id:
            try:
                line = get_imessage_line(
                    dashboard_token,
                    dashboard_id,
                    create_if_missing=False,
                )
            except Exception as e:
                logger.debug(
                    "photon: could not refresh iMessage line for status: %s", e
                )
            else:
                if line and line.get("phoneNumber"):
                    assigned = str(line["phoneNumber"])

    store_user_numbers(
        phone_number=phone,
        assigned_phone_number=assigned,
        user_id=str(user_id) if user_id else None,
        dashboard_project_id=dashboard_id,
    )
    return phone, assigned


def _configured_operator_phone() -> Optional[str]:
    """Infer the operator's E.164 number from existing Photon env settings."""
    home = _get_config_env_value("PHOTON_HOME_CHANNEL")
    if home:
        normalized = _normalize_phone(home)
        if E164_RE.match(normalized):
            return normalized

    allowed = _get_config_env_value("PHOTON_ALLOWED_USERS")
    if not allowed:
        return None
    candidates = []
    for part in re.split(r"[,\s]+", allowed):
        normalized = _normalize_phone(part)
        if E164_RE.match(normalized):
            candidates.append(normalized)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _get_config_env_value(key: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value
    except Exception:
        return os.getenv(key)
    return get_env_value(key)


# ---------------------------------------------------------------------------
# Dashboard API: iMessage lines (the assigned number inventory)

def list_lines(token: str, project_id: str) -> List[Dict[str, Any]]:
    """GET ``/api/projects/{id}/lines`` → ``[{id, platform, phoneNumber, status}]``."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/projects/{project_id}/lines"
    resp = httpx.get(url, headers=_bearer(token), timeout=30.0)
    resp.raise_for_status()
    return _unwrap_list(resp.json())


def add_line(
    token: str, project_id: str, *, platform: str = "imessage",
) -> Dict[str, Any]:
    """POST ``/api/projects/{id}/lines`` to provision a new line."""
    if httpx is None:
        raise RuntimeError("httpx is required for Photon")
    url = f"{_dashboard_host()}/api/projects/{project_id}/lines"
    resp = httpx.post(
        url, json={"platform": platform}, headers=_bearer(token), timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json() or {}
    if data.get("error"):
        raise RuntimeError(f"Photon add-line failed: {data['error']}")
    return data.get("line") or data


def get_imessage_line(
    token: str, project_id: str, *, create_if_missing: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return the project's iMessage line (the number to text the agent).

    If none exists and ``create_if_missing`` is set, provision one.  Returns
    ``None`` if there is no line and provisioning failed.
    """
    for line in list_lines(token, project_id):
        if (line.get("platform") or "").lower() == "imessage":
            return line
    if create_if_missing:
        try:
            return add_line(token, project_id, platform="imessage")
        except Exception as e:
            logger.warning("photon: could not auto-provision iMessage line: %s", e)
            return None
    return None


# ---------------------------------------------------------------------------
# Credential status (display-only — never emits raw secret material)

def print_credential_summary(emit: Any = print) -> None:
    """Pretty-print the credential status table via the *emit* callback.

    Every secret-bearing read is reduced to a display literal inside this
    function (``"✓ stored"`` / ``"✗ missing"`` / a non-secret id); the
    callback only ever receives the assembled banner string, so no tainted
    value escapes into the caller's scope.
    """
    labels: Dict[str, str] = {}
    labels["device_token"] = (
        "✓ stored" if load_photon_token()
        else "✗ missing (run `hermes photon setup`)"
    )
    sid, sec = load_project_credentials()
    # Dashboard id and Spectrum id are the same value now (ids unified), so
    # there's a single project id to show.
    labels["project_id"] = sid if sid else "✗ missing"
    labels["project_key"] = "✓ stored" if sec else "✗ missing"
    phone, assigned = load_user_numbers()
    labels["phone_number"] = (
        phone if phone else "✗ missing (run `hermes photon setup --phone ...`)"
    )
    labels["assigned_phone_number"] = (
        assigned if assigned else "✗ missing (run `hermes photon setup`)"
    )

    rows = [
        "Photon iMessage status",
        "──────────────────────",
        "  device token        : " + labels["device_token"],
        "  project id          : " + labels["project_id"],
        "  project secret      : " + labels["project_key"],
        "  my number           : " + labels["phone_number"],
        "  assigned number     : " + labels["assigned_phone_number"],
    ]
    emit("\n".join(rows))


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
