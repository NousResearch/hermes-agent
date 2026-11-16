"""Google Drive client backed by a service account via ADC.

Credential resolution mirrors the Google Chat adapter
(``plugins/platforms/google_chat/adapter.py``) so the gateway can run on a
GCP VM / Cloud Run with an *attached* service account and zero key files:

  1. Explicit ``GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON`` env (path or inline JSON)
  2. ``GOOGLE_APPLICATION_CREDENTIALS`` env (path)
  3. Application Default Credentials via ``google.auth.default()`` — picks up
     the VM's workload identity automatically (the everafter-deployed path),
     or ``gcloud auth application-default login`` locally.

The agent acts as the *service account itself* (no domain-wide delegation):
share Drive files/folders with the SA's email to grant access. This keeps the
blast radius to exactly what's shared. Per-user access is enforced on top by
:mod:`plugins.google_drive_sa.access` — the SA's share is necessary, the
requesting user's own ACL entry is what actually grants a tool call.

Heavy google-* imports are lazy so the plugin can register its tools at
startup without the deps installed; first real use triggers a venv-scoped
install via :mod:`tools.lazy_deps`.

The built ``googleapiclient`` service wraps an ``httplib2.Http`` transport
that is NOT thread-safe, but ``plugins/google_drive_sa/access.py::filter_listing``
fans ACL fetches out over a thread pool (and two concurrent gateway turns can
call in on the same process besides). Credentials are safe to share across
threads, so they're cached process-wide; the *service* objects are cached
per-thread (``threading.local()``) so each worker thread gets its own
transport instead of racing on one.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

# Full-drive scope is required to see folders *shared with* the SA (drive.file
# only sees files the app itself created). Sheets/Docs structured editing each
# need their own scope. Override via env for least-privilege deployments.
# NOTE: scopes are only *requested* on the token — actual access still comes
# from sharing the file with the SA's email (Editor for writes).
_DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
)

_LAZY_FEATURE = "plugin.google_drive_sa"

_lock = threading.Lock()
_credentials: Any | None = None
# Multiplex only: credentials per profile home (``hermes_home_key``). The
# process-wide slot above is the single-profile cache and stays as it was.
_profile_credentials: dict[str, Any] = {}
_thread_services = threading.local()
_avail: bool | None = None
_deps_ready = False


def _multiplex() -> bool:
    from agent.secret_scope import is_multiplex_active

    return is_multiplex_active()


def _env(name: str) -> str:
    """Profile-scoped read under multiplex, plain ``os.getenv`` otherwise."""
    if _multiplex():
        from agent.fork_secrets import read_secret

        return read_secret(name, tool="google_drive")
    return os.getenv(name) or ""


def _scopes() -> list[str]:
    raw = _env("HERMES_GDRIVE_SCOPES").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(_DEFAULT_SCOPES)


def _ensure_deps(*, prompt: bool = False) -> None:
    """Install google-api-python-client / google-auth on first use."""
    from tools import lazy_deps

    lazy_deps.ensure(_LAZY_FEATURE, prompt=prompt)


def _ensure_deps_once() -> None:
    """Run ``_ensure_deps`` at most once process-wide.

    ``_ensure_deps`` can shell out to ``pip install`` on a cold process, and
    up to ``_LIST_POOL_WORKERS`` (see ``access.py``) threads may reach this
    concurrently on first use — a pip install must never run concurrently
    from the listing pool (parallel installs into the same venv can corrupt
    it). Double-checked so the common warm-process case never takes the lock.
    """
    global _deps_ready
    if _deps_ready:
        return
    with _lock:
        if not _deps_ready:
            _ensure_deps()
            _deps_ready = True


def _load_credentials() -> Any:
    """Resolve service-account credentials (see module docstring for order)."""
    scopes = _scopes()
    sa = _env("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON") or _env("GOOGLE_APPLICATION_CREDENTIALS")
    if sa:
        from google.oauth2 import service_account

        if sa.lstrip().startswith("{"):
            try:
                info = json.loads(sa)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Inline SA JSON is not valid JSON: {exc}") from exc
            return service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
        if not os.path.exists(sa):
            raise FileNotFoundError(
                f"Service Account JSON file not found at configured path: {sa}"
            )
        return service_account.Credentials.from_service_account_file(sa, scopes=scopes)

    if _multiplex():
        # ADC reads the process environment (GOOGLE_APPLICATION_CREDENTIALS,
        # the default profile's) and the VM metadata server — neither belongs
        # to the profile that owns this call. Fail closed.
        raise RuntimeError(
            "Google Drive service account not configured for this profile "
            "(set GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON in its .env)"
        )

    # No explicit key — Application Default Credentials (the VM-attached SA).
    import google.auth

    creds, _project = google.auth.default(scopes=scopes)
    return creds


def _get_credentials() -> Any:
    """Return the process-wide credentials, loading once.

    Credentials are safe to share across threads (unlike the built service's
    ``httplib2.Http`` transport — see module docstring), so this is cached
    globally rather than per-thread.
    """
    global _credentials
    if _multiplex():
        from hermes_constants import hermes_home_key

        key = hermes_home_key()
        creds = _profile_credentials.get(key)
        if creds is not None:
            return creds
        _ensure_deps_once()
        with _lock:
            creds = _profile_credentials.get(key)
            if creds is None:
                creds = _load_credentials()
                _profile_credentials[key] = creds
            return creds
    if _credentials is not None:
        return _credentials
    _ensure_deps_once()
    with _lock:
        if _credentials is None:
            _credentials = _load_credentials()
        return _credentials


def _get_service(api: str, version: str) -> Any:
    """Return a per-thread cached googleapiclient service, building once per
    (thread, api, version).

    The service is NOT process-wide: its ``httplib2.Http`` transport is not
    thread-safe, and callers (e.g. the listing-filter thread pool) may invoke
    it concurrently from multiple threads. Credentials are cached separately
    and shared.
    """
    key: tuple = (api, version)
    if _multiplex():
        # A service is bound to its credentials, so it is per profile too.
        from hermes_constants import hermes_home_key

        key = (hermes_home_key(), api, version)
    services: dict[tuple, Any] = getattr(_thread_services, "services", None)
    if services is None:
        services = {}
        _thread_services.services = services
    svc = services.get(key)
    if svc is not None:
        return svc
    _ensure_deps_once()
    from googleapiclient.discovery import build

    svc = build(api, version, credentials=_get_credentials(), cache_discovery=False)
    services[key] = svc
    return svc


def get_service() -> Any:
    """Cached Drive v3 service."""
    return _get_service("drive", "v3")


def get_sheets_service() -> Any:
    """Cached Sheets v4 service."""
    return _get_service("sheets", "v4")


def get_docs_service() -> Any:
    """Cached Docs v1 service."""
    return _get_service("docs", "v1")


def check_available() -> bool:
    """Cheap gate for the tool registry — True when ADC is resolvable.

    Never installs and never raises. If the google-* deps aren't importable
    yet we optimistically return True so the toolset can be enabled and the
    deps install on first dispatch; the handler surfaces any real ADC error.
    """
    global _avail
    if _multiplex():
        # Per profile: never cache one profile's verdict process-wide (the
        # registry's check_fn cache is already scoped per profile).
        try:
            import google.auth  # noqa: F401
        except Exception:
            return True
        try:
            _load_credentials()
            return True
        except Exception:
            return False
    if _avail is not None:
        return _avail
    try:
        import google.auth  # noqa: F401
    except Exception:
        # Deps not installed yet — let the user enable + use it (handler installs).
        return True
    try:
        _load_credentials()
        _avail = True
    except Exception:
        _avail = False
    return _avail


def reset_cache() -> None:
    """Drop cached services/credentials/availability (used by tests and after
    re-auth).

    Only the *calling* thread's service cache is cleared — the service cache
    is per-thread by design (see module docstring); other threads' cached
    services are dropped naturally as those threads exit (e.g. pool workers
    recycled between listings).
    """
    global _avail, _credentials, _deps_ready
    with _lock:
        _thread_services.services = {}
        _credentials = None
        _avail = None
        _deps_ready = False
