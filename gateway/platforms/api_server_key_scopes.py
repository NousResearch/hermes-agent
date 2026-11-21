"""Named, scoped API keys for the API server (``platforms.api_server.keys``).

``API_SERVER_KEY`` stays the full-access operator key and is still required to start. Each
``keys`` entry adds one more bearer whose secret lives in ``.env`` under ``key_env`` (never in
config.yaml) and whose reach is cut down by:

* ``scopes`` -- a closed set (:class:`Scope`). Every key-authenticated route maps to exactly one
  scope in ``ROUTE_SCOPES``; a route missing from the table answers only the full key, so a route
  added later fails closed for scoped keys until someone classifies it.
* ``session_ids`` -- routes that address a session in the path (or a run started on one) answer only
  for listed ids, and never with an ``X-Hermes-Session-Key``/``-Id`` header. Routes that enumerate or
  create sessions, or pick one from the request body (``/v1/runs``, ``/v1/chat/completions``,
  ``/v1/responses``, jobs), are refused, never filtered.
* ``read_only`` -- any method but GET/HEAD is refused.
* ``expires`` -- ISO-8601; at or past it the key gets a 401.

Scoped keys authenticate only the listener's own profile: ``/p/<profile>/`` routes keep requiring
that profile's own ``API_SERVER_KEY``. They share the listener's run namespace, so a key with
``runs`` may poll/steer/stop (and ``approvals:answer`` may answer) any run whose id it holds;
``session_ids`` narrows that to runs on listed sessions.

A malformed, weak or missing scoped key is skipped with a warning at load (the full key still
guards the listener); it never authenticates. Secrets are never logged.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from enum import Enum
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.platforms._shared import get_scoped_secret, secrets_match
from hermes_cli.auth import has_usable_secret

logger = logging.getLogger("gateway.platforms.api_server")


class Scope(str, Enum):
    MODELS_READ = "models:read"
    SESSIONS_READ = "sessions:read"
    SESSIONS_WRITE = "sessions:write"
    RUNS = "runs"
    APPROVALS_ANSWER = "approvals:answer"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"


class Reach(Enum):
    """How a route addresses sessions -- what a ``session_ids`` allowlist does with it."""
    GLOBAL = "global"          # no session data (catalog reads): allowed
    SESSION_PATH = "session"   # ``{session_id}`` in the path: allowed for listed ids
    RUN_PATH = "run"           # ``{run_id}`` in the path: allowed when the run's session is listed
    UNBOUNDED = "unbounded"    # enumerates/creates sessions or takes one from the body: refused


_S = Scope
_REACH_GLOBAL, _BY_SESSION, _BY_RUN, _UNBOUNDED = Reach.GLOBAL, Reach.SESSION_PATH, Reach.RUN_PATH, Reach.UNBOUNDED
# (method, route path without the ``/p/{profile}`` mirror prefix) -> (scope, reach).
# Not listed (full key only): /health/detailed and /v1/room-members/* (mints room grants).
# Not key-authenticated at all: /health, /v1/health, platform events, cron fire.
ROUTE_SCOPES: dict[tuple[str, str], tuple[Scope, Reach]] = {
    ("GET", "/v1/models"): (_S.MODELS_READ, _REACH_GLOBAL),
    ("GET", "/api/model/options"): (_S.MODELS_READ, _REACH_GLOBAL),
    ("GET", "/v1/capabilities"): (_S.MODELS_READ, _REACH_GLOBAL),
    ("GET", "/v1/skills"): (_S.MODELS_READ, _REACH_GLOBAL),
    ("GET", "/v1/toolsets"): (_S.MODELS_READ, _REACH_GLOBAL),
    ("GET", "/api/sessions"): (_S.SESSIONS_READ, _UNBOUNDED),
    ("GET", "/api/sessions/{session_id}"): (_S.SESSIONS_READ, _BY_SESSION),
    ("GET", "/api/sessions/{session_id}/messages"): (_S.SESSIONS_READ, _BY_SESSION),
    ("POST", "/api/sessions"): (_S.SESSIONS_WRITE, _UNBOUNDED),
    ("PATCH", "/api/sessions/{session_id}"): (_S.SESSIONS_WRITE, _BY_SESSION),
    ("DELETE", "/api/sessions/{session_id}"): (_S.SESSIONS_WRITE, _BY_SESSION),
    ("POST", "/api/sessions/{session_id}/fork"): (_S.SESSIONS_WRITE, _BY_SESSION),
    ("POST", "/api/sessions/{session_id}/model"): (_S.SESSIONS_WRITE, _BY_SESSION),
    ("POST", "/api/sessions/{session_id}/chat"): (_S.RUNS, _BY_SESSION),
    ("POST", "/api/sessions/{session_id}/chat/stream"): (_S.RUNS, _BY_SESSION),
    ("POST", "/v1/chat/completions"): (_S.RUNS, _UNBOUNDED),
    ("POST", "/v1/responses"): (_S.RUNS, _UNBOUNDED),
    ("GET", "/v1/responses/{response_id}"): (_S.RUNS, _UNBOUNDED),
    ("DELETE", "/v1/responses/{response_id}"): (_S.RUNS, _UNBOUNDED),
    ("POST", "/v1/runs"): (_S.RUNS, _UNBOUNDED),
    ("GET", "/v1/runs/{run_id}"): (_S.RUNS, _BY_RUN),
    ("GET", "/v1/runs/{run_id}/events"): (_S.RUNS, _BY_RUN),
    ("POST", "/v1/runs/{run_id}/steer"): (_S.RUNS, _BY_RUN),
    ("POST", "/v1/runs/{run_id}/stop"): (_S.RUNS, _BY_RUN),
    ("POST", "/v1/runs/{run_id}/resolve-unknown"): (_S.RUNS, _BY_RUN),
    ("POST", "/v1/runs/{run_id}/approval"): (_S.APPROVALS_ANSWER, _BY_RUN),
    ("POST", "/v1/runs/{run_id}/clarify"): (_S.APPROVALS_ANSWER, _BY_RUN),
    ("GET", "/api/jobs"): (_S.JOBS_READ, _UNBOUNDED),
    ("GET", "/api/jobs/{job_id}"): (_S.JOBS_READ, _UNBOUNDED),
    ("POST", "/api/jobs"): (_S.JOBS_WRITE, _UNBOUNDED),
    ("PATCH", "/api/jobs/{job_id}"): (_S.JOBS_WRITE, _UNBOUNDED),
    ("DELETE", "/api/jobs/{job_id}"): (_S.JOBS_WRITE, _UNBOUNDED),
    ("POST", "/api/jobs/{job_id}/pause"): (_S.JOBS_WRITE, _UNBOUNDED),
    ("POST", "/api/jobs/{job_id}/resume"): (_S.JOBS_WRITE, _UNBOUNDED),
    ("POST", "/api/jobs/{job_id}/run"): (_S.JOBS_WRITE, _UNBOUNDED)}

_PROFILE_MIRROR_PREFIX = "/p/{profile}"
_SAFE_METHODS = frozenset({"GET", "HEAD"})
_SESSION_HEADERS = ("X-Hermes-Session-Key", "X-Hermes-Session-Id")
_ENTRY_FIELDS = frozenset({"name", "key_env", "scopes", "session_ids", "read_only", "expires"})
_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_SCOPE_VALUES = frozenset(scope.value for scope in Scope)


@dataclass(frozen=True)
class ScopedKey:
    name: str
    secret: str = field(repr=False)
    scopes: frozenset[Scope]
    session_ids: Optional[frozenset[str]] = None
    read_only: bool = False
    expires_at: Optional[float] = None


def _parse_expires(value: Any) -> Optional[float]:
    """Epoch seconds for an ISO-8601 string or a YAML-parsed timestamp; naive means UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.strip())
    if not isinstance(value, datetime):
        raise ValueError("expires must be an ISO-8601 timestamp")
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()


def _parse_scopes(value: Any) -> frozenset[Scope]:
    if not isinstance(value, list) or not value:
        raise ValueError("scopes must be a non-empty list")
    unknown = [str(item) for item in value if not isinstance(item, str) or item not in _SCOPE_VALUES]
    if unknown:
        raise ValueError(f"unknown scope(s) {unknown}; valid: {sorted(_SCOPE_VALUES)}")
    return frozenset(Scope(item) for item in value)


def _parse_session_ids(value: Any) -> Optional[frozenset[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("session_ids must be a list of session id strings")
    return frozenset(item.strip() for item in value)


def _parse_entry(entry: Any) -> ScopedKey:
    """One ``keys`` entry -> :class:`ScopedKey`; ``ValueError`` names what is wrong (never the secret)."""
    if not isinstance(entry, dict):
        raise ValueError("entry must be a mapping")
    if "key" in entry:
        raise ValueError("literal secrets are not accepted; put the secret in .env and name it in key_env")
    unknown = sorted(set(entry) - _ENTRY_FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}")
    name, key_env = entry.get("name"), entry.get("key_env")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError("name must be 1-64 characters of [A-Za-z0-9._-]")
    if not isinstance(key_env, str) or not _ENV_NAME_RE.fullmatch(key_env):
        raise ValueError("key_env must name an environment variable (e.g. API_SERVER_KEY_DASHBOARD)")
    read_only = entry.get("read_only", False)
    if not isinstance(read_only, bool):
        raise ValueError("read_only must be true or false")
    expires = entry.get("expires")
    secret = get_scoped_secret(key_env, "") or ""
    if not has_usable_secret(secret, min_length=16):
        raise ValueError(f"{key_env} is missing, a placeholder or shorter than 16 characters")
    return ScopedKey(
        name=name, secret=secret.strip(), scopes=_parse_scopes(entry.get("scopes")),
        session_ids=_parse_session_ids(entry.get("session_ids")), read_only=read_only,
        expires_at=None if expires is None else _parse_expires(expires))


def load_scoped_keys(raw: Any, *, legacy_key: str) -> tuple[ScopedKey, ...]:
    """Parse ``platforms.api_server.keys``; an invalid or duplicate entry is skipped with a warning."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("[api_server] Ignoring platforms.api_server.keys: expected a list of key entries")
        return ()
    keys: list[ScopedKey] = []
    for index, entry in enumerate(raw):
        try:
            key = _parse_entry(entry)
        except ValueError as exc:
            logger.warning("[api_server] Ignoring API key keys[%d]: %s", index, exc)
            continue
        if any(key.name == other.name for other in keys):
            logger.warning("[api_server] Ignoring API key keys[%d]: duplicate name %r", index, key.name)
        elif key.secret == legacy_key or any(key.secret == other.secret for other in keys):
            logger.warning(
                "[api_server] Ignoring API key %r: its secret is already used by another key", key.name)
        else:
            if key.expires_at is not None and key.expires_at <= time.time():
                logger.warning("[api_server] API key %r is already expired; it will be refused", key.name)
            keys.append(key)
    return tuple(keys)


def _route_path(request: web.Request) -> str:
    resource = getattr(request.match_info.route, "resource", None)
    path = resource.canonical if resource is not None else ""
    return path.removeprefix(_PROFILE_MIRROR_PREFIX)


def _run_session_id(adapter: Any, request: web.Request) -> Optional[str]:
    """The session a run belongs to: live status first, else the caller-scoped saved record the
    route handler would hydrate next (so an allowed run stays reachable after a restart)."""
    run_id = request.match_info.get("run_id", "")
    status = adapter._run_statuses.get(run_id) or adapter._durable_run_status(request, run_id) or {}
    return status.get("session_id")


_REACH_TARGET = {
    Reach.SESSION_PATH: lambda adapter, request: request.match_info.get("session_id"),
    Reach.RUN_PATH: lambda adapter, request: _run_session_id(adapter, request),
    Reach.UNBOUNDED: lambda adapter, request: None}


def _refusal(status: int, code: str, message: str, *, scope: Optional[Scope] = None) -> web.Response:
    error: dict[str, Any] = {
        "message": message, "type": "gateway_auth_error" if status == 401 else "permission_error",
        "param": None, "code": code}
    if scope is not None:
        error["required_scope"] = scope.value
    return web.json_response({"error": error}, status=status)


def _policy_refusal(adapter: Any, request: web.Request, key: ScopedKey) -> Optional[web.Response]:
    """None when *key*'s policy admits this route, else the 401/403 explaining why not."""
    if key.expires_at is not None and time.time() >= key.expires_at:
        return _refusal(401, "api_key_expired", f"API key {key.name!r} has expired")
    method = "GET" if request.method == "HEAD" else request.method
    path = _route_path(request)
    rule = ROUTE_SCOPES.get((method, path))
    if rule is None:
        return _refusal(403, "insufficient_scope", f"{method} {path} requires the full API_SERVER_KEY")
    scope, reach = rule
    if scope not in key.scopes:
        return _refusal(
            403, "insufficient_scope", f"API key {key.name!r} lacks the {scope.value!r} scope "
            f"required for {method} {path}", scope=scope)
    if key.read_only and method not in _SAFE_METHODS:
        return _refusal(403, "read_only_key", f"API key {key.name!r} is read-only")
    if key.session_ids is not None and reach is not Reach.GLOBAL:
        # A declared conversation/memory key would reach another conversation's long-term memory.
        declared = any(request.headers.get(name) for name in _SESSION_HEADERS)
        if declared or _REACH_TARGET[reach](adapter, request) not in key.session_ids:
            return _refusal(
                403, "session_not_allowed", f"API key {key.name!r} is not allowed to reach this session")
    return None


def _bearer(request: web.Request) -> str:
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.startswith("Bearer ") else ""


def check_bearer(adapter: Any, request: web.Request, expected_key: str, *, scoped: bool) -> Optional[web.Response]:
    """None when the bearer is the full key, or a scoped key (*scoped*: this listener's own profile)
    whose policy admits the route; else the 401/403. Every configured secret is compared, no early
    exit, so response timing does not reveal which key -- or whether any scoped key -- matched."""
    token = _bearer(request)
    full = secrets_match(token, expected_key)
    hits = [key for key in (adapter._scoped_keys if scoped else ()) if secrets_match(token, key.secret)]
    if full:
        return None
    if not hits:
        logger.warning("API server rejected invalid API key: %s", adapter._request_audit_log_suffix(request))
        return adapter._auth_failed_response()
    refusal = _policy_refusal(adapter, request, hits[0])
    if refusal is not None:
        logger.warning(
            "API server refused scoped API key %r (%s %s, %s): %s", hits[0].name, request.method,
            _route_path(request), refusal.status, adapter._request_audit_log_suffix(request))
    return refusal
