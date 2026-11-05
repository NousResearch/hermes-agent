"""Lifecycle controller for exactly ONE media sidecar process tree.

``MediaRuntimeController`` owns one UnSloth-Studio-style backend: it takes a
residency grant from :mod:`hermes_cli.local_runtime.admission`, launches the
backend owned (own session/process group handed to the owner-death supervisor)
and credential-safe on an ephemeral ``127.0.0.1`` port, proves health, heart-
beats the lease via host-driven :meth:`MediaRuntimeController.tick` (NO hidden
thread), detects crashes, and stops the tree with a VERIFIED exit before the
grant is ever handed back.

Hard rules encoded here (each has a test in
``tests/hermes_cli/test_media_runtime.py``):

* An admission ``Refused`` changes nothing: no process, state stays ``stopped``,
  the refusal reason surfaces in ``detail``.
* Every start mints a fresh ``instance_id`` and a fresh random password. The
  password reaches the child through its STDIN ONLY — never argv, never env,
  never a log line, never ``RuntimeStatus``/repr/exception text.
* Spawning goes through ``spawn_server(owned=True)`` so the whole tree dies
  with this process by any cause (#126631 discipline).
* Startup failure (health timeout or early child exit) tears the tree down,
  VERIFIES exit, and only then releases the grant — an unverified exit leaves
  the grant live so it quarantines-by-lease instead of risking double residency.
* Teardown mirrors ``LlamaServerSupervisor``: enumerate children FIRST,
  terminate, escalate to kill, then drain/release the owned process group.
* A ``FencedError`` (another controller took over) stops only our own sidecar;
  the new holder's record is never mutated.
* The bound address the child reports must be ``127.0.0.1`` and never the
  shared ``:8888`` default; anything else fails the start.

The controller never downloads anything and never loads models. It calls only
``GET /api/health``, the sidecar's own auth routes (``/api/auth/login`` +
``/api/auth/refresh``, see the channel section for the cited contract), and —
through :meth:`MediaRuntimeController.request`/:meth:`MediaRuntimeController.stream_request`
— caller-supplied relative ``/api/...`` paths on its own loopback base URL
(SSRF-guarded; the sidecar password never survives start(), tokens never
surface publicly).
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from hermes_cli.local_runtime.admission import (
    AdmissionError,
    AdmissionStore,
    FencedError,
    Grant,
    Refused,
)
from hermes_cli.local_runtime.processes import spawn_server

logger = logging.getLogger(__name__)

SHARED_DEFAULT_PORT = 8888
LOOPBACK_HOST = "127.0.0.1"
HEALTH_PATH = "/api/health"
DEFAULT_RESOURCE = "media:studio"
DEFAULT_ENGINE = "unsloth-studio"

#: Names a strict disposable child environment may still inherit: locale/TLS
#: plumbing only (ported from the proven Studio boot probe contract).
_STRICT_KEEP = re.compile(
    r"^(PATH|TERM|LANG|LC_[A-Z_]+|SSL_CERT_FILE|SSL_CERT_DIR|CURL_CA_BUNDLE|PATH_CA_BUNDLE)$"
)

_RUNNING_RE = re.compile(r"running on https?://([^\s:/]+):(\d+)", re.IGNORECASE)
_TAURI_PORT_RE = re.compile(r"TAURI_PORT=(\d+)")

# Credential scrubbing for any text we might surface (ported from the probe's
# sanitize_credentials; replacements hold no further matchable material).
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-\.]+)?")
_BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+")
_KV_RE = re.compile(
    r"(?i)([\"']?(?:password|passwd|passphrase|bootstrap_password|secret|api_key|apikey|"
    r"access_token|refresh_token|token)[\"']?\s*[:=]\s*)([\"']?)(\S+?)([\"']?)(?=[\s,;\"']|$)"
)
_PHRASE_RE = re.compile(r"(?i)(password\s+is\s+)(\S+)")

_STATES = ("stopped", "starting", "running", "stopping", "failed")

# ---------------------------------------------------------------------------
# authenticated sidecar request channel: upstream auth facts (READ from the
# installed upstream backend, cited per fact, do not guess):
#
# * ``POST {base}/api/auth/login`` — the auth router is mounted at prefix
#   ``/api/auth`` (studio/backend/main.py:1906) and the route is
#   ``@router.post("/login")`` (studio/backend/routes/auth.py:445).
# * login body is ``AuthLoginRequest`` = ``{"username": str, "password": str}``
#   (studio/backend/models/auth.py:17-21); the seeded admin username is
#   ``DEFAULT_ADMIN_USERNAME = "unsloth"`` (studio/backend/auth/storage.py:28).
# * response is ``Token`` = ``access_token`` / ``refresh_token`` /
#   ``token_type == "bearer"`` (+ ``must_change_password``, ``account_id``)
#   (studio/backend/models/users.py:11-30; constructed at routes/auth.py:493-499).
# * ``POST {base}/api/auth/refresh`` with ``{"refresh_token": str}``
#   (RefreshTokenRequest, studio/backend/models/auth.py:30-33) returns a NEW
#   Token pair and rotates the refresh token SINGLE-USE; an invalid/expired
#   refresh token yields HTTP 401 (routes/auth.py:578-608).
# * unauthenticated calls to bearer-protected routes yield HTTP 401
#   ``{"detail": "Not authenticated"}`` via the ``_BearerOrKeyless(HTTPBearer)``
#   fall-through (studio/backend/auth/authentication.py:128-179, security
#   instance at :179) — matches the Phase-1 probe
#   (``GET /api/models/list`` unauthenticated -> 401 "Not authenticated").
# ---------------------------------------------------------------------------
AUTH_USERNAME = "unsloth"
LOGIN_PATH = "/api/auth/login"
REFRESH_PATH = "/api/auth/refresh"
CHANNEL_PATH_PREFIX = "/api/"
CHANNEL_PATH_MAX = 2048
AUTH_LOST_DETAIL = "auth lost - restart the runtime"
_LOGIN_TIMEOUT = 15.0
_REFRESH_TIMEOUT = 15.0

_CHANNEL_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})
#: the controller owns these; a caller may never set them. Credentials and the
#: target (authorization, cookie, host), and request FRAMING / connection handling
#: (content-length, transfer-encoding and the hop-by-hop set): a caller-set
#: Transfer-Encoding next to the transport's own Content-Length is the classic
#: CL.TE pair that lets a second request ride the controller's bearer.
_FORBIDDEN_CALLER_HEADERS = frozenset({
    "authorization", "cookie", "host", "content-length", "transfer-encoding",
    "connection", "keep-alive", "upgrade", "te", "trailer", "trailers",
    "proxy-authorization", "proxy-connection"})
#: hop-by-hop headers plus Set-Cookie (upstream token material can ride in
#: cookies) are stripped from every ChannelResponse/MediaStream header dict
_STRIPPED_RESPONSE_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "set-cookie"})


class MediaRuntimeUnavailable(RuntimeError):
    """The authenticated sidecar channel could not serve this call.

    The message is always safe: it never carries the (already discarded)
    sidecar password, any token, or any response body.
    """


@dataclass(frozen=True)
class ChannelResponse:
    """One sidecar reply on the controller's own authenticated channel."""

    status: int
    headers: dict          # lowercased keys; hop-by-hop + set-cookie removed
    body: bytes


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib surfaces the 3xx as an HTTPError instead
    of RE-SENDING our ``Authorization: Bearer *** to the ``Location`` host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


#: shared opener: loopback-only (the base URL is always our own bound 127.0.0.1
#: socket), strictly no-redirect, and NEVER proxied. ``build_opener`` installs
#: ``ProxyHandler()`` by default, which reads ``http_proxy`` & co. from the
#: environment; with ``NO_PROXY=localhost`` (the usual setting, which does not
#: name the 127.0.0.1 literal) that sends the login password and every bearer
#: request to the proxy, and the health probe never reaches the sidecar at all.
_NO_REDIRECT_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoRedirectHandler)


def _validate_channel_path(path: str) -> str:
    """Guard one channel path; the target host/port are NEVER taken from input.

    Only a relative path starting with ``/api/`` is ever sent, and only to the
    controller's own loopback ``base_url``. Raises ``ValueError`` (before any
    socket is opened, so the bearer is never attached) on: absolute URLs / any
    scheme, ``//host``, backslashes, ``..`` or percent-encoded dot segments in
    any case, encoded slashes (``%2f``/``%5c``), double encodings, control
    chars / newlines / NUL, userinfo ``@``, fragments ``#``, a query smuggled
    into the path (``?`` — queries go through the ``query=`` argument only),
    and paths over :data:`CHANNEL_PATH_MAX` characters.
    """
    if not isinstance(path, str):
        raise ValueError("channel path must be a str")
    if len(path) > CHANNEL_PATH_MAX:
        raise ValueError(f"channel path longer than {CHANNEL_PATH_MAX} chars")
    if not path.startswith(CHANNEL_PATH_PREFIX):
        raise ValueError(f"channel path must be a relative path starting with "
                         f"{CHANNEL_PATH_PREFIX!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        raise ValueError("channel path contains control characters")
    # Studio's API paths are plain ASCII. Reject look-alike dots (U+2025,
    # fullwidth), bidi overrides and any whitespace here rather than relying on
    # http.client happening to refuse them downstream.
    if not path.isascii() or " " in path:
        raise ValueError("channel path must be ASCII without whitespace")
    for ch, why in ((":", "a scheme/colon"), ("\\", "a backslash"),
                    ("@", "userinfo"), ("#", "a fragment"),
                    ("?", "a query (pass it via the query= argument)")):
        if ch in path:
            raise ValueError(f"channel path contains {why}")
    if "//" in path:
        raise ValueError("channel path contains an empty segment ('//')")
    low = path.lower()
    if ".." in low:
        raise ValueError("channel path contains a '..' segment")
    for token in ("%2e", "%2f", "%5c", "%25", "%00"):
        if token in low:
            raise ValueError(f"channel path contains the encoded sequence {token!r}")
    return path


def _caller_headers(headers) -> dict:
    """Sanitize caller-supplied headers; the controller owns the credentials."""
    out: dict[str, str] = {}
    for name, value in (headers or {}).items():
        lname = str(name).strip().lower()
        if lname in _FORBIDDEN_CALLER_HEADERS:
            raise ValueError(f"caller may not set the {lname!r} header on the "
                             "media runtime channel")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in f"{name}\u2000{value}"):
            raise ValueError("channel headers contain control characters")
        out[str(name)] = str(value)
    return out


def _response_headers(response) -> dict:
    """Lowercased response headers minus hop-by-hop and Set-Cookie material."""
    return {name.lower(): str(value)
            for name, value in response.headers.items()
            if name.lower() not in _STRIPPED_RESPONSE_HEADERS}


#: hard ceiling for one raw request body on the channel. Callers (the plugin
#: bridge) enforce their own, smaller product cap; this only bounds memory.
CHANNEL_BODY_MAX = 256 * 1024 * 1024


def _request_body_parts(json_body, raw_body) -> tuple[bytes | None, dict]:
    """JSON body, or a raw byte body (audio clip upload) — never both.

    Raw bytes go out verbatim with ``application/octet-stream`` unless the
    caller supplies its own Content-Type (the clip's mime). Content-Length is
    computed by the transport and can never be set by a caller.
    """
    if raw_body is None:
        return _json_body_parts(json_body)
    if json_body is not None:
        raise ValueError("pass either json_body or raw_body, not both")
    if not isinstance(raw_body, (bytes, bytearray, memoryview)):
        raise ValueError("raw_body must be bytes")
    data = bytes(raw_body)
    if len(data) > CHANNEL_BODY_MAX:
        raise ValueError(f"raw_body larger than {CHANNEL_BODY_MAX} bytes")
    return data, {"Content-Type": "application/octet-stream"}


def _json_body_parts(json_body) -> tuple[bytes | None, dict]:
    if json_body is None:
        return None, {}
    return (json.dumps(json_body).encode("utf-8"),
            {"Content-Type": "application/json"})


class MediaStream:
    """Iterator over sidecar response bytes for large media / SSE payloads.

    The bearer lives strictly inside: ``status``/``headers`` are sanitized
    (:func:`_response_headers`) and the connection closes when the iterator is
    exhausted or closed. ``text/event-stream`` bodies surface line-by-line so
    events arrive incrementally instead of filling a read buffer.
    """

    def __init__(self, response, *, chunk_size: int = 65536) -> None:
        self._response = response
        self._chunk_size = max(1, int(chunk_size))
        self.status = int(response.status)
        self.headers = _response_headers(response)

    def __iter__(self):
        try:
            if "text/event-stream" in self.headers.get("content-type", ""):
                while True:
                    line = self._response.readline()
                    if not line:
                        return
                    if line.strip():
                        yield line
            else:
                while True:
                    chunk = self._response.read(self._chunk_size)
                    if not chunk:
                        return
                    yield chunk
        finally:
            self.close()

    def close(self) -> None:
        with suppress(Exception):
            self._response.close()

    def __enter__(self) -> MediaStream:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:  # never shows headers/credentials
        return f"<MediaStream status={self.status}>"


def sanitize_credentials(text: str) -> str:
    """Redact JWTs, bearer tokens, and key/value or prose password values."""
    text = _JWT_RE.sub("[REDACTED:token]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED:token]", text)
    text = _KV_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(4)}", text)
    text = _PHRASE_RE.sub(r"\1[REDACTED]", text)
    return text


@dataclass(frozen=True)
class RuntimeStatus:
    """Public observation of the controller. NEVER carries a credential."""

    state: str = "stopped"          # stopped | starting | running | stopping | failed
    instance_id: str | None = None  # fresh uuid4 hex per start
    pid: int | None = None          # sidecar pid (None once cleanly stopped)
    base_url: str | None = None     # loopback base URL while live
    epoch: int | None = None        # admission fencing epoch of our grant
    started_at: float | None = None  # wall clock of the successful start
    exit_code: int | None = None    # last observed child exit code
    detail: str = ""                # sanitized human-readable status


def default_boot_id() -> str:
    """Opaque current-boot identity for admission (injectable override)."""
    import psutil

    return str(psutil.boot_time())


def default_process_identity(pid: int) -> str:
    """Opaque creation identity for ``pid`` (injectable override)."""
    import psutil

    return str(psutil.Process(pid).create_time())


def free_loopback_port() -> int:
    """A kernel-chosen free TCP port bound to loopback; never :8888."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        port = sock.getsockname()[1]
    if port == SHARED_DEFAULT_PORT:  # astronomically unlikely; contract is absolute
        return free_loopback_port()
    return port


def parse_bound_endpoint(log_text: str) -> tuple[str | None, int | None]:
    """``(host, port)`` the child reported it is running on, or ``(None, None)``.

    The proven contract prints ``Unsloth Studio running on
    http://127.0.0.1:<port>``; ``TAURI_PORT=<port>`` is the host-less fallback.
    """
    match = _RUNNING_RE.search(log_text or "")
    if match:
        return match.group(1), int(match.group(2))
    match = _TAURI_PORT_RE.search(log_text or "")
    if match:
        return None, int(match.group(1))
    return None, None


def _probe_health(url: str, timeout: float = 1.5) -> bool:
    """True on HTTP 200 from the health endpoint; anything else keeps polling."""
    try:
        with _NO_REDIRECT_OPENER.open(url, timeout=timeout) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, OSError):
        return False


def disposable_environment(data_root: Path) -> dict[str, str]:
    """Strict-allowlist disposable child env built under ``data_root``.

    Only locale/TLS plumbing is inherited; everything else (PYTHONPATH, every
    credential-shaped name, arbitrary app config) is dropped. HOME/XDG/TMPDIR
    and the HF cache with hard-offline flags all live under ``data_root`` so
    the sidecar can never steer or touch the user's real trees.
    """
    env = {name: value for name, value in os.environ.items()
           if _STRICT_KEEP.match(name) and value}
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    home = data_root / "home"
    hf_cache = data_root / "hf-cache"
    for sub in ("tmp", "config", "cache", "state", "data"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "TMPDIR": str(home / "tmp"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_DATA_HOME": str(home / "data"),
        "UNSLOTH_STUDIO_HOME": str(data_root),
        "STUDIO_HOME": str(data_root),
        "UNSLOTH_STUDIO_PROJECTS_HOME": str(data_root / "projects"),
        "DATA_DESIGNER_HOME": str(data_root / "data-designer"),
        "HF_HOME": str(hf_cache),
        "HF_HUB_CACHE": str(hf_cache / "hub"),
        # Hard offline: this controller must never download; make any attempt
        # fail fast inside the child instead of fetching silently.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def _default_spawn(cmd, **kwargs):
    """Production seam: ``spawn_server`` keeps the POSIX/Windows ownership split."""
    return spawn_server(cmd, **kwargs)


def _quiet(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


class _StartupFailed(Exception):
    """Internal control flow: startup proof failed with a useful detail."""

    def __init__(self, detail: str, *, exit_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.exit_code = exit_code


class MediaRuntimeController:
    """Owns one media sidecar tree: grant -> launch -> health -> lease -> stop.

    The host calls :meth:`tick` periodically (no hidden thread): it renews the
    admission lease and detects child crashes. All teardown verifies process
    exit before the grant is released — the admission store itself never
    touches processes, so that verification lives here.
    """

    def __init__(self, *, owner_scope: str, data_root: Path | str,
                 admission: AdmissionStore | None = None, spawn=None,
                 boot_id_fn=None, identity_fn=None, clock=time.monotonic,
                 health_timeout: float = 60.0, lease_seconds: float = 30.0,
                 require_channel_auth: bool = True) -> None:
        self._owner_scope = owner_scope
        self._data_root = Path(data_root)
        self._admission = admission
        self._spawn = spawn or _default_spawn
        self._boot_id_fn = boot_id_fn or default_boot_id
        self._identity_fn = identity_fn or default_process_identity
        self._clock = clock
        self._health_timeout = float(health_timeout)
        self._lease_seconds = float(lease_seconds)
        # Fail closed by default: a sidecar whose login cannot complete is a failed
        # start, never a 'running' runtime with a dead channel. False is the explicit
        # lifecycle-only mode (process supervision without an authenticated channel).
        self._require_channel_auth = bool(require_channel_auth)
        # lifecycle state (in memory; durable truth lives in the admission store)
        self._state = "stopped"
        self._instance_id: str | None = None
        self._resource = DEFAULT_RESOURCE
        self._engine = DEFAULT_ENGINE
        self._model = ""
        self._grant: Grant | None = None
        self._proc: subprocess.Popen | None = None
        self._job = None                      # Windows job object from spawn_server
        self._pid: int | None = None
        self._child_identity: str | None = None
        self._base_url: str | None = None
        self._started_at: float | None = None
        self._exit_code: int | None = None
        self._detail = ""
        self._log_path: Path | None = None
        self._log_handle = None
        # sidecar channel credentials: ONLY tokens live here (never the password)
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._auth_lost = False

    # -- public API ---------------------------------------------------------

    def start(self, *, python: str | Path, run_py: str | Path,
              resource: str = DEFAULT_RESOURCE, engine: str = DEFAULT_ENGINE,
              model: str = "") -> RuntimeStatus:
        """Acquire the grant, launch the sidecar owned+health-checked.

        Returns the resulting :class:`RuntimeStatus` (never raises for launch
        problems: failure lands in ``state='failed'`` with a sanitized detail).
        """
        if self._state in ("starting", "running", "stopping"):
            return self.status()
        if self._state == "failed" and self._proc is not None and self._proc.poll() is None:
            # A start that failed while its child could not be proven gone keeps the only handle to that process. Blanking it here would
            # orphan a sidecar that still holds GPU/RAM and a live grant, with nothing able to stop it (and stop() would then claim
            # "no live sidecar"). Refuse until it has been stopped; stop() already knows how to deal with this state.
            note = f"the previous sidecar is still running (pid {self._pid}); stop it before starting again"
            if note not in self._detail:
                self._detail = sanitize_credentials(f"{self._detail}; {note}" if self._detail else note)
            return self.status()
        self._resource, self._engine, self._model = resource, engine, model
        self._instance_id = uuid.uuid4().hex
        self._grant = None
        self._proc = None
        self._job = None
        self._pid = None
        self._child_identity = None
        self._base_url = None
        self._started_at = None
        self._exit_code = None
        self._detail = ""
        self._access_token = None
        self._refresh_token = None
        self._auth_lost = False
        self._state = "starting"
        password = secrets.token_urlsafe(32)
        try:
            return self._start_attempt(python, run_py, password)
        except _StartupFailed as failed:
            verified, rc = self._teardown_tree(timeout=10.0)
            self._exit_code = rc if rc is not None else failed.exit_code
            self._set_failed(f"{failed.detail}; {self._release_on_verified_exit(verified)}")
            return self.status()
        except Exception as exc:
            verified, rc = self._teardown_tree(timeout=10.0)
            self._exit_code = rc
            self._set_failed(f"start aborted: {type(exc).__name__}: {exc}; "
                             f"{self._release_on_verified_exit(verified)}")
            return self.status()
        finally:
            password = None  # never retained on the instance past the stdin write

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            state=self._state,
            instance_id=self._instance_id,
            pid=self._pid,
            base_url=self._base_url,
            epoch=self._grant.epoch if self._grant is not None else None,
            started_at=self._started_at,
            exit_code=self._exit_code,
            detail=self._detail,
        )

    def tick(self) -> RuntimeStatus:
        """Host-driven heartbeat: renew the lease, detect crashes. No auto-restart."""
        if self._state != "running":
            return self.status()
        proc = self._proc
        if proc is None:
            self._set_failed("controller lost its process handle while running")
            return self.status()
        rc = proc.poll()
        if rc is not None:
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc if wait_rc is not None else rc
            self._set_failed(f"sidecar crashed rc={self._exit_code}; "
                             f"{self._release_on_verified_exit(verified)}")
            return self.status()
        try:
            self._grant = self._admission_store().renew(
                self._grant, now=self._clock(), lease_seconds=self._lease_seconds)
        except FencedError as exc:
            # Another controller legitimately took over: stop OUR sidecar and
            # step down; the new holder's record is never touched.
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc
            self._set_failed(f"grant fenced ({exc}); own sidecar stopped "
                             f"(exit verified={verified}); the newer holder was not touched")
        except AdmissionError as exc:
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc
            self._set_failed(f"grant renewal failed ({exc}); own sidecar stopped; "
                             f"{self._release_on_verified_exit(verified)}")
        return self.status()

    def stop(self, timeout: float = 20.0) -> RuntimeStatus:
        """Terminate the tree (children first), VERIFY exit, only THEN release.

        If the exit cannot be verified the grant is deliberately NOT released:
        it stays live and quarantines-by-lease, which is safer than releasing
        residency while a process may still hold GPU memory.
        """
        if self._state == "stopped":
            return self.status()
        if self._state == "failed":
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._state = "stopped"
                self._pid = None
                self._base_url = None
                self._access_token = None
                self._refresh_token = None
                self._detail = "stopped (no live sidecar; grant already handed back or quarantined)"
                return self.status()
        self._state = "stopping"
        self._base_url = None
        self._access_token = None
        self._refresh_token = None
        self._detail = ""
        verified, rc = self._teardown_tree(timeout=timeout)
        self._exit_code = rc
        if verified:
            note = self._release_cleanly()
            self._pid = None
            if "NOT released" in note:
                self._set_failed(f"sidecar stopped but {note}")
            else:
                self._state = "stopped"
                self._detail = f"sidecar stopped with verified exit; {note}"
        else:
            self._detail = (f"sidecar exit NOT verified within {timeout}s (pid {self._pid}): "
                            "grant NOT released, it stays live and quarantines-by-lease")
        return self.status()

    def log_tail(self, max_lines: int = 50) -> str:
        """Credential-scrubbed tail of the captured sidecar stdout/stderr."""
        if self._log_path is None or not self._log_path.exists():
            return ""
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return sanitize_credentials("\n".join(text.splitlines()[-max_lines:]))

    def __repr__(self) -> str:
        # never shows tokens, the (discarded) password, or grant internals
        return (f"<MediaRuntimeController state={self._state!r} "
                f"instance={self._instance_id!r} base_url={self._base_url!r}>")

    # -- authenticated sidecar request channel --------------------------------

    def request(self, method: str, path: str, *, json_body=None, raw_body=None,
                query=None, headers=None, timeout: float = 60.0) -> ChannelResponse:
        """One authenticated request to OUR OWN sidecar (never another host).

        ``path`` must be a relative ``/api/...`` path (guarded by
        :func:`_validate_channel_path` — anything else raises ``ValueError``
        before anything is sent); queries go through ``query=`` only. The
        controller attaches its own ``Authorization: Bearer <access token>``;
        the caller may not set Authorization/Cookie/Host/Content-Length. A 401
        triggers exactly ONE token refresh and one retry; if the refresh fails
        the channel is marked ``auth lost`` and :class:`MediaRuntimeUnavailable`
        is raised (there is no password to fall back to — it is gone).
        Redirects are returned as-is, never followed. Only state ``running``
        serves.
        """
        verb, target, caller = self._channel_prelude(method, path, headers)
        body, type_headers = _request_body_parts(json_body, raw_body)
        response = self._channel_send(verb, target, body, {**type_headers, **caller},
                                      query, timeout)
        try:
            return ChannelResponse(status=int(response.status),
                                   headers=_response_headers(response),
                                   body=response.read())
        finally:
            with suppress(Exception):
                response.close()

    def stream_request(self, method: str, path: str, *, json_body=None, raw_body=None,
                       query=None, headers=None, timeout: float = 60.0,
                       chunk_size: int = 65536) -> MediaStream:
        """Streaming variant of :meth:`request` for large media / SSE bodies.

        Same guards, same single bearer inside (never exposed), same one-shot
        refresh-and-retry. The returned :class:`MediaStream` carries
        ``status``/sanitized ``headers`` and yields byte chunks; the connection
        closes when the iterator is exhausted or closed.
        """
        verb, target, caller = self._channel_prelude(method, path, headers)
        body, type_headers = _request_body_parts(json_body, raw_body)
        response = self._channel_send(verb, target, body, {**type_headers, **caller},
                                      query, timeout)
        return MediaStream(response, chunk_size=chunk_size)

    def _channel_prelude(self, method, path, headers) -> tuple[str, str, dict]:
        """State gate + SSRF path guard + caller-header sanitize, in that order."""
        if self._state != "running":
            raise MediaRuntimeUnavailable(
                f"media runtime is not running (state={self._state})")
        if self._auth_lost:
            raise MediaRuntimeUnavailable(AUTH_LOST_DETAIL)
        if self._access_token is None:
            raise MediaRuntimeUnavailable(
                "media runtime is serving without sidecar authentication; "
                "restart the runtime")
        verb = str(method).upper()
        if verb not in _CHANNEL_METHODS:
            raise ValueError(f"unsupported HTTP method {method!r}")
        return verb, _validate_channel_path(path), _caller_headers(headers)

    def _channel_send(self, verb: str, target: str, body, headers: dict,
                      query, timeout: float):
        """Open the channel request; refresh-once-then-retry on a 401."""
        url = (self._base_url or "") + target
        if query:
            url += "?" + urllib.parse.urlencode(query)
        response = self._channel_open(verb, url, dict(headers), body, timeout)
        if response.status == 401:
            with suppress(Exception):
                response.close()
            if self._refresh_tokens():
                response = self._channel_open(verb, url, dict(headers), body, timeout)
            else:
                self._lose_auth()
                raise MediaRuntimeUnavailable(AUTH_LOST_DETAIL)
        return response

    def _channel_open(self, verb: str, url: str, headers: dict, body,
                      timeout: float):
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {self._access_token}"
        request = urllib.request.Request(url, data=body, headers=headers, method=verb)
        try:
            return _NO_REDIRECT_OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # 3xx (never followed) and 4xx/5xx surface to the caller as-is
            return exc
        except (urllib.error.URLError, OSError) as exc:
            raise MediaRuntimeUnavailable(
                f"media runtime channel: sidecar unreachable ({type(exc).__name__})"
            ) from None

    def _authenticate(self, password: str) -> None:
        """Log in ONCE with the stdin password over loopback; keep only tokens.

        Contract facts (cited in the module header above). A REFUSED login
        (endpoint present but rejecting: 401/403/429/5xx, redirect, unusable
        Token body) raises :class:`_StartupFailed` so the start fails through
        the existing teardown/verified-release path. HTTP 404/405/501 or an
        unreachable auth endpoint means this sidecar exposes NO auth API at
        all (a legacy/test sidecar; the real backend always serves
        /api/auth/login): the controller then runs WITHOUT tokens and every
        channel call is refused — the password is dropped either way, and no
        unauthenticated or password-bearing channel request is ever sent.

        That tokenless mode exists ONLY behind ``require_channel_auth=False``
        (lifecycle-only supervision). By default a missing or unreachable auth
        API fails the start, like any other refused login.
        """
        payload = json.dumps({"username": AUTH_USERNAME, "password": password})
        request = urllib.request.Request(
            (self._base_url or "") + LOGIN_PATH, data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=_LOGIN_TIMEOUT) as response:
                status = int(getattr(response, "status", None) or 200)
                body = response.read()
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            with suppress(Exception):
                exc.close()
            if code in (404, 405, 501):
                if self._require_channel_auth:
                    raise _StartupFailed(
                        f"sidecar exposes no auth API (HTTP {code}); an "
                        "authenticated channel is required") from None
                logger.warning("sidecar exposes no auth API (HTTP %s); "
                               "request channel disabled", code)
                return
            raise _StartupFailed(f"sidecar login rejected with HTTP {code}") from None
        except (urllib.error.URLError, OSError, ValueError):
            if self._require_channel_auth:
                raise _StartupFailed(
                    "sidecar auth endpoint unreachable; an authenticated "
                    "channel is required") from None
            logger.warning("sidecar auth endpoint unreachable; request channel disabled")
            return
        try:
            token = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _StartupFailed("sidecar login returned an unparseable body") from None
        access = token.get("access_token")
        refresh = token.get("refresh_token")
        if (status != 200 or not isinstance(access, str) or not access
                or not isinstance(refresh, str) or not refresh):
            raise _StartupFailed("sidecar login returned an unusable token response")
        if str(token.get("token_type", "bearer")).lower() != "bearer":
            raise _StartupFailed("sidecar login returned an unexpected token_type")
        self._access_token = access
        self._refresh_token = refresh

    def _refresh_tokens(self) -> bool:
        """Exchange the refresh token ONCE (single-use rotation upstream); False = lost."""
        token = self._refresh_token
        if not token:
            return False
        payload = json.dumps({"refresh_token": token}).encode("utf-8")
        request = urllib.request.Request(
            (self._base_url or "") + REFRESH_PATH, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=_REFRESH_TIMEOUT) as response:
                status = int(getattr(response, "status", None) or 200)
                body = response.read()
        except urllib.error.HTTPError as exc:
            with suppress(Exception):
                exc.close()
            return False
        except (urllib.error.URLError, OSError):
            return False
        try:
            rotated = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        access = rotated.get("access_token")
        refresh = rotated.get("refresh_token")
        if (status != 200 or not isinstance(access, str) or not access
                or not isinstance(refresh, str) or not refresh):
            return False
        self._access_token = access
        self._refresh_token = refresh
        return True

    def _lose_auth(self) -> None:
        """Terminal for the channel: purge tokens, flag the state, never retry."""
        self._access_token = None
        self._refresh_token = None
        self._auth_lost = True
        self._detail = sanitize_credentials(AUTH_LOST_DETAIL)

    # -- start internals ------------------------------------------------------

    def _admission_store(self) -> AdmissionStore:
        if self._admission is None:
            self._admission = AdmissionStore()
        return self._admission

    def _start_attempt(self, python, run_py, password: str) -> RuntimeStatus:
        store = self._admission_store()
        boot_id = self._boot_id_fn()
        outcome = store.acquire(
            self._resource, owner_scope=self._owner_scope,
            controller_id=self._instance_id, pid=os.getpid(),
            process_start_identity=self._safe_identity(os.getpid()) or "identity-unavailable",
            engine=self._engine, model=self._model, boot_id=boot_id,
            now=self._clock(),
            # the lease must outlive a full health wait without a heartbeat
            lease_seconds=self._lease_seconds + self._health_timeout)
        if isinstance(outcome, Refused):
            # State unchanged, no process: a refusal is not a failure.
            self._state = "stopped"
            self._detail = (f"start refused by admission: {outcome.reason} — {outcome.detail}")
            return self.status()
        self._grant = outcome

        port = free_loopback_port()
        if port == SHARED_DEFAULT_PORT:  # defensive; free_loopback_port guarantees this
            raise _StartupFailed("refusing the shared :8888 default port")
        data_root = self._data_root
        logs = data_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._log_path = logs / f"sidecar-{self._instance_id}.log"
        # The raw log is captured BEFORE any scrubbing (sanitize_credentials runs when the tail is read), so it is private to the user.
        self._log_handle = open(
            os.open(self._log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
            "a", encoding="utf-8", errors="replace")
        # list-args, never a shell; the password is '-' (stdin), so the command
        # line itself is safe to record for forensics. These are run.py's own flags (the short -H/-p
        # belong to the `unsloth studio` CLI and make run.py exit 2); --no-cloudflare forces a public
        # tunnel off even if a stored "start remote access automatically" setting asks for one.
        cmd = [str(python), str(run_py), "--api-only", "--host", LOOPBACK_HOST,
               "--port", str(port), "--no-cloudflare", "--password", "-"]
        self._log_handle.write(f"\n# spawn: {cmd}\n")
        self._log_handle.flush()
        proc, job = self._spawn(cmd, owned=True, stdin=subprocess.PIPE,
                                stdout=self._log_handle, stderr=subprocess.STDOUT,
                                cwd=str(data_root), env=disposable_environment(data_root),
                                text=True)
        self._proc, self._job = proc, job
        self._pid = proc.pid
        self._child_identity = self._safe_identity(proc.pid)
        self._deliver_password(proc, password)

        deadline = self._clock() + self._health_timeout
        host, bound_port = self._await_bound_endpoint(deadline)
        if host is not None and host != LOOPBACK_HOST:
            raise _StartupFailed(
                f"sidecar reported bound address {host!r}:{bound_port}; only "
                f"{LOOPBACK_HOST} loopback is permitted")
        if bound_port == SHARED_DEFAULT_PORT:
            raise _StartupFailed(f"sidecar reported the shared :{SHARED_DEFAULT_PORT} port")
        base_url = f"http://{LOOPBACK_HOST}:{bound_port}"
        self._await_health(base_url, deadline)

        self._base_url = base_url
        # Authenticate ONCE over loopback while the password still exists, then
        # drop it for good: only tokens are retained, and a refused login fails
        # the start through the same verified-teardown path as any other
        # startup failure.
        self._authenticate(password)
        del password
        with suppress(AdmissionError):  # residency advance is best-effort; tick() re-drives it
            self._grant = store.renew(self._grant, now=self._clock(), residency="resident")
        self._started_at = time.time()
        self._state = "running"
        self._detail = f"sidecar healthy at {base_url}"
        logger.info("media sidecar running pid=%s url=%s instance=%s",
                    self._pid, base_url, self._instance_id)
        return self.status()

    @staticmethod
    def _deliver_password(proc, password: str) -> None:
        """Password goes to the child's STDIN ONLY, then is forgotten here."""
        stdin = getattr(proc, "stdin", None)
        if stdin is None:
            return
        with suppress(OSError, ValueError):  # early exit surfaces via poll() below
            stdin.write(password + "\n")
            stdin.flush()
            stdin.close()

    def _read_log(self) -> str:
        if self._log_path is None or not self._log_path.exists():
            return ""
        return self._log_path.read_text(encoding="utf-8", errors="replace")

    def _await_bound_endpoint(self, deadline: float) -> tuple[str | None, int]:
        proc = self._proc
        while True:
            rc = proc.poll()
            if rc is not None:
                raise _StartupFailed(
                    f"sidecar exited rc={rc} during startup before reporting a bound port "
                    f"(log tail: {sanitize_credentials(self.log_tail(5)) or 'empty'})",
                    exit_code=rc)
            host, port = parse_bound_endpoint(self._read_log())
            if port is not None:
                return host, port
            if self._clock() >= deadline:
                raise _StartupFailed(
                    f"sidecar did not report a bound port within {self._health_timeout}s "
                    f"(log tail: {sanitize_credentials(self.log_tail(5)) or 'empty'})")
            time.sleep(0.1)

    def _await_health(self, base_url: str, deadline: float) -> None:
        proc = self._proc
        url = base_url + HEALTH_PATH
        while True:
            rc = proc.poll()
            if rc is not None:
                raise _StartupFailed(f"sidecar exited rc={rc} during health check", exit_code=rc)
            if _probe_health(url):
                return
            if self._clock() >= deadline:
                raise _StartupFailed(
                    f"sidecar never returned HTTP 200 from GET {url} within "
                    f"{self._health_timeout}s")
            time.sleep(0.15)

    # -- teardown internals (mirrors LlamaServerSupervisor discipline) --------

    def _teardown_tree(self, timeout: float) -> tuple[bool, int | None]:
        """Terminate children then root, escalate to kill; returns (verified, rc)."""
        proc = self._proc
        if proc is None:
            return True, self._exit_code
        rc = proc.poll()
        children: list = []
        with suppress(Exception):
            import psutil

            with suppress(psutil.Error):
                children = psutil.Process(proc.pid).children(recursive=True)
        if rc is None:
            half = max(0.5, timeout / 2.0)
            for child in children:
                _quiet(child.terminate)
            _quiet(proc.terminate)
            try:
                rc = proc.wait(timeout=half)
            except Exception:
                _quiet(proc.kill)
                try:
                    rc = proc.wait(timeout=half)
                except Exception:
                    rc = None
        for child in children:
            with suppress(Exception):
                if child.is_running():
                    child.kill()
        if self._job is not None:
            _quiet(self._job.close)
            self._job = None
        group_empty = True
        if os.name == "posix":
            from hermes_cli import process_identity as pi

            # A crashed leader can leave its children holding the group: stop
            # them, then release the group number only once it is empty, so a
            # tree we failed to kill stays covered by the owner-death supervisor.
            _quiet(lambda: pi.stop_owned_group_survivors(
                proc.pid, proc=proc, grace=min(3.0, max(0.5, timeout / 4.0))))
            with suppress(Exception):
                group_empty = pi.release_process_group_once_empty(
                    proc.pid, proc=proc, timeout=max(1.0, timeout / 3.0))
        if self._log_handle is not None:
            _quiet(self._log_handle.close)
            self._log_handle = None
        verified = self._process_gone(proc.pid, proc) and (os.name != "posix" or group_empty)
        return verified, rc

    def _safe_identity(self, pid: int) -> str | None:
        try:
            return self._identity_fn(pid)
        except Exception:
            return None

    def _process_gone(self, pid: int | None, proc: subprocess.Popen | None = None) -> bool:
        """True only when exit is PROVEN. Anything ambiguous fails closed.

        Proof is either (a) we reaped the Popen handle ourselves — the kernel
        has then released that pid, whatever a later process called ``pid``
        looks like — or (b) the pid is gone / is a zombie, or (c) it now carries
        a DIFFERENT creation identity than the one recorded for our child.

        A live process whose identity cannot be read, or one we never managed
        to identify, is ambiguous (it may be our child or a recycled pid): the
        caller must keep the grant held rather than treat that as permission.
        """
        if pid is None:
            return True
        if proc is not None and getattr(proc, "returncode", None) is not None:
            return True
        try:
            import psutil

            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return True
            with suppress(psutil.Error):
                if process.status() == psutil.STATUS_ZOMBIE:
                    return True
        except ImportError:
            pass
        current = self._safe_identity(pid)
        if current is None or self._child_identity is None:
            return False  # live and not comparable: ambiguous, never "gone"
        return current != self._child_identity

    # -- grant handback internals ---------------------------------------------

    def _release_cleanly(self) -> str:
        """Terminal handback after a clean verified stop. Never on unverified exit."""
        if self._grant is None:
            return "no grant held"
        try:
            self._admission_store().release(self._grant, now=self._clock())
            return "grant released"
        except FencedError as exc:
            return f"grant NOT released (fenced: {exc})"
        except AdmissionError as exc:
            return f"grant NOT released ({exc})"

    def _release_on_verified_exit(self, verified: bool) -> str:
        """Crash/failure handback: attest the exit through ``recover`` itself."""
        if self._grant is None:
            return "no grant held"
        store = self._admission_store()
        if not verified:
            # No proof the recorded process is gone: never attest. The record
            # stays live and quarantines-by-lease (recover with
            # verified_exit=False would need external verification too).
            return "process exit NOT verified: grant intentionally left live (quarantined-by-lease)"
        try:
            outcome = store.recover(self._resource, self._clock(), self._boot_id_fn(),
                                    verified_exit=True, expected_epoch=self._grant.epoch)
            return f"grant recovery: {outcome}"
        except AdmissionError as exc:
            return f"grant recovery failed ({exc})"

    def _set_failed(self, detail: str) -> None:
        self._state = "failed"
        self._base_url = None
        self._access_token = None
        self._refresh_token = None
        self._detail = sanitize_credentials(detail)


__all__ = [
    "AUTH_LOST_DETAIL",
    "AUTH_USERNAME",
    "DEFAULT_ENGINE",
    "DEFAULT_RESOURCE",
    "HEALTH_PATH",
    "LOOPBACK_HOST",
    "SHARED_DEFAULT_PORT",
    "ChannelResponse",
    "MediaRuntimeController",
    "MediaRuntimeUnavailable",
    "MediaStream",
    "RuntimeStatus",
    "disposable_environment",
    "free_loopback_port",
    "parse_bound_endpoint",
    "sanitize_credentials",
]
