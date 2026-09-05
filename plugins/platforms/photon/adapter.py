"""
Photon Spectrum (iMessage) platform adapter for Hermes Agent.

Both directions of traffic flow through a small supervised Node sidecar
(see ``sidecar/index.mjs``) that runs the ``spectrum-ts`` SDK — the SDK is
TypeScript-only and there is no public HTTP message API, so a sidecar is
unavoidable.

Inbound:
    The SDK's ``app.messages`` is a long-lived **gRPC** stream. The sidecar
    serializes each message to a normalized JSON event and streams it to this
    adapter over a loopback ``GET /inbound`` (NDJSON). A background task here
    consumes that stream, dedupes on ``messageId``, and dispatches a
    ``MessageEvent`` to the gateway via ``BasePlatformAdapter.handle_message``.
    No webhook, no public URL, no signing secret.

Outbound:
    ``send`` / ``send_typing`` are loopback POSTs to the sidecar's control
    endpoints, authenticated with a shared bearer token. URL-only messages can
    route through spectrum-ts' ``richlink()`` builder via ``/send-richlink``.
    Outbound media (images, voice notes, video, documents) goes through
    spectrum-ts' ``attachment()`` / ``voice()`` content builders via the
    sidecar's ``/send-attachment`` endpoint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    # Type checkers see ``httpx`` as the always-imported module, so every use
    # site type-checks cleanly. The runtime fallback below keeps the optional
    # dependency truly optional (each use site is guarded by HTTPX_AVAILABLE).
    import httpx
    HTTPX_AVAILABLE = True
else:
    try:
        import httpx
        HTTPX_AVAILABLE = True
    except ImportError:  # pragma: no cover - httpx is already a Hermes dep
        HTTPX_AVAILABLE = False
        httpx = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.platforms.helpers import compile_mention_patterns, strip_markdown

from .auth import load_project_credentials

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read with the default-profile startup fallback.

    Secondary profiles construct their adapters under a profile secret
    scope -- the scope is authoritative and a scoped miss returns ``default``
    (no cross-profile borrow from ``os.environ``, which may hold another
    profile's value). The DEFAULT profile's adapter constructs and sends
    *unscoped* under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash this path; there ``os.environ`` is that
    profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and
    ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants

_DEFAULT_SIDECAR_PORT = 8789
_DEFAULT_SIDECAR_BIND = "127.0.0.1"

# Photon iMessage messages from the SDK side have no documented hard
# limit, but the underlying iMessage protocol limits practical message
# size to ~16 KB.  Keep a conservative cap that matches BlueBubbles.
_MAX_MESSAGE_LENGTH = 8000

# ---------------------------------------------------------------------------
# Sidecar runtime record
#
# Out-of-process senders (cron subprocesses, `hermes send`, the dashboard)
# go through ``_standalone_send`` and need the live sidecar's port + token —
# but the token is generated at spawn time and otherwise exists only in the
# gateway process memory and the sidecar child env (issue #69960). The
# gateway persists this record once the sidecar passes its /healthz
# readiness check, and removes it on every stop / failed-start path so a
# stale record never outlives a dead sidecar.

_RUNTIME_RECORD_NAME = "photon-sidecar.json"


def _runtime_record_path() -> Path:
    # get_hermes_home() honors profile overrides — never hardcode ~/.hermes.
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "runtime" / _RUNTIME_RECORD_NAME


def _write_runtime_record(port: int, token: str, pid: int) -> None:
    """Atomically persist ``{port, token, pid}`` with owner-only perms."""
    import tempfile

    try:
        path = _runtime_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=".photon-sidecar.", suffix=".tmp"
        )
        try:
            # Restrict perms BEFORE the token hits disk (mkstemp is already
            # 0600 on POSIX; the explicit chmod is a belt-and-braces guard,
            # wrapped so Windows/exotic filesystems can't crash the write).
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover - Windows / odd fs
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"port": port, "token": token, "pid": pid}, fh)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:  # best-effort: gateway sends still work without it
        logger.warning("[photon] failed to write sidecar runtime record: %s", e)


def _read_runtime_record() -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(_runtime_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _delete_runtime_record() -> None:
    try:
        _runtime_record_path().unlink(missing_ok=True)
    except OSError:
        pass


def _sidecar_pid_alive(pid: Any) -> bool:
    """Best-effort liveness check for the recorded sidecar pid."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        # Cross-platform (psutil-backed, Windows-safe — see its docstring).
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid_int))
    except Exception:
        pass
    if os.name == "posix":
        try:
            os.kill(pid_int, 0)  # windows-footgun: ok — inside os.name == "posix" guard
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    # Windows without gateway.status/psutil: can't probe safely
    # (os.kill(pid, 0) is destructive there) — assume alive and let the
    # HTTP send itself be the arbiter.
    return True

# Dedup parameters — the gRPC stream is at-least-once, and a sidecar
# reconnect can replay, so keep at least 1k ids for ~48h.
_DEDUP_MAX_SIZE = 4000
_DEDUP_WINDOW_SECONDS = 48 * 3600

_FFFC_WAIT_SECONDS = 15.0  # Timeout for waiting on an attachment after a U+FFFC placeholder.

# Resolved lazily on first use: the installed plugin tree when writable (dev
# installs), or a mirror on the durable data volume when the install tree
# is immutable and the baked deps are missing/stale (hosted images, NS-606).
# See sidecar_paths.resolve_sidecar_dir for the full decision table.
#
# Resolution is deliberately NOT done at import time: resolve_sidecar_dir()
# probes the filesystem (touch/unlink) and may mirror files to the data
# volume — side effects that must not fire just because something imported
# this module (hermes status, test collection, plugin discovery).
from .sidecar_paths import dir_writable as _dir_writable, resolve_sidecar_dir

# Tests monkeypatch these module globals directly; the accessors below
# honor a non-None value and only resolve/derive when unset.
_SIDECAR_DIR: Optional[Path] = None
_NPM_ERROR_LOG: Optional[Path] = None
_NPM_ERROR_LOG_MAX_CHARS = 300


def _sidecar_dir() -> Path:
    """Sidecar runtime dir, resolved once on first use (never at import)."""
    global _SIDECAR_DIR
    if _SIDECAR_DIR is None:
        _SIDECAR_DIR = resolve_sidecar_dir()
    return _SIDECAR_DIR


def _npm_error_log() -> Path:
    """Path of the persisted npm-failure log (derived from the sidecar dir)."""
    if _NPM_ERROR_LOG is not None:
        return _NPM_ERROR_LOG
    return _sidecar_dir() / ".photon-npm-error.log"

# Cap on a self-heal `npm ci`/`npm install` of the sidecar deps. A cold
# install of the pinned spectrum-ts tree normally takes well under a minute;
# a wedged npm (dead registry, network blackhole) must not stall the photon
# connect path indefinitely.
_NPM_REINSTALL_TIMEOUT = 600

# Photon / Envoy / spectrum-ts error substrings that indicate a transient
# upstream overload rather than a permanent failure.  These are not in the
# core _RETRYABLE_ERROR_PATTERNS because they are specific to this adapter.
_PHOTON_RETRYABLE_PATTERNS = (
    "internal sidecar error",
    "upstream connect error",
    "upstream unavailable",
    "connection dropped",
    "reset reason: overflow",
    "upstream_overflow",
    "upstream_unavailable",
)

# iMessage may emit the Open Graph preview art for a rich link as one or more
# image attachments immediately after the URL/richlink message. Suppress those
# artifacts so Hermes sees the link once, not a follow-up "(attachment)" prompt.
_RICHLINK_PREVIEW_SUPPRESS_SECONDS = 30.0
_RICHLINK_PREVIEW_ATTACHMENT_SUFFIX = ".pluginpayloadattachment"

# Minimum seconds between typing-indicator calls for the same chat.
# iMessage is a personal channel — suppressing rapid repeats reduces
# upstream gRPC pressure during Photon overflow events.
_TYPING_COOLDOWN_SECONDS = 5.0

# Group-chat mention wake words. When ``require_mention`` is enabled, group
# messages are ignored unless they match one of these patterns — same
# behavior and defaults as the BlueBubbles iMessage channel so the two
# iMessage adapters gate group chats identically.
_DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]


# ---------------------------------------------------------------------------
# Module-level helpers — also used by check_fn / standalone send

class PhotonSidecarStartupError(RuntimeError):
    """Typed startup failure raised by ``_start_sidecar`` (OOF-156).

    Carries the fatal-error classification instead of relying on
    ``connect()``'s catch-all, which used to collapse every startup failure
    into ``SIDECAR_FAILED, retryable=True`` — including deterministic ones
    (deps that can't install on an immutable image, missing node binary)
    that then retried forever with zero owner signal. ``retryable`` defaults
    to True: only failures known to be deterministic mark themselves False.
    """

    def __init__(self, message: str, *, code: str = "SIDECAR_FAILED", retryable: bool = True) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class PhotonSidecarError(RuntimeError):
    """Structured failure returned by the supervised Photon sidecar."""

    def __init__(
        self,
        *,
        path: str,
        status_code: int,
        error: str,
        error_class: str = "sidecar_error",
        retryable: bool = False,
    ) -> None:
        self.path = path
        self.status_code = status_code
        self.error = error
        self.error_class = error_class
        self.retryable = retryable
        super().__init__(
            f"Photon sidecar {path} returned {status_code} "
            f"({error_class}, retryable={retryable}): {error}"
        )


def _sidecar_error_from_response(
    path: str,
    status_code: int,
    text: str,
    data: Optional[Dict[str, Any]] = None,
) -> PhotonSidecarError:
    if data is None:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}
        data = parsed if isinstance(parsed, dict) else {}

    error = str(data.get("error") or text[:200] or "sidecar error")
    error_class = str(data.get("error_class") or "sidecar_error")
    retryable = bool(data.get("retryable"))
    if error_class == "target_not_allowed":
        # Structured code path: replace whatever the sidecar echoed with the
        # canonical user-facing explanation — never raw upstream error text.
        error = _TARGET_NOT_ALLOWED_MESSAGE
        retryable = False
    return PhotonSidecarError(
        path=path,
        status_code=status_code,
        error=error,
        error_class=error_class,
        retryable=retryable,
    )

# User-facing explanation for the Spectrum "Target not allowed for this
# project" AuthenticationError. Shared/free-tier lines can only reply to
# conversations the target initiated; they cannot open new outbound threads.
_TARGET_NOT_ALLOWED_MESSAGE = (
    "shared/free-tier Photon lines cannot initiate outbound sends to new "
    "targets — upgrade to a dedicated line or use another delivery channel"
)

def _coerce_port(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sidecar_deps_installed() -> bool:
    """True when spectrum-ts is present under node_modules/.

    Checks the dependency's own directory, not just node_modules/'s
    existence: npm creates node_modules/ before aborting on ENOSPC, a
    network timeout, or EACCES, so an empty/partial node_modules/ would
    otherwise read as "installed". Shared by check_requirements(),
    _start_sidecar(), and `hermes photon status` so all three agree on
    what "installed" means.
    """
    return (_sidecar_dir() / "node_modules" / "spectrum-ts").exists()


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_set(*values: Any) -> Any:
    """Return the first value that is not None.

    Unlike ``a or b``, this preserves an explicit falsy value (e.g. ``0`` or
    ``""``) so config can intentionally set a zero/empty without it silently
    falling through to a later source or a default.
    """
    for v in values:
        if v is not None:
            return v
    return None


def _is_timeout_error(exc: BaseException) -> bool:
    """True when *exc* indicates the request timed out (call hung)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if HTTPX_AVAILABLE:
        timeout_exc = getattr(httpx, "TimeoutException", None)
        if timeout_exc is not None and isinstance(exc, timeout_exc):
            return True
    return "timeout" in type(exc).__name__.lower()


def check_requirements() -> bool:
    """Return True when both Python deps and the Node sidecar are available."""
    if not HTTPX_AVAILABLE:
        logger.warning("photon: httpx not installed — pip install httpx")
        return False
    if not shutil.which(os.getenv("PHOTON_NODE_BIN") or "node"):
        logger.warning(
            "photon: node binary '%s' not found on PATH",
            os.getenv("PHOTON_NODE_BIN") or "node",
        )
        return False
    if not sidecar_deps_installed():
        # spectrum-ts not installed yet, or node_modules/ was partially created
        # by an aborted npm install (ENOSPC, network timeout, EACCES).
        # Checking spectrum-ts presence — not just node_modules/ existence —
        # prevents a false positive where an empty/broken node_modules/ dir
        # causes check_requirements() to return True while the sidecar crashes
        # at runtime with an unrelated-looking missing-module error.
        #
        # NS-606: if we can self-install at connect time — npm on PATH and
        # the (resolved, possibly mirrored) sidecar dir is writable — report
        # available so the gateway creates the adapter and ``_start_sidecar``
        # cold-installs from the committed lockfile (on hosted images the
        # user has no CLI to run `hermes photon setup`, so the connect path
        # must self-heal). Otherwise keep returning False so
        # `hermes setup` / status surface the missing-deps state.
        if bool(shutil.which("npm")) and _dir_writable(_sidecar_dir()):
            return True
        # DEBUG (not WARNING): this is the normal pre-setup state.
        # check_fn() is called from multiple hot paths in the core
        # (load_gateway_config, hermes status, GET /api/status polling) —
        # WARNING here would spam logs on every probe for unconfigured photon.
        npm_error = ""
        try:
            if _npm_error_log().exists():
                npm_error = _npm_error_log().read_text(encoding="utf-8").strip()[:_NPM_ERROR_LOG_MAX_CHARS]
        except OSError:
            pass
        if npm_error:
            logger.debug(
                "photon: spectrum-ts not installed at %s "
                "(last npm error: %s) — run: hermes photon setup",
                _sidecar_dir(),
                npm_error,
            )
        else:
            logger.debug(
                "photon: spectrum-ts not installed at %s — run: hermes photon setup",
                _sidecar_dir(),
            )
        return False
    return True


def _sidecar_deps_stale() -> bool:
    """True when node_modules exists but is older than the committed lockfile.

    `hermes update` rewrites ``package-lock.json`` when the spectrum-ts pin is
    bumped, but does not reinstall ``node_modules``. npm records the state of
    the last install in ``node_modules/.package-lock.json``; when the top-level
    lockfile is newer than that marker, the install is out of date. This is the
    same signal ``npm ci`` uses. Returns False (do nothing) if either file is
    missing or unreadable, so a first-run or odd filesystem never blocks start.
    """
    lockfile = _sidecar_dir() / "package-lock.json"
    marker = _sidecar_dir() / "node_modules" / ".package-lock.json"
    try:
        return lockfile.stat().st_mtime > marker.stat().st_mtime
    except OSError:
        return False


def _reinstall_sidecar_deps() -> None:
    """Reinstall the sidecar's node_modules from the lockfile (blocking).

    Mirrors ``hermes photon install-sidecar``: ``npm ci`` for an exact,
    reproducible install, falling back to ``npm install`` if the lockfile is
    missing or drifted. Runs the postinstall patch as part of the install.
    Best-effort — a failure here just leaves the (stale) deps in place and the
    normal ``_start_sidecar`` readiness check reports the real error.
    """
    npm = shutil.which("npm")
    if not npm:
        logger.warning("[photon] cannot reinstall stale sidecar deps: npm not on PATH")
        return
    # Windows: suppress the console flash these short-lived npm runs would
    # otherwise pop (0 elsewhere). Same helper as the sidecar spawn below.
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        result = subprocess.run(  # noqa: S603
            [npm, "ci"],
            cwd=str(_sidecar_dir()),
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
            timeout=_NPM_REINSTALL_TIMEOUT,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            logger.warning(
                "[photon] sidecar `npm ci` failed; falling back to `npm install`"
            )
            result = subprocess.run(  # noqa: S603
                [npm, "install"],
                cwd=str(_sidecar_dir()),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False,
                timeout=_NPM_REINSTALL_TIMEOUT,
                creationflags=windows_hide_flags(),
            )
    except subprocess.TimeoutExpired:
        # A wedged npm (dead registry, network blackhole) must not stall the
        # photon connect forever — give up, leave the stale deps in place, and
        # let the readiness check report the real error. Retried on the next
        # reconnect tick.
        logger.error(
            "[photon] sidecar dependency reinstall timed out after %ss",
            _NPM_REINSTALL_TIMEOUT,
        )
        return
    if result.returncode != 0:
        logger.error(
            "[photon] sidecar dependency reinstall failed: %s",
            (result.stderr or result.stdout or "").strip(),
        )
    else:
        logger.info("[photon] sidecar dependencies reinstalled from lockfile")


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    project_id = extra.get("project_id") or os.getenv("PHOTON_PROJECT_ID")
    project_secret = extra.get("project_secret") or _get_scoped_secret("PHOTON_PROJECT_SECRET")
    if not project_id or not project_secret:
        # Fall back to auth.json
        stored_id, stored_sec = load_project_credentials()
        return bool(stored_id and stored_sec)
    return True


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so env-only setups appear in status.

    The special ``home_channel`` key is handled by the core plugin hook and
    becomes a proper ``HomeChannel`` on ``PlatformConfig``.
    """
    project_id, project_secret = load_project_credentials()
    if not (project_id and project_secret):
        return None
    seed: dict = {"project_id": project_id, "project_secret": project_secret}
    home = os.getenv("PHOTON_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("PHOTON_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


def _markdown_enabled() -> bool:
    """Send agent replies as markdown (spectrum-ts ``markdown()`` builder).

    iMessage renders it natively; other Spectrum platforms degrade to
    readable plain text. On-device rendering can't be unit-tested, so
    ``PHOTON_MARKDOWN=false`` is the kill-switch back to stripped plain
    text without a release.
    """
    return os.getenv("PHOTON_MARKDOWN", "true").strip().lower() not in {
        "false", "0", "no",
    }


def _url_only_candidate(text: str) -> Optional[str]:
    candidate = (text or "").strip()
    if not re.fullmatch(r"https?://\S+", candidate, flags=re.IGNORECASE):
        return None
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def _richlink_candidate(text: str) -> Optional[str]:
    """Return a URL that should be sent via spectrum-ts ``richlink()``.

    Keep this intentionally narrow: only exact http(s) URL messages become
    rich links. Prose containing URLs and Markdown links stay on the normal
    markdown/text path so Hermes does not drop labels or rewrite intent.
    """
    if not _markdown_enabled():
        return None
    return _url_only_candidate(text)


def _format_richlink_content(content: Dict[str, Any]) -> str:
    url = str(content.get("url") or "").strip()
    title = str(content.get("title") or "").strip()
    summary = str(content.get("summary") or "").strip()
    parts: List[str] = []
    if title:
        parts.append(title)
    if summary and summary != title:
        parts.append(summary)
    if url:
        parts.append(url)
    return "\n".join(parts) if parts else "[Photon rich link received with no URL]"


def _richlink_url_from_content(content: Dict[str, Any]) -> Optional[str]:
    ctype = content.get("type")
    if ctype == "text":
        return _url_only_candidate(content.get("text") or "")
    if ctype == "richlink":
        return _url_only_candidate(content.get("url") or "")
    if ctype == "group":
        for item in content.get("items") or []:
            if not isinstance(item, dict):
                continue
            item_content = item.get("content") or {}
            if isinstance(item_content, dict):
                url = _richlink_url_from_content(item_content)
                if url:
                    return url
    return None


def _is_richlink_preview_attachment(payload: Dict[str, Any]) -> bool:
    if payload.get("type") != "attachment":
        return False
    name = str(payload.get("name") or "").lower()
    attachment_id = str(payload.get("id") or "").lower()
    is_preview_payload = (
        name.endswith(_RICHLINK_PREVIEW_ATTACHMENT_SUFFIX)
        or attachment_id.endswith(_RICHLINK_PREVIEW_ATTACHMENT_SUFFIX)
        or _RICHLINK_PREVIEW_ATTACHMENT_SUFFIX in name
        or _RICHLINK_PREVIEW_ATTACHMENT_SUFFIX in attachment_id
    )
    # Live iMessage rich-link preview art can arrive with this marker but an
    # opaque document MIME such as application/octet-stream. The marker is the
    # reliable signal; the recent-link window guards real attachments.
    return is_preview_payload


def _richlink_preview_label(content: Dict[str, Any]) -> str:
    if content.get("type") == "attachment":
        return str(content.get("name") or content.get("id") or "(unnamed)")
    if content.get("type") == "group":
        labels = []
        for item in content.get("items") or []:
            item_content = item.get("content") if isinstance(item, dict) else None
            if isinstance(item_content, dict):
                labels.append(
                    str(item_content.get("name") or item_content.get("id") or "(unnamed)")
                )
        return ", ".join(labels) or "(group)"
    return "(unknown)"


def _is_richlink_preview_content(content: Dict[str, Any]) -> bool:
    if _is_richlink_preview_attachment(content):
        return True
    if content.get("type") != "group":
        return False
    items = content.get("items") or []
    if not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        item_content = item.get("content") or {}
        if not isinstance(item_content, dict):
            return False
        if not _is_richlink_preview_attachment(item_content):
            return False
    return True

# ---------------------------------------------------------------------------
# Adapter


class PhotonAdapter(BasePlatformAdapter):
    """Bidirectional bridge to Photon Spectrum via the Node spectrum-ts sidecar.

    Inbound: consume the sidecar's ``/inbound`` gRPC stream.
    Outbound: loopback POSTs to the sidecar's control channel.
    """

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH
    # Photon (iMessage) has no real edit API for already-sent messages.
    # Mark it explicitly so streaming suppresses the visible cursor instead
    # of leaving a stale tofu square (▉) behind when edit attempts fail.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("photon"))
        extra = config.extra or {}

        # Project credentials (env wins, then config.extra, then auth.json).
        # ``project_id`` here is the project's spectrumProjectId — the value
        # the spectrum-ts SDK authenticates with.
        stored_id, stored_sec = load_project_credentials()
        self._project_id: str = (
            os.getenv("PHOTON_PROJECT_ID")
            or extra.get("project_id")
            or stored_id
            or ""
        )
        self._project_secret: str = (
            _get_scoped_secret("PHOTON_PROJECT_SECRET")
            or extra.get("project_secret")
            or stored_sec
            or ""
        )

        # Sidecar
        self._sidecar_port = _coerce_port(
            extra.get("sidecar_port") or os.getenv("PHOTON_SIDECAR_PORT"),
            _DEFAULT_SIDECAR_PORT,
        )
        self._sidecar_bind = _DEFAULT_SIDECAR_BIND
        self._sidecar_token = (
            _get_scoped_secret("PHOTON_SIDECAR_TOKEN") or secrets.token_hex(16)
        )
        self._autostart_sidecar = str(
            os.getenv("PHOTON_SIDECAR_AUTOSTART", "true")
        ).lower() not in ("0", "false", "no")
        self._node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node") or "node"

        # Presence watchdog. spectrum-ts only reconnects when its inbound
        # iterator throws or ends; a half-open ("zombie") gRPC socket makes the
        # iterator hang forever (no error, no end), so inbound silently dies
        # until the sidecar is restarted. The sidecar owns primary zombie
        # detection (stream-staleness + upstream probe -> degraded -> exit 75;
        # surfaced here via /healthz in _monitor_sidecar_health). This adapter-
        # side watchdog is a conservative second layer that only respawns the
        # sidecar when the sidecar's own HTTP loop stops responding (probe
        # HTTP call hangs) — an "inconclusive" probe (sidecar answered but
        # could not prove upstream liveness) NEVER counts toward a respawn:
        # the network may simply be down, and restarting cannot fix that.
        # Thresholds are deliberately conservative (10 min interval) — shared
        # lines can be legitimately quiet for hours, so we never restart on
        # silence alone.
        # Behavioural settings -> config.yaml (extra), bridged to env.
        # Use _first_set (not ``or``) so an explicit 0 is honored — ``0 or X``
        # would silently fall through to the default and you could never
        # disable the watchdog with probe_interval_seconds: 0.
        self._probe_interval = _coerce_float(
            _first_set(
                extra.get("probe_interval_seconds"),
                os.getenv("PHOTON_PROBE_INTERVAL_SECONDS"),
            ),
            600.0,
        )
        self._probe_timeout = _coerce_float(
            _first_set(
                extra.get("probe_timeout_seconds"),
                os.getenv("PHOTON_PROBE_TIMEOUT_SECONDS"),
            ),
            10.0,
        )
        self._probe_max_failures = _coerce_int(
            _first_set(
                extra.get("probe_max_failures"),
                os.getenv("PHOTON_PROBE_MAX_FAILURES"),
            ),
            3,
        )
        # A non-positive interval disables the watchdog entirely (escape hatch).
        self._probe_enabled = self._probe_interval > 0

        # With markdown on, format_message preserves fences and the sidecar's
        # markdown() builder renders them (or degrades them readably).
        self.supports_code_blocks = _markdown_enabled()

        # Runtime state
        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._sidecar_supervisor_task: Optional[asyncio.Task] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._sidecar_health_task: Optional[asyncio.Task] = None
        self._inbound_running = False
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._sidecar_health_interval = 15.0
        # Presence-watchdog runtime state.
        self._watchdog_task: Optional[asyncio.Task] = None
        self._watchdog_running = False
        self._probe_failures = 0
        # Monotonic timestamp of the last real upstream activity (inbound
        # message or a successful probe). The watchdog skips its own probe when
        # natural traffic already proved the channel live within the interval.
        self._last_upstream_activity = 0.0
        self._respawn_lock: Optional[asyncio.Lock] = None
        # Lightweight in-memory dedup. The gRPC stream is at-least-once, so we
        # may see the same messageId more than once (e.g. after a reconnect).
        self._seen_messages: Dict[str, float] = {}
        # Ids of messages WE sent (bounded, insertion-order eviction). Inbound
        # reaction events are only routed to the agent when they target one of
        # these — a tapback on a human↔human message is not addressed to us.
        self._sent_message_ids: Dict[str, float] = {}
        # Latest inbound message id per chat (bounded). Lets the agent-facing
        # react action default to "the message that triggered me" without
        # requiring the model to thread message ids through tool calls.
        self._last_inbound_by_chat: Dict[str, str] = {}
        # Latest inbound URL/richlink per chat. iMessage can send rich-link
        # preview artwork as separate image attachments immediately after the
        # URL bubble; this lets us coalesce those artifacts.
        self._recent_richlinks_by_chat: Dict[str, float] = {}
        # Last time we sent a typing indicator per chat, for cooldown gating.
        self._typing_last_sent: Dict[str, float] = {}

        self._pending_fffc: Dict[str, tuple[float, Any]] = {}  # chat_key → (timestamp, asyncio.Task)

        # Group-chat mention gating (parity with BlueBubbles). When enabled,
        # group messages are ignored unless they match a wake word; DMs are
        # always processed. Config key wins, then env var.
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("PHOTON_REQUIRE_MENTION")
        self.require_mention = str(_require_mention).strip().lower() in {
            "true", "1", "yes", "on",
        }
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("PHOTON_MENTION_PATTERNS")
        )

    # -- Group-mention gating (parity with BlueBubbles) -------------------

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> "list[re.Pattern]":
        """Compile group-mention wake words from config/env.

        ``raw`` is a list (config or env JSON), a string (env var: JSON
        list, or comma/newline-separated), or None (use Hermes defaults).
        Mirrors the BlueBubbles implementation so both iMessage channels
        accept the same configuration shapes.
        """
        return compile_mention_patterns(
            raw,
            log_prefix="photon",
            defaults=_DEFAULT_MENTION_PATTERNS,
            logger_=logger,
        )

    def _message_matches_mention_patterns(self, text: str) -> bool:
        if not text or not self._mention_patterns:
            return False
        return any(pattern.search(text) for pattern in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip a leading wake word before dispatch.

        Custom mention patterns are regexes, so we only strip a leading
        match to avoid deleting ordinary words later in the prompt.
        """
        if not text:
            return text
        for pattern in self._mention_patterns:
            match = pattern.match(text.lstrip())
            if match:
                cleaned = text.lstrip()[match.end():].lstrip(" ,:-")
                return cleaned or text
        return text

    # -- Connection lifecycle ---------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not HTTPX_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP", "httpx not installed", retryable=False
            )
            return False
        if not self._project_id or not self._project_secret:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "PHOTON_PROJECT_ID and PHOTON_PROJECT_SECRET are required. "
                "Run: hermes photon setup",
                retryable=False,
            )
            return False

        client = httpx.AsyncClient(timeout=30.0, trust_env=False)
        self._http_client = client

        # The sidecar holds the gRPC stream for BOTH directions, so it is
        # required now (not just for outbound).
        if self._autostart_sidecar:
            try:
                await self._start_sidecar()
            except Exception as e:
                # Honor typed classification from _start_sidecar (OOF-153):
                # deterministic failures (deps can't install on an immutable
                # image, node binary missing) are retryable=False so they
                # surface as fatal instead of silently spinning in the
                # reconnect queue. Everything else — sidecar crashed before
                # ready, /healthz timeout — stays retryable=True; ambiguity
                # resolves toward retry, with the gateway's NEEDS_ATTENTION
                # escalation as the backstop for long-lived loops.
                if isinstance(e, PhotonSidecarStartupError):
                    self._set_fatal_error(e.code, str(e), retryable=e.retryable)
                else:
                    self._set_fatal_error(
                        "SIDECAR_FAILED",
                        f"failed to start Photon sidecar: {e}",
                        retryable=True,
                    )
                # No live sidecar — make sure no stale runtime record
                # survives to mislead standalone senders.
                _delete_runtime_record()
                await client.aclose()
                self._http_client = None
                return False
        else:
            logger.warning(
                "[photon] sidecar autostart disabled — inbound + outbound will fail"
            )

        # Start consuming the inbound gRPC stream from the sidecar.
        self._inbound_running = True
        self._inbound_task = asyncio.get_event_loop().create_task(
            self._inbound_loop()
        )
        self._sidecar_health_task = asyncio.get_event_loop().create_task(
            self._monitor_sidecar_health()
        )

        # Start the presence watchdog (detects a half-open/zombie gRPC stream
        # the SDK can't see and respawns the sidecar to recover inbound).
        self._last_upstream_activity = time.monotonic()
        if self._probe_enabled and self._autostart_sidecar:
            self._respawn_lock = asyncio.Lock()
            self._watchdog_running = True
            self._watchdog_task = asyncio.get_event_loop().create_task(
                self._presence_watchdog()
            )

        self._mark_connected()
        logger.info(
            "[photon] connected — sidecar on %s:%d, streaming inbound over gRPC",
            self._sidecar_bind, self._sidecar_port,
        )
        return True

    async def disconnect(self) -> None:
        self._inbound_running = False
        # Stop the watchdog first so it can't trigger a respawn while we tear
        # the sidecar down.
        await self._stop_watchdog()
        if self._sidecar_health_task is not None:
            task = self._sidecar_health_task
            self._sidecar_health_task = None
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        if self._inbound_task is not None:
            task = self._inbound_task
            self._inbound_task = None
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        # Cancel any pending U+FFFC placeholder tasks.
        for chat_key, (_, fffc_task) in list(self._pending_fffc.items()):
            if fffc_task and not fffc_task.done():
                fffc_task.cancel()
        self._pending_fffc.clear()
        await self._stop_sidecar()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._mark_disconnected()

    def _dispatch_fatal_notification(self) -> None:
        """Notify the gateway of a fatal error from a detached task.

        ``_monitor_sidecar_health`` and ``_supervise_sidecar`` run as
        ``self._sidecar_health_task`` / ``self._sidecar_supervisor_task`` and
        used to call ``await self._notify_fatal_error()`` directly, on their
        own call stack. That routes straight into
        ``GatewayRunner._handle_adapter_fatal_error_impl``, which tears the
        adapter down via ``_safe_adapter_disconnect`` -> ``disconnect()``.
        ``disconnect()`` cancels ``self._sidecar_health_task`` and awaits it
        to make sure it's really gone -- but when the *health task itself* is
        what's driving this whole notification (the common case: the health
        poll is what detected the fatal condition), that task IS the one
        awaiting several frames up the stack, so ``disconnect()`` cancels and
        awaits its own caller. ``disconnect()``'s
        ``task is not asyncio.current_task()`` guard doesn't catch this,
        because the wrapper task ``_await_adapter_cleanup_with_timeout``
        creates around ``disconnect()`` (via ``asyncio.ensure_future``) is
        the current task there, not the health/supervisor task further up
        the plain-await chain. The self-cancellation throws an unhandled
        ``CancelledError`` through the health/supervisor task with nothing
        to catch it -- ``CancelledError`` stopped subclassing ``Exception``
        in Python 3.8, so the ``except Exception`` around the notify call
        never saw it -- and the task dies silently mid-handoff: no log line,
        no retry, the platform stranded with no automated way back.

        Dispatching the notification onto a brand-new task instead -- the
        same pattern ``DiscordAdapter._handle_bot_task_done`` already uses --
        means ``disconnect()`` can cancel the health/supervisor task freely
        without that cancellation ever reaching back into the code that's
        still running the handoff, so the handoff always reaches the
        reconnect queue.
        """
        asyncio.create_task(self._notify_fatal_error_logged())

    async def _notify_fatal_error_logged(self) -> None:
        try:
            await self._notify_fatal_error()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[photon] fatal-error notification failed: %s", exc)

    # -- Inbound stream consumer ------------------------------------------

    async def _inbound_loop(self) -> None:
        """Consume the sidecar's ``/inbound`` NDJSON stream, with reconnect.

        The sidecar owns the gRPC reconnect/heartbeat to Photon; this loop
        only has to re-open the loopback HTTP stream if it drops (e.g. the
        sidecar restarts).
        """
        client = self._http_client
        if client is None:
            return
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}/inbound"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        backoff = 1.0
        while self._inbound_running:
            try:
                async with client.stream(
                    "GET", url, headers=headers, timeout=None,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"/inbound returned {resp.status_code}")
                    backoff = 1.0  # reset on a successful connect
                    async for line in resp.aiter_lines():
                        if not self._inbound_running:
                            break
                        line = line.strip()
                        if not line:
                            continue  # heartbeat
                        await self._on_inbound_line(line)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._inbound_running:
                    break
                logger.warning(
                    "[photon] inbound stream dropped (%s); reconnecting in %.1fs",
                    e, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _monitor_sidecar_health(self) -> None:
        """Promote degraded upstream Photon stream health into reconnect.

        The sidecar HTTP process can stay alive while spectrum-ts repeatedly
        fails to maintain the upstream inbound gRPC stream. Polling `/healthz`
        keeps that from becoming a silent inbound outage.
        """
        while self._inbound_running:
            await asyncio.sleep(self._sidecar_health_interval)
            if not self._inbound_running:
                break
            try:
                data = await self._sidecar_call("/healthz", {})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[photon] sidecar health check failed: %s", exc)
                continue

            stream = data.get("stream") if isinstance(data, dict) else None
            if not isinstance(stream, dict):
                continue

            # Surface the sidecar's zombie-stream staleness state early: a
            # suspected half-open stream (silence past threshold + probe-proven
            # connectivity) is worth a loud log line even before the sidecar's
            # degraded->exit-75 path fires.
            staleness = stream.get("staleness")
            if (
                isinstance(staleness, dict)
                and staleness.get("zombieSuspected") is True
            ):
                logger.warning(
                    "[photon] sidecar reports suspected zombie stream"
                    " (silentForMs=%s, lastProbeOutcome=%s)",
                    staleness.get("silentForMs"),
                    staleness.get("lastProbeOutcome"),
                )

            if stream.get("ok") is not False:
                continue

            state = str(stream.get("state") or "unknown")
            degraded_for_ms = stream.get("degradedForMs")
            last_issue = str(stream.get("lastIssue") or "unknown stream issue")
            message = (
                "Photon upstream stream degraded"
                f" (state={state}, degradedForMs={degraded_for_ms}): "
                f"{last_issue}"
            )
            logger.error("[photon] %s", message)
            self._set_fatal_error(
                "UPSTREAM_STREAM_DEGRADED",
                message,
                retryable=True,
            )
            self._dispatch_fatal_notification()
            break

    async def _on_inbound_line(self, line: str) -> None:
        # Any line on the inbound stream — message OR heartbeat — proves the
        # upstream gRPC channel is live, so reset the watchdog's failure count
        # and activity clock. (Heartbeats arrive as blank lines, already
        # filtered before this method, but a real message is the strongest
        # liveness signal we have.)
        self._note_upstream_activity()
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[photon] skipping non-JSON inbound line")
            return
        msg_id = event.get("messageId")
        if msg_id and self._is_duplicate(msg_id):
            return
        try:
            await self._dispatch_inbound(event)
        except Exception:
            logger.exception("[photon] inbound dispatch failed")

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        seen = self._seen_messages
        t = seen.get(msg_id)
        if t is not None and now - t < _DEDUP_WINDOW_SECONDS:
            return True  # seen, unexpired
        # New or expired: record and enforce a HARD size bound (evict oldest,
        # insertion-order) so a burst of unique ids within the window can't grow
        # the dict without limit — not just the expired-only prune.
        if msg_id in seen:
            del seen[msg_id]  # refresh insertion order
        seen[msg_id] = now
        if len(seen) > _DEDUP_MAX_SIZE:
            for old in list(seen.keys())[: len(seen) - _DEDUP_MAX_SIZE]:
                del seen[old]
        return False

    async def _fffc_timeout_handler(self, chat_key: str, message_id: str) -> None:
        await asyncio.sleep(_FFFC_WAIT_SECONDS)
        if self._pending_fffc.pop(chat_key, None):
            logger.warning(
                "[photon] wait for attachment was too long, can't retrieve attachment data "
                "(message %s, chat %s)", message_id, chat_key,
            )

    async def _dispatch_inbound(self, event: Dict[str, Any]) -> None:
        """Normalize a sidecar inbound event and dispatch it to the gateway.

        Event shape (from ``sidecar/index.mjs``)::

            {
              "messageId": "...",
              "platform": "iMessage",
              "space": {"id": "...", "type": "dm"|"group", "phone": "+E164"},
              "sender": {"id": "+E164"},
              "content": {"type": "text", "text": "..."}
                       | {"type": "attachment"|"voice", "id", "name",
                          "mimeType", "size", "duration"?, "data"?,
                          "encoding"?}
                       | {"type": "reaction", "emoji": "❤️",
                          "targetMessageId": "..." | null,
                          "targetDirection": "inbound"|"outbound" | null,
                          "targetText": "..." | null},
              "timestamp": "2026-05-14T19:06:32.000Z"

        Attachment and voice content carry the bytes inline as base64 ``data``
        (with ``encoding == "base64"``) when the sidecar could read them
        within its size cap; otherwise only metadata is present and we surface
        a marker.
            }
        """
        space = event.get("space") or {}
        sender = event.get("sender") or {}
        content = event.get("content") or {}

        space_id = space.get("id") or ""
        if not space_id:
            logger.warning("[photon] inbound missing space.id")
            return

        # iMessage spaces carry their type directly — no id string-sniffing.
        chat_type = "group" if space.get("type") == "group" else "dm"
        sender_id = sender.get("id") or space.get("phone") or space_id

        ts_str = event.get("timestamp") or ""
        try:
            timestamp = (
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_str
                else datetime.now(tz=timezone.utc)
            )
        except ValueError:
            timestamp = datetime.now(tz=timezone.utc)

        # Media attachments (local cached paths) handed to the agent via the
        # gateway's image-routing path, exactly like the BlueBubbles channel.
        media_urls: List[str] = []
        media_types: List[str] = []

        def _normalize_binary_payload(
            payload: Dict[str, Any]
        ) -> tuple[str, MessageType, List[str], List[str]]:
            is_voice = payload.get("type") == "voice"
            name = payload.get("name") or ("voice" if is_voice else "(unnamed)")
            mime = payload.get("mimeType") or ""
            # Promote CAF attachments to VOICE (iMessage voice notes use CAF).
            # Check both filename and MIME: the sidecar may send "(unnamed)"
            # when no name is supplied, so the MIME type is the reliable signal.
            if not is_voice and (name.lower().endswith(".caf") or mime == "audio/x-caf"):
                is_voice = True
            mtype = MessageType.VOICE if is_voice else _attachment_message_type(mime)
            cached = _cache_inbound_attachment(
                payload, name, mime, force_audio=is_voice
            )
            if cached:
                return (
                    "(voice)" if is_voice else "(attachment)",
                    mtype,
                    [cached],
                    [mime or ("audio/mp4" if is_voice else "application/octet-stream")],
                )
            label = "voice" if is_voice else "attachment"
            duration = payload.get("duration")
            duration_text = (
                f", duration: {duration}s"
                if isinstance(duration, (int, float))
                else ""
            )
            return (
                f"[Photon {label} received: {name} "
                f"({mime or 'unknown MIME'}{duration_text})]",
                mtype,
                [],
                [],
            )

        ctype = content.get("type")
        if ctype == "reaction":
            # Route only tapbacks on messages WE sent — those are implicitly
            # addressed to the bot (feishu precedent: synthetic text event).
            # Reactions on human↔human messages are not for us. Checked before
            # the mention gate: a tapback never carries a wake word.
            target_id = content.get("targetMessageId")
            is_ours = content.get("targetDirection") == "outbound" or (
                target_id and target_id in self._sent_message_ids
            )
            if not is_ours:
                logger.debug(
                    "[photon] ignoring reaction on a message we didn't send"
                )
                return
            emoji = content.get("emoji") or ""
            source = self.build_source(
                chat_id=space_id,
                chat_name=space_id,
                chat_type=chat_type,
                user_id=sender_id,
                user_name=sender_id or None,
            )
            # Correlate the tapback to the message it reacted to, so the agent
            # sees WHAT was reacted to. `is_ours` above guarantees the target is
            # one of the bot's own messages, so reply_to_is_own_message holds and
            # the gateway injects `[Replying to your previous message: "..."]`.
            # reply_to_text comes from the sidecar (hydrated reaction target);
            # it's None for attachment/voice-only targets, and the gateway only
            # injects the pointer when both id and text are present.
            await self.handle_message(
                MessageEvent(
                    text=f"reaction:added:{emoji}",
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=event.get("messageId"),
                    reply_to_message_id=target_id,
                    reply_to_text=content.get("targetText") or None,
                    reply_to_is_own_message=True,
                    raw_message=event,
                    timestamp=timestamp,
                )
            )
            return
        # U+FFFC placeholder — wait for the real attachment instead of
        # dispatching. Detected before _record_last_inbound so the placeholder
        # is not recorded as the reaction target — the real attachment will be.
        if ctype == "text" and (content.get("text") or "").strip() == "\ufffc":
            chat_key = space_id
            prev = self._pending_fffc.pop(chat_key, None)
            if prev and prev[1] and not prev[1].done():
                prev[1].cancel()
            task = asyncio.create_task(
                self._fffc_timeout_handler(chat_key, event.get("messageId") or "")
            )
            self._pending_fffc[chat_key] = (time.monotonic(), task)
            logger.debug("[photon] U+FFFC placeholder received — waiting for attachment")
            return

        # Cancel any pending U+FFFC timeout — the real message arrived.
        if ctype in {"attachment", "voice"}:
            prev = self._pending_fffc.pop(space_id, None)
            if prev and prev[1] and not prev[1].done():
                prev[1].cancel()
                logger.debug("[photon] attachment arrived — cancelling U+FFFC timeout")

        # Preview art for an immediately preceding URL/richlink should not
        # become a second user prompt. Suppress before recording it as the
        # latest reactable inbound or decoding/caching image bytes.
        if self._is_recent_richlink_preview(space_id, content):
            logger.info(
                "[photon] suppressing rich-link preview attachment: %s",
                _richlink_preview_label(content),
            )
            return
        # Anything past here is a real (reactable) message — remember it as
        # the chat's latest inbound so `add_reaction` can target it when the
        # caller doesn't pass an explicit message id. Recorded before the
        # mention gate: a reaction to a non-wake-word group message is valid.
        self._record_last_inbound(space_id, event.get("messageId"))
        if ctype == "poll_option":
            # A native poll vote. A *selection* carries the chosen option text
            # straight to the agent as if the user had typed it — the gateway's
            # pending-clarify text-intercept then resolves the open clarify and
            # unblocks the agent. A *deselection* (selected=false) is dropped:
            # there's no answer to record, and forwarding "" would mis-resolve.
            if content.get("selected") is False:
                logger.debug("[photon] ignoring poll deselection")
                return
            choice = (content.get("title") or "").strip()
            if not choice:
                logger.debug("[photon] ignoring poll vote with empty title")
                return
            source = self.build_source(
                chat_id=space_id,
                chat_name=space_id,
                chat_type=chat_type,
                user_id=sender_id,
                user_name=sender_id or None,
            )
            await self.handle_message(
                MessageEvent(
                    text=choice,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=event.get("messageId"),
                    raw_message=event,
                    timestamp=timestamp,
                )
            )
            return
        if ctype == "text":
            text = content.get("text") or ""
            mtype = MessageType.TEXT
        elif ctype in {"attachment", "voice"}:
            text, mtype, media_urls, media_types = _normalize_binary_payload(content)
        elif ctype == "richlink":
            text = _format_richlink_content(content)
            mtype = MessageType.TEXT
        elif ctype == "group":
            text_parts: List[str] = []
            mtype = MessageType.TEXT
            for item in content.get("items") or []:
                if not isinstance(item, dict):
                    continue
                item_content = item.get("content") or {}
                if not isinstance(item_content, dict):
                    continue
                item_type = item_content.get("type")
                if item_type == "text":
                    item_text = item_content.get("text") or ""
                    if item_text:
                        text_parts.append(item_text)
                    continue
                if item_type == "richlink":
                    text_parts.append(_format_richlink_content(item_content))
                    continue
                if item_type in {"attachment", "voice"}:
                    marker, item_mtype, item_urls, item_types = _normalize_binary_payload(
                        item_content
                    )
                    if mtype == MessageType.TEXT:
                        mtype = item_mtype
                    media_urls.extend(item_urls)
                    media_types.extend(item_types)
                    if not item_urls:
                        text_parts.append(marker)
                    continue
                if item_type:
                    text_parts.append(f"[Photon content type not handled: {item_type}]")
            if media_urls and mtype == MessageType.TEXT:
                mtype = MessageType.DOCUMENT
            text = "\n".join(part for part in text_parts if part).strip()
            if not text:
                text = "(attachment)" if media_urls else "[Photon empty group received]"
        else:
            text = f"[Photon content type not handled: {ctype}]"
            mtype = MessageType.TEXT

        # Group-mention gating (parity with BlueBubbles). In group chats with
        # require_mention enabled, drop messages that don't hit a wake word;
        # strip the leading wake word from the ones that do. DMs are never
        # gated.
        if chat_type == "group" and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                logger.debug(
                    "[photon] ignoring group message "
                    "(require_mention=true, no mention pattern matched)"
                )
                return
            text = self._clean_mention_text(text)

        self._record_recent_richlink(space_id, _richlink_url_from_content(content) or text)

        source = self.build_source(
            chat_id=space_id,
            chat_name=space_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_id or None,
        )
        message_event = MessageEvent(
            text=text,
            message_type=mtype,
            source=source,
            message_id=event.get("messageId"),
            raw_message=event,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(message_event)

    # -- Sidecar lifecycle -------------------------------------------------

    @staticmethod
    def _find_listener_pids(port: int) -> List[int]:
        """PIDs listening on a local TCP port (empty if none/undeterminable)."""
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [int(tok) for tok in out.stdout.split() if tok.strip().isdigit()]

    @staticmethod
    def _pid_is_sidecar(pid: int) -> bool:
        """True if ``pid``'s command line is a Photon sidecar process."""
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        # Checkout-agnostic: any Hermes checkout's sidecar entry point.
        return "photon/sidecar/index.mjs" in out.stdout

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)  # windows-footgun: ok — only called from _reap_stale_sidecar which win32-guards early
            return True
        except OSError:
            return False

    async def _reap_stale_sidecar(self) -> None:
        """Kill an orphaned sidecar squatting our port before spawning ours.

        A hard gateway exit (crash, SIGKILL, supervisor restart) used to leave
        the detached sidecar running with a token the new gateway doesn't
        know, so it can't be told to ``/shutdown`` — and every replacement
        spawn died on EADDRINUSE, failing each reconnect attempt. The
        stdin-EOF watch prevents new orphans; this reclaims the port from
        orphans that predate it (or survived it). Listeners are verified by
        command line before being signalled.
        """
        if sys.platform == "win32":  # lsof/ps; orphaning is a POSIX-only path
            return
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                await client.post(
                    f"http://{self._sidecar_bind}:{self._sidecar_port}/healthz",
                    headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                )
        except httpx.RequestError:
            return  # nothing listening — the normal case
        # Off the event loop: _find_listener_pids shells out to `lsof`
        # (timeout=5s) and _pid_is_sidecar runs a `ps` per candidate pid
        # (timeout=5s each), so this inspection can hold the loop for
        # 5 + 5·N seconds. _reap_stale_sidecar is awaited from
        # _start_sidecar, which runs on every reconnect — exactly when a
        # crashed gateway left an orphan behind — so the stall lands on a
        # live gateway that is still serving every other platform. One hop
        # covers the whole inspection instead of N+1 round trips.
        def _inspect():
            found = self._find_listener_pids(self._sidecar_port)
            mine = [pid for pid in found if self._pid_is_sidecar(pid)]
            return mine, [pid for pid in found if pid not in mine]

        stale, foreign = await asyncio.to_thread(_inspect)
        if not stale:
            raise RuntimeError(
                f"port {self._sidecar_port} is in use by another process "
                f"(pids: {foreign or 'unknown'}, not a Photon sidecar) — "
                f"free it or set PHOTON_SIDECAR_PORT to a different port"
            )
        for pid in stale:
            logger.warning(
                "[photon] reaping orphaned sidecar (pid %d) on port %d",
                pid, self._sidecar_port,
            )
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.time() + 3.0
        while time.time() < deadline and any(self._pid_alive(p) for p in stale):
            await asyncio.sleep(0.1)
        for pid in stale:
            if self._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — unreachable on win32 (early return above)
                except OSError:
                    pass
        # Give the OS a beat to release the listening socket.
        await asyncio.sleep(0.2)
        if foreign:
            raise RuntimeError(
                f"port {self._sidecar_port} is also held by non-sidecar "
                f"processes (pids: {foreign}) — free it or set "
                f"PHOTON_SIDECAR_PORT to a different port"
            )

    async def _start_sidecar(self) -> None:
        if not sidecar_deps_installed():
            # Cold install (NS-606): on hosted/managed images the install
            # tree is immutable and the user has no CLI to run
            # `hermes photon setup`, so the connect path must be able to
            # bootstrap the deps itself. _sidecar_dir() has already been
            # resolved to a writable location (or mirrored to the data
            # volume) by sidecar_paths.resolve_sidecar_dir; `npm ci` off
            # the committed lockfile is deterministic and bounded by
            # _NPM_REINSTALL_TIMEOUT. Uses sidecar_deps_installed() — not a
            # bare node_modules/ existence check — so a partial/aborted npm
            # install (empty node_modules/) also triggers the reinstall.
            logger.info(
                "[photon] sidecar deps not installed; installing into %s",
                _sidecar_dir(),
            )
            await asyncio.to_thread(_reinstall_sidecar_deps)
            if not sidecar_deps_installed():
                # Deterministic on managed/immutable images: npm ci failed and
                # will fail identically on every retry (OOF-153) — surface as
                # non-retryable so it doesn't spin silently in the reconnect
                # queue for weeks.
                raise PhotonSidecarStartupError(
                    f"Photon sidecar deps could not be installed into "
                    f"{_sidecar_dir()} (see log for the npm error). "
                    f"Run: cd {_sidecar_dir()} && npm ci   (or `hermes photon setup`)",
                    code="SIDECAR_DEPS_MISSING",
                    retryable=False,
                )
        # A `hermes update` that bumps the spectrum-ts pin rewrites
        # package-lock.json but never reinstalls node_modules, so the sidecar
        # spawns against stale deps and dies on every reconnect (the v8 patch
        # script can't find @spectrum-ts/imessage/dist that only v8 ships).
        # Self-heal by reinstalling when the lockfile is newer than npm's
        # install marker. Runs off the event loop so a cold install can't
        # freeze every other platform's traffic.
        if _sidecar_deps_stale():
            logger.warning(
                "[photon] sidecar deps are stale (lockfile newer than install); "
                "reinstalling before start"
            )
            await asyncio.to_thread(_reinstall_sidecar_deps)
        await self._reap_stale_sidecar()

        env = os.environ.copy()
        env["PHOTON_PROJECT_ID"] = self._project_id
        env["PHOTON_PROJECT_SECRET"] = self._project_secret
        env["PHOTON_SIDECAR_PORT"] = str(self._sidecar_port)
        env["PHOTON_SIDECAR_BIND"] = self._sidecar_bind
        env["PHOTON_SIDECAR_TOKEN"] = self._sidecar_token
        # The sidecar exits when its stdin (the pipe below) hits EOF, so a
        # gateway death of ANY kind — including SIGKILL, where disconnect()
        # never runs — can't leave it orphaned on the port.
        env["PHOTON_SIDECAR_WATCH_STDIN"] = "1"

        # Windows: hide the child console (0 elsewhere). Same helper the
        # discord/whatsapp adapters use for their sidecar spawns.
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            # Off the event loop, for the same reason the dep reinstall above
            # hops to a thread: this spawns node and *waits* for it (up to 10s).
            # Run inline it holds the shared gateway loop for that whole window,
            # so every other platform's traffic stalls — and _start_sidecar runs
            # on every reconnect (connect(is_reconnect=True)), not just startup,
            # so the stall recurs on a live gateway.
            patch = await asyncio.to_thread(
                subprocess.run,  # noqa: S603
                [
                    self._node_bin,
                    str(_sidecar_dir() / "patch-spectrum-mixed-attachments.mjs"),
                    str(_sidecar_dir()),
                ],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=10,
                check=False,
                # Windows: suppress the brief console flash this short-lived
                # node patch run would otherwise pop on every sidecar start.
                creationflags=windows_hide_flags(),
            )
            if patch.returncode != 0:
                raise RuntimeError((patch.stderr or patch.stdout or "").strip())
            if patch.stderr.strip():
                logger.debug("[photon] %s", patch.stderr.strip())
        except Exception as exc:
            logger.warning(
                "[photon] failed to apply Spectrum mixed attachment patch: %s",
                exc,
            )

        try:
            self._sidecar_proc = subprocess.Popen(  # noqa: S603
                [self._node_bin, str(_sidecar_dir() / "index.mjs")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=(sys.platform != "win32"),
                # Windows: run the persistent sidecar headless so it does not open
                # (or leave) a visible console window. CREATE_NO_WINDOW only (no
                # DETACHED_PROCESS) so the stdin/stdout pipes above stay usable.
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            # Deterministic: node isn't on PATH / PHOTON_NODE_BIN points at
            # nothing. Retrying can never fix a missing binary (OOF-153).
            raise PhotonSidecarStartupError(
                f"node binary not found ({self._node_bin!r}) — install Node.js "
                f"or set PHOTON_NODE_BIN: {exc}",
                code="SIDECAR_NODE_MISSING",
                retryable=False,
            ) from exc
        # Pump sidecar stderr/stdout into our logger so users see crashes.
        loop = asyncio.get_event_loop()
        self._sidecar_supervisor_task = loop.create_task(
            self._supervise_sidecar(self._sidecar_proc)
        )

        # Wait for /healthz to come up — give it up to 15s on cold start.
        deadline = time.time() + 15.0
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while time.time() < deadline:
                if self._sidecar_proc.poll() is not None:
                    _delete_runtime_record()
                    raise RuntimeError(
                        f"Photon sidecar exited with code "
                        f"{self._sidecar_proc.returncode} before becoming ready"
                    )
                try:
                    resp = await client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/healthz",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                    )
                    if resp.status_code == 200:
                        # Persist port/token/pid so out-of-process senders
                        # (cron, `hermes send`) can reach this sidecar
                        # (see _standalone_send / issue #69960).
                        _write_runtime_record(
                            self._sidecar_port,
                            self._sidecar_token,
                            self._sidecar_proc.pid,
                        )
                        return
                except httpx.RequestError as e:
                    last_err = e
                await asyncio.sleep(0.2)
        _delete_runtime_record()
        raise RuntimeError(
            f"Photon sidecar did not become ready within 15s: {last_err}"
        )

    async def _supervise_sidecar(self, proc: subprocess.Popen) -> None:
        """Pump the sidecar's stdout/stderr into our logger."""
        if proc.stdout is None:  # subprocess was launched without stdout=PIPE
            return
        stdout = proc.stdout
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, stdout.readline)
                if not line:
                    break
                logger.info("[photon-sidecar] %s", line.decode("utf-8", "replace").rstrip())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[photon-sidecar] supervisor exited: %s", e)
        if self._inbound_running:
            exit_code = proc.poll()
            logger.error(
                "[photon] sidecar exited unexpectedly (code %s) — triggering reconnect",
                exit_code,
            )
            self._set_fatal_error(
                "SIDECAR_CRASHED",
                f"Photon sidecar exited unexpectedly (code {exit_code})",
                retryable=True,
            )
            self._dispatch_fatal_notification()

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        if proc is None:
            # Nothing to stop, but never leave a record behind on disconnect
            # (e.g. autostart was disabled or startup failed earlier).
            _delete_runtime_record()
            return
        try:
            # Closing our end of the stdin pipe is itself a shutdown signal
            # (the sidecar watches for EOF), and covers the case where the
            # HTTP call below can't get through.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            # Polite shutdown first.
            if self._http_client is not None:
                try:
                    await self._http_client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/shutdown",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                        timeout=2.0,
                    )
                except Exception:
                    pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # windows-footgun: ok
                    except (ProcessLookupError, PermissionError):
                        proc.terminate()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            self._sidecar_proc = None
            # The sidecar is gone (or going) — remove the runtime record so
            # standalone senders don't chase a dead port/token.
            _delete_runtime_record()
            if self._sidecar_supervisor_task is not None:
                # _stop_sidecar() is called both from external cleanup
                # (Gateway shutdown, explicit disconnect) AND, indirectly,
                # from WITHIN the supervisor task's own crash-handling
                # chain: _supervise_sidecar() detects the sidecar exit,
                # calls _set_fatal_error() + self._notify_fatal_error(),
                # which the Gateway's fatal-error handler answers by
                # calling adapter.disconnect() -> this same _stop_sidecar().
                # In that second case, self._sidecar_supervisor_task IS the
                # currently-running task. Cancelling it raises
                # CancelledError into its own call stack (at the next
                # await point in _notify_fatal_error() or here), which
                # aborts the fatal-error handler before the Gateway ever
                # reaches the "queue for background reconnection" step --
                # Photon then stays permanently dead until a manual
                # restart, since asyncio.CancelledError inherits from
                # BaseException (not Exception) and isn't caught by the
                # handler's `except Exception` guards (issue #73159).
                # A task cannot legally cancel itself anyway (the
                # cancellation would only take effect at its own next
                # await, which is exactly the corruption described above),
                # so skip it here and let the task finish exiting on its
                # own instead.
                current_task = asyncio.current_task()
                if self._sidecar_supervisor_task is not current_task:
                    self._sidecar_supervisor_task.cancel()
                self._sidecar_supervisor_task = None

    # -- Presence watchdog -------------------------------------------------

    def _note_upstream_activity(self) -> None:
        """Record proof the upstream gRPC channel is live, and clear failures.

        Called on every inbound line and after every successful probe.
        """
        self._last_upstream_activity = time.monotonic()
        self._probe_failures = 0

    async def _probe_once(self) -> str:
        """Drive one liveness probe via the sidecar's ``/probe``.

        Returns a strict tri-state verdict:

        - ``"alive"``        — the sidecar completed a real upstream gRPC
          round-trip (HTTP 200). Only this counts as proof of liveness.
        - ``"hung"``         — the probe HTTP call itself timed out: the
          sidecar's event loop is unresponsive. Counts toward respawn.
        - ``"inconclusive"`` — anything else (503 from the sidecar's strict
          probe, connection refused, transport error). NEVER counts as alive
          and NEVER counts toward a respawn: a rejected upstream probe may
          just mean the network is down, and a dead sidecar process is the
          supervisor's job, not ours.
        """
        client = self._http_client
        if client is None:
            return "inconclusive"
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}/probe"
        try:
            resp = await client.post(
                url,
                headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                timeout=self._probe_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_timeout_error(e):
                logger.debug("[photon] probe HTTP call hung: %s", e)
                return "hung"
            logger.debug("[photon] probe transport error (inconclusive): %s", e)
            return "inconclusive"
        return "alive" if resp.status_code == 200 else "inconclusive"

    async def _respawn_sidecar(self, reason: str) -> None:
        """Tear down and restart the sidecar to recover a dead gRPC stream.

        A fresh ``Spectrum()`` re-subscribes the inbound stream and
        re-registers presence with Photon cloud. The inbound loop's existing
        reconnect logic re-opens the loopback NDJSON stream automatically once
        the new sidecar's ``/inbound`` is up. Guarded by a lock so overlapping
        triggers (watchdog + a manual call) can't double-spawn.
        """
        lock = self._respawn_lock
        if lock is None:
            lock = self._respawn_lock = asyncio.Lock()
        if lock.locked():
            logger.info("[photon] respawn already in progress; skipping")
            return
        async with lock:
            logger.warning(
                "[photon] presence watchdog: %s — respawning sidecar", reason
            )
            try:
                await self._stop_sidecar()
            except Exception:
                logger.exception("[photon] error stopping sidecar during respawn")
            try:
                await self._start_sidecar()
            except Exception:
                logger.exception(
                    "[photon] failed to respawn sidecar; watchdog will retry"
                )
                return
            # Fresh sidecar -> fresh stream. Reset the activity clock so we give
            # it a full interval before probing again, and clear failures.
            self._note_upstream_activity()
            logger.info("[photon] presence watchdog: sidecar respawned, gRPC stream renewed")

    async def _presence_watchdog(self) -> None:
        """Periodically confirm the sidecar (and its stream) is responsive.

        The sidecar owns primary zombie-stream detection (silence tracking +
        upstream probe -> degraded /healthz -> exit 75). This loop is the
        adapter's conservative second layer: it probes on a long interval,
        skips the probe when natural inbound traffic already proved liveness,
        and only counts a *hung* probe (the sidecar HTTP call itself timing
        out) toward a respawn. Inconclusive probes — the sidecar answered but
        couldn't prove upstream liveness — reset nothing and trigger nothing:
        the network may just be down, and restarting can't fix that. After
        ``_probe_max_failures`` consecutive hung probes we respawn the sidecar
        to force a fresh stream.
        """
        # Stagger the first probe so a fleet of restarts doesn't synchronize,
        # and so a freshly-started sidecar isn't probed before it's warm.
        await asyncio.sleep(self._probe_interval)
        while self._watchdog_running:
            try:
                # Skip our own probe if inbound traffic already proved the
                # channel live within the last interval (cheaper, and avoids
                # piling synthetic reads on a busy line).
                idle = time.monotonic() - self._last_upstream_activity
                if idle < self._probe_interval:
                    await asyncio.sleep(self._probe_interval - idle)
                    continue

                verdict = await self._probe_once()
                if verdict == "alive":
                    self._note_upstream_activity()
                elif verdict == "hung":
                    # Only a hung sidecar HTTP loop counts toward respawn —
                    # strict success-only-on-probe-OK semantics mean an
                    # inconclusive probe is never evidence in either direction.
                    self._probe_failures += 1
                    logger.warning(
                        "[photon] presence probe hung (%d/%d)",
                        self._probe_failures, self._probe_max_failures,
                    )
                    if self._probe_failures >= self._probe_max_failures:
                        await self._respawn_sidecar(
                            f"{self._probe_failures} consecutive hung probes"
                        )
                else:
                    logger.debug(
                        "[photon] presence probe inconclusive; taking no action"
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[photon] presence watchdog iteration failed")
            await asyncio.sleep(self._probe_interval)

    async def _stop_watchdog(self) -> None:
        self._watchdog_running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._watchdog_task = None

    # -- Outbound ----------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._sidecar_send(chat_id, self.format_message(content))

    # -- Clarify (native iMessage poll) ------------------------------------
    #
    # iMessage has a native poll bubble; spectrum-ts exposes it via the
    # `poll()` content builder. A multiple-choice clarify renders as that poll
    # and the user taps a choice instead of typing a number — the vote streams
    # back inbound as a `poll_option` event, which `_dispatch_inbound`
    # translates into a plain-text message carrying the chosen option. We flip
    # the clarify into text-capture mode (exactly like the base text fallback)
    # so the gateway's pending-clarify intercept resolves it with that choice.
    # Open-ended clarifies (no choices) keep the plain-text path.

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not choices:
            # No choices → open-ended. Base behaviour (plain text; the next
            # message resolves it) is exactly right.
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, metadata
            )
        # The poll vote comes back as a normal text message, so enable
        # text-capture and let the gateway intercept resolve the clarify.
        from tools.clarify_gateway import mark_awaiting_text

        mark_awaiting_text(clarify_id)
        result = await self._sidecar_send_poll(chat_id, question, list(choices))
        if not result.success:
            # Native poll failed (old sidecar without /send-poll, or a send
            # error) — fall back to the numbered-text clarify so the user can
            # still answer. The base impl also calls mark_awaiting_text (a
            # second call is harmless).
            logger.warning(
                "[photon] poll clarify failed (%s); falling back to text list",
                result.error,
            )
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, metadata
            )
        return result

    # -- Outbound media (parity with the BlueBubbles iMessage channel) -----
    #
    # Photon ships outbound attachments via spectrum-ts' `attachment()` /
    # `voice()` content builders. The sidecar's `/send-attachment` endpoint
    # wraps `space.send(attachment(path, {...}))`. These overrides mirror
    # BlueBubbles: URL-based helpers cache to a local path first, file-based
    # helpers pass the path straight through.

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url

            local_path = await cache_image_from_url(image_url)
        except Exception:
            # Couldn't fetch the URL — fall back to sending it as text.
            return await super().send_image(chat_id, image_url, caption, reply_to)
        return await self._sidecar_send_attachment(
            chat_id, local_path, caption=caption,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._sidecar_send_attachment(
            chat_id, image_path, caption=caption,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._sidecar_send_attachment(
            chat_id, audio_path, caption=caption, kind="voice",
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._sidecar_send_attachment(
            chat_id, video_path, caption=caption,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._sidecar_send_attachment(
            chat_id, file_path, name=file_name, caption=caption,
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # iMessage renders GIFs inline as ordinary image attachments.
        return await self.send_image(
            chat_id, animation_url, caption, reply_to, metadata,
        )

    async def send_poll(
        self,
        chat_id: str,
        title: str,
        options: list[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native iMessage poll through Photon Spectrum.

        Thin public wrapper over :meth:`_sidecar_send_poll`, the single
        implementation of the sidecar's ``/send-poll`` primitive (also used
        by the poll-backed clarify path).
        """
        return await self._sidecar_send_poll(chat_id, title, list(options or []))

    async def send_effect(
        self,
        chat_id: str,
        text: str,
        effect: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text with a native iMessage bubble or screen effect."""
        if not text.strip() or not effect.strip():
            return SendResult(success=False, error="text and effect are required")
        try:
            data = await self._sidecar_call(
                "/send-effect",
                {"spaceId": chat_id, "text": text.strip(), "effect": effect.strip()},
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        now = time.time()
        if now - self._typing_last_sent.get(chat_id, 0.0) < _TYPING_COOLDOWN_SECONDS:
            return
        self._typing_last_sent[chat_id] = now
        try:
            await self._sidecar_call(
                "/typing", {"spaceId": chat_id, "state": "start"}
            )
        except Exception as e:
            logger.debug("[photon] send_typing failed: %s", e)

    async def stop_typing(self, chat_id: str) -> None:
        self._typing_last_sent.pop(chat_id, None)
        try:
            await self._sidecar_call(
                "/typing", {"spaceId": chat_id, "state": "stop"}
            )
        except Exception as e:
            logger.debug("[photon] stop_typing failed: %s", e)

    # -- Reactions (tapbacks) -----------------------------------------------
    #
    # Same lifecycle-hook pattern as Telegram/Discord: 👀 while processing,
    # swapped for 👍/👎 on completion. Opt-in via PHOTON_REACTIONS — iMessage
    # is a personal-texting channel, and a tapback on every text is noisy.

    _SENT_IDS_MAX = 1000
    _LAST_INBOUND_CHATS_MAX = 200

    def _record_sent_message(self, message_id: Optional[str]) -> None:
        if not message_id:
            return
        sent = self._sent_message_ids
        if message_id in sent:
            del sent[message_id]  # refresh insertion order
        sent[message_id] = time.time()
        if len(sent) > self._SENT_IDS_MAX:
            for old in list(sent.keys())[: len(sent) - self._SENT_IDS_MAX]:
                del sent[old]

    # A DM space is addressable two ways — the chat GUID (`any;-;+1555...`)
    # that inbound events carry, and the bare E.164 phone that home-channel
    # config typically uses. The sidecar's resolveSpace treats them as the
    # same space; normalize to the bare phone so the last-inbound tracker
    # does too (mirrors phoneTargetFromSpaceId in sidecar/index.mjs).
    _DM_CHAT_GUID_RE = re.compile(r"^any;-;(\+\d{6,})$")

    @classmethod
    def _normalize_chat_key(cls, chat_id: str) -> str:
        match = cls._DM_CHAT_GUID_RE.match(chat_id)
        return match.group(1) if match else chat_id

    def _record_last_inbound(
        self, chat_id: Optional[str], message_id: Optional[str]
    ) -> None:
        if not chat_id or not message_id:
            return
        key = self._normalize_chat_key(chat_id)
        last = self._last_inbound_by_chat
        if key in last:
            del last[key]  # refresh insertion order
        last[key] = message_id
        if len(last) > self._LAST_INBOUND_CHATS_MAX:
            for old in list(last.keys())[
                : len(last) - self._LAST_INBOUND_CHATS_MAX
            ]:
                del last[old]

    def _record_recent_richlink(self, chat_id: str, text: str) -> None:
        if not chat_id or not _url_only_candidate(text):
            return
        key = self._normalize_chat_key(chat_id)
        recent = self._recent_richlinks_by_chat
        if key in recent:
            del recent[key]  # refresh insertion order
        recent[key] = time.time()
        if len(recent) > self._LAST_INBOUND_CHATS_MAX:
            for old in list(recent.keys())[
                : len(recent) - self._LAST_INBOUND_CHATS_MAX
            ]:
                del recent[old]

    def _is_recent_richlink_preview(self, chat_id: str, content: Dict[str, Any]) -> bool:
        if not chat_id or not _is_richlink_preview_content(content):
            return False
        key = self._normalize_chat_key(chat_id)
        last = self._recent_richlinks_by_chat.get(key)
        if last is None:
            return False
        if time.time() - last > _RICHLINK_PREVIEW_SUPPRESS_SECONDS:
            self._recent_richlinks_by_chat.pop(key, None)
            return False
        return True

    def _reactions_enabled(self) -> bool:
        return os.getenv("PHOTON_REACTIONS", "false").strip().lower() in {
            "true", "1", "yes", "on",
        }

    async def _add_reaction(
        self, chat_id: str, message_id: str, emoji: str
    ) -> bool:
        """Tapback ``emoji`` onto a message. Soft-fails (False), never raises."""
        try:
            await self._sidecar_call(
                "/react",
                {"spaceId": chat_id, "messageId": message_id, "emoji": emoji},
            )
            return True
        except Exception as e:
            logger.debug("[photon] add_reaction failed: %s", e)
            return False

    async def _remove_reaction(self, chat_id: str, message_id: str) -> bool:
        """Retract our tapback from a message. Soft-fails (False), never raises.

        The sidecar tracks one reaction handle per target message; after a
        sidecar restart the handle is gone and removal is best-effort (the
        stale tapback self-heals when the next reaction replaces it).
        """
        try:
            await self._sidecar_call(
                "/unreact", {"spaceId": chat_id, "messageId": message_id},
            )
            return True
        except Exception as e:
            logger.debug("[photon] remove_reaction failed: %s", e)
            return False

    # -- Agent-facing reactions (send_message action="react") ---------------
    #
    # Unlike the lifecycle hooks below, these are deliberate agent intents,
    # so they are NOT gated by PHOTON_REACTIONS (that env var exists to mute
    # the automatic per-message tapback noise, not explicit requests).

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tapback ``emoji`` onto a message in ``chat_id``.

        Without ``message_id``, targets the chat's most recent inbound
        message (typically the one the agent is responding to). iMessage
        maps ❤️👍👎😂‼️❓ to native tapbacks; anything else uses Apple's
        custom-emoji reaction.
        """
        target = message_id or self._last_inbound_by_chat.get(
            self._normalize_chat_key(chat_id)
        )
        if not target:
            return {
                "success": False,
                "error": "no message to react to — pass message_id (no "
                "inbound message seen in this chat since the gateway started)",
            }
        ok = await self._add_reaction(chat_id, target, emoji)
        if not ok:
            return {
                "success": False,
                "error": "reaction failed (see gateway debug log)",
            }
        return {"success": True, "message_id": target}

    async def remove_reaction(
        self, chat_id: str, message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retract our tapback from a message (best-effort)."""
        target = message_id or self._last_inbound_by_chat.get(
            self._normalize_chat_key(chat_id)
        )
        if not target:
            return {
                "success": False,
                "error": "no message to unreact — pass message_id",
            }
        ok = await self._remove_reaction(chat_id, target)
        if not ok:
            return {
                "success": False,
                "error": "unreact failed (see gateway debug log)",
            }
        return {"success": True, "message_id": target}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Tapback 👀 on the triggering message while the agent works."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._add_reaction(chat_id, message_id, "\U0001f440")

    # Shared reaction-ack flow (base.on_processing_complete): swap the 👀
    # progress tapback for a 👍/👎 result. Remove-then-add rather than a bare
    # replace: deterministic whether the platform replaces a sender's previous
    # tapback or stacks them, and it keeps the sidecar's reaction-handle slot
    # coherent. CANCELLED: leave the message unreacted.
    _OK_EMOJI = "\U0001f44d"
    _FAIL_EMOJI = "\U0001f44e"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return whatever we know about a Spectrum space id.

        Photon's ``space.id`` is opaque; the inbound event also carries the
        DM/group type, but here we only have the id, so infer conservatively.
        """
        return {"name": chat_id, "type": "dm", "id": chat_id}

    def format_message(self, content: str) -> str:
        # Markdown is passed through verbatim — the sidecar sends it with the
        # markdown() builder and iMessage renders it. The strip path remains
        # as the PHOTON_MARKDOWN=false kill-switch.
        if _markdown_enabled():
            return content
        return strip_markdown(content)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        if BasePlatformAdapter._is_retryable_error(error):
            return True
        if not error:
            return False
        lowered = error.lower()
        if "retryable=false" in lowered or "auth_or_config" in lowered:
            return False
        return any(pat in lowered for pat in _PHOTON_RETRYABLE_PATTERNS)

    @staticmethod
    def _is_permanent_sidecar_failure(result: SendResult) -> bool:
        """True when the sidecar classified the failure as permanent.

        ``auth_or_config`` and ``target_not_allowed`` cannot be fixed by
        retrying or by the plain-text fallback resend — attempting either
        just double-sends a doomed request (issue #50971).
        """
        raw = result.raw_response
        return (
            isinstance(raw, dict)
            and raw.get("retryable") is False
            and raw.get("error_class") in ("auth_or_config", "target_not_allowed")
        )

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
        max_retries: int = 1,
        base_delay: float = 2.0,
    ) -> SendResult:
        """Retry sends without the generic Markdown banner.

        Photon replies are markdown (rendered by iMessage) or stripped plain
        text under ``PHOTON_MARKDOWN=false`` — either way the gateway's
        generic banner never applies.
        """
        text = self.format_message(content)
        result = await self.send(
            chat_id=chat_id,
            content=text,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result.success:
            return result

        if self._is_permanent_sidecar_failure(result):
            # Permanent failure classes: retrying cannot succeed and the
            # unconditional plain-text fallback below would double-send the
            # same doomed request. Return the structured failure as-is
            # (with the user-facing explanation already attached for
            # target_not_allowed in _sidecar_send).
            return result

        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)
        if not is_network and self._is_timeout_error(error_str):
            return result

        if is_network:
            for attempt in range(1, max_retries + 1):
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[photon] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt, max_retries, delay, error_str,
                )
                await asyncio.sleep(delay)
                result = await self.send(
                    chat_id=chat_id,
                    content=text,
                    reply_to=reply_to,
                    metadata=metadata,
                )
                if result.success:
                    return result
                error_str = result.error or ""
                if self._is_permanent_sidecar_failure(result):
                    # A retry surfaced a permanent class — don't fall through
                    # to the plain-text resend either.
                    return result
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break
            else:
                logger.error(
                    "[photon] Failed to deliver response after %d retries: %s",
                    max_retries, error_str,
                )
                # Fall through to the plain-text fallback below. For URL-only
                # responses this bypasses ``richlink()`` so a rich-link-specific
                # outage or sidecar skew does not strand an otherwise sendable URL.

        logger.warning(
            "[photon] Send failed: %s - retrying plain-text message",
            error_str,
        )
        fallback_result = await self._sidecar_send(
            chat_id,
            text[: self.MAX_MESSAGE_LENGTH],
            richlink=False,
            markdown=False,
        )
        if not fallback_result.success:
            logger.error("[photon] Plain-text retry also failed: %s", fallback_result.error)
        return fallback_result

    async def _sidecar_send_richlink(self, space_id: str, url: str) -> SendResult:
        try:
            data = await self._sidecar_call(
                "/send-richlink", {"spaceId": space_id, "url": url}
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send(
        self,
        space_id: str,
        text: str,
        *,
        richlink: bool = True,
        markdown: bool = True,
    ) -> SendResult:
        rich_url = _richlink_candidate(text) if richlink else None
        if rich_url:
            rich_result = await self._sidecar_send_richlink(space_id, rich_url)
            if rich_result.success:
                return rich_result
            logger.warning(
                "[photon] rich-link send failed, falling back to plain text: %s",
                rich_result.error,
            )
            markdown = False
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[photon] truncating outbound from %d to %d chars",
                len(text), self.MAX_MESSAGE_LENGTH,
            )
            text = text[: self.MAX_MESSAGE_LENGTH]
        body: Dict[str, Any] = {"spaceId": space_id, "text": text}
        # Omit the key when disabled so an older sidecar (pre-`format`)
        # keeps accepting the body during a half-upgraded restart.
        if markdown and _markdown_enabled():
            body["format"] = "markdown"
        try:
            data = await self._sidecar_call("/send", body)
        except PhotonSidecarError as e:
            return SendResult(
                success=False,
                error=str(e),
                raw_response={
                    "error_class": e.error_class,
                    "retryable": e.retryable,
                },
                retryable=e.retryable,
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send_poll(
        self, space_id: str, title: str, options: list,
    ) -> SendResult:
        """POST a poll to the sidecar's ``/send-poll`` endpoint.

        Renders a native iMessage poll. ``options`` are choice strings; the
        sidecar's ``poll()`` builder degrades to a numbered text list on
        platforms without native polls.
        """
        opts = [str(o).strip() for o in (options or []) if str(o).strip()]
        if not title or not title.strip():
            return SendResult(success=False, error="poll title is required")
        if len(opts) < 2:
            return SendResult(success=False, error="poll needs at least two options")
        body: Dict[str, Any] = {
            "spaceId": space_id,
            "title": title.strip()[: self.MAX_MESSAGE_LENGTH],
            "options": opts,
        }
        try:
            data = await self._sidecar_call("/send-poll", body)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send_attachment(
        self,
        space_id: str,
        path: str,
        *,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        caption: Optional[str] = None,
        kind: str = "attachment",
    ) -> SendResult:
        """POST a local file to the sidecar's ``/send-attachment`` endpoint.

        ``kind`` is ``"voice"`` for audio sent as a voice note (downgrades
        to a plain audio attachment on platforms without voice notes),
        otherwise ``"attachment"``. spectrum-ts infers ``name`` and
        ``mimeType`` from the file extension; we only pass overrides when
        Hermes supplied them.
        """
        # Defense-in-depth: re-validate the path before handing it to the
        # Node sidecar. The gateway already filters MEDIA paths, but
        # send_*_file / cron callers may pass arbitrary strings.
        safe_path = self.validate_media_delivery_path(str(path))
        if not safe_path:
            return SendResult(
                success=False, error=f"unsafe or missing attachment path: {path}"
            )
        if not mime_type:
            import mimetypes

            guessed, _ = mimetypes.guess_type(safe_path)
            mime_type = guessed or None
        body: Dict[str, Any] = {
            "spaceId": space_id,
            "path": safe_path,
            "kind": "voice" if kind == "voice" else "attachment",
        }
        if name:
            body["name"] = name
        if mime_type:
            body["mimeType"] = mime_type
        if caption:
            body["caption"] = caption
        try:
            data = await self._sidecar_call("/send-attachment", body)
        except PhotonSidecarError as e:
            return SendResult(
                success=False,
                error=str(e),
                raw_response={
                    "error_class": e.error_class,
                    "retryable": e.retryable,
                },
                retryable=e.retryable,
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        # Guard: adapter not yet connected (no sidecar address known).
        if self._http_client is None:
            raise RuntimeError("Photon adapter not connected")
        # Use a fresh client per call so this method is safe when invoked from
        # a worker thread that owns a different event loop than the one the
        # persistent _http_client was created on (e.g. via _run_async in
        # send_message_tool).  The inbound streaming loop continues to use
        # _http_client directly — it always runs on the gateway's loop.
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}{path}"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise _sidecar_error_from_response(path, resp.status_code, resp.text)
        data = resp.json() or {}
        if not data.get("ok"):
            raise _sidecar_error_from_response(path, resp.status_code, resp.text, data)
        return data


# ---------------------------------------------------------------------------
# Helpers

def _attachment_message_type(mime: str) -> MessageType:
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    if mime.startswith("application/"):
        return MessageType.DOCUMENT
    return MessageType.DOCUMENT


# MIME → file-extension maps for caching inbound attachment bytes. These mirror
# the BlueBubbles iMessage channel so both adapters name cached media the same.
_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".jpg",
    "image/heif": ".jpg",
    "image/tiff": ".jpg",
}
_AUDIO_EXT_BY_MIME = {
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-caf": ".caf",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
}


def _cache_inbound_attachment(
    content: Dict[str, Any],
    name: str,
    mime: str,
    *,
    force_audio: bool = False,
) -> Optional[str]:
    """Decode a base64-inlined inbound attachment and cache it locally.

    The sidecar inlines the attachment bytes as ``content["data"]`` (base64).
    We decode them and route to the shared media cache by MIME type, returning
    the cached absolute path so the caller can populate ``media_urls`` (which
    the gateway then hands to the model). Returns ``None`` when there are no
    bytes (over the sidecar's inline cap or a failed read) or when caching
    fails, so the caller can fall back to a text marker.
    """
    data_b64 = content.get("data")
    if not data_b64:
        return None
    try:
        raw = base64.b64decode(data_b64)
    except (ValueError, TypeError) as exc:
        logger.warning("[photon] failed to decode inbound attachment bytes: %s", exc)
        return None

    from gateway.platforms.base import (
        cache_audio_from_bytes,
        cache_document_from_bytes,
        cache_image_from_bytes,
    )

    mime = (mime or "").lower()
    # Prefer the real extension from the filename; fall back to the MIME map.
    suffix = Path(name).suffix if name else ""
    try:
        if mime.startswith("image/"):
            ext = suffix or _IMAGE_EXT_BY_MIME.get(mime, ".jpg")
            try:
                return cache_image_from_bytes(raw, ext)
            except ValueError:
                # Bytes don't look like a supported image (e.g. HEIC magic) —
                # still deliver them as a document rather than dropping them.
                return cache_document_from_bytes(raw, name)
        if force_audio or mime.startswith("audio/"):
            ext = suffix or _AUDIO_EXT_BY_MIME.get(
                mime, ".m4a" if force_audio else ".mp3"
            )
            return cache_audio_from_bytes(raw, ext)
        # Video, application/*, and everything else → document cache.
        return cache_document_from_bytes(raw, name)
    except Exception as exc:
        logger.warning("[photon] failed to cache inbound attachment %s: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Standalone (out-of-process) send for cron deliveries when the gateway
# is not co-resident.  Reuses a live sidecar already listening on the
# configured port (cron processes cannot spawn the sidecar themselves).

def _standalone_error(resp: Any) -> Dict[str, Any]:
    """Build the structured error dict for a failed standalone sidecar call.

    Mirrors ``_sidecar_error_from_response``: parse the sidecar body
    independently, carry the error class + retryability, and swap in the
    canonical user-facing message for ``target_not_allowed`` (never the raw
    upstream text).
    """
    try:
        data = resp.json() or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    error_class = str(data.get("error_class") or "sidecar_error")
    retryable = bool(data.get("retryable"))
    if error_class == "target_not_allowed":
        error = _TARGET_NOT_ALLOWED_MESSAGE
        retryable = False
    elif resp.status_code != 200:
        error = f"sidecar returned {resp.status_code}: {resp.text[:200]}"
    else:
        error = str(data.get("error") or "sidecar reported failure")
    return {"error": error, "error_class": error_class, "retryable": retryable}


# Cron jobs, notification fanout, and `hermes send` all deliver through
# _standalone_send; a brief Photon overflow or connection reset would
# otherwise drop a scheduled message outright.
# One retry adds at most 2s of backoff per text or attachment request;
# sequential fanout accumulates that delay for each failed request.
_STANDALONE_SEND_RETRIES = 1
_STANDALONE_RETRY_BASE_DELAY_SECONDS = 2.0


async def _standalone_post_with_retry(
    client: Any,
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """POST to the sidecar, retrying transient Photon failures with backoff.

    Returns ``(data, None)`` on success or ``(None, error_dict)`` once the
    retries are spent or the failure is classified permanent.
    """
    last_error: Optional[Dict[str, Any]] = None
    for attempt in range(_STANDALONE_SEND_RETRIES + 1):
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code == 200:
            try:
                data = resp.json() or {}
            except Exception:
                data = {}
            if isinstance(data, dict) and data.get("ok"):
                return data, None
        last_error = _standalone_error(resp)
        if attempt >= _STANDALONE_SEND_RETRIES:
            break
        retryable = bool(last_error.get("retryable")) or PhotonAdapter._is_retryable_error(
            last_error.get("error")
        )
        if not retryable:
            break
        delay = _STANDALONE_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
        logger.warning(
            "[photon] standalone send failed (attempt %d/%d, retrying in %.1fs): %s",
            attempt + 1, _STANDALONE_SEND_RETRIES + 1, delay,
            last_error.get("error"),
        )
        await asyncio.sleep(delay)
    return None, last_error


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,  # noqa: ARG001 — Spectrum has no threads yet
    media_files: Optional[list] = None,
    force_document: bool = False,  # noqa: ARG001 — iMessage auto-detects file kind
) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not installed"}
    port = _coerce_port(
        (pconfig.extra or {}).get("sidecar_port") or os.getenv("PHOTON_SIDECAR_PORT"),
        _DEFAULT_SIDECAR_PORT,
    )
    token = _get_scoped_secret("PHOTON_SIDECAR_TOKEN")
    if not token:
        # Fall back to the runtime record the gateway persists once its
        # sidecar passes /healthz (issue #69960) — the token only exists in
        # the gateway process env otherwise, so cron/`hermes send` would be
        # structurally unable to authenticate.
        record = _read_runtime_record()
        stale_hint = ""
        if record and record.get("token"):
            if _sidecar_pid_alive(record.get("pid")):
                token = str(record["token"])
                port = _coerce_port(record.get("port"), port)
            else:
                stale_hint = (
                    " A stale sidecar runtime record was found (pid "
                    f"{record.get('pid')} is not running) — the gateway "
                    "appears to be down."
                )
        if not token:
            return {
                "error": (
                    "Photon standalone send requires a running sidecar. "
                    "Start the Hermes gateway (which spawns the sidecar and "
                    "records its address under <hermes-home>/runtime/"
                    f"{_RUNTIME_RECORD_NAME}), or set PHOTON_SIDECAR_TOKEN "
                    "in this process's environment." + stale_hint
                )
            }
    base = f"http://{_DEFAULT_SIDECAR_BIND}:{port}"
    headers = {"X-Hermes-Sidecar-Token": token}
    last_message_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            # 1. Text body first (if any), so it leads the conversation.
            if message:
                rich_url = _richlink_candidate(message)
                if rich_url:
                    resp = await client.post(
                        f"{base}/send-richlink",
                        json={"spaceId": chat_id, "url": rich_url},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json() or {}
                        if data.get("ok"):
                            last_message_id = data.get("messageId")
                        else:
                            rich_url = None
                    else:
                        rich_url = None
                if not rich_url:
                    send_body: Dict[str, Any] = {
                        "spaceId": chat_id,
                        "text": message[:_MAX_MESSAGE_LENGTH],
                    }
                    if _markdown_enabled() and not _richlink_candidate(message):
                        send_body["format"] = "markdown"
                    data, error = await _standalone_post_with_retry(
                        client, f"{base}/send", send_body, headers,
                    )
                    if error is not None:
                        return error
                    last_message_id = data.get("messageId")

            # 2. Each attachment as a separate /send-attachment call.
            #    media_files is List[Tuple[path, is_voice]] (see
            #    BasePlatformAdapter.filter_media_delivery_paths).
            import mimetypes

            for media_path, is_voice in media_files or []:
                safe_path = BasePlatformAdapter.validate_media_delivery_path(str(media_path))
                if not safe_path:
                    logger.warning("[photon] standalone send skipping unsafe path")
                    continue
                guessed, _ = mimetypes.guess_type(safe_path)
                att_body: Dict[str, Any] = {
                    "spaceId": chat_id,
                    "path": safe_path,
                    "kind": "voice" if is_voice else "attachment",
                }
                if guessed:
                    att_body["mimeType"] = guessed
                data, error = await _standalone_post_with_retry(
                    client, f"{base}/send-attachment", att_body, headers,
                )
                if error is not None:
                    return error
                last_message_id = data.get("messageId") or last_message_id

        return {"success": True, "message_id": last_message_id}
    except Exception as e:
        return {"error": f"Photon standalone send failed: {e}"}


# ---------------------------------------------------------------------------
# Plugin entry point

def register(ctx) -> None:
    """Called by the Hermes plugin loader at startup."""
    # Local import to avoid argparse work at module load; reused for both the
    # gateway-setup hook and the `hermes photon` CLI command below.
    from . import cli as _cli

    ctx.register_platform(
        name="photon",
        label="iMessage via Photon",
        adapter_factory=lambda cfg: PhotonAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["PHOTON_PROJECT_ID", "PHOTON_PROJECT_SECRET"],
        install_hint=(
            "Run: hermes photon setup  (logs in via device flow, creates a "
            "Spectrum project, links your phone number, installs the "
            "spectrum-ts sidecar)."
        ),
        # Surfaces Photon in `hermes gateway setup` alongside every other
        # channel — same unified onboarding wizard, no Photon-only detour.
        setup_fn=_cli.gateway_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="PHOTON_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="PHOTON_ALLOWED_USERS",
        allow_all_env="PHOTON_ALLOW_ALL_USERS",
        max_message_length=_MAX_MESSAGE_LENGTH,
        emoji="📱",
        # iMessage carries E.164 phone numbers — treat session descriptions
        # as PII-sensitive so they get redacted before reaching the LLM
        # (matches the BlueBubbles iMessage channel in _PII_SAFE_PLATFORMS).
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via Photon Spectrum (iMessage). "
            "Treat replies like regular text messages — short and friendly. "
            "Markdown is rendered (bold, italics, lists, code), but keep "
            "formatting light and conversational. Recipient identifiers are "
            "E.164 phone numbers; never expose them in responses unless the "
            "user asked. Attachments arrive as metadata only."
        ),
    )

    # Register CLI subcommands — `hermes photon ...`
    ctx.register_cli_command(
        name="photon",
        help="Set up and manage the Photon iMessage integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )
