"""Photon Spectrum (iMessage) platform adapter.

Both directions flow through a supervised Node sidecar (``sidecar/index.mjs``) running
the TypeScript-only ``spectrum-ts`` SDK. Inbound: the SDK's gRPC stream re-emitted as
NDJSON over loopback ``GET /inbound`` (no webhook / public URL). Outbound: loopback
POSTs to the sidecar's control endpoints with a shared bearer token.
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:  # type checkers see httpx as always-imported; runtime keeps it optional
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
from gateway.platforms._shared import coerce_port as _coerce_port
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.helpers import compile_mention_patterns, strip_markdown

from .auth import load_project_credentials
# Sidecar dir resolution is lazy (never at import): it probes the filesystem and may
# mirror files. Tests monkeypatch sidecar_paths._SIDECAR_DIR.
from .sidecar_paths import _NPM_ERROR_LOG_MAX_CHARS, _lock_newer_than_install, _npm_error_log, _sidecar_dir
from .sidecar_paths import dir_writable as _dir_writable
import contextlib

logger = logging.getLogger(__name__)

_DEFAULT_SIDECAR_PORT = 8789
_DEFAULT_SIDECAR_BIND = "127.0.0.1"
_MAX_MESSAGE_LENGTH = 8000  # iMessage caps practical size at ~16 KB; conservative, matches BlueBubbles
# Out-of-process senders (cron, `hermes send`) need the live sidecar's port + spawn-time
# token; persisted once /healthz passes, removed on every stop / failed-start path.
# --------------------------------------------------------------------------- Sidecar runtime record The
# gateway persists this record once the sidecar passes its /healthz readiness check, and removes it on every
# stop / failed-start path so a stale record never outlives a dead sidecar. See #69960.
_RUNTIME_RECORD_NAME = "photon-sidecar.json"
_DEDUP_MAX_SIZE = 4000  # the gRPC stream is at-least-once and a reconnect can replay
_DEDUP_WINDOW_SECONDS = 48 * 3600
_FFFC_WAIT_SECONDS = 15.0  # wait for the real attachment after a U+FFFC placeholder
_NPM_REINSTALL_TIMEOUT = 600  # a wedged self-heal `npm ci` must not stall connect indefinitely
# Photon / Envoy / spectrum-ts substrings meaning transient upstream overload.
_PHOTON_RETRYABLE_PATTERNS = (
    "internal sidecar error", "upstream connect error", "upstream unavailable", "connection dropped",
    "reset reason: overflow", "upstream_overflow", "upstream_unavailable")
# iMessage emits Open Graph preview art as attachments right after a URL message;
# suppress those so Hermes sees the link once.
_RICHLINK_PREVIEW_SUPPRESS_SECONDS = 30.0
_RICHLINK_PREVIEW_ATTACHMENT_SUFFIX = ".pluginpayloadattachment"
_TYPING_COOLDOWN_SECONDS = 5.0  # per chat; reduces gRPC pressure during overflow
# Group-chat wake words — same defaults as BlueBubbles so both iMessage adapters gate alike.
_DEFAULT_MENTION_PATTERNS = [r"(?<![\w@])@?hermes\s+agent\b[,:\-]?", r"(?<![\w@])@?hermes\b[,:\-]?"]
# Shared/free-tier lines can only reply to conversations the target initiated.
_TARGET_NOT_ALLOWED_MESSAGE = (
    "shared/free-tier Photon lines cannot initiate outbound sends to new "
    "targets — upgrade to a dedicated line or use another delivery channel")


# -- Sidecar runtime record ----------------------------------------------------

def _runtime_record_path() -> Path:
    from hermes_constants import get_hermes_home  # honors profile overrides
    return get_hermes_home() / "runtime" / _RUNTIME_RECORD_NAME


def _write_runtime_record(port: int, token: str, pid: int) -> None:
    """Atomically persist ``{port, token, pid}`` with owner-only perms (best-effort)."""
    import tempfile
    try:
        path = _runtime_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".photon-sidecar.", suffix=".tmp")
        try:
            with contextlib.suppress(OSError):  # perms BEFORE the token hits disk (Windows / odd fs)
                os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"port": port, "token": token, "pid": pid}, fh)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except Exception as e:
        logger.warning("[photon] failed to write sidecar runtime record: %s", e)


def _read_runtime_record() -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(_runtime_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _delete_runtime_record() -> None:
    with contextlib.suppress(OSError):
        _runtime_record_path().unlink(missing_ok=True)


def _sidecar_pid_alive(pid: Any) -> bool:
    """Best-effort liveness check for the recorded sidecar pid."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        from gateway.status import _pid_exists  # psutil-backed, Windows-safe
        return bool(_pid_exists(pid_int))
    except Exception:
        pass
    if os.name != "posix":  # os.kill(pid, 0) is destructive on Windows — assume alive; the HTTP send arbitrates
        return True
    try:
        os.kill(pid_int, 0)  # windows-footgun: ok — inside os.name == "posix" guard
    except PermissionError:
        return True
    except OSError:  # incl. ProcessLookupError
        return False
    return True


# -- Errors ---------------------------------------------------------------------

class PhotonSidecarStartupError(RuntimeError):
    """Startup failure from ``_start_sidecar``; only deterministic failures (deps can't
    install, node missing) set ``retryable=False`` so they surface as fatal."""

    def __init__(self, message: str, *, code: str = "SIDECAR_FAILED", retryable: bool = True) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class PhotonSidecarError(RuntimeError):
    """Structured failure returned by the supervised Photon sidecar."""

    def __init__(self, *, path: str, status_code: int, error: str, error_class: str = "sidecar_error",
                 retryable: bool = False) -> None:
        self.path = path
        self.status_code = status_code
        self.error = error
        self.error_class = error_class
        self.retryable = retryable
        super().__init__(
            f"Photon sidecar {path} returned {status_code} "
            f"({error_class}, retryable={retryable}): {error}")


def _sidecar_error_from_response(path: str, status_code: int, text: str,
                                 data: Optional[Dict[str, Any]] = None) -> PhotonSidecarError:
    if data is None:
        with contextlib.suppress(Exception):
            data = json.loads(text)
        data = data if isinstance(data, dict) else {}
    error = str(data.get("error") or text[:200] or "sidecar error")
    error_class = str(data.get("error_class") or "sidecar_error")
    retryable = bool(data.get("retryable"))
    if error_class == "target_not_allowed":
        # Canonical user-facing explanation, never raw upstream error text.
        error = _TARGET_NOT_ALLOWED_MESSAGE
        retryable = False
    return PhotonSidecarError(
        path=path, status_code=status_code, error=error, error_class=error_class, retryable=retryable)


# -- Module-level helpers (also used by check_fn / standalone send) ---------------

def sidecar_deps_installed() -> bool:
    """True when spectrum-ts is present under node_modules/ (not just node_modules/
    itself: npm creates it before aborting on ENOSPC/timeout/EACCES)."""
    return (_sidecar_dir() / "node_modules" / "spectrum-ts").exists()


def _is_timeout_error(exc: BaseException) -> bool:
    """True when *exc* indicates the request timed out (call hung)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if HTTPX_AVAILABLE and isinstance(exc, httpx.TimeoutException):
        return True
    return "timeout" in type(exc).__name__.lower()


def check_requirements() -> bool:
    """Return True when both Python deps and the Node sidecar are available."""
    if not HTTPX_AVAILABLE:
        logger.warning("photon: httpx not installed — pip install httpx")
        return False
    node_bin = _get_scoped_secret("PHOTON_NODE_BIN") or "node"
    if not shutil.which(node_bin):
        logger.warning("photon: node binary '%s' not found on PATH", node_bin)
        return False
    if not sidecar_deps_installed():
        # Self-install is possible at connect time (npm on PATH + writable sidecar dir):
        # report available so _start_sidecar cold-installs — hosted images have no CLI.
        if bool(shutil.which("npm")) and _dir_writable(_sidecar_dir()):
            return True
        # DEBUG, not WARNING: normal pre-setup state, and check_fn is polled from hot paths.
        npm_error = ""
        with contextlib.suppress(OSError):
            if _npm_error_log().exists():
                npm_error = _npm_error_log().read_text(encoding="utf-8").strip()[:_NPM_ERROR_LOG_MAX_CHARS]
        hint = f" (last npm error: {npm_error})" if npm_error else ""
        logger.debug("photon: spectrum-ts not installed at %s%s — run: hermes photon setup", _sidecar_dir(), hint)
        return False
    return True


def _sidecar_deps_stale() -> bool:
    """True when node_modules predates the lockfile (`hermes update` rewrites it without
    reinstalling); False if either file is missing."""
    return _lock_newer_than_install(_sidecar_dir())


def _reinstall_sidecar_deps() -> None:
    """``npm ci`` (fallback ``npm install``); blocking, best-effort — on failure the stale
    deps stay and the readiness check reports the real error."""
    npm = shutil.which("npm")
    if not npm:
        logger.warning("[photon] cannot reinstall stale sidecar deps: npm not on PATH")
        return
    from hermes_cli._subprocess_compat import windows_hide_flags  # no console flash on Windows

    def _run(verb: str) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            [npm, verb], cwd=str(_sidecar_dir()), capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, timeout=_NPM_REINSTALL_TIMEOUT, creationflags=windows_hide_flags())
    try:
        result = _run("ci")
        if result.returncode != 0:
            logger.warning("[photon] sidecar `npm ci` failed; falling back to `npm install`")
            result = _run("install")
    except subprocess.TimeoutExpired:  # retried on the next reconnect tick
        logger.error("[photon] sidecar dependency reinstall timed out after %ss", _NPM_REINSTALL_TIMEOUT)
        return
    if result.returncode == 0:
        logger.info("[photon] sidecar dependencies reinstalled from lockfile")
    else:
        logger.error("[photon] sidecar dependency reinstall failed: %s", (result.stderr or result.stdout or "").strip())


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    if (extra.get("project_id") or _get_scoped_secret("PHOTON_PROJECT_ID")) and (
            extra.get("project_secret") or _get_scoped_secret("PHOTON_PROJECT_SECRET")):
        return True
    stored_id, stored_sec = load_project_credentials()  # auth.json fallback
    return bool(stored_id and stored_sec)


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so env-only setups appear in status
    (``home_channel`` becomes a ``HomeChannel`` via the core plugin hook)."""
    project_id, project_secret = load_project_credentials()
    if not (project_id and project_secret):
        return None
    seed: dict = {"project_id": project_id, "project_secret": project_secret}
    home = _get_scoped_secret("PHOTON_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": _get_scoped_secret("PHOTON_HOME_CHANNEL_NAME", "Home")}
    return seed


def _markdown_enabled() -> bool:
    """Replies go out as markdown; ``PHOTON_MARKDOWN=false`` is the kill-switch to plain text."""
    return _get_scoped_secret("PHOTON_MARKDOWN", "true").strip().lower() not in {"false", "0", "no"}


def _url_only_candidate(text: str) -> Optional[str]:
    candidate = (text or "").strip()
    if not re.fullmatch(r"https?://\S+", candidate, flags=re.IGNORECASE):
        return None
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    return candidate if parsed.scheme.lower() in {"http", "https"} and parsed.netloc else None


def _richlink_candidate(text: str) -> Optional[str]:
    """URL to send via ``richlink()`` — only exact http(s) URL messages; prose with
    URLs and Markdown links stay on the text path so labels aren't dropped."""
    return _url_only_candidate(text) if _markdown_enabled() else None


def _format_richlink_content(content: Dict[str, Any]) -> str:
    url, title, summary = (str(content.get(k) or "").strip() for k in ("url", "title", "summary"))
    parts = [p for p in (title, summary if summary != title else "", url) if p]
    return "\n".join(parts) if parts else "[Photon rich link received with no URL]"


def _group_item_contents(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The dict ``content`` of every well-formed item in a ``group`` payload."""
    items = (item.get("content") if isinstance(item, dict) else None for item in content.get("items") or [])
    return [c for c in items if isinstance(c, dict)]


def _richlink_url_from_content(content: Dict[str, Any]) -> Optional[str]:
    ctype = content.get("type")
    if ctype in ("text", "richlink"):
        return _url_only_candidate(content.get("text" if ctype == "text" else "url") or "")
    if ctype == "group":
        return next((u for u in map(_richlink_url_from_content, _group_item_contents(content)) if u), None)
    return None


def _is_richlink_preview_attachment(payload: Dict[str, Any]) -> bool:
    # Preview art can carry an opaque MIME; the name/id marker is the reliable signal,
    # the recent-link window guards real files.
    return payload.get("type") == "attachment" and any(
        _RICHLINK_PREVIEW_ATTACHMENT_SUFFIX in str(payload.get(k) or "").lower() for k in ("name", "id"))


def _richlink_preview_label(content: Dict[str, Any]) -> str:
    def _label(c: Dict[str, Any]) -> str:
        return str(c.get("name") or c.get("id") or "(unnamed)")
    if content.get("type") == "attachment":
        return _label(content)
    if content.get("type") == "group":
        return ", ".join(_label(c) for c in _group_item_contents(content)) or "(group)"
    return "(unknown)"


def _is_richlink_preview_content(content: Dict[str, Any]) -> bool:
    """A preview attachment, or a non-empty group made ONLY of preview attachments."""
    if _is_richlink_preview_attachment(content):
        return True
    if content.get("type") != "group":
        return False
    items = content.get("items") or []
    contents = _group_item_contents(content)
    return bool(items) and len(contents) == len(items) and all(
        _is_richlink_preview_attachment(c) for c in contents)


def _parse_timestamp(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(tz=timezone.utc)
    except ValueError:
        return datetime.now(tz=timezone.utc)


_Normalized = Tuple[str, MessageType, List[str], List[str]]  # text, type, media_urls, media_types


def _normalize_binary_payload(payload: Dict[str, Any]) -> _Normalized:
    """Cache an inline attachment/voice payload; fall back to a text marker."""
    is_voice = payload.get("type") == "voice"
    name = payload.get("name") or ("voice" if is_voice else "(unnamed)")
    mime = payload.get("mimeType") or ""
    # iMessage voice notes are CAF and may arrive "(unnamed)", so MIME is a signal too.
    is_voice = is_voice or name.lower().endswith(".caf") or mime == "audio/x-caf"
    mtype = MessageType.VOICE if is_voice else _attachment_message_type(mime)
    label = "voice" if is_voice else "attachment"
    cached = _cache_inbound_attachment(payload, name, mime, force_audio=is_voice)
    if cached:
        return f"({label})", mtype, [cached], [mime or ("audio/mp4" if is_voice else "application/octet-stream")]
    duration = payload.get("duration")
    duration_text = f", duration: {duration}s" if isinstance(duration, (int, float)) else ""
    return f"[Photon {label} received: {name} ({mime or 'unknown MIME'}{duration_text})]", mtype, [], []


def _normalize_group_content(content: Dict[str, Any]) -> _Normalized:
    text_parts: List[str] = []
    mtype = MessageType.TEXT
    media_urls: List[str] = []
    media_types: List[str] = []
    for item_content in _group_item_contents(content):
        item_type = item_content.get("type")
        if item_type in {"attachment", "voice"}:
            marker, item_mtype, item_urls, item_types = _normalize_binary_payload(item_content)
            if mtype == MessageType.TEXT:
                mtype = item_mtype
            media_urls.extend(item_urls)
            media_types.extend(item_types)
            if not item_urls:
                text_parts.append(marker)
        elif item_type == "text":
            text_parts.append(item_content.get("text") or "")
        elif item_type == "richlink":
            text_parts.append(_format_richlink_content(item_content))
        elif item_type:
            text_parts.append(f"[Photon content type not handled: {item_type}]")
    if media_urls and mtype == MessageType.TEXT:
        mtype = MessageType.DOCUMENT
    text = "\n".join(part for part in text_parts if part).strip()
    return text or ("(attachment)" if media_urls else "[Photon empty group received]"), mtype, media_urls, media_types


_CONTENT_NORMALIZERS: Dict[Any, Callable[[Dict[str, Any]], _Normalized]] = {
    "text": lambda c: (c.get("text") or "", MessageType.TEXT, [], []),
    "attachment": _normalize_binary_payload, "voice": _normalize_binary_payload,
    "richlink": lambda c: (_format_richlink_content(c), MessageType.TEXT, [], []),
    "group": _normalize_group_content,
}
_BINARY_CONTENT_TYPES = {"attachment", "voice", "group"}  # may decode/cache media bytes → run off the event loop


def _normalize_content(content: Dict[str, Any]) -> _Normalized:
    """Turn a sidecar ``content`` payload into (text, type, media_urls, media_types)."""
    ctype = content.get("type")
    normalize = _CONTENT_NORMALIZERS.get(ctype) if isinstance(ctype, str) else None
    if normalize is None:
        return f"[Photon content type not handled: {ctype}]", MessageType.TEXT, [], []
    return normalize(content)


def _attachment_body(space_id: str, safe_path: str, *, kind: str, name: Optional[str] = None,
                     mime_type: Optional[str] = None, caption: Optional[str] = None) -> Dict[str, Any]:
    """``/send-attachment`` body; spectrum-ts infers name/mimeType from the extension,
    so optional keys are only sent when Hermes supplied them."""
    body: Dict[str, Any] = {
        "spaceId": space_id, "path": safe_path, "kind": "voice" if kind == "voice" else "attachment"}
    body.update({k: v for k, v in (("name", name), ("mimeType", mime_type), ("caption", caption)) if v})
    return body


def _guess_mime(path: str) -> Optional[str]:
    import mimetypes
    return mimetypes.guess_type(path)[0] or None


def _bounded_put(store: Dict[str, Any], key: str, value: Any, max_size: int) -> None:
    """Insert with insertion-order refresh and a HARD size bound (evict oldest)."""
    if key in store:
        del store[key]
    store[key] = value
    if len(store) > max_size:
        for old in list(store.keys())[: len(store) - max_size]:
            del store[old]


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    """Cancel *task* and wait for it, unless it is the current task."""
    if task is None:
        return
    task.cancel()
    if task is not asyncio.current_task():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


# -- Adapter -------------------------------------------------------------------

class PhotonAdapter(BasePlatformAdapter):
    """Bidirectional bridge to Photon Spectrum via the Node spectrum-ts sidecar."""

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False  # no edit API: streaming must not leave a stale cursor (▉)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("photon"))
        extra = config.extra or {}
        # Project credentials: env wins, then config.extra, then auth.json.
        stored_id, stored_sec = load_project_credentials()
        self._project_id: str = _get_scoped_secret("PHOTON_PROJECT_ID") or extra.get("project_id") or stored_id or ""
        self._project_secret: str = (
            _get_scoped_secret("PHOTON_PROJECT_SECRET") or extra.get("project_secret") or stored_sec or "")
        self._sidecar_port = _coerce_port(
            extra.get("sidecar_port") or _get_scoped_secret("PHOTON_SIDECAR_PORT"), _DEFAULT_SIDECAR_PORT)
        self._sidecar_bind = _DEFAULT_SIDECAR_BIND
        self._sidecar_token = _get_scoped_secret("PHOTON_SIDECAR_TOKEN") or secrets.token_hex(16)
        autostart = str(_get_scoped_secret("PHOTON_SIDECAR_AUTOSTART", "true")).lower()
        self._autostart_sidecar = autostart not in ("0", "false", "no")
        self._node_bin = _get_scoped_secret("PHOTON_NODE_BIN") or shutil.which("node") or "node"
        # Presence watchdog (second layer behind the sidecar's own zombie-stream detection):
        # respawns only when the sidecar's HTTP loop hangs; 10-min interval because shared
        # lines are quiet for hours. Config key wins, then env; None-aware so 0 disables it.
        def _setting(key: str, env: str, default: Any, cast: Callable[[Any], Any]) -> Any:
            value = extra.get(key)
            if value is None:
                value = _get_scoped_secret(env)
            try:
                return cast(value)
            except (TypeError, ValueError):
                return default
        self._probe_interval = _setting("probe_interval_seconds", "PHOTON_PROBE_INTERVAL_SECONDS", 600.0, float)
        self._probe_timeout = _setting("probe_timeout_seconds", "PHOTON_PROBE_TIMEOUT_SECONDS", 10.0, float)
        self._probe_max_failures = _setting("probe_max_failures", "PHOTON_PROBE_MAX_FAILURES", 3, int)
        self._probe_enabled = self._probe_interval > 0
        self.supports_code_blocks = _markdown_enabled()  # markdown on => fences pass through
        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._respawn_lock: Optional[asyncio.Lock] = None
        self._sidecar_supervisor_task = self._inbound_task = self._sidecar_health_task = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._inbound_running = self._watchdog_running = False
        self._sidecar_health_interval = 15.0
        self._probe_failures = 0
        self._last_upstream_activity = 0.0  # monotonic; watchdog skips probe if traffic proved liveness
        self._seen_messages: Dict[str, float] = {}  # at-least-once stream dedup
        self._sent_message_ids: Dict[str, float] = {}  # only reactions targeting OUR sends are routed
        self._last_inbound_by_chat: Dict[str, str] = {}  # default target for the react action
        self._recent_richlinks_by_chat: Dict[str, float] = {}  # coalesce preview-art attachments
        self._typing_last_sent: Dict[str, float] = {}
        self._pending_fffc: Dict[str, tuple[float, Any]] = {}  # chat_key → (timestamp, asyncio.Task)
        # Group-chat mention gating (parity with BlueBubbles); DMs are never gated.
        require_mention = extra.get("require_mention")
        if require_mention is None:
            require_mention = _get_scoped_secret("PHOTON_REQUIRE_MENTION")
        self.require_mention = str(require_mention).strip().lower() in {"true", "1", "yes", "on"}
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"] if "mention_patterns" in extra
            else _get_scoped_secret("PHOTON_MENTION_PATTERNS"))

    # -- Group-mention gating (parity with BlueBubbles) ----------------------------

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> "list[re.Pattern]":
        """``raw``: list, string (JSON list or comma/newline-separated) or None (defaults)."""
        return compile_mention_patterns(
            raw, log_prefix="photon", defaults=_DEFAULT_MENTION_PATTERNS, logger_=logger)

    def _message_matches_mention_patterns(self, text: str) -> bool:
        return bool(text) and any(pattern.search(text) for pattern in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip a leading wake word only (patterns are regexes; never touch later words)."""
        stripped = text.lstrip() if text else ""
        for match in filter(None, (pattern.match(stripped) for pattern in self._mention_patterns)):
            return stripped[match.end():].lstrip(" ,:-") or text
        return text

    # -- Sidecar HTTP plumbing ------------------------------------------------------

    def _sidecar_url(self, path: str) -> str:
        return f"http://{self._sidecar_bind}:{self._sidecar_port}{path}"

    def _sidecar_headers(self) -> Dict[str, str]:
        return {"X-Hermes-Sidecar-Token": self._sidecar_token}

    # -- Connection lifecycle ------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not HTTPX_AVAILABLE:
            self._set_fatal_error("MISSING_DEP", "httpx not installed", retryable=False)
            return False
        if not self._project_id or not self._project_secret:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "PHOTON_PROJECT_ID and PHOTON_PROJECT_SECRET are required. Run: hermes photon setup",
                retryable=False)
            return False
        client = httpx.AsyncClient(timeout=30.0, trust_env=False)
        self._http_client = client
        if self._autostart_sidecar:  # the sidecar holds the gRPC stream for BOTH directions: required now
            try:
                await self._start_sidecar()
            except Exception as e:
                # Typed deterministic failures may be retryable=False (fatal); everything else
                # stays retryable, with the gateway's NEEDS_ATTENTION escalation as the backstop.
                if isinstance(e, PhotonSidecarStartupError):
                    self._set_fatal_error(e.code, str(e), retryable=e.retryable)
                else:
                    self._set_fatal_error("SIDECAR_FAILED", f"failed to start Photon sidecar: {e}", retryable=True)
                _delete_runtime_record()  # no live sidecar — don't mislead standalone senders
                await client.aclose()
                self._http_client = None
                return False
        else:
            logger.warning("[photon] sidecar autostart disabled — inbound + outbound will fail")
        loop = asyncio.get_event_loop()
        self._inbound_running = True
        self._inbound_task = loop.create_task(self._inbound_loop())
        self._sidecar_health_task = loop.create_task(self._monitor_sidecar_health())
        self._last_upstream_activity = time.monotonic()
        if self._probe_enabled and self._autostart_sidecar:
            self._respawn_lock = asyncio.Lock()
            self._watchdog_running = True
            self._watchdog_task = loop.create_task(self._presence_watchdog())
        self._mark_connected()
        logger.info("[photon] connected — sidecar on %s:%d, streaming inbound over gRPC",
                    self._sidecar_bind, self._sidecar_port)
        self._wire_plugin_handlers(None)  # ctx.register_platform_handler natives
        return True

    async def disconnect(self) -> None:
        self._inbound_running = False
        await self._stop_watchdog()  # first, so it can't respawn while we tear the sidecar down
        task, self._sidecar_health_task = self._sidecar_health_task, None
        await _cancel_task(task)
        task, self._inbound_task = self._inbound_task, None
        await _cancel_task(task)
        for _, fffc_task in list(self._pending_fffc.values()):
            if fffc_task and not fffc_task.done():
                fffc_task.cancel()
        self._pending_fffc.clear()
        await self._stop_sidecar()
        if self._http_client is not None:
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None
        self._mark_disconnected()

    def _dispatch_fatal_notification(self) -> None:
        """Notify the gateway of a fatal error from a detached task. The health/supervisor
        tasks must NOT await ``_notify_fatal_error()`` inline: the gateway answers with
        ``disconnect()``, which cancels those very tasks (via its own wrapper task, so the
        ``current_task()`` guard can't help) — the CancelledError would kill the notifier
        mid-handoff: no log, no reconnect. A fresh task can be cancelled freely."""
        asyncio.create_task(self._notify_fatal_error_logged())

    async def _notify_fatal_error_logged(self) -> None:
        try:
            await self._notify_fatal_error()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[photon] fatal-error notification failed: %s", exc)

    # -- Inbound stream consumer ---------------------------------------------------

    async def _inbound_loop(self) -> None:
        """Consume the sidecar's ``/inbound`` NDJSON stream, re-opening it if it drops
        (the sidecar owns the gRPC reconnect to Photon)."""
        client = self._http_client
        if client is None:
            return
        url = self._sidecar_url("/inbound")
        headers = self._sidecar_headers()
        backoff = 1.0
        while self._inbound_running:
            try:
                async with client.stream("GET", url, headers=headers, timeout=None) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"/inbound returned {resp.status_code}")
                    backoff = 1.0
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
                logger.warning("[photon] inbound stream dropped (%s); reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _monitor_sidecar_health(self) -> None:
        """Promote a degraded upstream stream (per ``/healthz``) into a reconnect, so a
        live sidecar HTTP process with a dead gRPC stream isn't a silent outage."""
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
            # Loud line for a suspected zombie stream before the sidecar's degraded->exit-75 fires.
            staleness = stream.get("staleness")
            if isinstance(staleness, dict) and staleness.get("zombieSuspected") is True:
                logger.warning("[photon] sidecar reports suspected zombie stream"
                               " (silentForMs=%s, lastProbeOutcome=%s)",
                               staleness.get("silentForMs"), staleness.get("lastProbeOutcome"))
            if stream.get("ok") is not False:
                continue
            message = (
                f"Photon upstream stream degraded (state={stream.get('state') or 'unknown'}, "
                f"degradedForMs={stream.get('degradedForMs')}): {stream.get('lastIssue') or 'unknown stream issue'}")
            logger.error("[photon] %s", message)
            self._set_fatal_error("UPSTREAM_STREAM_DEGRADED", message, retryable=True)
            self._dispatch_fatal_notification()
            break

    async def _on_inbound_line(self, line: str) -> None:
        self._note_upstream_activity()  # any inbound line proves the upstream gRPC channel is live
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
        t = self._seen_messages.get(msg_id)
        if t is not None and now - t < _DEDUP_WINDOW_SECONDS:
            return True
        _bounded_put(self._seen_messages, msg_id, now, _DEDUP_MAX_SIZE)
        return False

    async def _fffc_timeout_handler(self, chat_key: str, message_id: str) -> None:
        await asyncio.sleep(_FFFC_WAIT_SECONDS)
        if self._pending_fffc.pop(chat_key, None):
            logger.warning("[photon] wait for attachment was too long, can't retrieve attachment data "
                           "(message %s, chat %s)", message_id, chat_key)

    def _cancel_pending_fffc(self, chat_key: str) -> bool:
        """Pop and cancel a pending U+FFFC timeout; True when a live task was cancelled."""
        prev = self._pending_fffc.pop(chat_key, None)
        live = bool(prev and prev[1] and not prev[1].done())
        if live:
            prev[1].cancel()
        return live

    async def _dispatch_inbound(self, event: Dict[str, Any]) -> None:
        """Normalize a sidecar inbound event ``{messageId, space: {id, type: dm|group, phone},
        sender: {id}, content: {type: text|attachment|voice|reaction|richlink|group|
        poll_option|read, ...}, timestamp}`` and dispatch it. Attachment/voice bytes arrive
        inline as base64 ``data`` under the sidecar's cap; otherwise metadata only → marker."""
        space = event.get("space") or {}
        sender = event.get("sender") or {}
        content = event.get("content") or {}
        space_id = space.get("id") or ""
        if not space_id:
            logger.warning("[photon] inbound missing space.id")
            return
        chat_type = "group" if space.get("type") == "group" else "dm"
        sender_id = sender.get("id") or space.get("phone") or space_id
        timestamp = _parse_timestamp(event.get("timestamp") or "")
        message_id = event.get("messageId")
        ctype = content.get("type")

        def _event(text: str, mtype: MessageType = MessageType.TEXT, **kwargs: Any) -> MessageEvent:
            source = self.build_source(chat_id=space_id, chat_name=space_id, chat_type=chat_type,
                                       user_id=sender_id, user_name=sender_id or None)
            return MessageEvent(text=text, message_type=mtype, source=source, message_id=message_id,
                                raw_message=event, timestamp=timestamp, **kwargs)
        if ctype in {"read", "read_receipt"}:  # presence signal, not a user turn (receipts for our sends)
            logger.debug("[photon] outbound message read: %s", content.get("targetMessageId") or "unknown")
            return
        if ctype == "reaction":
            # Only tapbacks on messages WE sent are addressed to the bot. Checked before the
            # mention gate: a tapback never carries a wake word.
            target_id = content.get("targetMessageId")
            is_ours = content.get("targetDirection") == "outbound" or (
                target_id and target_id in self._sent_message_ids)
            if not is_ours:
                logger.debug("[photon] ignoring reaction on a message we didn't send")
                return
            # reply_to_is_own_message holds by construction, so the gateway injects
            # `[Replying to your previous message: "..."]` when targetText is present.
            await self.handle_message(_event(
                f"reaction:added:{content.get('emoji') or ''}", reply_to_message_id=target_id,
                reply_to_text=content.get("targetText") or None, reply_to_is_own_message=True))
            return
        # U+FFFC placeholder: wait for the real attachment. Detected before _record_last_inbound
        # so the placeholder isn't the reaction target.
        if ctype == "text" and (content.get("text") or "").strip() == "\ufffc":
            self._cancel_pending_fffc(space_id)
            task = asyncio.create_task(self._fffc_timeout_handler(space_id, message_id or ""))
            self._pending_fffc[space_id] = (time.monotonic(), task)
            logger.debug("[photon] U+FFFC placeholder received — waiting for attachment")
            return
        if ctype in {"attachment", "voice"} and self._cancel_pending_fffc(space_id):
            logger.debug("[photon] attachment arrived — cancelling U+FFFC timeout")
        # Preview art for a just-received URL must not become a second user prompt;
        # suppress before recording it as reactable or decoding image bytes.
        if self._is_recent_richlink_preview(space_id, content):
            logger.info("[photon] suppressing rich-link preview attachment: %s", _richlink_preview_label(content))
            return
        # Everything past here is a real (reactable) message. Recorded before the mention
        # gate: reacting to a non-wake-word group message is valid.
        self._record_last_inbound(space_id, message_id)
        if ctype == "poll_option":
            # Native poll vote: a selection is forwarded as if typed (the gateway's
            # pending-clarify intercept resolves it); a deselection is dropped.
            if content.get("selected") is False:
                logger.debug("[photon] ignoring poll deselection")
                return
            choice = (content.get("title") or "").strip()
            if not choice:
                logger.debug("[photon] ignoring poll vote with empty title")
                return
            await self.handle_message(_event(choice))
            return
        if ctype in _BINARY_CONTENT_TYPES:
            # Base64 decode + media-cache write of possibly multi-MB payloads — keep it off the event loop.
            text, mtype, media_urls, media_types = await asyncio.to_thread(_normalize_content, content)
        else:
            text, mtype, media_urls, media_types = _normalize_content(content)
        if chat_type == "group" and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                logger.debug("[photon] ignoring group message (require_mention=true, no mention pattern matched)")
                return
            text = self._clean_mention_text(text)
        self._record_recent_richlink(space_id, _richlink_url_from_content(content) or text)
        await self.handle_message(_event(text, mtype, media_urls=media_urls, media_types=media_types))

    # -- Sidecar lifecycle ---------------------------------------------------------

    @staticmethod
    def _quick_stdout(cmd: List[str]) -> Optional[str]:
        """stdout of a short shell-out, or None if it failed to run."""
        try:
            return subprocess.run(  # noqa: S603, S607
                cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5.0,
                check=False).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None

    @classmethod
    def _find_listener_pids(cls, port: int) -> List[int]:
        """PIDs listening on a local TCP port (empty if none/undeterminable)."""
        out = cls._quick_stdout(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"])
        return [int(tok) for tok in out.split() if tok.strip().isdigit()] if out is not None else []

    @classmethod
    def _pid_is_sidecar(cls, pid: int) -> bool:
        """True if ``pid``'s command line is a Photon sidecar (any Hermes checkout)."""
        out = cls._quick_stdout(["ps", "-p", str(pid), "-o", "command="])
        return out is not None and "photon/sidecar/index.mjs" in out

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)  # windows-footgun: ok — only called from _reap_stale_sidecar which win32-guards early
        except OSError:
            return False
        return True

    async def _reap_stale_sidecar(self) -> None:
        """Kill an orphaned sidecar squatting our port (a SIGKILLed gateway leaves one whose
        token we don't know, so every respawn dies on EADDRINUSE). Listeners are verified
        by command line before being signalled."""
        if sys.platform == "win32":  # lsof/ps; orphaning is a POSIX-only path
            return
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                await client.post(self._sidecar_url("/healthz"), headers=self._sidecar_headers())
        except httpx.RequestError:
            return  # nothing listening — the normal case
        # Off the loop: lsof + one `ps` per pid can hold it 5+5·N s, on every reconnect.
        def _inspect():
            found = self._find_listener_pids(self._sidecar_port)
            mine = [pid for pid in found if self._pid_is_sidecar(pid)]
            return mine, [pid for pid in found if pid not in mine]
        stale, foreign = await asyncio.to_thread(_inspect)
        fix = "free it or set PHOTON_SIDECAR_PORT to a different port"
        if not stale:
            raise RuntimeError(f"port {self._sidecar_port} is in use by another process "
                               f"(pids: {foreign or 'unknown'}, not a Photon sidecar) — {fix}")

        def _kill(pid: int, sig: int) -> None:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)  # windows-footgun: ok — unreachable on win32 (early return above)
        for pid in stale:
            logger.warning("[photon] reaping orphaned sidecar (pid %d) on port %d", pid, self._sidecar_port)
            _kill(pid, signal.SIGTERM)
        deadline = time.time() + 3.0
        while time.time() < deadline and any(self._pid_alive(p) for p in stale):
            await asyncio.sleep(0.1)
        for pid in stale:
            if self._pid_alive(pid):
                _kill(pid, signal.SIGKILL)  # windows-footgun: ok — unreachable on win32 (early return above)
        await asyncio.sleep(0.2)  # let the OS release the listening socket
        if foreign:
            raise RuntimeError(
                f"port {self._sidecar_port} is also held by non-sidecar processes (pids: {foreign}) — {fix}")

    async def _ensure_sidecar_deps(self) -> None:
        """Cold-install or refresh sidecar node_modules before spawn (off the loop)."""
        if not sidecar_deps_installed():
            # Hosted images have no CLI for `hermes photon setup`: connect bootstraps deps itself.
            logger.info("[photon] sidecar deps not installed; installing into %s", _sidecar_dir())
            await asyncio.to_thread(_reinstall_sidecar_deps)
            if not sidecar_deps_installed():
                # Deterministic on immutable images — non-retryable so it doesn't spin silently.
                raise PhotonSidecarStartupError(
                    f"Photon sidecar deps could not be installed into "
                    f"{_sidecar_dir()} (see log for the npm error). "
                    f"Run: cd {_sidecar_dir()} && npm ci   (or `hermes photon setup`)",
                    code="SIDECAR_DEPS_MISSING", retryable=False)
        # `hermes update` bumps the lockfile without reinstalling node_modules; the sidecar
        # would spawn against stale deps and die on every reconnect.
        if _sidecar_deps_stale():
            logger.warning("[photon] sidecar deps are stale (lockfile newer than install); reinstalling before start")
            await asyncio.to_thread(_reinstall_sidecar_deps)

    async def _apply_spectrum_patch(self, hide_flags: int) -> None:
        """Run the mixed-attachment patch script (best-effort, off the loop: up to 10s, every reconnect)."""
        try:
            patch = await asyncio.to_thread(
                subprocess.run,  # noqa: S603
                [self._node_bin, str(_sidecar_dir() / "patch-spectrum-mixed-attachments.mjs"), str(_sidecar_dir())],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10, check=False,
                creationflags=hide_flags)
            if patch.returncode != 0:
                raise RuntimeError((patch.stderr or patch.stdout or "").strip())
            if patch.stderr.strip():
                logger.debug("[photon] %s", patch.stderr.strip())
        except Exception as exc:
            logger.warning("[photon] failed to apply Spectrum mixed attachment patch: %s", exc)

    async def _start_sidecar(self) -> None:
        await self._ensure_sidecar_deps()
        await self._reap_stale_sidecar()
        env = os.environ.copy()
        env.update({
            "PHOTON_PROJECT_ID": self._project_id, "PHOTON_PROJECT_SECRET": self._project_secret,
            "PHOTON_SIDECAR_PORT": str(self._sidecar_port), "PHOTON_SIDECAR_BIND": self._sidecar_bind,
            "PHOTON_SIDECAR_TOKEN": self._sidecar_token,
            # Exit on stdin EOF so ANY gateway death (incl. SIGKILL) can't orphan it on the port.
            "PHOTON_SIDECAR_WATCH_STDIN": "1"})
        from hermes_cli._subprocess_compat import windows_hide_flags  # hide child console on Windows
        await self._apply_spectrum_patch(windows_hide_flags())
        try:
            self._sidecar_proc = subprocess.Popen(  # noqa: S603
                [self._node_bin, str(_sidecar_dir() / "index.mjs")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                start_new_session=(sys.platform != "win32"),
                creationflags=windows_hide_flags())  # CREATE_NO_WINDOW only (no DETACHED_PROCESS): pipes stay usable
        except FileNotFoundError as exc:  # deterministic: retrying can never fix a missing binary
            raise PhotonSidecarStartupError(
                f"node binary not found ({self._node_bin!r}) — install Node.js or set PHOTON_NODE_BIN: {exc}",
                code="SIDECAR_NODE_MISSING", retryable=False) from exc
        loop = asyncio.get_event_loop()
        self._sidecar_supervisor_task = loop.create_task(self._supervise_sidecar(self._sidecar_proc))
        deadline = time.time() + 15.0  # wait for /healthz — up to 15s on cold start
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while time.time() < deadline:
                if self._sidecar_proc.poll() is not None:
                    _delete_runtime_record()
                    raise RuntimeError(
                        f"Photon sidecar exited with code {self._sidecar_proc.returncode} before becoming ready")
                try:
                    resp = await client.post(self._sidecar_url("/healthz"), headers=self._sidecar_headers())
                    if resp.status_code == 200:  # let out-of-process senders (cron) reach this sidecar
                        _write_runtime_record(self._sidecar_port, self._sidecar_token, self._sidecar_proc.pid)
                        return
                except httpx.RequestError as e:
                    last_err = e
                await asyncio.sleep(0.2)
        _delete_runtime_record()
        raise RuntimeError(f"Photon sidecar did not become ready within 15s: {last_err}")

    async def _supervise_sidecar(self, proc: subprocess.Popen) -> None:
        """Pump the sidecar's stdout/stderr into our logger."""
        if proc.stdout is None:  # launched without stdout=PIPE
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
            logger.error("[photon] sidecar exited unexpectedly (code %s) — triggering reconnect", exit_code)
            self._set_fatal_error(
                "SIDECAR_CRASHED", f"Photon sidecar exited unexpectedly (code {exit_code})", retryable=True)
            self._dispatch_fatal_notification()

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        if proc is None:
            _delete_runtime_record()  # never leave a record behind on disconnect
            return
        try:
            if proc.stdin is not None:  # closing our stdin end is itself a shutdown signal (EOF watch)
                with contextlib.suppress(Exception):
                    proc.stdin.close()
            if self._http_client is not None:  # polite shutdown first
                with contextlib.suppress(Exception):
                    await self._http_client.post(
                        self._sidecar_url("/shutdown"), headers=self._sidecar_headers(), timeout=2.0)
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
            _delete_runtime_record()
            if self._sidecar_supervisor_task is not None:
                # May run INSIDE the supervisor task's own crash chain (sidecar exit -> fatal
                # notify -> gateway disconnect() -> here). A task cancelling itself raises
                # CancelledError into the fatal handler before the reconnect is queued, leaving
                # Photon permanently dead — so let it finish exiting on its own.
                if self._sidecar_supervisor_task is not asyncio.current_task():
                    # _stop_sidecar() is called both from external cleanup (Gateway shutdown, explicit
                    # disconnect) AND, indirectly, from WITHIN the supervisor task's own crash-handling
                    # chain: _supervise_sidecar() detects the sidecar exit, calls _set_fatal_error() +
                    # self._notify_fatal_error(), which the Gateway's fatal-error handler answers by calling
                    # adapter.disconnect() -> this same _stop_sidecar(). In that second case,
                    # self._sidecar_supervisor_task IS the currently-running task. Cancelling it raises
                    # CancelledError into its own call stack (at the next await point in
                    # _notify_fatal_error() or here), which aborts the fatal-error handler before the
                    # Gateway ever reaches the "queue for background reconnection" step -- Photon then stays
                    # permanently dead until a manual restart, since asyncio.CancelledError inherits from
                    # BaseException (not Exception) and isn't caught by the handler's `except Exception`
                    # guards (issue #73159). A task cannot legally cancel itself anyway (the cancellation
                    # would only take effect at its own next await, which is exactly the corruption
                    # described above), so skip it here and let the task finish exiting on its own instead.
                    self._sidecar_supervisor_task.cancel()
                self._sidecar_supervisor_task = None

    # -- Presence watchdog ---------------------------------------------------------

    def _note_upstream_activity(self) -> None:
        """Record proof the upstream gRPC channel is live (inbound line or good probe)."""
        self._last_upstream_activity = time.monotonic()
        self._probe_failures = 0

    async def _probe_once(self) -> str:
        """One ``/probe`` round-trip → ``"alive"`` (HTTP 200, the only proof of liveness),
        ``"hung"`` (the HTTP call timed out; counts toward respawn) or ``"inconclusive"``
        (503/refused/transport error: never counts — the network may just be down and a
        dead process is the supervisor's job)."""
        client = self._http_client
        if client is None:
            return "inconclusive"
        try:
            resp = await client.post(
                self._sidecar_url("/probe"), headers=self._sidecar_headers(), timeout=self._probe_timeout)
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
        """Restart the sidecar to recover a dead gRPC stream (a fresh ``Spectrum()``
        re-subscribes; the inbound loop re-opens ``/inbound`` on its own). Locked so
        overlapping triggers can't double-spawn."""
        if self._respawn_lock is None:
            self._respawn_lock = asyncio.Lock()
        lock = self._respawn_lock
        if lock.locked():
            logger.info("[photon] respawn already in progress; skipping")
            return
        async with lock:
            logger.warning("[photon] presence watchdog: %s — respawning sidecar", reason)
            try:
                await self._stop_sidecar()
            except Exception:
                logger.exception("[photon] error stopping sidecar during respawn")
            try:
                await self._start_sidecar()
            except Exception:
                logger.exception("[photon] failed to respawn sidecar; watchdog will retry")
                return
            self._note_upstream_activity()  # fresh stream: give it a full interval before probing again
            logger.info("[photon] presence watchdog: sidecar respawned, gRPC stream renewed")

    async def _presence_watchdog(self) -> None:
        """Probe on a long interval, skipping when inbound traffic already proved liveness;
        only *hung* probes count toward respawn (``_probe_max_failures``)."""
        await asyncio.sleep(self._probe_interval)  # stagger the first probe (fleet restarts, warm-up)
        while self._watchdog_running:
            try:
                idle = time.monotonic() - self._last_upstream_activity
                if idle < self._probe_interval:
                    await asyncio.sleep(self._probe_interval - idle)
                    continue
                verdict = await self._probe_once()
                if verdict == "alive":
                    self._note_upstream_activity()
                elif verdict == "hung":
                    self._probe_failures += 1
                    logger.warning("[photon] presence probe hung (%d/%d)",
                                   self._probe_failures, self._probe_max_failures)
                    if self._probe_failures >= self._probe_max_failures:
                        await self._respawn_sidecar(f"{self._probe_failures} consecutive hung probes")
                else:
                    logger.debug("[photon] presence probe inconclusive; taking no action")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[photon] presence watchdog iteration failed")
            await asyncio.sleep(self._probe_interval)

    async def _stop_watchdog(self) -> None:
        self._watchdog_running = False
        task, self._watchdog_task = self._watchdog_task, None
        await _cancel_task(task)

    # -- Outbound ------------------------------------------------------------------

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._sidecar_send(chat_id, self.format_message(content))

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
                           session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Multiple-choice renders as a native poll; the vote comes back as a `poll_option`
        event that _dispatch_inbound turns into plain text, so the clarify is flipped into
        text-capture mode like the base fallback."""
        if not choices:  # open-ended: base plain-text behaviour is right
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata)
        from tools.clarify_gateway import mark_awaiting_text
        mark_awaiting_text(clarify_id)
        result = await self._sidecar_send_poll(chat_id, question, list(choices))
        if not result.success:
            # Old sidecar without /send-poll or a send error: numbered-text clarify fallback
            # (base also calls mark_awaiting_text; harmless).
            logger.warning("[photon] poll clarify failed (%s); falling back to text list", result.error)
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata)
        return result

    # -- Outbound media (parity with BlueBubbles): URL-based helpers cache to a local path
    # first; file-based ones pass the path straight to /send-attachment.

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url
            local_path = await cache_image_from_url(image_url)
        except Exception:  # couldn't fetch — send the URL as text
            return await super().send_image(chat_id, image_url, caption, reply_to)
        return await self._sidecar_send_attachment(chat_id, local_path, caption=caption)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                              **kwargs) -> SendResult:
        return await self._sidecar_send_attachment(chat_id, image_path, caption=caption)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        return await self._sidecar_send_attachment(chat_id, audio_path, caption=caption, kind="voice")

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        return await self._sidecar_send_attachment(chat_id, video_path, caption=caption)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        return await self._sidecar_send_attachment(chat_id, file_path, name=file_name, caption=caption)

    # send_animation: base falls back to send_image (iMessage renders GIFs inline as images).

    async def _sidecar_try(self, path: str, body: Dict[str, Any], what: str) -> bool:
        """Soft-failing sidecar call: True on success, False (debug-logged) on any error."""
        try:
            await self._sidecar_call(path, body)
            return True
        except Exception as e:
            logger.debug("[photon] %s failed: %s", what, e)
            return False

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        now = time.time()
        if now - self._typing_last_sent.get(chat_id, 0.0) < _TYPING_COOLDOWN_SECONDS:
            return
        self._typing_last_sent[chat_id] = now
        await self._sidecar_try("/typing", {"spaceId": chat_id, "state": "start"}, "send_typing")

    async def stop_typing(self, chat_id: str) -> None:
        self._typing_last_sent.pop(chat_id, None)
        await self._sidecar_try("/typing", {"spaceId": chat_id, "state": "stop"}, "stop_typing")

    # -- Reactions (tapbacks). Lifecycle hooks (👀 while processing, 👍/👎 on completion)
    # are opt-in via PHOTON_REACTIONS — noisy on a personal channel.

    _SENT_IDS_MAX = 1000
    _LAST_INBOUND_CHATS_MAX = 200

    def _record_sent_message(self, message_id: Optional[str]) -> None:
        if message_id:
            _bounded_put(self._sent_message_ids, message_id, time.time(), self._SENT_IDS_MAX)

    # A DM space is addressable as the chat GUID (`any;-;+1555...`) inbound events carry, or
    # the bare E.164 phone home-channel config uses; the sidecar's resolveSpace treats them
    # as one space, so normalize to the bare phone (mirrors phoneTargetFromSpaceId in index.mjs).
    _DM_CHAT_GUID_RE = re.compile(r"^any;-;(\+\d{6,})$")

    @classmethod
    def _normalize_chat_key(cls, chat_id: str) -> str:
        match = cls._DM_CHAT_GUID_RE.match(chat_id)
        return match.group(1) if match else chat_id

    def _put_by_chat(self, store: Dict[str, Any], chat_id: str, value: Any) -> None:
        _bounded_put(store, self._normalize_chat_key(chat_id), value, self._LAST_INBOUND_CHATS_MAX)

    def _record_last_inbound(self, chat_id: Optional[str], message_id: Optional[str]) -> None:
        if chat_id and message_id:
            self._put_by_chat(self._last_inbound_by_chat, chat_id, message_id)

    def _record_recent_richlink(self, chat_id: str, text: str) -> None:
        if chat_id and _url_only_candidate(text):
            self._put_by_chat(self._recent_richlinks_by_chat, chat_id, time.time())

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
        return _get_scoped_secret("PHOTON_REACTIONS", "false").strip().lower() in {"true", "1", "yes", "on"}

    async def _add_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Tapback ``emoji`` onto a message. Soft-fails (False), never raises."""
        return await self._sidecar_try(
            "/react", {"spaceId": chat_id, "messageId": message_id, "emoji": emoji}, "add_reaction")

    async def _remove_reaction(self, chat_id: str, message_id: str) -> bool:
        """Retract our tapback (best-effort: the sidecar's per-message reaction handle is
        lost on restart). Soft-fails (False), never raises."""
        return await self._sidecar_try("/unreact", {"spaceId": chat_id, "messageId": message_id}, "remove_reaction")

    # -- Agent-facing reactions (send_message action="react"): deliberate intents, so NOT
    # gated by PHOTON_REACTIONS.

    async def add_reaction(self, chat_id: str, emoji: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        """Tapback ``emoji`` onto a message (default: the chat's latest inbound). iMessage
        maps ❤️👍👎😂‼️❓ to native tapbacks; anything else is a custom-emoji reaction."""
        target = message_id or self._last_inbound_by_chat.get(self._normalize_chat_key(chat_id))
        if not target:
            return {"success": False, "error": "no message to react to — pass message_id (no "
                    "inbound message seen in this chat since the gateway started)"}
        if not await self._add_reaction(chat_id, target, emoji):
            return {"success": False, "error": "reaction failed (see gateway debug log)"}
        return {"success": True, "message_id": target}

    async def remove_reaction(self, chat_id: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        """Retract our tapback from a message (best-effort)."""
        target = message_id or self._last_inbound_by_chat.get(self._normalize_chat_key(chat_id))
        if not target:
            return {"success": False, "error": "no message to unreact — pass message_id"}
        if not await self._remove_reaction(chat_id, target):
            return {"success": False, "error": "unreact failed (see gateway debug log)"}
        return {"success": True, "message_id": target}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Tapback 👀 on the triggering message while the agent works."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._add_reaction(chat_id, message_id, "\U0001f440")

    # base.on_processing_complete swaps 👀 for 👍/👎 (remove-then-add keeps the sidecar's
    # reaction-handle slot coherent); CANCELLED leaves it unreacted.
    _OK_EMOJI = "\U0001f44d"
    _FAIL_EMOJI = "\U0001f44e"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Photon's ``space.id`` is opaque; with only the id, infer conservatively."""
        return {"name": chat_id, "type": "dm", "id": chat_id}

    def format_message(self, content: str) -> str:
        # Markdown passes through verbatim (sidecar markdown() builder); PHOTON_MARKDOWN=false strips.
        return content if _markdown_enabled() else strip_markdown(content)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        if BasePlatformAdapter._is_retryable_error(error):
            return True
        lowered = (error or "").lower()
        if not lowered or "retryable=false" in lowered or "auth_or_config" in lowered:
            return False
        return any(pat in lowered for pat in _PHOTON_RETRYABLE_PATTERNS)

    @staticmethod
    def _is_permanent_sidecar_failure(result: SendResult) -> bool:
        """``auth_or_config`` / ``target_not_allowed`` can't be fixed by retrying or by the
        plain-text resend — either would just double-send a doomed request.

        See #50971.
        """
        raw = result.raw_response
        return (isinstance(raw, dict) and raw.get("retryable") is False
                and raw.get("error_class") in ("auth_or_config", "target_not_allowed"))

    async def _send_with_retry(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                               metadata: Any = None, max_retries: int = 1, base_delay: float = 2.0) -> SendResult:
        """Retry sends without the generic Markdown banner (replies are markdown or
        already-stripped plain text, so it never applies)."""
        text = self.format_message(content)

        async def _send() -> SendResult:
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

        result = await _send()
        if result.success:
            return result
        if self._is_permanent_sidecar_failure(result):
            return result  # structured failure already carries the user-facing explanation
        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)
        if not is_network and self._is_timeout_error(error_str):
            return result
        if is_network:
            for attempt in range(1, max_retries + 1):
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("[photon] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                               attempt, max_retries, delay, error_str)
                await asyncio.sleep(delay)
                result = await _send()
                if result.success:
                    return result
                error_str = result.error or ""
                if self._is_permanent_sidecar_failure(result):
                    return result
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break
            else:
                logger.error("[photon] Failed to deliver response after %d retries: %s", max_retries, error_str)
                # Fall through to plain text; for URL-only responses this bypasses richlink()
                # so a rich-link outage doesn't strand a sendable URL.
        logger.warning("[photon] Send failed: %s - retrying plain-text message", error_str)
        fallback_result = await self._sidecar_send(
            chat_id, text[: self.MAX_MESSAGE_LENGTH], richlink=False, markdown=False)
        if not fallback_result.success:
            logger.error("[photon] Plain-text retry also failed: %s", fallback_result.error)
        return fallback_result

    async def _post_send(self, path: str, body: Dict[str, Any], *, structured: bool = False) -> SendResult:
        """POST a send-like body and wrap the outcome as a SendResult. ``structured`` carries
        a ``PhotonSidecarError``'s class/retryability so ``_send_with_retry`` can recognise
        permanent failures."""
        try:
            data = await self._sidecar_call(path, body)
        except PhotonSidecarError as e:
            if structured:
                return SendResult(success=False, error=str(e), retryable=e.retryable,
                                  raw_response={"error_class": e.error_class, "retryable": e.retryable})
            return SendResult(success=False, error=str(e))
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send(self, space_id: str, text: str, *, richlink: bool = True,
                            markdown: bool = True) -> SendResult:
        rich_url = _richlink_candidate(text) if richlink else None
        if rich_url:
            rich_result = await self._post_send("/send-richlink", {"spaceId": space_id, "url": rich_url})
            if rich_result.success:
                return rich_result
            logger.warning("[photon] rich-link send failed, falling back to plain text: %s", rich_result.error)
            markdown = False
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning("[photon] truncating outbound from %d to %d chars", len(text), self.MAX_MESSAGE_LENGTH)
            text = text[: self.MAX_MESSAGE_LENGTH]
        body: Dict[str, Any] = {"spaceId": space_id, "text": text}
        if markdown and _markdown_enabled():  # key omitted when disabled: pre-`format` sidecars still accept
            body["format"] = "markdown"
        return await self._post_send("/send", body, structured=True)

    async def _sidecar_send_poll(self, space_id: str, title: str, options: list) -> SendResult:
        """POST a native poll to ``/send-poll`` (degrades to a numbered list elsewhere)."""
        opts = [str(o).strip() for o in (options or []) if str(o).strip()]
        if not title or not title.strip():
            return SendResult(success=False, error="poll title is required")
        if len(opts) < 2:
            return SendResult(success=False, error="poll needs at least two options")
        body = {"spaceId": space_id, "title": title.strip()[: self.MAX_MESSAGE_LENGTH], "options": opts}
        return await self._post_send("/send-poll", body)

    async def _sidecar_send_attachment(self, space_id: str, path: str, *, name: Optional[str] = None,
                                       mime_type: Optional[str] = None, caption: Optional[str] = None,
                                       kind: str = "attachment") -> SendResult:
        """POST a local file to ``/send-attachment``. ``kind="voice"`` sends audio as a voice
        note (downgrades to a plain audio attachment where unsupported)."""
        safe_path = self.validate_media_delivery_path(str(path))  # send_*_file / cron may pass arbitrary strings
        if not safe_path:
            return SendResult(success=False, error=f"unsafe or missing attachment path: {path}")
        body = _attachment_body(
            space_id, safe_path, kind=kind, name=name, mime_type=mime_type or _guess_mime(safe_path), caption=caption)
        return await self._post_send("/send-attachment", body, structured=True)

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._http_client is None:
            raise RuntimeError("Photon adapter not connected")
        # Fresh client per call so this is safe from a worker thread with its own loop
        # (send_message_tool via _run_async); the inbound loop keeps using _http_client.
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.post(self._sidecar_url(path), json=body, headers=self._sidecar_headers())
        if resp.status_code != 200:
            raise _sidecar_error_from_response(path, resp.status_code, resp.text)
        data = resp.json() or {}
        if not data.get("ok"):
            raise _sidecar_error_from_response(path, resp.status_code, resp.text, data)
        return data


# -- Inbound media helpers -------------------------------------------------------

def _attachment_message_type(mime: str) -> MessageType:
    mime = (mime or "").lower()
    for prefix, mtype in (("image/", MessageType.PHOTO), ("video/", MessageType.VIDEO), ("audio/", MessageType.AUDIO)):
        if mime.startswith(prefix):
            return mtype
    return MessageType.DOCUMENT


# MIME → extension maps for cached inbound bytes (mirror BlueBubbles naming).
_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
    "image/heic": ".jpg", "image/heif": ".jpg", "image/tiff": ".jpg"}
_AUDIO_EXT_BY_MIME = {
    "audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/x-caf": ".caf", "audio/mp4": ".m4a", "audio/aac": ".m4a"}


def _cache_inbound_attachment(content: Dict[str, Any], name: str, mime: str, *,
                              force_audio: bool = False) -> Optional[str]:
    """Decode base64-inlined ``content["data"]`` into the shared media cache by MIME; None
    when there are no bytes (over the inline cap) or caching fails → marker."""
    if not content.get("data"):
        return None
    try:
        raw = base64.b64decode(content["data"])
    except (ValueError, TypeError) as exc:
        logger.warning("[photon] failed to decode inbound attachment bytes: %s", exc)
        return None
    from gateway.platforms.base import cache_audio_from_bytes, cache_document_from_bytes, cache_image_from_bytes
    mime = (mime or "").lower()
    suffix = Path(name).suffix if name else ""  # prefer the real extension
    try:
        if mime.startswith("image/"):
            ext = suffix or _IMAGE_EXT_BY_MIME.get(mime, ".jpg")
            try:
                return cache_image_from_bytes(raw, ext)
            except ValueError:  # unsupported image bytes (e.g. HEIC magic): deliver as a document
                return cache_document_from_bytes(raw, name)
        if force_audio or mime.startswith("audio/"):
            ext = suffix or _AUDIO_EXT_BY_MIME.get(mime, ".m4a" if force_audio else ".mp3")
            return cache_audio_from_bytes(raw, ext)
        return cache_document_from_bytes(raw, name)  # video, application/*, everything else
    except Exception as exc:
        logger.warning("[photon] failed to cache inbound attachment %s: %s", name, exc)
        return None


# -- Standalone (out-of-process) send for cron deliveries when the gateway is not
# co-resident. Reuses a live sidecar (cron processes cannot spawn one). -----------

def _standalone_error(resp: Any) -> Dict[str, Any]:
    """Structured error dict for a failed standalone call (mirrors
    ``_sidecar_error_from_response``, incl. the canonical target_not_allowed text)."""
    data: Any = {}
    with contextlib.suppress(Exception):
        data = resp.json() or {}
    data = data if isinstance(data, dict) else {}
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


def _standalone_token_from_record(port: int) -> Tuple[Optional[str], int, str]:
    """``(token, port, error)`` from the runtime record the gateway persists once the
    sidecar passes /healthz — the token otherwise exists only in the gateway env."""
    # See #69960.
    record = _read_runtime_record()
    stale_hint = ""
    if record and record.get("token"):
        if _sidecar_pid_alive(record.get("pid")):
            return str(record["token"]), _coerce_port(record.get("port"), port), ""
        stale_hint = (f" A stale sidecar runtime record was found (pid {record.get('pid')} is not running)"
                      " — the gateway appears to be down.")
    return None, port, (
        "Photon standalone send requires a running sidecar. Start the Hermes gateway (which spawns "
        f"the sidecar and records its address under <hermes-home>/runtime/{_RUNTIME_RECORD_NAME}), "
        "or set PHOTON_SIDECAR_TOKEN in this process's environment." + stale_hint)


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
    pconfig: PlatformConfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None,  # noqa: ARG001 — Spectrum has no threads yet
    media_files: Optional[list] = None,
    force_document: bool = False,  # noqa: ARG001 — iMessage auto-detects file kind
) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not installed"}
    port = _coerce_port(
        (pconfig.extra or {}).get("sidecar_port") or _get_scoped_secret("PHOTON_SIDECAR_PORT"), _DEFAULT_SIDECAR_PORT)
    token = _get_scoped_secret("PHOTON_SIDECAR_TOKEN")
    if not token:
        token, port, error = _standalone_token_from_record(port)
        if not token:
            return {"error": error}
    base = f"http://{_DEFAULT_SIDECAR_BIND}:{port}"
    headers = {"X-Hermes-Sidecar-Token": token}
    last_message_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            async def _post(path: str, body: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
                """(response, data-if-ok-else-None)."""
                resp = await client.post(f"{base}{path}", json=body, headers=headers)
                if resp.status_code != 200:
                    return resp, None
                data = resp.json() or {}
                return resp, (data if data.get("ok") else None)
            if message:  # 1. text body first, so it leads the conversation
                rich_url = _richlink_candidate(message)
                data = None
                if rich_url:
                    _resp, data = await _post("/send-richlink", {"spaceId": chat_id, "url": rich_url})
                if not data:  # no URL-only message, or the rich-link send failed: plain text
                    send_body: Dict[str, Any] = {"spaceId": chat_id, "text": message[:_MAX_MESSAGE_LENGTH]}
                    if _markdown_enabled() and not rich_url:
                        send_body["format"] = "markdown"
                    data, error = await _standalone_post_with_retry(
                        client, f"{base}/send", send_body, headers,
                    )
                    if error is not None:
                        return error
                last_message_id = data.get("messageId")
            # 2. Each attachment as a separate /send-attachment call; media_files is
            #    List[Tuple[path, is_voice]] (filter_media_delivery_paths).
            for media_path, is_voice in media_files or []:
                safe_path = BasePlatformAdapter.validate_media_delivery_path(str(media_path))
                if not safe_path:
                    logger.warning("[photon] standalone send skipping unsafe path")
                    continue
                att_body = _attachment_body(
                    chat_id, safe_path, kind="voice" if is_voice else "attachment", mime_type=_guess_mime(safe_path))
                data, error = await _standalone_post_with_retry(
                    client, f"{base}/send-attachment", att_body, headers,
                )
                if error is not None:
                    return error
                last_message_id = data.get("messageId") or last_message_id
        return {"success": True, "message_id": last_message_id}
    except Exception as e:
        return {"error": f"Photon standalone send failed: {e}"}


# -- Plugin entry point ----------------------------------------------------------

def register(ctx) -> None:
    """Called by the Hermes plugin loader at startup."""
    from . import cli as _cli  # local: avoid argparse work at module load
    ctx.register_platform(
        name="photon", label="iMessage via Photon", adapter_factory=lambda cfg: PhotonAdapter(cfg),
        check_fn=check_requirements, validate_config=validate_config, is_connected=is_connected,
        required_env=["PHOTON_PROJECT_ID", "PHOTON_PROJECT_SECRET"],
        install_hint=(
            "Run: hermes photon setup  (logs in via device flow, creates a "
            "Spectrum project, links your phone number, installs the "
            "spectrum-ts sidecar)."),
        setup_fn=_cli.gateway_setup,  # surfaces Photon in the unified `hermes gateway setup` wizard
        env_enablement_fn=_env_enablement, cron_deliver_env_var="PHOTON_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, allowed_users_env="PHOTON_ALLOWED_USERS",
        allow_all_env="PHOTON_ALLOW_ALL_USERS", max_message_length=_MAX_MESSAGE_LENGTH, emoji="📱",
        pii_safe=True,  # E.164 phone numbers: redact session descriptions before they reach the LLM
        allow_update_command=True,
        platform_hint=(
            "You are communicating via Photon Spectrum (iMessage). "
            "Treat replies like regular text messages — short and friendly. "
            "Markdown is rendered (bold, italics, lists, code), but keep "
            "formatting light and conversational. Recipient identifiers are "
            "E.164 phone numbers; never expose them in responses unless the "
            "user asked. Attachments arrive as metadata only."))
    ctx.register_cli_command(
        name="photon", help="Set up and manage the Photon iMessage integration",
        setup_fn=_cli.register_cli, handler_fn=_cli.dispatch)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'ProcessingOutcome': ('gateway.platforms.base', 'ProcessingOutcome'),
    'resolve_sidecar_dir': ('plugins.platforms.photon.sidecar_paths', 'resolve_sidecar_dir'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
