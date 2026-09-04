"""WhatsApp platform adapter (Baileys bridge): a Node.js bridge process runs the WhatsApp Web
client; messages are polled over a local HTTP API and responses are posted back through it."""

import asyncio
import logging
import os
import platform
import re
import signal
import subprocess
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any

from gateway.platforms._shared import get_scoped_secret
from gateway.native_document_guard import mark_native_document_guard
from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
from hermes_constants import (find_node_executable, get_hermes_dir, with_hermes_node_path)

_IS_WINDOWS = platform.system() == "Windows"


def _wenv(name: str, default: str = "") -> str:
    """WHATSAPP_* env via the profile secret scope (multiplexed profiles see their own .env, not the process-global value)."""
    return get_scoped_secret(name, default)

logger = logging.getLogger(__name__)

# Owner-typed inbound text is prefixed at MessageEvent construction so transcripts stay disambiguated before silent_ingest.
_OWNER_REPLY_PREFIX = "[owner reply] "

_RUN_TEXT = dict(capture_output=True, text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL)


def _listener_pids_on_port(port: int) -> list:
    """PIDs *listening* on ``port`` (POSIX), never clients — a bare ``lsof -i :PORT`` once killed the user's browser."""
    pids: list = []
    with suppress(FileNotFoundError):  # lsof not installed — fall through to ss
        pids = _safe_ints(subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], timeout=5, **_RUN_TEXT).stdout.strip().splitlines())
        if pids:
            return pids
    with suppress(FileNotFoundError):
        pids.extend(int(m.group(1)) for m in re.finditer(r"pid=(\d+)", subprocess.run(["ss", "-ltnHp", f"sport = :{port}"], timeout=5, **_RUN_TEXT).stdout))
    return pids


def _safe_ints(tokens) -> list:
    out: list = []
    for tok in tokens:
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def _windows_listener_pids(port: int) -> list:
    """PIDs in LISTENING state on ``port`` via netstat (Windows)."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    result = subprocess.run(["netstat", "-ano", "-p", "TCP"], timeout=5, creationflags=windows_hide_flags(), **_RUN_TEXT)
    rows = (line.split() for line in result.stdout.splitlines())
    return _safe_ints(p[4] for p in rows if len(p) >= 5 and p[3] == "LISTENING" and p[1].endswith(f":{port}"))


def _pid_looks_like_node_bridge(pid: int) -> bool:
    """Fail-closed: the live process must be a ``node`` executable (a scan-time PID can be a stranger by kill time).

    ``_kill_port_process`` discovers PIDs from a netstat/lsof scan of a TCP port — a bare number naming a
    *stranger* process (#89614 class: an unverified scan-time PID force-killed later can be anything,
    including a critical system process). Before any kill, require the live process to actually look like
    our Baileys bridge: a ``node`` executable. Any ambiguity (process gone, unreadable cmdline) refuses the
    kill.
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        return "node" in (proc.name() or "").lower() or "node" in " ".join(proc.cmdline() or []).lower().split(" ", 1)[0]
    except Exception:
        return False


def _kill_port_process(port: int) -> None:
    """Kill any node bridge *listening* on the given TCP port (never a client); SIGTERM on POSIX, taskkill /F on Windows."""
    with suppress(Exception):
        for pid in (_windows_listener_pids(port) if _IS_WINDOWS else _listener_pids_on_port(port)):
            # Killing a mistyped or recycled PID is unrecoverable — verify first.
            if pid <= 0 or not _pid_looks_like_node_bridge(pid):
                logger.warning("[whatsapp] Not killing PID %s on port %d: process is not a node bridge (or identity unverifiable)", pid, port)
                continue
            if _IS_WINDOWS:
                from hermes_cli._subprocess_compat import windows_hide_flags
                # Only SubprocessError is swallowed per-PID; an OSError (e.g. taskkill missing) aborts the scan.
                with suppress(subprocess.SubprocessError):
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, stdin=subprocess.DEVNULL, timeout=5, creationflags=windows_hide_flags())
            else:
                with suppress(OSError):  # ProcessLookupError/PermissionError are OSError subclasses
                    os.kill(pid, signal.SIGTERM)


def _bridge_pid_is_ours(pid: int, session_path: Path, expected_start) -> bool:
    """``pid`` alive AND still our bridge: kernel start time (definitive), else legacy ``node`` + session path in cmdline."""
    from gateway import status
    if not status._pid_exists(pid):
        return False
    if expected_start is not None:
        return status.get_process_start_time(pid) == expected_start
    cmdline = status._read_process_cmdline(pid)
    return bool(cmdline) and ("node" in cmdline) and (str(session_path) in cmdline)


def _unlink_quietly(path: Path) -> None:
    with suppress(OSError):
        path.unlink()


def _kill_stale_bridge_by_pidfile(session_path: Path) -> None:
    """Kill an orphaned bridge recorded in ``bridge.pid``, after :func:`_bridge_pid_is_ours`."""
    from gateway.status import _pid_exists
    pid_file = session_path / "bridge.pid"
    if not pid_file.exists():
        return
    try:  # Line 1 = pid, optional line 2 = kernel start time (legacy files: pid only).
        lines = [ln.strip() for ln in pid_file.read_text(encoding="utf-8").split("\n")]
        pid = int(lines[0])
        recorded_start = int(lines[1]) if len(lines) > 1 and lines[1] else None
    except (ValueError, OSError, TypeError, IndexError):
        _unlink_quietly(pid_file)
        return
    if _bridge_pid_is_ours(pid, session_path, recorded_start):
        with suppress(OSError):  # ProcessLookupError / PermissionError included
            os.kill(pid, signal.SIGTERM)
            logger.info("[whatsapp] Killed stale bridge PID %d from pidfile", pid)
    elif _pid_exists(pid):
        logger.warning("[whatsapp] Not killing pidfile PID %d: it is no longer the bridge (recycled onto an unrelated process); "
                       "skipping to avoid killing a stranger.", pid)
    _unlink_quietly(pid_file)


def _write_bridge_pidfile(session_path: Path, pid: int) -> None:
    """Write the bridge PID plus its kernel start time (line 2) for identity-checked cleanup."""
    with suppress(OSError):
        from gateway.status import get_process_start_time
        start = get_process_start_time(pid)
        (session_path / "bridge.pid").write_text(str(pid) if start is None else f"{pid}\n{start}", encoding="utf-8")


def _terminate_bridge_process(proc, *, force: bool = False) -> None:
    """Terminate the bridge process using process-tree semantics where possible."""
    action = "kill" if force else "terminate"
    if _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T"] + (["/F"] if force else []),
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
            )
        except FileNotFoundError:
            return getattr(proc, action)()
        if result.returncode != 0:
            raise OSError((result.stderr or result.stdout or "").strip() or f"taskkill failed for PID {proc.pid}")
        return
    import psutil
    with suppress(psutil.NoSuchProcess):
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            with suppress(psutil.NoSuchProcess):
                getattr(child, action)()
        getattr(parent, action)()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.whatsapp_identity import to_whatsapp_jid
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult, SUPPORTED_DOCUMENT_TYPES, cache_image_from_url, cache_audio_from_url,
)
from utils import env_int


def _cache_dirs() -> tuple:
    """``(image, audio, video, document)`` cache dirs, resolved per call so a profile override's cache matches."""
    from gateway.platforms.base import get_audio_cache_dir, get_document_cache_dir, get_image_cache_dir, get_video_cache_dir
    return get_image_cache_dir(), get_audio_cache_dir(), get_video_cache_dir(), get_document_cache_dir()


def _is_allowed_bridge_path(url: str) -> bool:
    """Absolute bridge path resolves (symlinks included) inside a Hermes cache dir — a rogue bridge could hand back /etc/passwd."""
    try:
        resolved = Path(url).resolve()
    except (OSError, ValueError):
        return False
    for root in _cache_dirs():
        with suppress(OSError, ValueError):
            if resolved.is_relative_to(Path(root).resolve()):
                return True
    return False


def _file_content_hash(path: Path) -> str:
    """First 16 hex chars of SHA-256 of *path* ("" if unreadable); bridge.js reports its own as ``/health`` ``scriptHash``."""
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def check_whatsapp_requirements() -> bool:
    """Node.js (Hermes-managed first, so a bad system Node on PATH can't break Windows) is available."""
    _node = find_node_executable("node")
    try:
        return bool(_node) and subprocess.run([_node, "--version"], timeout=5, **_RUN_TEXT).returncode == 0
    except Exception:
        return False


# Env vars bridge.js consumes; injected because a multiplexed subprocess's os.environ lacks the secondary profile's .env.
_BRIDGE_PASSTHROUGH_ENV = (
    "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_FROM", "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY",
    "WHATSAPP_GROUP_ALLOWED_USERS", "WHATSAPP_GROUP_ALLOW_FROM", "WHATSAPP_REQUIRE_MENTION",
    "WHATSAPP_MENTION_PATTERNS", "WHATSAPP_FREE_RESPONSE_CHATS", "WHATSAPP_DEBUG",
    "WHATSAPP_FORWARD_OWNER_MESSAGES", "WHATSAPP_REPLY_PREFIX", "WHATSAPP_MAX_MESSAGE_LENGTH",
    "WHATSAPP_CHUNK_DELAY_MS", "WHATSAPP_SEND_TIMEOUT_MS",
)
_TEXT_INJECT_EXTS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css"}
_MAX_TEXT_INJECT_BYTES = 100 * 1024  # matches Telegram/Discord/Slack
_NATIVE_MEDIA_TYPES = {"location": MessageType.LOCATION, "live_location": MessageType.LOCATION, "sticker": MessageType.STICKER}
# Inbound mediaType substring → kind; ptt = WhatsApp voice note, so "ptt" must precede "audio".
_MEDIA_NEEDLES = (("image", MessageType.PHOTO), ("video", MessageType.VIDEO), ("ptt", MessageType.VOICE), ("audio", MessageType.AUDIO))
# MessageType → (bridge label, default mime); documents take their mime from SUPPORTED_DOCUMENT_TYPES instead.
_MEDIA_INFO = {
    MessageType.PHOTO: ("image", "image/jpeg"), MessageType.VOICE: ("audio", "audio/ogg"), MessageType.AUDIO: ("audio", "audio/mpeg"),
    MessageType.VIDEO: ("video", "video/mp4"), MessageType.DOCUMENT: ("document", ""),
}


def _needs_bridge(method):
    """Adapter-method decorator: return ``SendResult(error=...)`` when ``_bridge_unavailable()`` says so."""
    @wraps(method)
    async def _guarded(self, *args, **kwargs):
        unavailable = await self._bridge_unavailable()
        return SendResult(success=False, error=unavailable) if unavailable else await method(self, *args, **kwargs)
    return _guarded


class WhatsAppAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """Transport over a local Node.js (Baileys) HTTP bridge; behavior lives in ``WhatsAppBehaviorMixin``. config.extra: bridge_script /
    bridge_port (3000) / session_path, dm_policy / group_policy (open|allowlist|disabled|pairing), allow_from / group_allow_from, send_read_receipts."""

    _DEFAULT_BRIDGE_DIR = None  # resolved in __init__
    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP)
        if WhatsAppAdapter._DEFAULT_BRIDGE_DIR is None:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            WhatsAppAdapter._DEFAULT_BRIDGE_DIR = resolve_whatsapp_bridge_dir()
        extra = config.extra
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_port: int = extra.get("bridge_port", 3000)
        self._bridge_script: str = extra.get("bridge_script", str(self._DEFAULT_BRIDGE_DIR / "bridge.js"))
        self._session_path = Path(extra.get("session_path", get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")))
        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        self._dm_policy = str(extra.get("dm_policy") or _wenv("WHATSAPP_DM_POLICY", "pairing")).strip().lower()
        self._allow_from = self._coerce_allow_list(self._select_dm_allowlist(extra, ("WHATSAPP_ALLOWED_USERS",), _wenv))
        self._group_policy = str(extra.get("group_policy") or _wenv("WHATSAPP_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = self._coerce_allow_list(extra.get("group_allow_from") or extra.get("groupAllowFrom"))
        rr = extra.get("send_read_receipts", False)
        self._send_read_receipts = rr if isinstance(rr, bool) else str(rr or "").strip().lower() in {"1", "true", "yes", "on"}
        self._mention_patterns = self._compile_mention_patterns()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._bridge_log_fh = self._bridge_log = self._poll_task = self._http_session = None
        # Set by disconnect() before SIGTERMing so _check_managed_bridge_exit() can tell an intentional exit (-15/-2/0) from a crash.
        self._shutting_down = False
        # Text debounce batching: rapid bursts (forwards, paste-splits) would otherwise each trigger a separate agent turn.
        self._text_batch_delay_seconds = self._coerce_float_extra("text_batch_delay_seconds", 5.0)
        self._text_batch_split_delay_seconds = self._coerce_float_extra("text_batch_split_delay_seconds", 10.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

    def _coerce_float_extra(self, key: str, default: float) -> float:
        """Read a float from ``config.extra``; NaN/Inf/negative/unparseable → ``default`` (fed to asyncio.sleep)."""
        import math
        try:  # float(None) → TypeError → default
            parsed = float(self.config.extra.get(key) if getattr(self.config, "extra", None) else None)
        except (TypeError, ValueError):
            return float(default)
        return parsed if math.isfinite(parsed) and parsed >= 0 else float(default)

    def _bridge_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._bridge_port}/{path}"

    def _bridge_req(self, method: str, path: str, timeout: float, **kwargs):
        """``session.<method>`` context manager for a bridge endpoint (caller must ``async with``)."""
        import aiohttp
        return getattr(self._http_session, method)(self._bridge_url(path), **kwargs, timeout=aiohttp.ClientTimeout(total=timeout))

    async def _probe_bridge_health(self) -> tuple[bool, Any]:
        """GET /health with a fresh session → ``(http_200, json)``; unparseable 200 body → ``(True, None)``; connection errors propagate."""
        import aiohttp
        async with aiohttp.ClientSession() as session, session.get(self._bridge_url("health"), timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status != 200:
                return False, None
            try:
                return True, await resp.json()
            except Exception:
                return True, None

    def _ensure_bridge_deps(self, bridge_dir: Path) -> bool:
        """npm install when node_modules is missing OR package.json hash != stamp file. False = fatal error set."""
        _dep_stamp = bridge_dir / "node_modules" / ".hermes-pkg-hash"  # holds the package.json hash of the last install
        _pkg_hash = _file_content_hash(bridge_dir / "package.json")
        try:
            if (bridge_dir / "node_modules").exists() and _dep_stamp.read_text(encoding="utf-8").strip() == _pkg_hash and bool(_pkg_hash):
                return True
        except OSError:
            pass
        print(f"[{self.name}] Installing WhatsApp bridge dependencies...")
        # Hermes-managed portable Node's npm.cmd first (Windows), then PATH.
        _npm_bin = find_node_executable("npm") or "npm"
        detail = ""
        try:  # Default 300s accommodates slow systems like an Unraid NAS.
            install_result = subprocess.run([_npm_bin, "install", "--silent"], cwd=str(bridge_dir), timeout=env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300),
                                            env=with_hermes_node_path(), **_RUN_TEXT)
            if install_result.returncode == 0:
                print(f"[{self.name}] Dependencies installed")
                with suppress(OSError):  # Stamp is an optimization; install still succeeded
                    if _pkg_hash:
                        _dep_stamp.write_text(_pkg_hash, encoding="utf-8")
                return True
            print(f"[{self.name}] npm install failed: {install_result.stderr}")
        except Exception as e:
            print(f"[{self.name}] Failed to install dependencies: {e}")
            detail = f" ({e})"
        self._set_fatal_error("whatsapp_npm_install_failed", f"WhatsApp bridge npm install failed{detail}. Run `cd {bridge_dir} && {_npm_bin} install` "
                              "manually, then restart `hermes gateway`.", retryable=False)
        return False

    def _attach_to_bridge(self, managed_process) -> None:
        import aiohttp
        self._bridge_process = managed_process
        self._http_session = aiohttp.ClientSession()
        self._poll_task = asyncio.create_task(self._poll_messages())

    async def _reuse_running_bridge(self, bridge_path: Path) -> bool:
        """Adopt a connected bridge serving the on-disk bridge.js + same read-receipt config; else say why it restarts."""
        try:
            ok, data = await self._probe_bridge_health()
            if not ok or data is None:
                return False
            bridge_status = data.get("status", "unknown")
            if bridge_status != "connected":
                print(f"[{self.name}] Bridge found but not connected (status: {bridge_status}), restarting")
                return False
            running_hash, disk_hash = data.get("scriptHash", ""), _file_content_hash(bridge_path)
            if running_hash and disk_hash and running_hash == disk_hash and bool(data.get("sendReadReceipts", False)) == self._send_read_receipts:
                print(f"[{self.name}] Using existing bridge (status: {bridge_status})")
                self._mark_connected()
                self._attach_to_bridge(None)  # Not managed by us
                self._wire_plugin_handlers(None)
                return True
            stale_reason = f"running={running_hash or 'unversioned'}, disk={disk_hash}" if running_hash != disk_hash else "send_read_receipts config changed"
            print(f"[{self.name}] Running bridge is stale ({stale_reason}), restarting")
        except Exception:
            pass  # Bridge not running, start a new one
        return False

    def _bridge_env(self) -> dict:
        """Subprocess env: profile-resolved WHATSAPP_* values + profile-aware cache dirs."""
        # with_hermes_node_path() copies os.environ when called with no arg.
        bridge_env = with_hermes_node_path()
        if self._reply_prefix is not None:
            bridge_env["WHATSAPP_REPLY_PREFIX"] = self._reply_prefix
        bridge_env["WHATSAPP_SEND_READ_RECEIPTS"] = "true" if self._send_read_receipts else "false"
        for _key, _v in [("WHATSAPP_MODE", _wenv("WHATSAPP_MODE", "self-chat"))] + [(k, _wenv(k)) for k in _BRIDGE_PASSTHROUGH_ENV]:
            if _v:
                bridge_env[_key] = _v
        # Without these the bridge hardcodes ~/.hermes/{image,audio,document}_cache (wrong under HERMES_HOME/profiles/cache layout).
        img_dir, audio_dir, _video_dir, doc_dir = _cache_dirs()
        bridge_env.update(HERMES_IMAGE_CACHE_DIR=str(img_dir), HERMES_AUDIO_CACHE_DIR=str(audio_dir), HERMES_DOCUMENT_CACHE_DIR=str(doc_dir))
        return bridge_env

    def _bridge_died(self, detail: str) -> bool:
        print(f"[{self.name}] {detail}")
        print(f"[{self.name}] Check log: {self._bridge_log}")
        self._close_bridge_log()
        return False

    async def _poll_bridge_health(self, died_msg: str) -> tuple[Optional[bool], bool, dict]:
        """Poll /health up to 15×1s → ``(connected, http_ready, data)``; connected False = process died (reported), None = timeout."""
        http_ready = False
        data: dict = {}
        for attempt in range(15):
            await asyncio.sleep(1)
            if self._bridge_process.poll() is not None:
                return self._bridge_died(died_msg.format(code=self._bridge_process.returncode)), http_ready, data
            try:
                ok, d = await self._probe_bridge_health()
                if ok:
                    http_ready = True
                    if d is not None:
                        data = d
                        if data.get("status") == "connected":
                            print(f"[{self.name}] Bridge ready (status: connected)")
                            return True, http_ready, data
            except Exception:
                continue
        return None, http_ready, data

    async def _wait_for_bridge(self) -> bool:
        """Phase 1: HTTP up (≤15s). Phase 2: ``status: connected`` (≤15s more; warns but proceeds if still connecting)."""
        connected, http_ready, data = await self._poll_bridge_health("Bridge process died (exit code {code})")
        if connected is False:
            return False
        if not http_ready:
            return self._bridge_died("Bridge HTTP server did not start in 15s")
        if data.get("status") != "connected":
            print(f"[{self.name}] Bridge HTTP ready, waiting for WhatsApp connection...")
            connected, _, _ = await self._poll_bridge_health("Bridge process died during connection")
            if connected is False:
                return False
            if connected is None:
                print(f"[{self.name}] ⚠ WhatsApp not connected after 30s")
                print(f"[{self.name}]   Bridge log: {self._bridge_log}")
                print(f"[{self.name}]   If session expired, re-pair: hermes whatsapp")
        return True

    def _preflight(self) -> bool:
        """Node + bridge script + creds.json present, else a non-retryable fatal error (an unpaired bridge only prints QR codes; retries would pay 30s each)."""
        bridge_path = Path(self._bridge_script)
        creds_path = self._session_path / "creds.json"
        checks = (
            (check_whatsapp_requirements, ("[%s] Node.js not found. WhatsApp requires Node.js.", self.name),
             "whatsapp_node_missing", "Node.js is not installed — install Node.js and re-run `hermes gateway`."),
            (bridge_path.exists, ("[%s] Bridge script not found: %s", self.name, bridge_path),
             "whatsapp_bridge_missing", f"WhatsApp bridge script missing at {bridge_path}."),
            (creds_path.exists, ("[%s] WhatsApp is enabled but not paired (no creds.json at %s). Pair from the dashboard or run "
                                 "`hermes whatsapp`; remove WHATSAPP_ENABLED from your .env to disable.", self.name, creds_path),
             "whatsapp_not_paired", "WhatsApp enabled but not paired — pair from the dashboard or run `hermes whatsapp`."),
        )
        for ok, warn_args, code, message in checks:
            if not ok():
                logger.warning(*warn_args)
                self._set_fatal_error(code, message, retryable=False)
                return False
        logger.info("[%s] Bridge found at %s", self.name, bridge_path)
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start (or adopt) the Node.js bridge and wait for it to be ready."""
        if not self._preflight():
            return False
        bridge_path = Path(self._bridge_script)
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('whatsapp-session', str(self._session_path), 'WhatsApp session'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("[%s] Could not acquire session lock (non-fatal): %s", self.name, e)
        try:
            if not self._ensure_bridge_deps(bridge_path.parent):
                return False
            self._session_path.mkdir(parents=True, exist_ok=True)
            if await self._reuse_running_bridge(bridge_path):
                return True
            _kill_stale_bridge_by_pidfile(self._session_path)
            _kill_port_process(self._bridge_port)
            await asyncio.sleep(1)
            # Bridge output goes to a log file so QR codes, errors, and reconnection messages survive for troubleshooting.
            self._bridge_log = self._session_path.parent / "bridge.log"
            self._bridge_log_fh = bridge_log_fh = open(self._bridge_log, "a", encoding="utf-8")
            self._bridge_process = subprocess.Popen(
                [find_node_executable("node") or "node", str(bridge_path), "--port", str(self._bridge_port), "--session", str(self._session_path),
                 "--mode", _wenv("WHATSAPP_MODE", "self-chat")], stdout=bridge_log_fh, stderr=bridge_log_fh, env=self._bridge_env(), **windows_detach_popen_kwargs())
            _write_bridge_pidfile(self._session_path, self._bridge_process.pid)
            if not await self._wait_for_bridge():
                return False
            self._attach_to_bridge(self._bridge_process)
            self._mark_connected()
            print(f"[{self.name}] Bridge started on port {self._bridge_port}")
            self._wire_plugin_handlers(None)
            return True
        except Exception as e:
            logger.error("[%s] Failed to start bridge: %s", self.name, e, exc_info=True)
            return False
        finally:
            if not self._running:
                if lock_acquired:
                    self._release_platform_lock()
                self._close_bridge_log()

    def _close_bridge_log(self) -> None:
        if self._bridge_log_fh:
            with suppress(Exception):
                self._bridge_log_fh.close()
            self._bridge_log_fh = None

    async def _check_managed_bridge_exit(self) -> Optional[str]:
        returncode = self._bridge_process.poll() if self._bridge_process is not None else None
        if returncode is None:
            return None
        # getattr-with-default: tests build the adapter via ``__new__`` without __init__.
        if getattr(self, "_shutting_down", False) and returncode in {0, -2, -15}:
            logger.info("[%s] Bridge exited during shutdown (code %d).", self.name, returncode)
            return None
        message = f"WhatsApp bridge process exited unexpectedly (code {returncode})."
        if not self.has_fatal_error:
            logger.error("[%s] %s", self.name, message)
            self._set_fatal_error("whatsapp_bridge_exited", message, retryable=True)
            self._close_bridge_log()
            await self._notify_fatal_error()
        return self.fatal_error_message or message

    def _terminate_bridge(self, *, force: bool) -> None:
        try:
            _terminate_bridge_process(self._bridge_process, force=force)
        except (ProcessLookupError, PermissionError):
            getattr(self._bridge_process, "kill" if force else "terminate")()

    async def disconnect(self) -> None:
        """Stop the WhatsApp bridge and clean up any orphaned processes."""
        self._shutting_down = True  # flip BEFORE signalling so send()/poll loop don't report the intentional exit as fatal
        if not self._bridge_process:
            print(f"[{self.name}] Disconnecting (external bridge left running)")
        else:
            try:
                self._terminate_bridge(force=False)
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    self._terminate_bridge(force=True)
            except Exception as e:
                print(f"[{self.name}] Error stopping bridge: {e}")
        _unlink_quietly(self._session_path / "bridge.pid")
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._poll_task
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._poll_task = self._http_session = self._bridge_process = None
        self._release_platform_lock()
        self._mark_disconnected()
        self._close_bridge_log()
        print(f"[{self.name}] Disconnected")

    async def _bridge_unavailable(self) -> Optional[str]:
        return "Not connected" if not self._running or not self._http_session else (await self._check_managed_bridge_exit() or None)

    async def _post_bridge_message(self, path: str, payload: Dict[str, Any], *, timeout: float) -> SendResult:
        """POST to the bridge; 200 → SendResult(messageId, raw_response), else the error text."""
        try:
            async with self._bridge_req("post", path, timeout, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(success=True, message_id=data.get("messageId"), raw_response=data)
                return SendResult(success=False, error=await resp.text())
        except Exception as e:
            return SendResult(success=False, error=str(e))

    @_needs_bridge
    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Format markdown for WhatsApp, chunk preserving code blocks, send sequentially."""
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        chat_id = to_whatsapp_jid(chat_id)
        try:
            chunks = self.truncate_message(self.format_message(content), self._outgoing_chunk_limit())
            sent_message_ids: list[str] = []
            last_message_id = None
            for idx, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {"chatId": chat_id, "message": chunk}
                if reply_to and idx == 0:
                    payload["replyTo"] = reply_to  # Reply-to on the first chunk only.
                result = await self._post_bridge_message("send", payload, timeout=30)
                if not result.success:
                    return SendResult(success=False, error=result.error)
                last_message_id = result.message_id
                if last_message_id:
                    sent_message_ids.append(str(last_message_id))
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)  # avoid rate limiting between chunks
            return SendResult(success=True, message_id=last_message_id, continuation_message_ids=tuple(sent_message_ids[:-1]),
                              raw_response={"message_ids": sent_message_ids})
        except Exception as e:
            return SendResult(success=False, error=str(e))

    @_needs_bridge
    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        try:
            async with self._bridge_req("post", "edit", 15, json={"chatId": to_whatsapp_jid(chat_id), "messageId": message_id, "message": content}) as resp:
                return SendResult(success=True, message_id=message_id) if resp.status == 200 else SendResult(success=False, error=await resp.text())
        except Exception as e:
            return SendResult(success=False, error=str(e))

    @_needs_bridge
    async def _send_media_to_bridge(self, chat_id: str, file_path: str, media_type: str, caption: Optional[str] = None, file_name: Optional[str] = None) -> SendResult:
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        payload: Dict[str, Any] = {"chatId": to_whatsapp_jid(chat_id), "filePath": file_path, "mediaType": media_type}
        payload.update({k: v for k, v in (("caption", caption), ("fileName", file_name)) if v})
        return await self._post_bridge_message("send-media", payload, timeout=120)

    @_needs_bridge
    async def send_poll(self, chat_id: str, question: str, options: list[str], *, selectable_count: int = 1) -> SendResult:
        """Native WhatsApp poll (low-level transport primitive; approval UX stays gateway-owned)."""
        payload: Dict[str, Any] = {"chatId": to_whatsapp_jid(chat_id), "question": question, "options": list(options or []), "selectableCount": selectable_count}
        return await self._post_bridge_message("send-poll", payload, timeout=30)

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str, session_key: str,
                           metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Multiple-choice clarify as a native poll (the pick arrives as message text for the normal intercept); else text prompt."""
        clean_choices = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
        if 2 <= len(clean_choices) <= 12:
            result = await self.send_poll(chat_id, str(question or "").strip(), clean_choices, selectable_count=1)
            if result.success:
                return result
            logger.warning("[%s] Native WhatsApp clarify poll failed; falling back to text: %s", self.name, result.error)
        return await super().send_clarify(chat_id=chat_id, question=question, choices=choices, clarify_id=clarify_id, session_key=session_key, metadata=metadata)

    @_needs_bridge
    async def send_location(self, chat_id: str, latitude: float, longitude: float, *, name: Optional[str] = None, address: Optional[str] = None,
                            reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        try:
            payload: Dict[str, Any] = {"chatId": to_whatsapp_jid(chat_id), "latitude": float(latitude), "longitude": float(longitude)}
        except Exception as e:
            return SendResult(success=False, error=str(e))
        payload.update({k: v for k, v in (("name", name), ("address", address)) if v})
        return await self._post_bridge_message("send-location", payload, timeout=30)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Download image URL to cache, send natively via bridge (``metadata`` honors the base contract)."""
        try:
            local_path = await cache_image_from_url(image_url)
            return await self._send_media_to_bridge(chat_id, local_path, "image", caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_to_bridge(chat_id, image_path, "image", caption)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_to_bridge(chat_id, video_path, "video", caption)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_to_bridge(chat_id, audio_path, "audio", caption)

    @mark_native_document_guard
    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
                            reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_to_bridge(chat_id, file_path, "document", caption, file_name or os.path.basename(file_path))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if await self._bridge_unavailable():
            return
        with suppress(Exception):
            import aiohttp
            # ``async with`` — a bare ``await session.post(...)`` leaves the response (and its CLOSE_WAIT socket) alive until GC.
            async with self._http_session.post(self._bridge_url("typing"), json={"chatId": to_whatsapp_jid(chat_id)}, timeout=aiohttp.ClientTimeout(total=5)):
                pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if not self._running or not self._http_session:
            return {"name": "Unknown", "type": "dm"}
        if not await self._check_managed_bridge_exit():
            try:
                async with self._bridge_req("get", f"chat/{to_whatsapp_jid(chat_id)}", 10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"name": data.get("name", chat_id), "type": "group" if data.get("isGroup") else "dm", "participants": data.get("participants", [])}
            except Exception as e:
                logger.debug("Could not get WhatsApp chat info for %s: %s", chat_id, e)
        return {"name": chat_id, "type": "dm"}

    async def _report_bridge_exit(self) -> bool:
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            print(f"[{self.name}] {bridge_exit}")
        return bool(bridge_exit)

    async def _poll_messages(self) -> None:
        while self._running:
            if not self._http_session or await self._report_bridge_exit():
                break
            try:
                async with self._bridge_req("get", "messages", 30) as resp:
                    if resp.status == 200:
                        for msg_data in await resp.json():
                            event = await self._build_message_event(msg_data)
                            if event:
                                # Fire-and-forget: a slow bridge /read must not delay dispatch.
                                asyncio.create_task(self._send_read_receipt(msg_data))
                                if event.message_type == MessageType.TEXT:
                                    self._enqueue_text_event(event)
                                else:
                                    await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if await self._report_bridge_exit():
                    break
                print(f"[{self.name}] Poll error: {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(1)  # Poll interval

    async def _send_read_receipt(self, data: Dict[str, Any]) -> None:
        key = data.get("readReceiptKey")
        if not self._send_read_receipts or not self._http_session or not isinstance(key, dict):
            return
        try:
            async with self._bridge_req("post", "read", 5, json={"key": key}) as resp:
                if resp.status != 200:
                    logger.warning("[%s] WhatsApp read receipt failed with HTTP %s", self.name, resp.status)
        except Exception as exc:
            logger.warning("[%s] WhatsApp read receipt failed: %s", self.name, exc)

    _SPLIT_THRESHOLD = 6000  # WhatsApp supports ~65K chars; generous threshold

    async def _flush_text_batch(self, key: str) -> None:
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            await asyncio.sleep(self._text_batch_split_delay_seconds if last_len >= self._SPLIT_THRESHOLD else self._text_batch_delay_seconds)
            event = self._pending_text_batches.pop(key, None)
            if event:
                await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    @staticmethod
    def _classify_bridge_message(data: Dict[str, Any]) -> MessageType:
        media_type = str(data.get("mediaType", "") or "")
        if media_type in _NATIVE_MEDIA_TYPES:
            return _NATIVE_MEDIA_TYPES[media_type]
        if not data.get("hasMedia"):
            return MessageType.TEXT
        return next((kind for needle, kind in _MEDIA_NEEDLES if needle in media_type), MessageType.DOCUMENT)

    async def _collect_bridge_media(self, data: Dict[str, Any], msg_type: MessageType) -> tuple[list, list]:
        """``mediaUrls`` → ``(cached_urls, media_types)``: remote image/audio cached locally; absolute paths only inside a cache dir."""
        accepted: list[tuple] = []  # (url_or_path, mime)
        label, default_mime = _MEDIA_INFO.get(msg_type, (None, ""))
        bridge_mime = str(data.get("mime") or "").strip()
        for url in data.get("mediaUrls", []):
            mime = bridge_mime or (SUPPORTED_DOCUMENT_TYPES.get(Path(url).suffix.lower(), "application/octet-stream") if msg_type == MessageType.DOCUMENT else default_mime)
            if url.startswith(("http://", "https://")) and msg_type in {MessageType.PHOTO, MessageType.VOICE, MessageType.AUDIO}:
                cacher, ext = (cache_image_from_url, ".jpg") if msg_type == MessageType.PHOTO else (cache_audio_from_url, ".ogg")
                try:
                    url = await cacher(url, ext=ext)
                    print(f"[{self.name}] Cached user {label}: {url}", flush=True)
                except Exception as e:
                    print(f"[{self.name}] Failed to cache {label}: {e}", flush=True)
                accepted.append((url, mime))
            elif label is not None and os.path.isabs(url):
                if _is_allowed_bridge_path(url):
                    accepted.append((url, mime))
                    print(f"[{self.name}] Using bridge-cached {label}: {url}", flush=True)
                else:
                    print(f"[{self.name}] Rejected bridge {label} path outside cache dir: {url}", flush=True)
            else:
                accepted.append((url, "unknown"))
        return [u for u, _ in accepted], [m for _, m in accepted]

    def _inject_document_text(self, cached_urls: list, body: str) -> str:
        """Prepend text-readable document contents (≤100KB) so the agent reads them inline."""
        for doc_path in cached_urls:
            p = Path(doc_path)
            if p.suffix.lower() not in _TEXT_INJECT_EXTS:
                continue
            try:
                file_size = p.stat().st_size
                if file_size > _MAX_TEXT_INJECT_BYTES:
                    print(f"[{self.name}] Skipping text injection for {doc_path} ({file_size} bytes > {_MAX_TEXT_INJECT_BYTES})", flush=True)
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                parts = p.name.split("_", 2)  # strip the doc_<hex>_ prefix for display
                injection = f"[Content of {parts[2] if len(parts) >= 3 else p.name}]:\n{content}"
                body = f"{injection}\n\n{body}" if body else injection
                print(f"[{self.name}] Injected text content from: {doc_path}", flush=True)
            except Exception as e:
                print(f"[{self.name}] Failed to read document text: {e}", flush=True)
        return body

    async def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Build a MessageEvent from bridge message data, downloading images to cache."""
        try:
            if not self._should_process_message(data):
                return None
            msg_type = self._classify_bridge_message(data)
            raw_is_group = data.get("isGroup")
            is_group = raw_is_group is True
            source = self.build_source(
                chat_id=data.get("chatId", ""), chat_name=data.get("chatName"), chat_type="group" if is_group else "dm",
                user_id=data.get("senderId"), user_name=data.get("senderName"),
                is_bot=data.get("fromMe") is True and data.get("fromOwner") is not True)
            source.is_one_to_one = raw_is_group is False
            source.message_is_edit = bool(
                data.get("isEdited") is True
                or str(data.get("nativeType") or "").casefold() in {"editedmessage", "protocolmessage:message_edit"}
            )
            cached_urls, media_types = await self._collect_bridge_media(data, msg_type)
            body = data.get("body", "")
            if data.get("isGroup"):
                body = self._clean_bot_mention_text(body, data)
            if msg_type == MessageType.VOICE and cached_urls and str(body).strip().lower() == "[ptt received]":
                body = ""  # Bridge placeholder for captionless voice notes; the audio is the payload.
            # Quoted message stays in structured fields only — GatewayRunner renders the "[Replying to: ...]" pointer.
            quoted = bool(data.get("hasQuotedMessage"))
            raw_reply_id = data.get("quotedMessageId") if quoted else None
            if msg_type == MessageType.DOCUMENT and cached_urls:
                body = self._inject_document_text(cached_urls, body)
            native_metadata = data.get("nativeMetadata")
            metadata: Dict[str, Any] = {k: v for k, v in (
                ("whatsapp_native_type", str(data.get("nativeType") or "").strip()),
                ("whatsapp_native", native_metadata if isinstance(native_metadata, dict) else None),
            ) if v}
            if source.message_is_edit:
                metadata["message_is_edit"] = True
            # ``fromOwner`` = owner-typed inbound fromMe (gated by WHATSAPP_FORWARD_OWNER_MESSAGES at the bridge); surfaced as
            # metadata AND a text prefix so the marker survives downstream failures before silent_ingest.
            if data.get("fromOwner"):
                metadata["whatsapp_from_owner"] = True
                if not body.startswith(_OWNER_REPLY_PREFIX):
                    body = f"{_OWNER_REPLY_PREFIX}{body}"
            return MessageEvent(
                text=body, message_type=msg_type, source=source, raw_message=data, message_id=data.get("messageId"),
                media_urls=cached_urls, media_types=media_types, metadata=metadata,
                reply_to_message_id=str(raw_reply_id) if raw_reply_id is not None else None,
                reply_to_text=str(data.get("quotedText") or "").strip() or None,
                reply_to_author_id=(self._normalize_whatsapp_id(data.get("quotedParticipant")) or None) if quoted else None,
                reply_to_is_own_message=self._message_is_reply_to_bot(data) if quoted else False,
            )
        except Exception as e:
            print(f"[{self.name}] Error building event: {e}")
            return None


# ── Plugin glue: register(ctx) plus the hooks for gateway/run.py, gateway/config.py, hermes_cli/gateway.py, send_message_tool.py.

_WA_EXT_MEDIA_TYPE = {
    **dict.fromkeys((".jpg", ".jpeg", ".png", ".webp", ".gif"), "image"),
    **dict.fromkeys((".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"), "video"),
    **dict.fromkeys((".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"), "audio"),
}


def _bridge_media_type(file_path: str, is_voice: bool, force_document: bool) -> str:
    """Local file → bridge ``mediaType`` (image|video|audio|document); ``force_document`` = the [[as_document]] directive."""
    return "document" if force_document else "audio" if is_voice else _WA_EXT_MEDIA_TYPE.get(os.path.splitext(file_path)[1].lower(), "document")


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False, caption=None):
    """Out-of-process delivery via the bridge HTTP API (standalone_sender_fn: cron apart from the gateway); ``caption`` rides on the media bubble."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        bridge_port = (getattr(pconfig, "extra", {}) or {}).get("bridge_port", 3000)
        normalized_chat_id = to_whatsapp_jid(chat_id)
        media = media_files or []
        # A caption only applies to a single media file — never repeat it across a multi-file send.
        media_caption = caption if (caption and len(media) == 1) else None
        last_message_id = None
        async with aiohttp.ClientSession() as session:
            async def _post(path, payload, total, error_label=None):
                """``(messageId, None)`` on 200, else ``(None, error_dict)`` (body read only when labelled)."""
                url = f"http://localhost:{bridge_port}/{path}"
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=total)) as resp:
                    if resp.status == 200:
                        return (await resp.json()).get("messageId"), None
                    return None, {} if error_label is None else {"error": f"WhatsApp {error_label} error ({resp.status}): {await resp.text()}"}
            # 1) Text first (skipped when media-only or when the text rides as the caption).
            if (message or "").strip() and not media_caption:
                last_message_id, err = await _post("send", {"chatId": normalized_chat_id, "message": message}, 30, "bridge")
                if err:
                    return err
            # 2) Each media file as a native attachment (mediaType picks the WhatsApp kind).
            for media_path, is_voice in media:
                if not os.path.exists(media_path):
                    # In caption mode the words would vanish with the missing file — deliver the caption as a plain message.
                    if media_caption:
                        try:
                            await _post("send", {"chatId": normalized_chat_id, "message": media_caption}, 30)
                        except Exception:
                            logger.warning("WhatsApp caption-fallback send failed for missing media")
                    return {"error": f"WhatsApp media file not found: {media_path}"}
                media_type = _bridge_media_type(media_path, is_voice, force_document)
                payload: Dict[str, Any] = {"chatId": normalized_chat_id, "filePath": media_path, "mediaType": media_type}
                payload.update({k: v for k, v in (("fileName", os.path.basename(media_path) if media_type == "document" else None), ("caption", media_caption)) if v})
                mid, err = await _post("send-media", payload, 120, "media")
                if err:
                    return err
                last_message_id = mid or last_message_id
        return {"success": True, "platform": "whatsapp", "chat_id": normalized_chat_id, "message_id": last_message_id}
    except Exception as e:
        return {"error": f"WhatsApp send failed: {e}"}


def interactive_setup() -> None:
    """Guide the user through WhatsApp setup (CLI helpers lazy-imported)."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import prompt, prompt_yes_no, print_header, print_info, print_success
    print_header("WhatsApp")
    print_info("WhatsApp uses a local Node.js bridge (WhatsApp Web client).")
    print_info("Start the bridge separately; the gateway connects to it over HTTP.")
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() in {"true", "1", "yes"}:
        print_info("WhatsApp: already enabled")
        if not prompt_yes_no("Reconfigure WhatsApp?", False):
            return
    if not prompt_yes_no("Enable WhatsApp?", True):
        save_env_value("WHATSAPP_ENABLED", "false")
        print_info("WhatsApp left disabled")
        return
    save_env_value("WHATSAPP_ENABLED", "true")
    print_success("WhatsApp enabled")
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for no allowlist)")
    if allowed_users:
        save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("WhatsApp allowlist configured")
    home_channel = prompt("Home chat ID for cron delivery (leave empty to skip)").strip()
    if home_channel:
        save_env_value("WHATSAPP_HOME_CHANNEL", home_channel)
    elif remove_env_value("WHATSAPP_HOME_CHANNEL"):
        print_info("Home channel cleared.")


# config.yaml whatsapp: key → env var. Env vars take precedence over YAML.
_YAML_LOWERCASE_KEYS = (("require_mention", "WHATSAPP_REQUIRE_MENTION"), ("dm_policy", "WHATSAPP_DM_POLICY"), ("group_policy", "WHATSAPP_GROUP_POLICY"))
_YAML_LIST_KEYS = (("free_response_chats", "WHATSAPP_FREE_RESPONSE_CHATS"), ("allow_from", "WHATSAPP_ALLOWED_USERS"), ("group_allow_from", "WHATSAPP_GROUP_ALLOWED_USERS"))


def _apply_yaml_config(yaml_cfg: dict, whatsapp_cfg: dict) -> dict | None:
    """config.yaml whatsapp: keys → WHATSAPP_* env vars (apply_yaml_config_fn contract; returns None).

    Mirrors the legacy whatsapp_cfg block from gateway/config.py::load_gateway_config(). Env vars take
    precedence over YAML. Returns None — everything flows through env. See #24849.
    """
    import json as _json
    for key, env in _YAML_LOWERCASE_KEYS:
        if key in whatsapp_cfg and not os.getenv(env):
            os.environ[env] = str(whatsapp_cfg[key]).lower()
    if "mention_patterns" in whatsapp_cfg and not os.getenv("WHATSAPP_MENTION_PATTERNS"):
        os.environ["WHATSAPP_MENTION_PATTERNS"] = _json.dumps(whatsapp_cfg["mention_patterns"])
    for key, env in _YAML_LIST_KEYS:
        val = whatsapp_cfg.get(key)
        if val is not None and not os.getenv(env):
            os.environ[env] = ",".join(str(v) for v in val) if isinstance(val, list) else str(val)
    return None


def _is_connected(config) -> bool:
    """Connected == WHATSAPP_ENABLED opt-in (or an enabled PlatformConfig with extras); auth lives in the bridge."""
    if config is not None and getattr(config, "enabled", False) and (getattr(config, "extra", {}) or {}):
        return True
    # Via hermes_cli.gateway.get_env_value (not os.getenv) so setup-status callers that patch it observe the same value.
    import hermes_cli.gateway as gateway_mod
    return (gateway_mod.get_env_value("WHATSAPP_ENABLED") or "").strip().lower() in {"true", "1", "yes"}


def _build_adapter(config):
    return WhatsAppAdapter(config)


def register(ctx) -> None:
    ctx.register_platform(
        name="whatsapp", label="WhatsApp", adapter_factory=_build_adapter, check_fn=check_whatsapp_requirements,
        is_connected=_is_connected, required_env=["WHATSAPP_ENABLED"],
        install_hint="WhatsApp requires a Node.js bridge — see the WhatsApp messaging docs",
        setup_fn=interactive_setup, apply_yaml_config_fn=_apply_yaml_config, allowed_users_env="WHATSAPP_ALLOWED_USERS",
        allow_all_env="WHATSAPP_ALLOW_ALL_USERS", cron_deliver_env_var="WHATSAPP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, max_message_length=4096, emoji="💬", allow_update_command=True,
    )
