"""Centralized logging setup for Hermes Agent.

Log files: agent.log (INFO+, everything), errors.log (WARNING+), gateway.log (INFO+,
gateway components; ``mode="gateway"``), gui.log (INFO+, dashboard/TUI-gateway;
``mode="gui"``). All are rotating files driven through one async queue and formatted
with ``RedactingFormatter`` so secrets never reach disk.
"""

import atexit
import copy
import io
import logging
import os
import queue
import sys
import threading
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional, Sequence

# Windows-ONLY swap (#44873): stdlib ``RotatingFileHandler.doRollover()`` calls
# ``os.rename()``, which fails with ``PermissionError [WinError 32]`` whenever
# another process holds an append handle on ``agent.log`` — essentially always
# in Hermes (TUI, gateway, hy_memory, MCP servers, CLI commands all log) —
# pinning the file at the size threshold and spamming stderr on every emit.
# ``concurrent-log-handler`` serializes rollover with a cross-process lock.
# POSIX keeps stdlib: renames of open files work, and managed mode (NixOS)
# relies on stdlib's exact ``_open()``/``doRollover()`` lifecycle for the
# 0660 chmod and eager file creation; CLH opens lazily and rotates differently.
if sys.platform == "win32":
    from concurrent_log_handler import (  # noqa: E402
        ConcurrentRotatingFileHandler as RotatingFileHandler,
    )
else:
    from logging.handlers import RotatingFileHandler  # noqa: E402


from hermes_constants import get_config_path, get_hermes_home, mkdir_under_hermes_home

# setup_logging() is idempotent: a second call is a no-op unless ``force=True``.
_logging_initialized = False

# Thread-local per-conversation session context.
_session_context = threading.local()

# ``%(session_tag)s`` exists on every LogRecord via _install_session_record_factory().
_LOG_FORMAT = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s"


def _safe_stderr():  # type: ignore[return]
    """Return a stderr stream that tolerates Unicode on all platforms.

    Wraps ``sys.stderr`` with ``errors='replace'`` so un-encodable characters become
    ``?`` instead of crashing the process.
    """
    stream = sys.stderr
    encoding = getattr(stream, "encoding", None) or "utf-8"
    if encoding.lower().replace("-", "") in ("utf8", "utf8surrogateescape"):
        return stream
    try:
        wrapped = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        # Prevent the wrapper from closing the underlying buffer when garbage-collected.
        wrapped.close = lambda: None  # type: ignore[assignment]
        return wrapped
    except Exception:
        return stream  # best-effort: no buffer / wrapping failed -> original stream


def _is_windows_concurrent_log_lock_timeout(exc: BaseException | None) -> bool:
    """True for concurrent-log-handler's Windows lock timeout.

    Slash-command workers and the gateway share rotating files on Windows Desktop;
    when another process holds the rollover lock too long CLH raises this
    RuntimeError, which must not escape into Desktop chat output.
    """
    return (
        sys.platform == "win32"
        and isinstance(exc, RuntimeError)
        and "Cannot acquire lock after 20 attempts" in str(exc)
    )


# Third-party loggers that are noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "openai", "openai._base_client", "httpx", "httpcore", "asyncio", "hpack", "hpack.hpack",
    "grpc", "modal", "urllib3", "urllib3.connectionpool", "websockets", "charset_normalizer",
    "markdown_it",
)


def _quiet_noisy_loggers() -> None:
    """Pin noisy third-party loggers at WARNING."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def set_session_context(session_id: str) -> None:
    """Set the session ID for the current thread."""
    _session_context.session_id = session_id


def clear_session_context() -> None:
    """Clear the session ID for the current thread."""
    _session_context.session_id = None


def _install_session_record_factory() -> None:
    """Replace the global LogRecord factory with one that adds ``session_tag``.

    Unlike a Filter, the record factory runs for EVERY record in the process (propagated
    and third-party-handled ones included), so ``%(session_tag)s`` never KeyErrors.
    Idempotent via a marker attribute.
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_hermes_session_injector", False):
        return

    def _session_record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        sid = getattr(_session_context, "session_id", None)
        record.session_tag = f" [{sid}]" if sid else ""  # type: ignore[attr-defined]
        # QueueListener formats on its own thread, after the profile-scoped
        # ContextVar is gone; keep the resolved home on the record so a
        # multiplex desktop ticker can route to the job owner's files (#97489).
        try:
            record.hermes_home = str(get_hermes_home().resolve())  # type: ignore[attr-defined]
        except Exception:
            record.hermes_home = ""  # type: ignore[attr-defined]
        return record
    _session_record_factory._hermes_session_injector = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(_session_record_factory)


# Install on import so session_tag exists on all records even before setup_logging().
_install_session_record_factory()


class _ComponentFilter(logging.Filter):
    """Only pass records whose logger name starts with one of *prefixes*."""

    def __init__(self, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


# Logger name prefixes per component; used by _ComponentFilter and ``hermes logs --component``.
COMPONENT_PREFIXES = {
    # ``plugins.platforms``: messaging adapters that migrated out of
    # ``gateway/platforms/`` into bundled plugins (#41112) are still gateway
    # components and belong in gateway.log.
    "gateway": ("gateway", "hermes_plugins", "plugins.platforms"),
    "agent": ("agent", "run_agent", "model_tools", "batch_runner"),
    "tools": ("tools",),
    "cli": ("hermes_cli", "cli"),
    "cron": ("cron",),
    "gui": ("hermes_cli.web_server", "hermes_cli.pty_bridge", "tui_gateway", "uvicorn"),
}


def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Configure the Hermes logging subsystem; returns the ``logs/`` directory.

    Safe to call multiple times; the second call is a no-op unless *force*. Level and
    rotation defaults come from config.yaml ``logging.*``. ``mode="gateway"`` adds
    ``gateway.log`` and ``mode="gui"`` adds ``gui.log``.
    """
    global _logging_initialized
    home = hermes_home or get_hermes_home()
    from hermes_constants import mkdir_under_hermes_home
    # --- worker-init RO evidence capture (read-only + bounded writability probe, best-effort) ---
    # If this process is a kanban worker (HERMES_KANBAN_TASK set), record
    # mount/writeability evidence BEFORE attempting to open agent.log so
    # an immediate Errno 30 is observable. Never raises, never mutates.
    # File writes limited to logs/<task>.worker_init.json via tmp rename.
    # Open-append probe is a bounded writability probe (not pure read-only):
    # it opens existing files with "ab" without writing then closes; it
    # bumps atime and requires W but does not mutate content.
    _kanban_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if _kanban_task:
        try:
            from hermes_cli.kanban_prespawn_evidence import (
                capture_prespawn_evidence as _kpe_cap2,
                write_prespawn_evidence_file as _kpe_write2,
            )

            _ev2 = _kpe_cap2(
                task_id=_kanban_task,
                profile=os.environ.get("HERMES_PROFILE", "").strip(),
                workspace=os.environ.get("HERMES_KANBAN_WORKSPACE", "") or str(home),
                board=os.environ.get("HERMES_KANBAN_BOARD"),
                hermes_home=str(home),
            )
            _ev2["capture_side"] = "worker_init_pre_logging"
            # log_dir not yet known here; resolve via home/logs directly to avoid
            # mkdir_under_hermes_home side-effects before capture
            _wd_log = Path(home) / "logs"
            try:
                _kpe_write2(_wd_log, _kanban_task, _ev2, suffix="worker_init")
            except Exception:
                pass
            # Also try board-scoped log dir if this is a board task (best-effort)
            try:
                from hermes_cli.kanban_db import worker_logs_dir as _wld

                _b = os.environ.get("HERMES_KANBAN_BOARD")
                if _b:
                    _bl = _wld(board=_b)
                    if _bl != _wd_log:
                        _kpe_write2(_bl, _kanban_task, _ev2, suffix="worker_init")
            except Exception:
                pass
        except Exception:
            pass
    log_dir = mkdir_under_hermes_home(home / "logs")

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()
    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    from agent.redact import RedactingFormatter  # lazy: circular at module load

    root = logging.getLogger()

    # (filename, level, max_bytes, backup_count, component) — a component gates
    # the file on ``mode`` and restricts it to that component's logger prefixes.
    handler_specs = (
        ("agent.log", level, max_bytes, backups, None),
        ("errors.log", logging.WARNING, 2 * 1024 * 1024, 2, None),
        ("gateway.log", logging.INFO, 5 * 1024 * 1024, 3, "gateway"),
        ("gui.log", logging.INFO, 10 * 1024 * 1024, 5, "gui"),
    )
    for filename, lvl, size, count, component in handler_specs:
        if component is not None and mode != component:
            continue
        _add_rotating_handler(
            log_dir / filename, level=lvl, max_bytes=size, backup_count=count,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES[component]) if component else None,
        )

    if _logging_initialized and not force:
        return log_dir

    # Root level must be low enough for the handlers to fire.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    _quiet_noisy_loggers()
    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """Enable DEBUG-level console logging for ``--verbose`` / ``-v`` mode."""
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    if any(getattr(h, "_hermes_verbose", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(_safe_stderr())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)
    _quiet_noisy_loggers()
    # rex-deploy at INFO for sandbox status.
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


def _quietly(fn) -> None:
    """Call *fn* (a ``close``/``stop`` bound method) swallowing errors — teardown must never raise."""
    try:
        fn()
    except Exception:
        pass


class _ManagedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler with managed-mode perms and external-rotation detection.

    In managed mode (NixOS) the setgid stateDir needs group-readable files, but
    ``_open()``/``doRollover()`` honor the umask (0644), so ``chmod 0660`` follows both.
    A rotating handler also holds an fd: if the file is rotated externally (logrotate,
    ``mv``) writes silently go to the old inode, so each emit compares the path's inode
    to the open stream's and reopens on mismatch (the ``WatchedFileHandler`` pattern).
    """

    def __init__(self, *args, **kwargs):
        from hermes_cli.config import is_managed
        self._managed = is_managed()
        super().__init__(*args, **kwargs)
        self._record_stream_stat()

    def _chmod_if_managed(self):
        if self._managed:
            try:
                os.chmod(self.baseFilename, 0o660)
            except OSError:
                pass

    def _record_stream_stat(self, st: Optional[os.stat_result] = None) -> None:
        """Snapshot dev/ino of ``baseFilename`` so emit() can detect external rotation."""
        try:
            st = st or os.stat(self.baseFilename)
            self._stat_dev, self._stat_ino = st.st_dev, st.st_ino
        except OSError:
            self._stat_dev, self._stat_ino = None, None

    def _reopen_stream(self, stat_result=None) -> None:
        """Close and reopen ``baseFilename`` (best-effort).

        On failure the stream is left ``None`` so the next emit bails rather than
        writing to a stale inode.
        """
        if self.stream is not None:
            _quietly(self.stream.close)
        self.stream = None  # type: ignore[assignment]
        try:
            self.stream = self._open()
        except Exception:
            return
        self._record_stream_stat(stat_result)

    def _reopen_if_externally_rotated(self) -> None:
        """Reopen when ``baseFilename`` was renamed, unlinked, or replaced by another inode.

        Silent + best-effort: any error falls back to the existing (possibly stale)
        stream so logging keeps working instead of dying on a stat failure.
        """
        try:
            st = os.stat(self.baseFilename)
        except FileNotFoundError:
            self._reopen_stream()  # rotated/unlinked underneath us: recreate at the path
            return
        except OSError:
            return  # transient — try again on the next emit

        if self._stat_dev is None or self._stat_ino is None:
            self._record_stream_stat(st)
        elif (st.st_dev, st.st_ino) != (self._stat_dev, self._stat_ino):
            self._reopen_stream(st)

    def emit(self, record: logging.LogRecord) -> None:
        # The kernel caches inode metadata, so this stat is sub-microsecond on a hot file.
        if self.stream is not None or os.path.exists(self.baseFilename):
            self._reopen_if_externally_rotated()
        super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Suppress the known Windows ``concurrent-log-handler`` lock timeout.

        CLH's ``emit()`` routes that RuntimeError here, so this is the single point to
        silence it before stdlib prints to stderr (which the Desktop slash-worker
        captures into chat output).
        """
        if not _is_windows_concurrent_log_lock_timeout(sys.exc_info()[1]):
            super().handleError(record)

    def _open(self):
        stream = super()._open()
        self._chmod_if_managed()
        return stream

    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()
        # Our own rollover writes a new baseFilename; refresh the snapshot so
        # the next emit doesn't mistake it for external rotation.
        self._record_stream_stat()


def _new_file_handler(
    path: Path, *, level: int, max_bytes: int, backup_count: int, formatter
) -> "_ManagedRotatingFileHandler":
    """Create the ``logs/`` directory and a configured ``_ManagedRotatingFileHandler``."""
    mkdir_under_hermes_home(path.parent)
    handler = _ManagedRotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


class _ProfileRoutingFileHandler(logging.Handler):
    """Route queued records to the log file for their Hermes home.

    Used only behind the QueueListener, so its small routing lock never blocks an agent
    or dashboard event loop. Per-home handlers keep rotation, redaction and managed perms.
    """

    def __init__(self, existing: RotatingFileHandler, profile_homes: Sequence[Path]) -> None:
        """Take over *existing*'s path, level, rotation, formatter and filters."""
        super().__init__(level=existing.level)
        resolved = Path(existing.baseFilename).resolve()
        self.baseFilename = str(resolved)
        self._hermes_routed_log_path = resolved
        self._default_home = resolved.parent.parent.resolve()
        self._profile_homes = {Path(home).expanduser().resolve() for home in profile_homes}
        self._filename = resolved.name
        self._max_bytes = getattr(existing, "maxBytes", 0)
        self._backup_count = getattr(existing, "backupCount", 0)
        self._profile_handlers: dict[Path, _ManagedRotatingFileHandler] = {}
        self._profile_handlers_lock = threading.RLock()
        self.setFormatter(existing.formatter)
        for log_filter in existing.filters:
            self.addFilter(log_filter)

    def _home_for_record(self, record: logging.LogRecord) -> Path:
        raw_home = getattr(record, "hermes_home", "")
        try:
            candidate = Path(raw_home).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            candidate = self._default_home
        return candidate if candidate in self._profile_homes else self._default_home

    def _handler_for_home(self, home: Path) -> _ManagedRotatingFileHandler:
        with self._profile_handlers_lock:
            if home not in self._profile_handlers:
                self._profile_handlers[home] = _new_file_handler(
                    home / "logs" / self._filename, level=self.level, max_bytes=self._max_bytes,
                    backup_count=self._backup_count, formatter=self.formatter,
                )
            return self._profile_handlers[home]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._handler_for_home(self._home_for_record(record)).handle(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._profile_handlers_lock:
            handlers = list(self._profile_handlers.values())
            self._profile_handlers.clear()
        for handler in handlers:
            _quietly(handler.close)
        super().close()


# Asynchronous file logging: an ``emit`` can block on the cross-process
# rotation lock (module header); on an asyncio thread that stalls the loop and
# drops WebSocket clients. Every file handler is therefore driven by a single
# QueueListener thread; loggers only do a non-blocking enqueue.

_log_queue: "Optional[queue.SimpleQueue]" = None
_queue_listener: Optional[QueueListener] = None
_queued_file_handlers: list = []
_queue_atexit_registered = False
# Guards every read-modify-write of the four globals above. setup_logging()
# holds no lock and its _logging_initialized guard runs AFTER handler
# registration, so _register_queued_handler() can race a flush/reset from
# another thread (gateway init vs a plugin/CLI path); without this, two
# threads can interleave stop()/reassign/start() and leave two live listeners.
_queue_state_lock = threading.Lock()


class _NonFormattingQueueHandler(QueueHandler):
    """``QueueHandler`` for an in-process queue.

    Stdlib ``prepare()`` formats and strips ``args``/``exc_info`` for cross-process
    pickling; ours is in-process, so targets get the unformatted record and apply their
    own ``RedactingFormatter`` on the listener thread. A shallow copy is returned because
    the emitting thread's synchronous handlers may mutate ``record.message`` meanwhile.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return copy.copy(record)


def _stop_queue_listener() -> None:
    """Flush and stop the background log listener (idempotent; atexit hook, so it takes the lock)."""
    global _queue_listener
    with _queue_state_lock:
        listener, _queue_listener = _queue_listener, None
        if listener is not None:
            _quietly(listener.stop)


def _start_queue_listener_locked() -> None:
    """(Re)build + start a listener over the current handler set (``_queue_state_lock`` held).

    A running listener is stopped first; this only happens while handlers are being
    added (queue empty), so ``stop()`` returns immediately.
    """
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
    _queue_listener = QueueListener(_log_queue, *_queued_file_handlers, respect_handler_level=True)
    _queue_listener.start()


def _register_queued_handler(handler: logging.Handler) -> None:
    """Route *handler* through the shared async queue instead of attaching it to root.

    Emitting threads never block on file I/O or the rotation lock; the ``QueueListener``
    applies each handler's own level and filters on its worker thread.
    """
    global _log_queue, _queue_atexit_registered
    with _queue_state_lock:
        if _log_queue is None:
            _log_queue = queue.SimpleQueue()
            qh = _NonFormattingQueueHandler(_log_queue)
            qh._hermes_queue = True  # type: ignore[attr-defined]
            # Always on the root logger so records from any logger reach the queue.
            logging.getLogger().addHandler(qh)
        _queued_file_handlers.append(handler)
        _start_queue_listener_locked()
        if not _queue_atexit_registered:
            # Runs before logging.shutdown (registered earlier at import time),
            # so the listener stops before its file handlers are closed.
            atexit.register(_stop_queue_listener)
            _queue_atexit_registered = True


def flush_log_queue() -> None:
    """Block until all queued records have been written, then resume.

    Stops the listener (which processes every pending record before joining) and
    restarts it. ``stop()`` joins the worker thread — do NOT call this on a hard-exit
    path where the listener may be wedged on the rotation lock; use
    ``drain_log_queue()`` there, which bounds the wait.
    """
    with _queue_state_lock:
        listener = _queue_listener
        if listener is not None:
            listener.stop()
            listener.start()


def drain_log_queue(timeout: float = 1.0) -> None:
    """Best-effort, time-bounded drain for hard-exit paths (no restart).

    If the listener's worker is wedged on the cross-process rotation lock — the very
    failure async logging exists to survive — an unbounded join would re-freeze shutdown.
    """
    listener = _queue_listener
    if listener is None:
        return
    t = threading.Thread(target=lambda: _quietly(listener.stop), name="hermes-log-drain", daemon=True)
    t.start()
    t.join(timeout)


def enable_profile_log_routing(profile_homes: Sequence[str | Path]) -> bool:
    """Make the queued file logs follow a desktop profile context.

    ``setup_logging`` binds handlers to one process home; the desktop dashboard's
    embedded cron ticker may run jobs for every profile, so its static file handlers
    are replaced with profile routers once the profile list is known. Returns ``True``
    when routing is (or already was) enabled; a single-profile caller is left untouched.
    """
    global _queue_listener
    homes: list[Path] = []
    for entry in profile_homes:
        try:
            resolved = Path(entry[1] if isinstance(entry, tuple) else entry).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            continue
        if resolved not in homes:
            homes.append(resolved)
    if len(homes) < 2:
        return False

    with _queue_state_lock:
        if not _queued_file_handlers:
            return False
        if any(isinstance(h, _ProfileRoutingFileHandler) for h in _queued_file_handlers):
            return True
        listener = _queue_listener
        if listener is not None:
            listener.stop()
            _queue_listener = None
        replacement = []
        for existing in _queued_file_handlers:
            if isinstance(existing, RotatingFileHandler):
                replacement.append(_ProfileRoutingFileHandler(existing, homes))
                _quietly(existing.close)
            else:
                replacement.append(existing)
        _queued_file_handlers[:] = replacement
        if listener is not None:
            _start_queue_listener_locked()
        return True


def _reset_queued_handlers() -> None:
    """Tear down the async logging queue + listener (test-isolation helper)."""
    global _log_queue
    _stop_queue_listener()
    with _queue_state_lock:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_hermes_queue", False):
                root.removeHandler(h)
        for h in list(_queued_file_handlers):
            _quietly(h.close)
        _queued_file_handlers.clear()
        _log_queue = None


def _add_rotating_handler(
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    log_filter: Optional[logging.Filter] = None,
) -> None:
    """Register a queued ``RotatingFileHandler`` for *path*; idempotent per resolved path."""
    resolved = path.resolve()
    for existing in _queued_file_handlers:
        # Already attached directly, or already covered by the profile router.
        if getattr(existing, "_hermes_routed_log_path", None) == resolved or (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return
    handler = _new_file_handler(
        path, level=level, max_bytes=max_bytes, backup_count=backup_count, formatter=formatter,
    )
    if log_filter is not None:
        handler.addFilter(log_filter)
    # Queue, not ``addHandler``: the rotation-lock wait never runs on the caller's thread.
    _register_queued_handler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml."""
    try:
        # Prefer the shared (mtime, size)-keyed raw-config cache so this reuses
        # hermes_cli.main's early parse (one config.yaml parse per process);
        # fall back to a direct parse for bare hermes_logging consumers.
        try:
            from hermes_cli.config import read_raw_config as _rrc
            cfg = _rrc() or {}
        except Exception:
            from utils import fast_safe_load
            config_path = get_config_path()
            cfg = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = fast_safe_load(f) or {}
        if not cfg:
            return (None, None, None)
        # Managed scope: an administrator can pin logging.* too (fail-open overlay).
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
        log_cfg = cfg.get("logging", {})
        if isinstance(log_cfg, dict):
            return (log_cfg.get("level"), log_cfg.get("max_size_mb"), log_cfg.get("backup_count"))
    except Exception:
        pass
    return (None, None, None)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def rotating_file_handlers() -> list:
    """Return the live rotating file handlers.

    They are attached to the async ``QueueListener`` rather than the root
    logger, so callers/tests must use this instead of scanning
    ``logging.getLogger().handlers``."""
    return list(_queued_file_handlers)
# ---- END PLUGIN-COMPAT ----
