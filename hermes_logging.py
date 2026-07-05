"""Centralized logging setup for Hermes Agent.

Provides a single ``setup_logging()`` entry point that both the CLI and
gateway call early in their startup path.  All log files live under
``~/.hermes/logs/`` (profile-aware via ``get_hermes_home()``).

Log files produced:
    agent.log   — INFO+, all agent/tool/session activity (the main log)
    errors.log  — WARNING+, errors and warnings only (quick triage)
    gateway.log — INFO+, gateway-only events (created when mode="gateway")
    gui.log     — INFO+, dashboard/websocket/TUI-gateway events
                  (created when mode="gui")

All files use ``RotatingFileHandler`` with ``RedactingFormatter`` so
secrets are never written to disk.

Component separation:
    gateway.log only receives records from ``gateway.*`` loggers —
    platform adapters, session management, slash commands, delivery.
    gui.log receives dashboard-side records from ``hermes_cli.web_server``,
    ``hermes_cli.pty_bridge``, ``tui_gateway.*``, and ``uvicorn.*``.
    agent.log remains the catch-all (everything goes there).

Session context:
    Call ``set_session_context(session_id)`` at the start of a conversation
    and ``clear_session_context()`` when done.  All log lines emitted on
    that thread will include ``[session_id]`` for filtering/correlation.
"""

import io
import logging
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

# On Windows, stdlib ``RotatingFileHandler`` calls ``os.rename()`` in
# ``doRollover()`` and fails with ``PermissionError [WinError 32]`` whenever
# another process holds an append-mode handle on ``agent.log`` — which is
# essentially always in Hermes (TUI, gateway, ``hy_memory`` server, MCP
# servers, and on-demand CLI commands all log from separate processes),
# pinning ``agent.log`` at the 5 MiB threshold and spamming stderr with
# a traceback on every emit. ``concurrent-log-handler`` wraps the rename in a
# cross-process file lock (via ``portalocker``: pywin32 on Windows) so only
# one process rotates at a time and the others wait their turn.
#
# This swap is Windows-ONLY and deliberately so:
#   * The bug (WinError 32 on rename-while-open) is specific to Windows file
#     locking semantics — POSIX renames an open file fine, so stdlib already
#     works correctly on Linux/macOS.
#   * On POSIX, managed-mode (NixOS) relies on the exact ``_open()`` /
#     ``doRollover()`` lifecycle of stdlib ``RotatingFileHandler`` (the
#     ``_ManagedRotatingFileHandler`` subclass chmods 0660 after each). CLH
#     opens lazily and rotates differently, which breaks the group-writable
#     guarantee and the eager file-creation those paths depend on.
# Aliasing keeps every existing ``RotatingFileHandler`` reference in this
# module (class declaration, ``isinstance`` checks, docstring) working
# unchanged. See #44873.
if sys.platform == "win32":
    from concurrent_log_handler import (  # noqa: E402
        ConcurrentRotatingFileHandler as RotatingFileHandler,
    )
else:
    from logging.handlers import RotatingFileHandler  # noqa: E402


from hermes_constants import get_config_path, get_hermes_home

# Sentinel to track whether setup_logging() has already run.  The function
# is idempotent — calling it twice is safe but the second call is a no-op
# unless ``force=True``.
_logging_initialized = False

# Idempotency flag for the HERMES_LOG_BLOCKING=1 startup warning -- see
# _warn_if_blocking_env_set(). Production callers do not need to clear
# it; only tests do (via fixture).
_blocking_warn_emitted = False

# Thread-local storage for per-conversation session context.
_session_context = threading.local()

# Default log format — includes timestamp, level, optional session tag,
# logger name, and message.  The ``%(session_tag)s`` field is guaranteed to
# exist on every LogRecord via _install_session_record_factory() below.
_LOG_FORMAT = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s"


def _safe_stderr():  # type: ignore[return]
    """Return a stderr stream that tolerates Unicode on all platforms.

    On Windows the console encoding is often a legacy MBCS codec
    (cp949, cp1252, …) that raises ``UnicodeEncodeError`` for characters
    like the em-dash (U+2014).  We wrap ``sys.stderr`` in a
    ``TextIOWrapper`` with ``errors='replace'`` so log lines are never
    lost — un-encodable characters are replaced with ``?`` instead of
    crashing the process.
    """
    stream = sys.stderr
    encoding = getattr(stream, "encoding", None) or "utf-8"
    # Already UTF-8 or surrogate-aware — no wrapping needed.
    if encoding.lower().replace("-", "") in ("utf8", "utf8surrogateescape"):
        return stream
    try:
        buf = getattr(stream, "buffer", None)
        if buf is not None:
            wrapped = io.TextIOWrapper(
                buf,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            # Prevent the wrapper from closing the underlying buffer
            # when it is garbage-collected.
            wrapped.close = lambda: None  # type: ignore[assignment]
            return wrapped
    except Exception:
        pass
    # Best-effort: if wrapping fails, return the original stream.
    return stream


_CONCURRENT_LOG_LOCK_TIMEOUT = "Cannot acquire lock after 20 attempts"


def _is_windows_concurrent_log_lock_timeout(exc: BaseException | None) -> bool:
    """Return True for concurrent-log-handler's Windows lock timeout.

    On Windows Desktop, slash-command workers and the gateway can all write to
    the same rotating log files. ``concurrent-log-handler`` serializes rollover
    with a cross-process lock, but when another process holds that lock too
    long it raises this RuntimeError. Logging failures should not escape into
    Desktop chat output.
    """
    return (
        sys.platform == "win32"
        and isinstance(exc, RuntimeError)
        and _CONCURRENT_LOG_LOCK_TIMEOUT in str(exc)
    )


# Third-party loggers that are noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
)


# ---------------------------------------------------------------------------
# Public session context API
# ---------------------------------------------------------------------------

def set_session_context(session_id: str) -> None:
    """Set the session ID for the current thread.

    All subsequent log records on this thread will include ``[session_id]``
    in the formatted output.  Call at the start of ``run_conversation()``.
    """
    _session_context.session_id = session_id


def clear_session_context() -> None:
    """Clear the session ID for the current thread."""
    _session_context.session_id = None


# ---------------------------------------------------------------------------
# Record factory — injects session_tag into every LogRecord at creation
# ---------------------------------------------------------------------------

def _install_session_record_factory() -> None:
    """Replace the global LogRecord factory with one that adds ``session_tag``.

    Unlike a ``logging.Filter`` on a handler or logger, the record factory
    runs for EVERY record in the process — including records that propagate
    from child loggers and records handled by third-party handlers.  This
    guarantees ``%(session_tag)s`` is always available in format strings,
    eliminating the KeyError that would occur if a handler used our format
    without having a ``_SessionFilter`` attached.

    Idempotent — checks for a marker attribute to avoid double-wrapping if
    the module is reloaded.
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_hermes_session_injector", False):
        return  # already installed

    def _session_record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        sid = getattr(_session_context, "session_id", None)
        record.session_tag = f" [{sid}]" if sid else ""  # type: ignore[attr-defined]
        return record

    _session_record_factory._hermes_session_injector = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(_session_record_factory)


# Install immediately on import — session_tag is available on all records
# from this point forward, even before setup_logging() is called.
_install_session_record_factory()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class _ComponentFilter(logging.Filter):
    """Only pass records whose logger name starts with one of *prefixes*.

    Used to route gateway-specific records to ``gateway.log`` while
    keeping ``agent.log`` as the catch-all.
    """

    def __init__(self, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


# Logger name prefixes that belong to each component.
# Used by _ComponentFilter and exposed for ``hermes logs --component``.
COMPONENT_PREFIXES = {
    # ``plugins.platforms`` covers messaging-platform adapters that migrated
    # out of ``gateway/platforms/`` into bundled plugins (#41112) — they are
    # still gateway components and their logs belong in gateway.log / match
    # ``hermes logs --component gateway``.
    "gateway": ("gateway", "hermes_plugins", "plugins.platforms"),
    "agent": ("agent", "run_agent", "model_tools", "batch_runner"),
    "tools": ("tools",),
    "cli": ("hermes_cli", "cli"),
    "cron": ("cron",),
    "gui": (
        "hermes_cli.web_server",
        "hermes_cli.pty_bridge",
        "tui_gateway",
        "uvicorn",
    ),
}


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Configure the Hermes logging subsystem.

    Safe to call multiple times — the second call is a no-op unless
    *force* is ``True``.

    Parameters
    ----------
    hermes_home
        Override for the Hermes home directory.  Falls back to
        ``get_hermes_home()`` (profile-aware).
    log_level
        Minimum level for the ``agent.log`` file handler.  Accepts any
        standard Python level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        Defaults to ``"INFO"`` or the value from config.yaml ``logging.level``.
    max_size_mb
        Maximum size of each log file in megabytes before rotation.
        Defaults to 5 or the value from config.yaml ``logging.max_size_mb``.
    backup_count
        Number of rotated backup files to keep.
        Defaults to 3 or the value from config.yaml ``logging.backup_count``.
    mode
        Caller context: ``"cli"``, ``"gateway"``, ``"gui"``, ``"cron"``.
        When ``"gateway"``, an additional ``gateway.log`` file is created
        that receives only gateway-component records.
        When ``"gui"``, an additional ``gui.log`` file is created that
        receives dashboard and TUI-gateway component records.
    force
        Re-run setup even if it has already been called.

    Returns
    -------
    Path
        The ``logs/`` directory where files are written.
    """
    global _logging_initialized
    home = hermes_home or get_hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # Lazy import to avoid circular dependency at module load time.
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # --- agent.log (INFO+) — the main activity log -------------------------
    _add_rotating_handler(
        root,
        log_dir / "agent.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backups,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- errors.log (WARNING+) — quick triage log --------------------------
    _add_rotating_handler(
        root,
        log_dir / "errors.log",
        level=logging.WARNING,
        max_bytes=2 * 1024 * 1024,
        backup_count=2,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- gateway.log (INFO+, gateway component only) ------------------------
    if mode == "gateway":
        _add_rotating_handler(
            root,
            log_dir / "gateway.log",
            level=logging.INFO,
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES["gateway"]),
        )

    # --- gui.log (INFO+, dashboard/tui-gateway components) -----------------
    if mode == "gui":
        _add_rotating_handler(
            root,
            log_dir / "gui.log",
            level=logging.INFO,
            max_bytes=10 * 1024 * 1024,
            backup_count=5,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES["gui"]),
        )

    if _logging_initialized and not force:
        return log_dir

    # Ensure root logger level is low enough for the handlers to fire.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Suppress noisy third-party loggers.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Defense-in-depth: surface HERMES_LOG_BLOCKING=1 to stderr so an
    # accidental opt-out of the non-blocking handler is loud, not silent.
    # Idempotent: prints at most once per process.
    _warn_if_blocking_env_set()

    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """Enable DEBUG-level console logging for ``--verbose`` / ``-v`` mode.

    Called by ``AIAgent.__init__()`` when ``verbose_logging=True``.
    """
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # Avoid adding duplicate stream handlers.
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            if getattr(h, "_hermes_verbose", False):
                return

    handler = logging.StreamHandler(_safe_stderr())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Lower root logger level so DEBUG records reach all handlers.
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # Keep third-party libraries at WARNING to reduce noise.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # rex-deploy at INFO for sandbox status.
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ManagedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that ensures group-writable perms in managed mode
    AND survives external rotation.

    Two responsibilities:

    1.  In managed mode (NixOS), the stateDir uses setgid (2770) so new files
        inherit the hermes group. However, both ``_open()`` (initial creation)
        and ``doRollover()`` create files via ``open()``, which uses the
        process umask — typically 0022, producing 0644. This subclass applies
        ``chmod 0660`` after both operations so the gateway and interactive
        users can share log files.

    2.  ``RotatingFileHandler`` keeps an open file descriptor.  If anything
        rotates the file *externally* (``logrotate``, manual ``mv``,
        another process rotating under us, a transient unlink), our fd
        keeps pointing at the renamed/unlinked inode and every subsequent
        write goes to ``gateway.log.1`` instead of ``gateway.log`` — silent
        log loss for the file every operator expects to read.  Before each
        emit we ``stat`` ``baseFilename`` and compare it against the open
        stream's inode; on mismatch we reopen.  This is the same pattern
        as stdlib ``WatchedFileHandler.reopenIfNeeded()``, adapted for
        rotating handlers.
    """

    def __init__(self, *args, **kwargs):
        from hermes_cli.config import is_managed
        self._managed = is_managed()
        super().__init__(*args, **kwargs)
        # Snapshot the inode of the currently open stream so emit() can
        # detect external rotation without an extra fstat per write.
        self._stat_dev: Optional[int] = None
        self._stat_ino: Optional[int] = None
        self._record_stream_stat()

    def _chmod_if_managed(self):
        if self._managed:
            try:
                os.chmod(self.baseFilename, 0o660)
            except OSError:
                pass

    def _record_stream_stat(self) -> None:
        """Snapshot dev/ino of ``baseFilename`` so we can detect external rotation."""
        try:
            st = os.stat(self.baseFilename)
            self._stat_dev, self._stat_ino = st.st_dev, st.st_ino
        except OSError:
            self._stat_dev, self._stat_ino = None, None

    def _reopen_if_externally_rotated(self) -> None:
        """Reopen the stream when ``baseFilename`` no longer matches our fd.

        Triggered when ``baseFilename`` was renamed (logrotate), unlinked,
        or replaced by a different inode.  Silent + best-effort: any error
        falls back to the existing (possibly stale) stream so logging keeps
        working instead of dying on a stat failure.
        """
        try:
            st = os.stat(self.baseFilename)
        except FileNotFoundError:
            # File was rotated/unlinked underneath us.  Close + reopen so a
            # fresh inode is created at the expected path.
            try:
                if self.stream is not None:
                    self.stream.close()
            except Exception:
                pass
            self.stream = None  # type: ignore[assignment]
            try:
                self.stream = self._open()
                self._record_stream_stat()
            except Exception:
                # Couldn't reopen — leave stream=None; next emit will
                # bail rather than write to a stale inode.
                pass
            return
        except OSError:
            return  # transient — try again on the next emit

        if self._stat_dev is None or self._stat_ino is None:
            self._stat_dev, self._stat_ino = st.st_dev, st.st_ino
            return

        if (st.st_dev, st.st_ino) != (self._stat_dev, self._stat_ino):
            # baseFilename now points at a DIFFERENT inode than the one we
            # hold open.  Close the old stream and open the new file.
            try:
                if self.stream is not None:
                    self.stream.close()
            except Exception:
                pass
            self.stream = None  # type: ignore[assignment]
            try:
                self.stream = self._open()
                self._stat_dev, self._stat_ino = st.st_dev, st.st_ino
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        # Cheap-ish stat-per-record check; the kernel caches inode metadata
        # so the syscall is sub-microsecond on a hot file.
        if self.stream is not None or os.path.exists(self.baseFilename):
            self._reopen_if_externally_rotated()
        super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Suppress the known Windows ``concurrent-log-handler`` lock timeout
        instead of printing a traceback.

        CLH's own ``emit()`` wraps its body in ``try/except Exception:
        self.handleError(record)``, so the ``"Cannot acquire lock after N
        attempts"`` RuntimeError raised in ``_do_lock()`` is caught inside CLH
        and routed here — it never propagates out of ``super().emit()``.  This
        override is the single point where that timeout can be silenced before
        the stdlib handler prints it to stderr (which, under the Desktop
        slash-worker, is captured and surfaced into chat output)."""
        exc = sys.exc_info()[1]
        if _is_windows_concurrent_log_lock_timeout(exc):
            return
        super().handleError(record)

    def _open(self):
        stream = super()._open()
        self._chmod_if_managed()
        return stream

    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()
        # Our own rollover writes a new baseFilename; refresh the snapshot
        # so the next emit doesn't mistake it for external rotation.
        self._record_stream_stat()


# ---------------------------------------------------------------------------
# Non-blocking emit (fix for gateway.log asyncio freeze — see comment below)
# ---------------------------------------------------------------------------

# Shared executor used by ``_NonBlockingRotatingFileHandler`` to push the
# actual write/flush off the calling (asyncio / agent) thread.  A single
# process-wide executor keeps the cost bounded: 2 workers is enough to absorb
# a slow disk without ever blocking more than one or two concurrent log emits.
# Daemon threads ensure we never keep the process alive at shutdown; the
# atexit hook below gives pending emits a chance to drain first.
_log_emit_executor: Optional[ThreadPoolExecutor] = None
_log_emit_executor_lock = threading.Lock()
# Default emit timeout: short enough that a wedged disk write can't freeze the
# gateway's event loop for more than half a second, but long enough that the
# normal hot-cache write path (sub-millisecond) always completes on the worker
# thread before we'd ever consider giving up. Operators can override at
# process start via the ``HERMES_LOG_EMIT_TIMEOUT_S`` env var.
_LOG_EMIT_DEFAULT_TIMEOUT_S = 0.5


def _get_log_emit_executor() -> ThreadPoolExecutor:
    """Return the process-wide emit executor, creating it lazily.

    Lazy creation avoids spinning up worker threads for short-lived CLI
    invocations that never emit a single record (most ``hermes`` commands).

    On Python 3.11+, ``concurrent.futures.thread._python_exit`` (registered
    with ``atexit``) shuts down every live ``ThreadPoolExecutor`` when the
    interpreter starts to exit.  Late stragglers — tests with helper threads
    that outlive the test fn, or workers spawned just before a SIGTERM —
    then race against that shutdown and see ``RuntimeError: cannot schedule
    new futures after shutdown``.  We absorb that by transparently
    re-creating the executor the next time ``emit()`` is invoked.
    """
    global _log_emit_executor
    while True:
        executor = _log_emit_executor
        if executor is None:
            with _log_emit_executor_lock:
                if _log_emit_executor is None:
                    new = ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="hermes-log-emit",
                    )
                    _log_emit_executor = new
                    return new
            continue  # raced; loop and re-read
        # If stdlib's _python_exit has marked this executor as shut down,
        # swap it for a fresh one.  We can't directly read the private
        # ``_shutdown`` flag, but ``submit`` after shutdown raises
        # ``RuntimeError``, so we probe lazily on the next emit.
        return executor


# NOTE: we intentionally do NOT register an ``atexit`` hook to call
# ``executor.shutdown(wait=True)``.  ``ThreadPoolExecutor`` defaults to daemon
# threads, so the worker pool is torn down automatically when the interpreter
# exits — pending in-flight emits are simply cut off, which is fine because:
#
#   1.  Any test, hook, or worker that races against interpreter shutdown has
#       no useful recovery path anyway — the process is about to die.
#   2.  A naive ``shutdown(wait=True)`` inside ``atexit`` can preempt late
#       stragglers that are still trying to submit() (e.g. reproduction
#       tests that spawn helper threads), producing noisy
#       ``RuntimeError: cannot schedule new futures after shutdown``
#       tracebacks in stderr AFTER pytest has already declared the run
#       successful.  That noise hides real failures.
#
# If we ever need an explicit drain (e.g. for a synchronous shutdown path
# in the gateway), it should live on a different trigger (SIGTERM handler,
# ``hermes gateway restart`` cleanup, etc.) and cap the wait at
# ``_LOG_EMIT_DEFAULT_TIMEOUT_S`` so it can never block shutdown itself.


class _NonBlockingRotatingFileHandler(_ManagedRotatingFileHandler):
    """Rotating handler whose ``emit()`` runs on a background thread.

    Why this exists
    ---------------
    ``_ManagedRotatingFileHandler`` writes synchronously to disk from the
    calling thread.  In the gateway, that calling thread is the asyncio
    event loop itself: ``gateway.run`` emits an ``INFO`` line on every
    inbound Telegram message, every outbound response, and every clarify
    intercept.  When the disk stalls (NFS server, full volume, AV scanner
    holding the file, USB disk glitch) ``flush()`` blocks the event loop,
    and **Telegram stops being polled** — even though the gateway process
    is alive and ``launchd`` is happy.  This is the documented
    ``telegram-clarify-deadlock`` pattern.

    The queue-based fix (``logging.handlers.QueueHandler`` /
    ``QueueListener``) is the architecturally clean answer, but it is also
    a bigger change: it has to wrap every handler in setup_logging(), add
    listener lifecycle + shutdown, and handle the ``None``-lock semantics
    of the queue listener's drain thread.  Instead, this subclass takes
    the minimal route: it runs the parent's ``emit()`` on a background
    thread and gives it a short deadline.  If the deadline expires the
    record is dropped (logged once to ``sys.stderr`` for ops visibility)
    and the caller — the asyncio loop or the tool executor — is free to
    keep going.

    Operational knobs
    -----------------
    * ``HERMES_LOG_EMIT_TIMEOUT_S`` — emit deadline in seconds (float).
      Defaults to ``0.5``.  ``0`` disables the timeout (records are
      submitted to the executor but the caller never waits; still
      non-blocking on the caller's thread, with the trade-off that
      ordering is best-effort and ordering can drift between concurrent
      handlers).
    * ``HERMES_LOG_BLOCKING=1`` — escape hatch: bypass this subclass
      entirely and use the synchronous parent.  Useful when diagnosing
      ordering issues or running under tools whose threads do not
      survive ``atexit`` drain.
    """

    _drop_count_attr = "_hermes_log_dropped_total"
    _warned_timeout_attr = "_hermes_log_timeout_warned"
    _EMIT_TIMEOUT_WARN_INTERVAL = 50  # warn every N dropped records

    def emit(self, record: logging.LogRecord) -> None:
        """Submit ``record`` to the emit executor, with a deadline.

        The deadline is what converts "disk stall freezes the event loop"
        into "disk stall drops a record and the event loop keeps running".
        If the deadline fires before the future completes we cancel it and
        silently drop the record; the caller's coroutine never blocks.
        ``emit()`` synchronously running inside a Python coroutine is
        *always* dangerous, so this method must remain non-blocking.

        We tolerate ``RuntimeError`` from ``submit()`` (the executor was
        shut down by stdlib's ``_python_exit`` atexit hook during interpreter
        shutdown) by lazily swapping in a fresh executor on the next call.
        """
        timeout_s = _LOG_EMIT_DEFAULT_TIMEOUT_S
        try:
            env_val = os.environ.get("HERMES_LOG_EMIT_TIMEOUT_S")
            if env_val is not None and env_val.strip() != "":
                timeout_s = float(env_val)
        except (TypeError, ValueError):
            pass

        try:
            executor = _get_log_emit_executor()
            future: Future = executor.submit(super().emit, record)
        except RuntimeError:
            # Executor was shut down (likely stdlib's _python_exit atexit
            # hook firing during interpreter shutdown).  Replace it with a
            # fresh one and retry once.  If the retry also fails we drop
            # the record — we're shutting down anyway, and a final stderr
            # flood would be worse than a missing log line.
            self._reinit_executor_after_shutdown()
            try:
                executor = _get_log_emit_executor()
                future = executor.submit(super().emit, record)
            except RuntimeError:
                return

        if timeout_s <= 0:
            # Best-effort: submit and return. Ordering is best-effort and
            # the caller cannot observe back-pressure, but the asyncio loop
            # is never blocked.
            self._track_dropped_future(future)
            return

        # Block *only* this emit call, not the caller's coroutine beyond the
        # timeout itself. We use ``future.result(timeout=...)`` rather than
        # blocking on the underlying ``Handler.lock``, so concurrent emits to
        # OTHER handlers (or other handlers' queues) are unaffected.
        try:
            future.result(timeout=timeout_s)
        except TimeoutError:
            # Disk is slower than the deadline. Cancel the future to release
            # its captured resources; the underlying handler.lock may stay
            # held briefly until the worker finishes its current write — at
            # worst a subsequent emit on the SAME handler will queue for the
            # remainder of that write. Records on other handlers continue
            # unaffected.
            future.cancel()
            self._record_drop(record)
        except Exception:
            # ``super().emit()`` already routes handler errors through
            # ``handleError()``; we swallow any unexpected exception here so
            # a buggy handler can never crash the gateway loop.
            self.handleError(record)

    @staticmethod
    def _reinit_executor_after_shutdown() -> None:
        """Replace the global emit executor with a fresh one.

        Called when ``submit()`` raises ``RuntimeError`` (executor was shut
        down by stdlib's atexit hook during interpreter shutdown).  We
        forcibly null the module-level reference and let the next
        ``_get_log_emit_executor()`` create a replacement.
        """
        global _log_emit_executor
        with _log_emit_executor_lock:
            _log_emit_executor = None

    def _track_dropped_future(self, future: Future) -> None:
        """Periodically report that drops are happening without blocking.

        With ``timeout_s <= 0`` the caller never blocks, so we just keep a
        process-wide counter and warn every N drops.  The counter is
        process-global because handler instances can be recreated on
        ``setup_logging(force=True)`` and we want a stable signal.
        """
        future.add_done_callback(_log_drop_audit_callback)

    def _record_drop(self, record: logging.LogRecord) -> None:
        """Bump the drop counter and warn periodically to ``sys.stderr``."""
        # Module-global counter so the warning cadence is stable across
        # handler instances and across forced re-initializations.
        global _log_dropped_total
        _log_dropped_total += 1
        if (
            _log_dropped_total % self._EMIT_TIMEOUT_WARN_INTERVAL == 0
            and not getattr(record, self._warned_timeout_attr, False)
        ):
            _warn_log_drop(_log_dropped_total)


# Module-level drop counter shared by all ``_NonBlockingRotatingFileHandler``
# instances. Initialised here so the symbol exists at import time even before
# any handler has been instantiated.
_log_dropped_total = 0


def _log_drop_audit_callback(future: Future) -> None:
    """Done-callback that audits a fire-and-forget emit.

    Logs an unexpected exception to ``sys.stderr`` if the worker raised (we
    cannot route it through the handler because the handler's lock is the
    very thing we are trying to avoid blocking on).  Failure here is a
    "missing log line" at worst.
    """
    try:
        exc = future.exception()
    except Exception:
        return
    if exc is not None:
        print(
            f"[hermes-log] non-blocking emit raised: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _warn_log_drop(dropped_total: int) -> None:
    """Surface sustained log drops to ``sys.stderr`` so operators see them.

    Stderr is intentional: it never goes through the very handler we're
    guarding against, so a wedged log file cannot silence this warning.
    """
    try:
        print(
            f"[hermes-log] WARNING: {_log_dropped_total} log records dropped "
            f"(gateway.log or agent.log write exceeded timeout). "
            f"Check disk health or raise HERMES_LOG_EMIT_TIMEOUT_S.",
            file=sys.stderr,
        )
    except Exception:
        pass


def _warn_if_blocking_env_set() -> None:
    """Surface the HERMES_LOG_BLOCKING escape hatch to stderr at startup.

    When ``HERMES_LOG_BLOCKING=1`` is set in the environment, the gateway
    reverts to the synchronous ``_ManagedRotatingFileHandler`` -- which
    reintroduces the asyncio-freeze risk that the ``a835f97`` fix
    removed.  An operator who accidentally leaves the var set in
    production (a launchd plist override, a deploy wrapper, etc.) would
    see logs "work" until the disk wedges, at which point the gateway
    dies exactly as in the 2026-07-04 outage.

    To prevent that silent regression we print a loud
    ``[hermes-log] WARNING`` to ``sys.stderr`` the first time
    ``setup_logging()`` runs with the var set.  Stderr is intentional:
    it never goes through the very handler we are guarding against,
    so a wedged log file cannot silence this warning.

    Idempotent across repeated ``setup_logging()`` calls -- exactly one
    warning per process.  Reset the module flag explicitly in tests.
    """
    global _blocking_warn_emitted
    if _blocking_warn_emitted:
        return
    if os.environ.get("HERMES_LOG_BLOCKING") != "1":
        return
    _blocking_warn_emitted = True
    try:
        print(
            "[hermes-log] WARNING: HERMES_LOG_BLOCKING=1 -- gateway.log "
            "emit() is SYNCHRONOUS; a wedged disk will freeze the asyncio "
            "loop. Unset this env var to restore non-blocking behavior.",
            file=sys.stderr,
        )
    except Exception:
        # Do not let a print failure (broken stderr, exotic platform) keep
        # the rest of setup_logging() from succeeding.
        pass


def _add_rotating_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    log_filter: Optional[logging.Filter] = None,
) -> None:
    """Add a ``RotatingFileHandler`` to *logger*, skipping if one already
    exists for the same resolved file path (idempotent).

    Parameters
    ----------
    log_filter
        Optional filter to attach to the handler (e.g. ``_ComponentFilter``
        for gateway.log).
    """
    resolved = path.resolve()
    for existing in logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # already attached

    path.parent.mkdir(parents=True, exist_ok=True)
    # ``_NonBlockingRotatingFileHandler`` runs the actual write/flush on a
    # process-wide executor so that a wedged disk cannot freeze the calling
    # thread — critical for the gateway's asyncio event loop (see
    # ``_NonBlockingRotatingFileHandler`` docstring for the full rationale).
    # Tests and operators that need synchronous, ordered writes can opt out
    # via ``HERMES_LOG_BLOCKING=1`` in the environment; in that mode we
    # still go through ``_ManagedRotatingFileHandler`` (the original) so the
    # managed-mode chmod and external-rotation detection are preserved.
    #
    # NOTE: ``HERMES_LOG_BLOCKING=1`` also triggers a one-shot stderr
    # WARNING from ``_warn_if_blocking_env_set()`` (called from
    # ``setup_logging()``) so an accidental opt-out -- e.g. a stale launchd
    # plist override -- is loud at startup instead of silently
    # reintroducing the asyncio-freeze risk the ``a835f97`` fix removed.
    blocking = os.environ.get("HERMES_LOG_BLOCKING") == "1"
    handler_cls = (
        _ManagedRotatingFileHandler if blocking else _NonBlockingRotatingFileHandler
    )
    handler = handler_cls(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    if log_filter is not None:
        handler.addFilter(log_filter)
    logger.addHandler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml.

    Returns ``(level, max_size_mb, backup_count)`` — any may be ``None``.
    """
    try:
        from utils import fast_safe_load
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = fast_safe_load(f) or {}
            # Managed scope: an administrator can pin logging.* too. Overlay via
            # the shared helper (fail-open) since this reads config.yaml directly.
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                return (
                    log_cfg.get("level"),
                    log_cfg.get("max_size_mb"),
                    log_cfg.get("backup_count"),
                )
    except Exception:
        pass
    return (None, None, None)
