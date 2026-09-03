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
_redacting_factory_installed = False
_original_log_record_factory = None  # type: ignore

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
        # QueueListener formats records on its own thread, after the
        # profile-scoped ContextVar has gone out of scope. Keep the resolved
        # home on the record so a multiplex desktop ticker can route the log
        # to the job owner's files (#97489).
        try:
            record.hermes_home = str(get_hermes_home().resolve())  # type: ignore[attr-defined]
        except Exception:
            record.hermes_home = ""  # type: ignore[attr-defined]
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


class _SecretRedactionFilter(logging.Filter):
    """Global secret redaction — truly global (handler-level + factory).

    Why handler-level: a Filter installed only on the root *logger* is
    NEVER checked for child logger records that propagate to root handlers
    (Logger.callHandlers traverses ancestor *handlers*, not ancestor logger
    filters — verified against stdlib). Installing on each *handler* makes
    the check run for every record via Handler.filter → Handler.handle.

    Factory covers queue-bypass / early logs / future handlers; handler
    filter covers propagation, extra, assembled, and makeLogRecord paths.
    """

    # Audit counter — not content, just count (for SOC, not for exfil)
    _redacted_total = 0
    _lock = threading.Lock()

    # Standard LogRecord attributes — anything else is caller-supplied `extra`
    _STANDARD_ATTRS = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime"}

    @staticmethod
    def _redact_text(text: str) -> str:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from agent.redact import _REDACT_ENABLED

            if not _REDACT_ENABLED:
                return True
        except Exception:
            return record.levelno >= logging.WARNING

        try:
            from agent.redact import redact_sensitive_text

            redacted_any = False

            def _bump():
                with self._lock:
                    type(self)._redacted_total += 1

            # 1) msg — including non-string/custom objects
            if record.msg is not None:
                if isinstance(record.msg, str):
                    red = redact_sensitive_text(record.msg)
                    if red != record.msg:
                        record.msg = red
                        _bump()
                        redacted_any = True
                else:
                    # Custom objects control their own __str__; redact that
                    # rendering without assuming it is safe.
                    try:
                        s = str(record.msg)
                        red = redact_sensitive_text(s)
                        if red != s:
                            record.msg = red
                            _bump()
                            redacted_any = True
                    except Exception:
                        pass

            # 2) args — tuple / dict, string values only (preserve types)
            if record.args:
                if isinstance(record.args, tuple):
                    new_args = []
                    changed = False
                    for arg in record.args:
                        if isinstance(arg, str):
                            red = redact_sensitive_text(arg)
                            if red != arg:
                                _bump()
                                changed = True
                                new_args.append(red)
                            else:
                                new_args.append(arg)
                        else:
                            # Nested structures inside args (e.g. dataclass) — stringify
                            # only if the string rendering contains a secret
                            try:
                                s = str(arg)
                                if redact_sensitive_text(s) != s:
                                    _bump()
                                    changed = True
                                    # keep original type but ensure logging won't leak via %s
                                    new_args.append(redact_sensitive_text(s))
                                else:
                                    new_args.append(arg)
                            except Exception:
                                new_args.append(arg)
                    if changed:
                        record.args = tuple(new_args)
                        redacted_any = True
                elif isinstance(record.args, dict):
                    new_args = {}
                    changed = False
                    for k, v in record.args.items():
                        if isinstance(v, str):
                            red = redact_sensitive_text(v)
                            if red != v:
                                _bump()
                                changed = True
                                new_args[k] = red
                            else:
                                new_args[k] = v
                        else:
                            try:
                                s = str(v)
                                if redact_sensitive_text(s) != s:
                                    _bump()
                                    changed = True
                                    new_args[k] = redact_sensitive_text(s)
                                else:
                                    new_args[k] = v
                            except Exception:
                                new_args[k] = v
                    if changed:
                        record.args = new_args
                        redacted_any = True

            # 3) assembled message — catches token formed only after % formatting
            # e.g. msg="token sk-%s" args=("a"*32,) → assembled "token sk-aaa..."
            try:
                assembled = record.getMessage()
                red_assembled = redact_sensitive_text(assembled)
                if red_assembled != assembled:
                    # Replace components with redacted assembled to avoid re-leak via formatter
                    record.msg = red_assembled
                    record.args = ()
                    _bump()
                    redacted_any = True
            except Exception:
                pass

            # 4) exc_info — value, traceback text, and chained causes
            if record.exc_info and record.exc_info[1] is not None:
                try:
                    exc_val = record.exc_info[1]
                    exc_str = str(exc_val)
                    red_exc = redact_sensitive_text(exc_str)
                    if red_exc != exc_str:
                        _bump()
                        try:
                            # mutate args if possible
                            if hasattr(exc_val, "args") and exc_val.args:
                                # keep exc type, redact first arg string
                                if isinstance(exc_val.args[0], str):
                                    exc_val.args = (red_exc,) + exc_val.args[1:]  # type: ignore
                        except Exception:
                            pass
                    # Also redact the formatted traceback skeleton (file lines don't leak, but exc message does)
                    # Store redacted exc_text for formatters that use it
                    try:
                        import traceback as _tb

                        exc_text = "".join(_tb.format_exception(*record.exc_info))
                        red_exc_text = redact_sensitive_text(exc_text)
                        if red_exc_text != exc_text:
                            record.exc_text = red_exc_text  # type: ignore[attr-defined]
                            _bump()
                    except Exception:
                        pass
                except Exception:
                    pass

            # 5) stack_info
            if record.stack_info and isinstance(record.stack_info, str):
                red_stack = redact_sensitive_text(record.stack_info)
                if red_stack != record.stack_info:
                    record.stack_info = red_stack
                    _bump()

            # 6) extra fields — merged into record.__dict__ after factory by Logger.makeRecord
            # Any caller-supplied `extra={'token': 'sk-...'}` ends up here
            for k, v in list(record.__dict__.items()):
                if k in self._STANDARD_ATTRS:
                    continue
                if isinstance(v, str):
                    red = redact_sensitive_text(v)
                    if red != v:
                        setattr(record, k, red)
                        _bump()
                        redacted_any = True
                elif isinstance(v, (dict, list, tuple)):
                    # shallow redact string members of extra structures
                    try:
                        s = str(v)
                        if redact_sensitive_text(s) != s:
                            setattr(record, k, redact_sensitive_text(s))
                            _bump()
                            redacted_any = True
                    except Exception:
                        pass

            # 7) logging.makeLogRecord path — dict already redacted via wrapped
            # function (see below), but handler filter is still the last gate

        except Exception:
            record.msg = "[REDACTION_ERROR] log record suppressed due to redaction failure"
            record.args = ()
            record.exc_info = None
            record.stack_info = None

        return True

    @classmethod
    def redacted_total(cls) -> int:
        with cls._lock:
            return cls._redacted_total


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
    from hermes_constants import mkdir_under_hermes_home
    log_dir = mkdir_under_hermes_home(home / "logs")

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # Lazy import to avoid circular dependency at module load time.
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # 98833 100× stronger: truly global redaction — handler-level Filter +
    # factory + Formatter defense in depth. Root logger Filter alone is NOT
    # global (Logger.callHandlers does not re-check ancestor logger filters for
    # child logger records). Installing on each *handler* via
    # Handler.filter → handle() makes it fire for gateway/agent/gui/console
    # regardless of propagation. Factory covers early logs / future handlers /
    # queue-bypass; Formatter covers final assembled string.
    #
    # Ensure any queued handlers that already exist also carry the secret
    # redaction filter (idempotent).
    def _ensure_handler_redaction_filter(h: logging.Handler) -> None:
        for f in list(h.filters):
            if isinstance(f, _SecretRedactionFilter):
                return
        h.addFilter(_SecretRedactionFilter())

    for _h in list(_queued_file_handlers):
        _ensure_handler_redaction_filter(_h)
    # Also remove any stale root Filter that previous head installed — it was
    # ineffective for child loggers and now superseded by handler filters.
    for f in list(root.filters):
        if isinstance(f, _SecretRedactionFilter):
            root.removeFilter(f)

    # 98833 100× stronger: install redacting LogRecord factory + makeLogRecord
    # wrap. Factory covers queue-bypass, early logs, and any handler without
    # RedactingFormatter. makeLogRecord wrap covers dict-reconstructed records
    # (logging.makeLogRecord) where Logger.makeRecord's extra-merge happens after
    # the factory.
    global _redacting_factory_installed, _original_log_record_factory
    _original_make_log_record = getattr(logging, "makeLogRecord", None)
    try:
        if not _redacting_factory_installed:
            _original_log_record_factory = logging.getLogRecordFactory()

            def _redacting_log_record_factory(*args, **kwargs):
                record = _original_log_record_factory(*args, **kwargs)
                # Reuse handler filter's redaction logic so factory and handler
                # stay in sync — instantiate a temporary filter and run it.
                try:
                    _SecretRedactionFilter().filter(record)
                except Exception:
                    try:
                        record.msg = "[REDACTION_ERROR]"
                        record.args = ()
                        record.exc_info = None
                        record.stack_info = None
                    except Exception:
                        pass
                return record

            logging.setLogRecordFactory(_redacting_log_record_factory)

            # Wrap makeLogRecord (dict → LogRecord) to also redact extra etc.
            if _original_make_log_record is not None and not getattr(
                _original_make_log_record, "_hermes_redacting_wrapped", False
            ):
                _orig = _original_make_log_record

                def _redacting_make_log_record(dict):  # type: ignore
                    rec = _orig(dict)
                    try:
                        _SecretRedactionFilter().filter(rec)
                    except Exception:
                        pass
                    return rec

                _redacting_make_log_record._hermes_redacting_wrapped = True  # type: ignore
                logging.makeLogRecord = _redacting_make_log_record  # type: ignore

            _redacting_factory_installed = True
    except Exception:
        pass

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
    handler.addFilter(_SecretRedactionFilter())
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


class _ProfileRoutingFileHandler(logging.Handler):
    """Route queued records to the log file for their Hermes home.

    Dashboard logging is initialized once for the process that launched it,
    while the desktop cron ticker can execute jobs for several profile homes.
    A normal ``RotatingFileHandler`` therefore pins every cron record to the
    dashboard profile. This handler keeps one rotating file handler per live
    profile and selects it from the home captured by the record factory.

    The handler itself is used only behind the existing QueueListener, so its
    small routing lock never blocks an agent or dashboard event loop. The
    underlying handlers retain the existing rotation, redaction, and managed
    permission behavior.
    """

    def __init__(
        self,
        *,
        default_path: Path,
        profile_homes: Sequence[Path],
        level: int,
        max_bytes: int,
        backup_count: int,
        formatter: logging.Formatter | None,
        log_filters: Sequence[logging.Filter],
    ) -> None:
        super().__init__(level=level)
        self.baseFilename = str(default_path.resolve())
        self._hermes_routed_log_path = Path(self.baseFilename)
        self._default_home = Path(self.baseFilename).parent.parent.resolve()
        self._profile_homes = {
            Path(home).expanduser().resolve()
            for home in profile_homes
        }
        self._filename = Path(self.baseFilename).name
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._profile_handlers: dict[Path, _ManagedRotatingFileHandler] = {}
        self._profile_handlers_lock = threading.RLock()
        if formatter is not None:
            self.setFormatter(formatter)
        for log_filter in log_filters:
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
            handler = self._profile_handlers.get(home)
            if handler is not None:
                return handler

            path = home / "logs" / self._filename
            from hermes_constants import mkdir_under_hermes_home

            mkdir_under_hermes_home(path.parent)
            handler = _ManagedRotatingFileHandler(
                str(path),
                maxBytes=self._max_bytes,
                backupCount=self._backup_count,
                encoding="utf-8",
            )
            handler.setLevel(self.level)
            handler.setFormatter(self.formatter)
            self._profile_handlers[home] = handler
            return handler

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
            try:
                handler.close()
            except Exception:
                pass
        super().close()


# ---------------------------------------------------------------------------
# Asynchronous file logging — keep the cross-process rotation lock off the loop
#
# The rotating file handlers serialize rollover with a cross-process lock (see
# the module header): when several Hermes processes log to the same file, an
# ``emit`` can block while another process holds that lock.  When the emitting
# thread is an asyncio event loop, that block stalls the loop and drops
# WebSocket clients.  To keep file I/O off the hot path, every file handler is
# driven by a single ``QueueListener`` on a dedicated thread; loggers only touch
# an in-memory queue (a non-blocking enqueue).
# ---------------------------------------------------------------------------

_log_queue: "Optional[queue.SimpleQueue]" = None
_queue_listener: Optional[QueueListener] = None
_queued_file_handlers: list = []
_queue_atexit_registered = False
# Guards every read-modify-write of the four globals above. setup_logging()
# holds no lock and its _logging_initialized guard runs AFTER handler
# registration, so _register_queued_handler() can run concurrently with a
# flush/reset from another thread (gateway init racing a plugin/CLI path).
# Without this, two threads can interleave listener.stop()/reassign/start()
# and leave the queue with two live listeners or an orphaned worker thread.
_queue_state_lock = threading.Lock()


class _NonFormattingQueueHandler(QueueHandler):
    """``QueueHandler`` for an in-process queue.

    Stdlib ``prepare()`` formats the record and drops ``args``/``exc_info`` so it
    can be pickled to another process.  Our queue is in-process, so we skip that
    and hand the target file handlers an unformatted record — they apply their
    own ``RedactingFormatter`` and component filters on the listener thread.

    We return a **shallow copy** rather than the original record: the same
    record is still owned by the emitting thread (and any synchronous handler
    on it, e.g. a ``StreamHandler``), which may format/mutate ``record.message``
    while our listener thread reads it. Copying preserves ``msg``/``args``/
    ``exc_info`` for the deferred format while removing the cross-thread
    mutation race on a shared object.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return copy.copy(record)


def _stop_queue_listener_locked() -> None:
    """Stop the listener assuming ``_queue_state_lock`` is already held."""
    global _queue_listener
    listener, _queue_listener = _queue_listener, None
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass


def _stop_queue_listener() -> None:
    """Flush and stop the background log listener (idempotent, thread-safe).

    This is the atexit hook, so it must acquire the state lock itself.
    """
    with _queue_state_lock:
        _stop_queue_listener_locked()


def _register_queued_handler(handler: logging.Handler) -> None:
    """Route *handler* through the shared async queue instead of attaching it to
    *root* directly, so emitting threads never block on file I/O or the
    cross-process rotation lock.  The ``QueueListener`` applies each handler's
    own level and filters on its worker thread."""
    global _log_queue, _queue_listener, _queue_atexit_registered
    with _queue_state_lock:
        if _log_queue is None:
            _log_queue = queue.SimpleQueue()
            qh = _NonFormattingQueueHandler(_log_queue)
            qh._hermes_queue = True  # type: ignore[attr-defined]
            # Always funnel through the root logger so records from any logger
            # (production passes root here; callers may pass a child) reach the
            # queue via propagation.
            logging.getLogger().addHandler(qh)
        _queued_file_handlers.append(handler)
        # Rebuild the listener with the full target set.  This only happens
        # while init_logging() adds handlers (2-3 times, queue empty), so
        # stop() returns immediately.
        if _queue_listener is not None:
            _queue_listener.stop()
        _queue_listener = QueueListener(
            _log_queue, *_queued_file_handlers, respect_handler_level=True
        )
        _queue_listener.start()
        if not _queue_atexit_registered:
            # Runs before logging.shutdown (registered earlier at import time),
            # so the listener stops before its file handlers are closed.
            atexit.register(_stop_queue_listener)
            _queue_atexit_registered = True


def flush_log_queue() -> None:
    """Block until all queued records have been written, then resume.

    Draining is done by stopping the listener (which processes every pending
    record before joining) and restarting it.  Used by tests that read a log
    file right after emitting to it.

    NOTE: ``stop()`` joins the worker thread, so this blocks until the queue
    is empty. Do NOT call this on a hard-exit path where the listener may be
    wedged on the rotation lock — use ``drain_log_queue()`` there instead,
    which bounds the wait.
    """
    with _queue_state_lock:
        listener = _queue_listener
        if listener is not None:
            listener.stop()
            listener.start()


def drain_log_queue(timeout: float = 1.0) -> None:
    """Best-effort, time-bounded drain for hard-exit paths (no restart).

    Unlike ``flush_log_queue()``, this stops the listener WITHOUT restarting it
    (the process is about to exit) and bounds the drain: if the listener's
    worker thread is wedged on the cross-process rotation lock — the very
    failure this async-logging change exists to survive — an unbounded
    ``stop()``/join would re-freeze the shutdown path. We run ``stop()`` on a
    throwaway thread and only wait ``timeout`` seconds for it; if it hasn't
    drained by then we abandon the last few records and let ``os._exit``
    proceed. Availability beats the last log line when the disk is already
    wedged.
    """
    listener = _queue_listener
    if listener is None:
        return

    def _drain() -> None:
        try:
            listener.stop()
        except Exception:
            pass

    t = threading.Thread(target=_drain, name="hermes-log-drain", daemon=True)
    t.start()
    t.join(timeout)


def rotating_file_handlers() -> list:
    """Return the live rotating file handlers.

    They are attached to the async ``QueueListener`` rather than the root
    logger, so callers/tests must use this instead of scanning
    ``logging.getLogger().handlers``."""
    return list(_queued_file_handlers)


def enable_profile_log_routing(profile_homes: Sequence[str | Path]) -> bool:
    """Make the queued file logs follow a desktop profile context.

    ``setup_logging`` normally binds handlers to one process home. The
    desktop dashboard is the exception: its embedded cron ticker may run
    jobs for every profile. Replace the existing static file handlers with
    profile routers after that profile list is known.

    Returns ``True`` when routing is enabled or was already enabled. A
    single-profile caller is left untouched because its existing handlers are
    already correctly scoped.
    """
    global _queue_listener

    homes = []
    for entry in profile_homes:
        home = entry[1] if isinstance(entry, tuple) else entry
        try:
            resolved = Path(home).expanduser().resolve()
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

        replacement = []
        for existing in _queued_file_handlers:
            if not isinstance(existing, RotatingFileHandler):
                replacement.append(existing)
                continue

            default_path = Path(existing.baseFilename)
            router = _ProfileRoutingFileHandler(
                default_path=default_path,
                profile_homes=homes,
                level=existing.level,
                max_bytes=getattr(existing, "maxBytes", 0),
                backup_count=getattr(existing, "backupCount", 0),
                formatter=existing.formatter,
                log_filters=list(existing.filters),
            )
            replacement.append(router)
            try:
                existing.close()
            except Exception:
                pass

        _queued_file_handlers[:] = replacement
        if listener is not None:
            _queue_listener = QueueListener(
                _log_queue, *_queued_file_handlers, respect_handler_level=True
            )
            _queue_listener.start()
        return True


def _reset_queued_handlers() -> None:
    """Tear down the async logging queue + listener (test-isolation helper)."""
    global _log_queue
    with _queue_state_lock:
        _stop_queue_listener_locked()
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_hermes_queue", False):
                root.removeHandler(h)
        for h in list(_queued_file_handlers):
            try:
                h.close()
            except Exception:
                pass
        _queued_file_handlers.clear()
        _log_queue = None


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
    for existing in _queued_file_handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # already attached
        if getattr(existing, "_hermes_routed_log_path", None) == resolved:
            return  # already covered by the profile router

    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(path.parent)
    handler = _ManagedRotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    # Truly global secret redaction — must be on handler, not root logger
    # (root logger Filter is never checked for child logger records).
    handler.addFilter(_SecretRedactionFilter())
    if log_filter is not None:
        handler.addFilter(log_filter)
    # Route through the async queue instead of ``logger.addHandler(handler)`` so
    # the rotation-lock wait never runs on the caller's (often event-loop) thread.
    _register_queued_handler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml.

    Returns ``(level, max_size_mb, backup_count)`` — any may be ``None``.
    """
    try:
        # Prefer the shared (mtime, size)-keyed raw-config cache so this read
        # reuses the parse hermes_cli.main's early bridge already did (one
        # config.yaml parse per process instead of 3-4). Fall back to a
        # direct parse when hermes_cli.config isn't importable (bare
        # hermes_logging consumers).
        try:
            from hermes_cli.config import read_raw_config as _rrc
            cfg = _rrc() or {}
        except Exception:
            from utils import fast_safe_load
            config_path = get_config_path()
            if not config_path.exists():
                return (None, None, None)
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = fast_safe_load(f) or {}
        if cfg:
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
