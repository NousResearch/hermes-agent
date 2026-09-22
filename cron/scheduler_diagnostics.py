"""Private run-document diagnostics, kept separate from delivery summaries."""

from pathlib import Path
from traceback import format_exception

from agent.redact import redact_sensitive_text

# Enough for a Python traceback's last frames plus the exception line.
_WORKER_STDERR_TAIL_CHARS = 1500


def format_run_error(exc: BaseException) -> str:
    """Retain chained causes without capturing locals or exposing URL credentials."""
    traceback_text = redact_sensitive_text(
        "".join(format_exception(exc)), force=True, redact_url_credentials=True,
    )
    return f"## Error\n\n```\n{traceback_text}\n```\n"


def external_worker_stderr_tail(path: Path) -> str:
    """Redacted tail of a restart-safe cron worker's captured stderr, as a message suffix.

    The gateway used to spawn the worker with ``stderr=DEVNULL``, so a worker that died before
    publishing its acknowledgement (an import error, a missing module in the interpreter it was
    handed) only ever reported ``exit 1`` and the reporter had to guess the cause (#112729).
    Returns an empty string when nothing was captured.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    tail = text[-_WORKER_STDERR_TAIL_CHARS:]
    if len(text) > _WORKER_STDERR_TAIL_CHARS:
        tail = "…" + tail
    tail = redact_sensitive_text(tail, force=True, redact_url_credentials=True)
    return f"; worker stderr: {tail}"
