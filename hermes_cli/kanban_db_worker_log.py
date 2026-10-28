"""Worker-log and exit-evidence helpers for Kanban dispatch.

Log files are append-only across task runs.  The dispatcher-written run marker
keeps current-run classification from inheriting a prior worker's evidence.
"""

from __future__ import annotations

import re
from typing import Mapping
from typing import Optional

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)
_WORKER_RUN_MARKER_PREFIX = "KANBAN_WORKER_RUN_START"
_WORKER_RUN_MARKER_RE = re.compile(
    r"^" + re.escape(_WORKER_RUN_MARKER_PREFIX) + r"\s+run_id=(\d+)\s*$", re.MULTILINE,
)
_REFUSAL_REASON_RE = re.compile(r"^hermes-refusal-reason:\s*([A-Z_]+)\s*$", re.MULTILINE)
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _kb():
    """Late-bind the Kanban facade to avoid its dispatcher re-export cycle."""
    from hermes_cli import kanban_db
    return kanban_db


def _exit_code_kind(code: int) -> tuple[str, int]:
    """``(kind, code)`` for a worker's exit code, however it was observed."""
    if code == 0:
        return ("clean_exit", 0)
    if code == _kb().KANBAN_RATE_LIMIT_EXIT_CODE:
        return ("rate_limited", code)
    if code == _kb().KANBAN_TERMINAL_PROVIDER_EXIT_CODE:
        return ("terminal_provider", code)
    return ("nonzero_exit", code)


def _classify_recent_worker_exit(
    exits: Mapping[int, tuple[int, float]], pid: int,
) -> tuple[str, Optional[int]]:
    """Classify a reaped worker PID from the dispatcher's local exit registry."""
    entry = exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    # Bit-level POSIX wait-status decode instead of os.WIFEXITED/WEXITSTATUS/
    # WIFSIGNALED/WTERMSIG: those helpers do not exist on Windows, where the
    # registry is fed by reap_worker_zombies' Popen poll. Low 7 bits = signal
    # (0 = normal exit, 0x7F = stopped), bits 8-15 = exit code.
    raw = int(raw)
    signal_number = raw & 0x7F
    if signal_number == 0:
        return _exit_code_kind((raw >> 8) & 0xFF)
    if signal_number != 0x7F:
        return ("signaled", signal_number)
    return ("unknown", None)


def _worker_log_run_marker(run_id: Optional[int]) -> bytes:
    """Dispatcher-written delimiter that correlates append-only log text to one task run."""
    if run_id is None:
        return b""
    return f"\n{_WORKER_RUN_MARKER_PREFIX} run_id={int(run_id)}\n".encode()


def _latest_log_run_segment(raw: Optional[str], *, current_run_id: Optional[int] = None) -> str:
    """The append-only worker-log segment that belongs to the latest observed run."""
    if not raw:
        return ""
    markers = list(_WORKER_RUN_MARKER_RE.finditer(raw))
    if current_run_id is not None and markers:
        target = str(int(current_run_id))
        for idx in range(len(markers) - 1, -1, -1):
            marker = markers[idx]
            if marker.group(1) != target:
                continue
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(raw)
            return raw[marker.end():end]
        # The log has run markers, but not for this in-flight run.  That means
        # the current worker produced no correlated log text; never reuse a
        # previous run's capacity/refusal evidence.
        return ""
    if markers:
        return raw[markers[-1].end():]
    trailers = list(_EXIT_TRAILER_RE.finditer(raw))
    if not trailers:
        return raw
    latest = trailers[-1]
    trailing = raw[latest.end():]
    if trailing.strip():
        # A later worker may crash before ``exit_single_query`` writes a new
        # trailer. Once it appends any output after the previous trailer, that
        # trailing text is the latest observed run; never reuse the prior run.
        return trailing
    previous_end = trailers[-2].end() if len(trailers) > 1 else 0
    return raw[previous_end:latest.start()]


def _latest_log_exit_code(raw: Optional[str], *, current_run_id: Optional[int] = None) -> Optional[int]:
    """Exit code for the latest observed log run, if that run wrote a trailer."""
    if not raw:
        return None
    segment = _latest_log_run_segment(raw, current_run_id=current_run_id)
    if segment != raw or _WORKER_RUN_MARKER_RE.search(raw):
        trailers = list(_EXIT_TRAILER_RE.finditer(segment))
        if not trailers:
            return None
        latest = trailers[-1]
        if segment[latest.end():].strip():
            return None
        return int(latest.group(1))
    trailers = list(_EXIT_TRAILER_RE.finditer(raw))
    if not trailers:
        return None
    latest = trailers[-1]
    if raw[latest.end():].strip():
        return None
    return int(latest.group(1))


def _refusal_reason_for_latest_log_run(
    raw: Optional[str], *, current_run_id: Optional[int] = None,
) -> Optional[str]:
    """Machine-readable refusal reason for the current/latest log run."""
    matches = _REFUSAL_REASON_RE.findall(
        _latest_log_run_segment(raw, current_run_id=current_run_id),
    )
    return matches[-1] if matches else None


def _worker_log_refusal_reason(
    task_id: str, board: Optional[str] = None, *, current_run_id: Optional[int] = None,
) -> Optional[str]:
    """Machine-readable active-session refusal reason from the worker log."""
    try:
        raw = _kb().read_worker_log(task_id, tail_bytes=4000, board=board)
    except OSError:
        return None
    return _refusal_reason_for_latest_log_run(raw, current_run_id=current_run_id)


def _worker_log_exit_code(
    task_id: str, board: Optional[str] = None, *, current_run_id: Optional[int] = None,
) -> Optional[int]:
    """Exit code from the trailer the worker CLI wrote to its own log, if present."""
    try:
        raw = _kb().read_worker_log(task_id, tail_bytes=4000, board=board)
    except OSError:
        return None
    return _latest_log_exit_code(raw, current_run_id=current_run_id)


def _exit_summary_marker() -> str:
    """The CLI exit-summary header in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _worker_final_output(
    task_id: str, board: Optional[str] = None, *, current_run_id: Optional[int] = None,
) -> str:
    """Best-effort, current-run diagnostic text from a dead worker's log."""
    try:
        raw = _kb().read_worker_log(task_id, tail_bytes=4000, board=board)
    except OSError:
        return ""
    if not raw:
        return ""
    raw = _latest_log_run_segment(raw, current_run_id=current_run_id)
    raw = _EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for line in raw.splitlines():
        line = _LOG_CHROME.sub("", line).strip()
        if line and not line.startswith(_log_noise_prefixes()):
            lines.append(line)
    return " ".join(lines)[-400:]
