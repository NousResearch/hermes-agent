"""Attempt-scoped diagnostics for append-only Kanban worker logs."""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


WORKER_ATTEMPT_START = "[hermes-kanban-worker-start] run_id="
_WORKER_ATTEMPT_RE = re.compile(
    r"^" + re.escape(WORKER_ATTEMPT_START) + r"(?:\d+|none)[ \t]*$", re.MULTILINE,
)
_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _latest_worker_attempt(raw: str) -> str:
    """Isolate this attempt before trimming summaries or looking up its exit code."""
    starts = list(_WORKER_ATTEMPT_RE.finditer(raw))
    if starts:
        return raw[starts[-1].end():]
    # Older logs have no start marker. A trailer at EOF ends this attempt;
    # a trailer followed by output ends the previous attempt instead.
    trailers = list(_EXIT_TRAILER_RE.finditer(raw))
    if trailers:
        boundary = trailers[-1] if raw[trailers[-1].end():].strip() else (
            trailers[-2] if len(trailers) > 1 else None
        )
        if boundary is not None:
            return raw[boundary.end():]
    return _legacy_worker_attempt(raw)


def _legacy_worker_attempt(raw: str) -> str:
    """Pre-trailer releases ended quiet runs with session_id and full runs with counts."""
    from agent.i18n import t

    counts = re.escape(t(
        "cli.session.exit_label_messages", count="__COUNT__", user="__USER__",
        tool_calls="__TOOLS__",
    ))
    for token in ("__COUNT__", "__USER__", "__TOOLS__"):
        counts = counts.replace(token, r"\d+")
    footer = rf"(?:session_id:[ \t]*\S+|{counts})"
    # Adjacent summary/session_id lines are one footer, not two attempts.
    footers = list(re.finditer(rf"^[ \t]*{footer}(?:\s+{footer})*[ \t]*$", raw, re.MULTILINE))
    if footers:
        suffix = _EXIT_TRAILER_RE.sub("", raw[footers[-1].end():]).strip()
        boundary = footers[-1] if suffix else (footers[-2] if len(footers) > 1 else None)
        if boundary is not None:
            return raw[boundary.end():]
    return raw


def _worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Durable exit witness for a sweep that did not reap the worker itself.

    Only the newest attempt's trailer counts. A worker killed before its
    epilogue must not inherit an earlier attempt's clean or provider exit.
    """
    from hermes_cli import kanban_db as kb

    try:
        raw = kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return None
    matches = _EXIT_TRAILER_RE.findall(_latest_worker_attempt(raw or ""))
    return int(matches[-1]) if matches else None


def _exit_summary_marker() -> str:
    """The CLI exit-summary header in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's latest output for the board diagnostic.

    A new start marker (flushed before spawn) fences diagnostics to the newest
    run, even when it exits before producing output or an exit trailer. Older
    logs fall back to exit trailers or the CLI's summary/session_id footer.
    Trim CLI chrome only after selecting the attempt: an older summary must
    never hide a newer pre-summary startup error. Missing/empty logs return "".

    ``board`` comes from the dispatching tick, never ambient current-board
    resolution, which would silently read the wrong log in a multi-board tick.
    """
    from hermes_cli import kanban_db as kb

    try:
        raw = kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return ""
    if not raw:
        return ""
    raw = _latest_worker_attempt(raw)
    raw = _EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            lines.append(ln)
    return " ".join(lines)[-400:]
