"""Worker-log diagnostics for the Kanban dispatcher: keep the reason, drop the handle.

Split out of ``hermes_cli.kanban_db_dispatch`` (facade line budget; see root AGENTS.md).
The facade's ``_worker_final_output`` owns the log read, because the late-bound
``kanban_db.read_worker_log`` seam lives there; this module owns the text scrubbing.
"""

from __future__ import annotations

import re

# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")

# A managed worker prints its own session handle (``Session: local-<sha256>``) on every owner
# refusal, and several refusals stack in one attempt's log. The handle is an internal
# identifier, not evidence: it must not reach the board's durable error text or event payload,
# and the repeated copies crowded the real reason out of the 400-char tail (a stuck card read
# as five identical refusals). Redact the handle, keep the reason.
_SESSION_HANDLE = re.compile(r"\blocal-[0-9a-f]{32,64}\b")

# The board diagnostic's durable bound (``_record_task_failure`` caps the stored error at 500).
_DIAGNOSTIC_CHARS = 400


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _collapse_adjacent(lines: list[str]) -> list[tuple[str, int]]:
    """One run of identical adjacent lines becomes one entry plus its repeat count."""
    collapsed: list[tuple[str, int]] = []
    for ln in lines:
        if collapsed and collapsed[-1][0] == ln:
            prev, count = collapsed[-1]
            collapsed[-1] = (prev, count + 1)
        else:
            collapsed.append((ln, 1))
    return collapsed


def final_output(raw: str, exit_trailer: re.Pattern[str]) -> str:
    """Trim one worker log's text into the board diagnostic (never raises).

    Drops the CLI exit summary, rule/chrome lines and the ``session_id:`` trailer, redacts
    printed session handles, then collapses adjacent repeats so a repeated refusal keeps its
    reason and its count instead of crowding the reason out of the 400-character tail.
    """
    raw = exit_trailer.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            # Redact BEFORE dedupe: identical refusals differ only by their session handle.
            lines.append(_SESSION_HANDLE.sub("local-<redacted>", ln))
    # One event echoed per refusal is one diagnostic: keep it once, note the count.
    return " ".join(ln if n == 1 else f"{ln} (x{n})" for ln, n in _collapse_adjacent(lines))[-_DIAGNOSTIC_CHARS:]
