"""Worker transport: how a host-side refusal is classified, booked, and recovered from.

``kanban_db_dispatch`` is over its ``FILE_LINES`` cap on main and may only go down, so the
transport lane lives here: the submitter's refusal exit codes and exception, the dead-worker
booking record, the classification of a refusal by *phase* (never submitted vs admitted-then-lost),
the host's own pressure numbers, and the durable sentence the retry worker reads.

Why this lane exists at all: the submitter funnels every transport-shaped failure into ``rc 1``
(``GatewayClientError`` subclasses ``ValueError``, so ``kanban_worker_client._submit``'s
``except (KeyError, ValueError, OSError)`` catches it), and ``rc 1`` books as a card crash. A host
that stalls for one tick therefore costs every card in that tick a counted failure, and the
systemic-identical-error breaker trips at 1. Four distinguishable outcomes share one exit code.

Phase is what separates them, and phase is already observable: the submitter prints
``Session: <id>`` only after ``kanban.run`` was answered, i.e. only after the owner checked the
claim and wrote its ``owner_admitted`` marker. The marker is the authority a booking reads, not the
exit code: an admitted attempt can still die as a raw non-restartable transport error, which maps
to 71 ("never reached the owner") whatever phase it died in.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

# Worker exit "the HOST refused the submitter before the owner ever saw the claim": the gateway
# websocket could not be opened (connect failure, open timeout, no ready owner). Nothing of this
# card ran, so an immediate retry duplicates nothing. 71 == BSD EX_OSERR — a system error at the
# host's edge, not a task failure. Booked on the infrastructure lane: recorded, NOT counted, the
# breaker cannot trip on it.
KANBAN_HOST_REFUSAL_EXIT_CODE = 71

# Worker exit "the owner admitted the claim and the outcome then became ambiguous": the transport
# died after ``kanban.run`` was answered, the answer never arrived although the frame was sent, or
# the owner's admission write never settled the board lock and it retained the attempt. The turn
# may have completed and a PUSH HEARTBEAT worker may already have pushed its branch, so the retry
# must RESUME the printed session instead of re-deriving the work. 70 == BSD EX_IOERR. Also
# infrastructure (never counted), but reconciled: ``check_respawn_guard`` spaces it.
KANBAN_TURN_UNKNOWN_EXIT_CODE = 70

# The dispatcher books both refusal kinds under this run outcome and event kind. A distinct kind is
# the point: board history that reads "crashed" is what made an infrastructure burst look, three
# times over, like cards failing.
TRANSPORT_REFUSED = "transport_refused"

#: ``_exit_code_kind`` kind for each refusal code.
REFUSAL_KIND_BY_CODE = {
    KANBAN_HOST_REFUSAL_EXIT_CODE: "host_refusal",
    KANBAN_TURN_UNKNOWN_EXIT_CODE: "turn_unknown",
}

# ``Session: <id>`` as the submitter prints it to stderr, which lands in the task worker log.
_SESSION_LINE_RE = re.compile(r"^Session:\s*(\S+)", re.MULTILINE)


def refusal_kind(code: Optional[int]) -> Optional[str]:
    """The refusal ``kind`` for exit ``code``, or None when it is not a refusal code.

    Only the submitter can exit with these codes, and no card's own work can produce them — which
    is what lets a refusal seen in ANY channel (the reap registry, the authority's
    ``worker_result``, or the worker log's exit trailer) outrank the interpreter's rc.
    """
    return REFUSAL_KIND_BY_CODE.get(int(code)) if code is not None else None


@dataclass
class DeadWorker:
    """How ``detect_crashed_workers`` should book one dead worker."""

    kind: str
    code: Optional[int]
    error_text: str
    event_kind: str
    event_payload: dict
    protocol_violation: bool = False
    rate_limited: bool = False
    terminal_provider: bool = False
    """``KANBAN_TERMINAL_PROVIDER_EXIT_CODE``: the provider rejected the worker's
    credential/model — trips the breaker on this first occurrence."""
    transport_refusal: bool = False
    """A host refusal (:func:`refusal_kind`): booked on the infrastructure lane, never counted."""
    reconcile_required: bool = False
    """The owner admitted the attempt before its outcome went ambiguous, so the retry has to
    reconcile that session rather than re-derive the work."""

    @property
    def run_outcome(self) -> str:
        # A rate-limited requeue is recorded as ``rate_limited`` and a host refusal as
        # ``transport_refused``, so board history does not show a phantom crash for a quota wall
        # or for the host's own transport.
        if self.rate_limited:
            return "rate_limited"
        return TRANSPORT_REFUSED if self.transport_refusal else "crashed"


def admitted_session_id(conn, task_id: str, board: Optional[str] = None) -> Optional[str]:
    """The session id the owner admitted for this task's current run, best effort.

    The ``owner_admitted`` event is the authority: the owner writes it in the same board
    transaction that commits the claim, so a run carrying one cannot be "never submitted". The
    worker log's ``Session:`` line is the fallback for an admission whose receipt ended
    ``unknown``. Callers must read this BEFORE the run closes: ``current_run_id`` is cleared with
    it.
    """
    from hermes_cli import kanban_db as _kb

    row = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id = e.run_id "
        "WHERE t.id = ? AND e.task_id = ? AND e.kind = 'owner_admitted' "
        "ORDER BY e.id DESC LIMIT 1",
        (task_id, task_id),
    ).fetchone()
    if row is not None:
        sid = _kb._json_dict(_kb._row_get(row, "payload")).get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    try:
        log = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except (OSError, ValueError, sqlite3.Error):
        # Board/path resolution and the log read can each fail their own way; this runs inside the
        # reclaim sweep, so a diagnostic lookup must never take the sweep down with it.
        return None
    found = _SESSION_LINE_RE.findall(log or "")
    return found[-1] if found else None


def _reconcile_text(pid: int, code: int, session_id: Optional[str]) -> str:
    """The retry worker's instruction for an admitted-then-ambiguous attempt.

    Stated first in the sentence because ``_record_task_failure`` truncates ``last_failure_error``
    at 500 characters and the sweep appends the worker's own log text after it: "do not redo" and
    the session id have to survive the cut — they are the whole recoverable outcome.
    """
    if session_id:
        return (
            f"pid {pid} exit {code}: admitted then lost its transport — the turn may have "
            f"completed and already pushed. Do not redo the work: reconcile or resume session "
            f"{session_id} first. Host refusal, not a card failure (nothing counted)."
        )
    return (
        f"pid {pid} exit {code}: admitted then lost its transport — the turn may have completed "
        f"and already pushed. Do not redo the work: check the branch and session this attempt "
        f"printed before resending it. Host refusal, not a card failure (nothing counted)."
    )


def classify_refusal(
    kind: str, code: int, pid: int, claimer: Optional[str], *,
    task_id: Optional[str] = None, board: Optional[str] = None, conn=None,
) -> DeadWorker:
    """Book a dead worker whose exit code is a host refusal: infrastructure, never counted.

    The two kinds need opposite retry policies, so the booking states which: a never-submitted
    claim is safe to resend at once, an admitted one must be reconciled. The sentence reaches the
    retry worker as the run error in ``## Prior attempts``; the session id, the reconcile flag and
    the host's pressure numbers reach it (and the board UI) as run metadata.
    """
    # The board's durable evidence outranks the code the submitter died with. The two disagree on a
    # reachable path: ``kanban_worker_client.run`` re-raises a NON-restartable dial raw, so an
    # attempt the owner admitted (marker committed, ``Session:`` printed) exits with a plain
    # ``GatewayUnavailableError`` and ``submitter_exit_code`` calls that 71 — "never reached the
    # owner", which is false, and whose "a retry duplicates nothing" sentence invites a second
    # worker to redo a possibly-pushed turn. A committed ``owner_admitted`` marker means the owner
    # saw this claim, whatever the exit code says; the reclaim fence cannot stand in for that
    # reading, because it steps aside once the worker is provably dead — which is exactly this
    # aftermath. Read the marker once; it is also the session id the retry has to resume.
    session_id = admitted_session_id(conn, task_id, board=board) if conn is not None and task_id else None
    reconcile = kind == "turn_unknown" or session_id is not None
    payload: dict[str, Any] = {
        "pid": pid, "claimer": claimer, "exit_kind": kind, "exit_code": code,
        "infrastructure": True, "reconcile_required": reconcile, "transport_refusal": True,
    }
    if session_id:
        payload["session_id"] = session_id
    error_text = (
        _reconcile_text(pid, code, session_id) if reconcile
        else (
            f"pid {pid} exit {code}: the host refused the submitter before the owner saw the "
            f"claim (no gateway connection, or no ready owner) — nothing of this attempt ran, so "
            f"a retry duplicates nothing. Host refusal, not a card failure (nothing counted)."
        )
    )
    return DeadWorker(
        kind, code, error_text, TRANSPORT_REFUSED, payload,
        transport_refusal=True, reconcile_required=reconcile,
    )


# --------------------------------------------------------------------------- submitter side

class KanbanHostRefusalError(RuntimeError):
    """The host refused the submitter before the owner saw the claim: nothing was submitted."""


class KanbanTurnUnknownError(RuntimeError):
    """The owner admitted (or may have admitted) the claim and the outcome became ambiguous.

    ``session_id`` is the one the submitter printed — exactly what the operator, and the retry
    worker, must resume instead of re-running.
    """

    def __init__(self, message: str, *, session_id: Optional[str] = None):
        super().__init__(message)
        self.session_id = session_id


def submitter_exit_code(exc: BaseException) -> int:
    """Worker exit code for an error that ended a claim submission.

    Ordered by what the host actually managed to do, because that is what decides whether a retry
    can duplicate work:
    * :class:`KanbanTurnUnknownError` — admitted, or the frame went out and no answer came back;
    * :class:`GatewayRPCError` — the owner ANSWERED and refused: a real problem with this claim, so
      it stays the counted ``rc 1`` and the breaker still reaches a genuinely poisoned card;
    * :class:`KanbanHostRefusalError` / :class:`GatewayUnavailableError` — no connection or no
      ready owner, nothing submitted;
    * anything else — the submitter's own bug, or an environment error: counted, as before.

    ``GatewayRPCError`` is a ``GatewayClientError`` subclass, so it must be tested first.
    """
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, GatewayUnavailableError

    if isinstance(exc, (KanbanTurnUnknownError, TimeoutError)):
        # A timeout means the request was sent and unanswered: the owner may admit after the
        # submitter gave up, which is precisely the "admission left unknown" case. Not "nothing ran".
        return KANBAN_TURN_UNKNOWN_EXIT_CODE
    if isinstance(exc, GatewayRPCError):
        return 1
    if isinstance(exc, (KanbanHostRefusalError, GatewayUnavailableError)):
        return KANBAN_HOST_REFUSAL_EXIT_CODE
    if isinstance(exc, GatewayClientError):
        # An ambiguous transport failure with no admission recorded (the websocket would not open).
        return KANBAN_HOST_REFUSAL_EXIT_CODE
    return 1


# --------------------------------------------------------------------------- host pressure

_PSI_FULL_AVG60_RE = re.compile(r"^full\s+avg10=\S+\s+avg60=(\S+)", re.MULTILINE)


def _psi_full_avg60(resource: str) -> Optional[float]:
    """PSI ``full`` stall percentage (avg60) for ``memory``/``io``; None where procfs has no PSI."""
    try:
        with open(f"/proc/pressure/{resource}", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    match = _PSI_FULL_AVG60_RE.search(text)
    if not match:
        return None
    with suppress(ValueError):
        return float(match.group(1))
    return None


def host_pressure_sample() -> dict[str, Any]:
    """The host's own numbers, best effort and never raising: {} where the kernel exposes none.

    Recorded into each refused run's metadata so the next operator reads a cause instead of
    "Gateway disconnected" — the misreading that cost this fleet three investigations. Keys match
    what ``reclaim-host-space.sh`` and the gateway heartbeat already print (PSI stall percentages,
    load average, memory/swap in KiB), so a burst lines up against those logs with no translation.

    Every leg is guarded because the caller is not a diagnostics command: ``_reclaim_dead_workers``
    samples INSIDE its ``write_txn``, so a raise here aborts the crash sweep for every card on the
    host — the failure mode this lane exists to remove. A leg that fails costs its own keys only.
    """
    sample: dict[str, Any] = {}
    with suppress(OSError, AttributeError):
        load = os.getloadavg()
        sample["load1"], sample["load15"] = load[0], load[2]
    for resource in ("memory", "io"):
        value = _psi_full_avg60(resource)
        if value is not None:
            sample[f"psi_{resource}_full_avg60"] = value
    try:
        from gateway.lifecycle_ledger import sample_memory
        mem = sample_memory()
    except Exception:  # health: allow BLE001 -- best-effort host telemetry inside the reclaim txn
        return sample
    # ``rss_kib`` is the *sweeping* process's own resident size, not the fleet's: recording it
    # against a burst invites exactly the misreading this sampler exists to prevent.
    mem.pop("rss_kib", None)
    sample.update(mem)
    return sample


# --------------------------------------------------------------------------- worker channels
#
# How an outcome reaches the dispatcher when the socket could not carry it: the exit trailer and
# the final printed text the worker CLI left in its own per-task log. Moved here verbatim from
# ``kanban_db_dispatch`` (which is over its ``FILE_LINES`` cap and may only go down); the patch
# seam for tests is this module, and the dispatcher calls these through it.

EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)


def worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Exit code from the trailer the worker CLI wrote to its own log; None when absent.

    The durable twin of ``_recent_worker_exits``: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running this sweep ever reaped the worker. Last trailer
    wins — the log is append-mode across re-runs.
    """
    from hermes_cli import kanban_db as _kb

    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:  # health: allow BLE001 -- best-effort diagnostic read; must never break the sweep
        return None
    matches = EXIT_TRAILER_RE.findall(raw or "")
    return int(matches[-1]) if matches else None


# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's last printed text, for the board diagnostic.

    A ``chat -q`` worker's stdout/stderr are redirected to its per-task log
    (``_default_spawn``), so when it exits without a terminal board call the
    reason is usually sitting there: the model's own explanation of why it could
    not comply (#88603), or the rendered provider error (#46593). The reap used to
    discard it in favour of a canned message on every retry. Trims the CLI exit
    summary, rule lines and the ``session_id:`` trailer; returns "" (never raises)
    on a missing/empty log.

    ``board`` must come from the dispatching tick: ambient current-board resolution
    is wrong for every board but the one the dispatcher thread happens to call
    "current", so the log would silently not be found.
    """
    from hermes_cli import kanban_db as _kb

    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:  # health: allow BLE001 -- best-effort diagnostic read; must never break the sweep
        return ""
    if not raw:
        return ""
    raw = EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            lines.append(ln)
    return " ".join(lines)[-400:]


@dataclass
class CrashSweep:
    """Everything ``detect_crashed_workers`` collects inside its reclaim txn."""

    crashed: list[str] = field(default_factory=list)
    rate_limited: list[str] = field(default_factory=list)
    # ``(task_id, pid, claimer, dead_worker)``: accounted after the txn via
    # ``_record_task_failure`` (needs its own write_txn).
    crash_details: list[tuple[str, int, str, DeadWorker]] = field(default_factory=list)
    # Host refusals, same tuple shape, accounted on the infrastructure lane instead of
    # ``crash_details``: recorded and retryable, never counted against the card.
    transport_refused: list[tuple[str, int, str, DeadWorker]] = field(default_factory=list)
    # Worker-exit observer payloads, fired only after every reclaim/accounting
    # txn has committed.
    exited_hook_payloads: list[dict] = field(default_factory=list)
