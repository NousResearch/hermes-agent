"""Per-request busy policy for ``POST /v1/runs`` (``if_busy``) and the queued-run withdraw
(``POST /v1/runs/{run_id}/cancel``).

A run aimed at a session whose turn is still running has always waited in that session's FIFO.
``if_busy`` lets the caller pick, for that one request, what happens instead:

- ``queue``: the run waits its turn (what an absent ``if_busy`` does too);
- ``steer``: the text joins the running turn at its next tool boundary and this run settles with it;
- ``interrupt``: the running turn is stopped and this run keeps its FIFO place.

Steer and interrupt only act on a running turn the same caller started (the same run owner scope
``POST /v1/runs/{id}/stop`` and ``/steer`` already require); any other busy session, and a run
carrying images, stays queued. The 202 names what happened in ``delivery``.
"""
import asyncio
import hmac
import logging
import sqlite3

from gateway.platforms.api_server_authority_runs import _authority, run_admission, run_projection
from gateway.session_contract import SessionActor, SessionRef
from gateway.session_queued_controls import transfer_to_running
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

logger = logging.getLogger(__name__)

IF_BUSY_VALUES = ("queue", "steer", "interrupt")


def parse_if_busy(body):
    """``(value or None, error message or None)`` for the request's optional ``if_busy``."""
    value = body.get("if_busy")
    if value is None or value in IF_BUSY_VALUES:
        return value, None
    return None, "'if_busy' must be one of: " + ", ".join(IF_BUSY_VALUES)


def _owner_scope(row):
    return ((row.get("payload") or {}).get("api_turn_v1") or {}).get("run_owner_scope") or ""


def _running_head(rows, row):
    """The started claim ahead of *row* in its FIFO, ``None`` when nothing is running ahead, or
    ``False`` when the session is idle (nothing at all ahead: the run starts)."""
    ahead = [r for r in rows if r["seq"] < row["seq"]]
    if not ahead:
        return False
    return next((r for r in ahead if r["status"] == "started"), None)


async def _steer(adapter, authority, ref, row, head, launch):
    try:
        await transfer_to_running(authority, ref, row["admission_id"], launch.user_message,
                                  generation=head["generation"])
    except RuntimeStoreError:
        return "queued"  # refused before the row moved: it keeps its FIFO place
    return "steered"


async def _interrupt(adapter, authority, ref, row, head, launch):
    actor = SessionActor("api", authority.profile_id, frozenset({"session:submit", "session:control"}),
                         "api-run:" + launch.run_id)
    try:
        await authority.interrupt(actor, ref, head["generation"])
    except RuntimeStoreError:
        return "queued"  # the head settled meanwhile: this run is next anyway
    adapter._stopping_run_ids.add(head["request_id"])  # as /stop marks the run it interrupts
    return "interrupted"  # the head's own run settles ``cancelled``; this one runs after it


_BUSY_ACTIONS = {"steer": _steer, "interrupt": _interrupt}


async def apply_if_busy(launch, if_busy):
    """Apply *if_busy* to an admitted, launched run -> its ``delivery`` (``None`` when the request
    sent no ``if_busy``: the 202 body stays exactly what it was). The decision is recorded on the
    run status, so an exact keyed retry (``_replay_or_conflict``) reports it without acting again."""
    if if_busy is None:
        return None
    delivery = await _decide(launch, if_busy)
    owner = launch.owner
    owner._set_run_status(launch.run_id, owner._run_statuses[launch.run_id]["status"], delivery=delivery)
    return delivery


async def _decide(launch, if_busy):
    if launch.admission is None:
        return "started"  # in-process execution (no session authority) or a live Bot Chat owner
    authority, ref, row = launch.admission
    try:
        rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
    except (sqlite3.Error, RuntimeStoreError):
        # The run is admitted and launched: it runs in FIFO order, which is the queue policy.
        logger.warning("if_busy: could not read the FIFO of %s; run %s stays queued", ref.session_id, launch.run_id)
        return "queued"
    head = _running_head(rows, row)
    if head is False:
        return "started"
    action = _BUSY_ACTIONS.get(if_busy)
    if (action is None or not head or head["principal_id"] != "api" or not isinstance(launch.user_message, str)
            or not _owner_scope(row) or not hmac.compare_digest(_owner_scope(head), _owner_scope(row))):
        return "queued"
    return await action(launch.owner, authority, ref, row, head, launch)


async def cancel_queued_run(adapter, run_id):
    """Withdraw the still-queued run *run_id* -> its cancelled projection, or ``None`` when it is
    not queued (running, settled, steered, or served without the session authority)."""
    owned = run_admission(adapter, run_id) if _authority(adapter) is not None else None
    if owned is None:
        return None
    authority, row = owned
    if row["status"] != "queued":
        return None
    ref = SessionRef(authority.profile_id, row["target_session_id"])
    actor = SessionActor("api", authority.profile_id, frozenset({"session:submit", "session:control"}),
                         "api-run:" + run_id)
    try:
        await authority.cancel_queued(actor, ref, row["admission_id"])
    except RuntimeStoreError as exc:
        if exc.reason != "stale_generation":
            raise
        return None  # the drain claimed it between the read and the cancel: it is running now
    adapter._stopping_run_ids.add(run_id)
    return {**(run_projection(adapter, run_id) or {}), "reason": "withdrawn"}

