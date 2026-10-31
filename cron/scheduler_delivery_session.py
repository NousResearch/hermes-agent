"""Cron delivery into the API-server session that scheduled the job.

An ``api_server`` session cannot receive a pushed message (``supports_async_delivery = False``,
``send()`` is a stub), so #69304 stopped stamping it as a cron origin. One case is still
reachable: a request that declared a server-history consumer (``session_history_delivery="1"``,
the gate async ``delegate_task`` uses, #98619). Its client reads the session transcript, so the
result is appended there through ``gateway.wake.persist_delegation_delivery``, the path detached
delegation results already take. No self-POST: that would start an agent turn the client did not
ask for (see the ``gateway/wake.py`` docstring). Nothing is pushed; the client sees the row on its
next ``GET /api/sessions/{id}/messages`` and the next turn folds it in as context.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import Optional

logger = logging.getLogger("cron.scheduler")

# ``_resolved_from`` tag of a session-transcript target; ``_deliver_result`` routes it here.
SESSION_ORIGIN = "origin_session"

# The append is refused while the session holds a live turn lease (the user is mid-turn) and when a
# compaction closes the tip between resolve and write. Both clear on their own, and nothing upstream
# retries a cron delivery error, so wait out about a minute before reporting one.
_BUSY_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0, 20.0, 30.0)
# One append is a single SessionDB write; the bound only stops a wedged gateway loop holding the thread.
_APPEND_TIMEOUT_SECONDS = 30.0


def session_target(job: dict) -> Optional[dict]:
    """The scheduling session as a delivery target, when the job's origin declared server history."""
    origin = job.get("origin")
    if not (isinstance(origin, dict) and origin.get("session_history") is True
            and str(origin.get("platform") or "").lower() == "api_server"):
        return None
    session_id = str(origin.get("chat_id") or "").strip()
    if not session_id:
        return None
    return {"platform": "api_server", "chat_id": session_id, "thread_id": None,
            "_resolved_from": SESSION_ORIGIN}


def deliver_to_session(job: dict, target: dict, text: str, *, adapters, loop, config,
                       for_failure: bool = False) -> Optional[str]:
    """Append one cron result to the target session's transcript. None on success, else an error.

    The write runs on the gateway loop that owns the api_server adapter, the way the live send lane
    and the other cron adapter calls run theirs (``safe_schedule_threadsafe`` + a bounded wait).
    """
    from agent.async_utils import safe_schedule_threadsafe
    from cron import scheduler as _sched
    from cron.scheduler_delivery import _resolve_target_transport
    from gateway.config import Platform
    from gateway.wake import persist_delegation_delivery
    from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError

    session_id = str(target["chat_id"])
    where = f"api_server session {session_id}"
    resolved, _ = _resolve_target_transport(
        job, Platform.API_SERVER, "api_server", target, adapters, config)
    adapter = resolved[2] if resolved else None
    if adapter is None or loop is None:
        return (f"cannot add the result to {where}: no gateway with the API server adapter is "
                "running in this process (start the gateway with the API server enabled); the "
                "output is saved in last_output")
    # Every scheduler path stamps execution_id before delivery (run_one_job / _run_one_job_body),
    # so a retried delivery of one fire matches its row and is not appended twice. A direct caller
    # without one gets a fresh key: delivered, but not de-duplicated.
    unit = f"cron:{job.get('id', '?')}:{job.get('execution_id') or 'unkeyed-' + uuid.uuid4().hex}"
    evt = {"delegation_id": unit + (":failure" if for_failure else ""),
           "results": [{"status": "failed" if for_failure else "completed"}]}
    last_error: Optional[Exception] = None
    for delay in (0.0, *_BUSY_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        if _sched._interpreter_shutting_down():
            return f"delivery to {where} skipped: the interpreter is shutting down"
        future = safe_schedule_threadsafe(
            persist_delegation_delivery(adapter, text=text, session_id=session_id, evt=evt), loop,
            logger=logger, log_message=f"Job '{job.get('id', '?')}': scheduling the {where} append failed")
        if future is None:
            return f"cannot add the result to {where}: the gateway event loop is not running"
        try:
            future.result(timeout=_APPEND_TIMEOUT_SECONDS)
        except (SessionTurnLeaseLostError, CompressionSessionClosedError) as exc:
            last_error = exc
            logger.info("Job '%s': %s is busy (%s); retrying", job.get("id", "?"), where,
                        type(exc).__name__)
            continue
        except TimeoutError:
            return (f"adding the result to {where} timed out after {int(_APPEND_TIMEOUT_SECONDS)}s "
                    "on the gateway loop; the append may still land")
        except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
            if _sched._interpreter_shutting_down(exc):
                return f"delivery to {where} skipped: the interpreter is shutting down"
            logger.warning("Job '%s': adding the result to %s failed", job.get("id", "?"), where,
                           exc_info=True)
            return f"cannot add the result to {where}: {type(exc).__name__}: {exc}"
        logger.info("Job '%s': result added to %s", job.get("id", "?"), where)
        return None
    waited = int(sum(_BUSY_RETRY_DELAYS_SECONDS))
    return (f"{where} stayed busy for {waited}s ({last_error}); the result was not added to it "
            "and is saved in last_output")
