"""Session-stream run lifecycle helpers for the API server adapter.

Extracted from ``gateway/platforms/api_server.py`` (godfile gate — epic
#78647, precedent #83546) plus the durable active-run control surface for the
session SSE stream (PR #96507 P1 follow-up).

Everything here is transport-independent: none of these helpers import
aiohttp, and the two payload-builders return plain dicts, so this module stays
small and importable from tests without pulling in the web stack.

Vocabulary mirrors PR #15492's ``ResponseRun``/subscriber separation: a run is
a durable server-side control object whose identity survives the SSE
subscriber lifecycle; the SSE connection is only a transport. Status values
are ``queued``/``running``/``completed``/``failed``/``cancelled``.
"""

import asyncio
from contextlib import suppress
from hashlib import sha256
from typing import Any, Dict, Optional


async def drain_session_stream_task_on_disconnect(
    adapter: Any,
    run_id: str,
    task: "asyncio.Task",
    *,
    interrupt_message: str,
    shield_wait: bool,
) -> None:
    """Preserve live run control refs until the executor-backed turn exits.

    Used on server shutdown (task cancellation), where the gateway is going
    away and letting the turn finish is pointless: interrupt the agent and
    wait for the executor-backed turn to drain.
    """
    agent = adapter._active_run_agents.get(run_id)
    if agent is None:
        if not task.done():
            task.cancel()
            with suppress(Exception):
                await task
        return
    with suppress(Exception):
        agent.interrupt(interrupt_message)
    if not task.done():
        with suppress(Exception):
            await (asyncio.shield(task) if shield_wait else task)


async def detach_session_stream_task_on_disconnect(
    adapter: Any,
    run_id: str,
    queue: "asyncio.Queue",
) -> None:
    """Detach a client-disconnected session stream without interrupting it.

    The session endpoint always persists to state.db, so a dropped SSE
    connection is only a dead transport, not a stop signal (ref issue #15026).
    The agent turn runs in ``_run_and_signal`` — already a
    ``_background_tasks`` member independent of the request handler — and keeps
    producing events into *queue*. Drain those events until the end sentinel so
    they don't accumulate in memory for the remainder of the turn. The Stop
    button halts a detached run via ``POST /v1/runs/{run_id}/stop``.
    """
    del run_id  # the drain loop needs no run id — it only consumes the queue

    async def _drain() -> None:
        with suppress(Exception):
            while True:
                if await queue.get() is None:
                    break

    drain_task = asyncio.create_task(_drain())
    try:
        adapter._background_tasks.add(drain_task)
    except TypeError:
        pass
    if hasattr(drain_task, "add_done_callback"):
        drain_task.add_done_callback(adapter._background_tasks.discard)


def session_run_key(
    session_id: str,
    user_message: Any,
    system_prompt: Optional[str],
    idempotency_header: Optional[str],
) -> str:
    """Derive the immutable request fingerprint for a session-stream run.

    Honors an explicit client ``Idempotency-Key`` header (hashed so raw client
    values never sit in the session row). Without one, falls back to a
    deterministic fingerprint of the admitted request — (session_id,
    system_prompt, message) — so a client retry of the same session/message
    maps to the same key and can be recognized as a duplicate.
    """
    if idempotency_header:
        return f"idem-{sha256(idempotency_header.encode('utf-8')).hexdigest()}"
    seed = repr((session_id, system_prompt or "", user_message))
    return f"fp-{sha256(seed.encode('utf-8')).hexdigest()}"


def run_already_active_error(run_id: str) -> Dict[str, Any]:
    """Deterministic conflict envelope for a duplicate session-stream start."""
    return {
        "error": {
            "message": "A run is already active for this session",
            "type": "invalid_request_error",
            "param": None,
            "code": "run_already_active",
            "run_id": run_id,
        },
        "run_id": run_id,
    }


async def claim_session_run_or_conflict(
    adapter: Any,
    session_id: str,
    run_id: str,
    run_key: str,
) -> Optional[str]:
    """Atomically claim the session's active-run slot, or report the live holder.

    Returns ``None`` when ``run_id`` won the claim. Returns the existing live
    run id (str) when another run already holds the slot and is still
    executing — the caller must surface a ``run_already_active`` conflict for
    that id. A stale marker (a prior run that ended without clearing, or a
    gateway restart) is reclaimed in place so a session can never be wedged
    behind a dead run id.
    """
    existing = await adapter._claim_session_active_run_async(
        session_id, run_id, run_key, "queued"
    )
    if existing and adapter._session_run_is_live(existing):
        return existing
    if existing:
        # Stale marker — reclaim it for this run, then re-claim. The second
        # claim serializes against a genuinely-live concurrent winner.
        await adapter._clear_session_active_run_async(
            session_id, expected_run_id=existing
        )
        existing = await adapter._claim_session_active_run_async(
            session_id, run_id, run_key, "queued"
        )
        if existing and adapter._session_run_is_live(existing):
            return existing
    return None

