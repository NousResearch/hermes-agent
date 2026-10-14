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
import json
from collections.abc import Mapping
from contextlib import suppress
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from aiohttp import web


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


def canonical_execution_value(value: Any) -> Any:
    """Deterministically canonicalise a JSON-ish value for fingerprinting.

    Mapping keys are ordered by their TEXT form (so equivalent
    ``model_options`` dicts built in a different insertion order hash the
    same), sequences keep their order, and scalars pass through unchanged.
    Crucially this replaces the old order-sensitive ``repr()`` of nested
    mappings, which made semantically identical retries look like changed
    requests (review F2).
    """
    if isinstance(value, Mapping):
        return {
            str(key): canonical_execution_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_execution_value(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


# Fields of the resolved runtime request that define what will be executed.
# ``requested`` carries the aliases already resolved (``model`` OR ``model_id``,
# ``provider`` OR ``provider_id``, plus the raw alias and the provider/model
# split), ``route`` is the concrete route execution will take, and
# ``require_model_lock``/``lock_active`` capture the confirmed-lock semantics
# that forward that resolved route into the turn (review F2).
_RUNTIME_IDENTITY_FIELDS = (
    "requested",
    "route",
    "route_source",
    "runtime_options",
    "require_model_lock",
    "model_options",
)


def resolved_execution_identity(
    runtime_request: Optional[Dict[str, Any]],
    lock_active: Any = None,
) -> Dict[str, Any]:
    """Project the RESOLVED execution request onto a stable comparison shape.

    Only the request-defining fields are kept: adding a new field to the
    runtime request cannot silently change fingerprints, and no volatile
    per-request bookkeeping leaks into the identity.
    """
    runtime_request = runtime_request if isinstance(runtime_request, Mapping) else {}
    identity: Dict[str, Any] = {
        field: runtime_request.get(field) for field in _RUNTIME_IDENTITY_FIELDS
    }
    identity["lock_active"] = bool(lock_active)
    return identity


def session_request_fingerprint(
    session_id: str,
    user_message: Any,
    system_prompt: Optional[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
    *,
    runtime_request: Optional[Dict[str, Any]] = None,
    lock_active: Any = None,
) -> str:
    """Fingerprint of the canonical execution request for an idempotency key.

    Key vs fingerprint mirrors POST /v1/runs: the idempotency KEY identifies
    the client's retry token, the FINGERPRINT identifies what was asked. A
    replayed key with a changed request is a conflict, not a replay.

    The identity is the execution request the prelude ACTUALLY RESOLVED
    (``runtime_request``: aliases, route, lock semantics, runtime options),
    not the raw request fields — a direct ``model`` and its ``model_id`` alias
    that resolve to different routes must not collide, and equivalent
    ``model_options`` in a different key order must not look changed
    (review F2). Raw ``model``/``provider``/``model_options`` remain supported
    as a fallback for callers without a resolved prelude.
    """
    if runtime_request is not None:
        execution: Any = resolved_execution_identity(runtime_request, lock_active)
    else:
        execution = {
            "model": model,
            "provider": provider,
            "model_options": model_options,
        }
    canonical = {
        "session_id": session_id,
        "system_prompt": system_prompt or "",
        "user_message": user_message,
        "execution": canonical_execution_value(execution),
    }
    seed = json.dumps(
        canonical_execution_value(canonical),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return f"fp-{sha256(seed.encode('utf-8')).hexdigest()}"


def replayed_session_run_payload(run_id: str, status: Dict[str, Any]) -> Dict[str, Any]:
    """202 replay BODY for a session-stream retry that already landed.

    The run already executed (or is executing) under this key: point the
    caller at the original run id instead of starting a second execution.
    The caller wraps this with its own json_response (transport stays out
    of this module).
    """
    return {
        "run_id": run_id,
        "status": str(status.get("status") or "queued"),
        "replayed": True,
        "session_stream": True,
    }


async def replay_or_reserve_header_run(
    adapter: Any,
    request: "web.Request",
    run_key: str,
    fingerprint: str,
    run_id: str,
    initial_status: Dict[str, Any],
) -> Optional["web.Response"]:
    """Idempotency receipt for an explicit-key session-stream run.

    Composed with the /v1/runs RunIdempotencyStore (same store, same scope
    resolution, same retention) so a retry AFTER the turn completed also
    replays instead of minting a second execution — the session-row active
    slot only guards live runs. Returns a replay/conflict response when the
    key is already spoken for, or None after reserving it for ``run_id``
    (which registers terminal-status persistence — the durable receipt).
    """
    scope = adapter._run_idempotency_scope(request)
    store = adapter._run_idempotency_store
    outcome, record = store.lookup(scope, run_key, fingerprint)
    if outcome == "reused" and record is not None:
        original = str(record["run_id"])
        status = adapter._durable_run_status(request, original) or record["status"]
        return ("replay", replayed_session_run_payload(original, status))
    if outcome == "conflict":
        return ("conflict", {
            "error": {
                "message": "Idempotency-Key was already used with a different request payload",
                "type": "invalid_request_error",
                "param": None,
                "code": "idempotency_key_conflict",
                "run_id": str(record["run_id"]) if record else None,
            },
        })
    outcome, record = store.reserve(
        scope, run_key, fingerprint, run_id, initial_status,
        owner_pid=adapter._run_owner_pid, owner_started=adapter._run_owner_started)
    if outcome == "conflict":
        # Lost a reserve race against a different fingerprint — deterministic
        # conflict, NOT a replay (review P2: the post-reserve branch used to
        # fold every non-created outcome into "replay").
        return ("conflict", {
            "error": {
                "message": "Idempotency-Key was already used with a different request payload",
                "type": "invalid_request_error",
                "param": None,
                "code": "idempotency_key_conflict",
                "run_id": str(record["run_id"]) if record else None,
            },
        })
    if outcome != "created":
        original = str(record["run_id"])
        status = adapter._durable_run_status(request, original) or record["status"]
        return ("replay", replayed_session_run_payload(original, status))
    adapter._run_idempotency_ids.add(run_id)
    return None


def release_header_run_reservation(
    adapter: Any,
    request: "web.Request",
    run_key: str,
    fingerprint: str,
    run_id: str,
) -> None:
    """Compensating delete: drop the receipt reserved for ``run_id`` when the
    admission is refused AFTER the reserve (busy session claim). Only the
    owner run id's row is removed, and only while it is non-terminal — a
    terminal receipt is a real execution and must stay (review P1: a 409
    rejection must never leave a durable "queued" receipt behind)."""
    store = adapter._run_idempotency_store
    scope = adapter._run_idempotency_scope(request)
    with suppress(Exception):
        store.release(scope, run_key, fingerprint, run_id)


async def durable_session_run_claim_holds(
    adapter: Any, session_id: str, run_id: str
) -> Optional[bool]:
    """Whether the durable session row already records ``run_id`` as its run.

    Used to RECONCILE an ambiguous pre-launch claim before compensating its
    idempotency receipt (review F1):

    ``True``  — the claim landed before the failure, so the run may have been
                admitted: never delete its receipt.
    ``False`` — the row is readable and holds no such run: the claim never
                landed, so the reserved receipt is a false acceptance record
                and must go.
    ``None``  — unknown (no DB handle, read failure, unreadable row): treat as
                ambiguous and keep the receipt rather than risk erasing work
                that was admitted.
    """
    try:
        db = await adapter._ensure_session_db_async()
        if db is None:
            return None
        row = await asyncio.to_thread(db.get_session, session_id)
    except Exception:
        return None
    if not isinstance(row, Mapping):
        return None
    return str(row.get("active_run_id") or "") == run_id


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


# ---------------------------------------------------------------------------
# Adapter-method extraction (architecture gate follow-up, PR #96507)
#
# The four DB-access/inspect methods below are defined on the adapter as
# one-line assignments so existing tests can still patch
# ``APIServerAdapter._set_run_status``-style attributes, while the godfile
# stops growing: the BODIES live here. ``replay_or_reserve...``/
# ``claim_session_run_or_conflict`` already reach them through the adapter
# duck-type, so the boundary is unchanged.
# ---------------------------------------------------------------------------


async def adapter_claim_session_active_run_async(
    adapter: Any, session_id: str, run_id: str, run_key: str, status: str
) -> Optional[str]:
    """Off-loop claim of the session's active-run slot (see SessionDB)."""
    db = await adapter._ensure_session_db_async()
    if db is None:
        return None
    return await asyncio.to_thread(
        db.claim_session_active_run, session_id, run_id, run_key, status
    )


async def adapter_set_session_active_run_status_async(
    adapter: Any, session_id: str, run_id: str, status: str
) -> bool:
    """Off-loop coarse-status update, guarded by run id."""
    db = await adapter._ensure_session_db_async()
    if db is None:
        return False
    return await asyncio.to_thread(
        db.set_session_active_run_status, session_id, run_id, status
    )


async def adapter_clear_session_active_run_async(
    adapter: Any, session_id: str, expected_run_id: Optional[str] = None
) -> bool:
    """Off-loop clear of the session's active-run slot on terminal."""
    db = await adapter._ensure_session_db_async()
    if db is None:
        return False
    return await asyncio.to_thread(
        db.clear_session_active_run, session_id, expected_run_id
    )


def adapter_session_run_is_live(adapter: Any, run_id: str) -> bool:
    """Whether a run id names a still-executing server-side run.

    ``_active_run_agents`` is the authoritative "the executor-backed turn
    is still alive" signal (a detached run stays registered until the turn
    exits). ``_run_statuses`` with a non-terminal status covers the brief
    queued window before the agent is registered.
    """
    if run_id in adapter._active_run_agents:
        return True
    status = adapter._run_statuses.get(run_id)
    return bool(status and status.get("status") in ("queued", "running", "stopping"))


async def admit_session_stream_run(
    adapter: Any,
    request: "web.Request",
    ctx: Dict[str, Any],
    session_id: str,
    user_message: Any,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """Idempotency receipt + durable claim for a session-stream run — the whole
    admission gate in one call (single call site: ``_handle_session_chat_stream``).

    Durable active-run claim (PR #96507 P1): the session row carries the live
    run id + immutable request fingerprint, so a client that lost the SSE body
    can rediscover the run via GET /api/sessions/{id} and a retry of the same
    admitted request is met with a deterministic conflict instead of a second
    execution. The caller registers the in-memory "queued" status BEFORE this
    call, so a concurrent loser observes the winner's run as live (queued)
    rather than mistaking it for a stale marker.

    Durable receipt (review P1 fix): the session-row active slot only guards
    LIVE runs — a same-key retry AFTER completion must replay the original
    run, not mint a second execution. Explicit-key runs compose with the
    /v1/runs ``RunIdempotencyStore`` (24h retention, terminal-status
    persistence) so the receipt survives the active slot being cleared.

    Returns ``None`` when the run is admitted and execution may proceed.
    Returns a transport-independent payload dict otherwise:

    - ``{"conflict": True, "payload": {...}}`` — another live run holds the
      session (``run_already_active``) or the key was reused with a changed
      body (``idempotency_key_conflict``); caller answers 409.
    - ``{"replay": True, "payload": {...}}`` — same key + same body after (or
      during) execution; caller answers 202 with the original run id.

    On every pre-launch rejection the just-reserved receipt is released, so a
    refused request never leaves a durable "queued" receipt behind (review P1).
    Unsuccessful exits at the claim seam are exception-safe and RECONCILED: an
    exception or cancellation compensates the receipt only when the durable
    session row proves the run was never admitted, so admitted work is never
    erased by a blind delete (review F1).
    """
    body = ctx["body"]
    system_prompt = body.get("system_message") or body.get("instructions")
    # Fingerprint the RESOLVED execution request (review F2): the prelude has
    # already normalised aliases (model_id/provider_id), split provider-prefixed
    # models and resolved the route/lock semantics, and those are what actually
    # executes. Raw body fields would let a changed alias select a different
    # runtime request under an identical fingerprint.
    request_fingerprint = session_request_fingerprint(
        session_id=session_id,
        user_message=user_message,
        system_prompt=system_prompt,
        runtime_request=ctx.get("runtime_request"),
        lock_active=ctx.get("lock_active"),
    )
    run_key = session_run_key(
        session_id=session_id,
        user_message=user_message,
        system_prompt=system_prompt,
        idempotency_header=request.headers.get("Idempotency-Key"),
    )
    idempotency_header = request.headers.get("Idempotency-Key")
    reserved = False
    conflict_run_id: Optional[str] = None
    # Reservation ownership must follow admission across EVERY unsuccessful
    # exit, not just a returned busy conflict (review F1): an exception or
    # cancellation at the claim seam used to leave a queued receipt behind, so
    # a later healthy retry replayed a run that never launched. The failure is
    # AMBIGUOUS (the durable claim may have landed before it), so reconcile
    # against the session row and only compensate when the row proves this run
    # was never admitted — never blind-delete a possibly-admitted receipt.
    try:
        if idempotency_header:
            receipt = await replay_or_reserve_header_run(
                adapter, request, run_key, request_fingerprint, run_id,
                adapter._run_statuses[run_id],
            )
            if receipt is not None:
                kind, payload = receipt
                return {"conflict": kind == "conflict", "payload": payload}
            reserved = True
        conflict_run_id = await claim_session_run_or_conflict(
            adapter, session_id, run_id, run_key
        )
    except BaseException:
        if reserved:
            admitted = await durable_session_run_claim_holds(
                adapter, session_id, run_id)
            if admitted is False:
                release_header_run_reservation(
                    adapter, request, run_key, request_fingerprint, run_id)
        raise
    if conflict_run_id:
        # Pre-launch rejection: the claim is known NOT to have been taken by
        # this run (another live run holds the session), so the just-reserved
        # receipt is released (review P1/F1).
        if reserved:
            release_header_run_reservation(
                adapter, request, run_key, request_fingerprint, run_id)
        return {"conflict": True, "payload": run_already_active_error(conflict_run_id)}
    return None

