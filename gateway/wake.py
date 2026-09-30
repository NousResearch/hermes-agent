"""Wake an existing agent session from a background completion event. Push-capable adapters
(``supports_async_delivery``) get a synthetic ``MessageEvent(internal=True)`` via handle_message;
stateless adapters (API server) would run that under a ``build_session_key()`` key that never
matches the raw ``X-Hermes-Session-Id`` real turns use (invisible parallel session), so we self-POST
``/v1/chat/completions`` with the raw id header to resume the REAL session. Exception:
async-delegation completions: the CLIENT owns the next turn, so they are never self-POSTed as a
new ``role=user`` prompt (could cross a pending human-confirmation gate); instead
``persist_delegation_delivery`` writes a durable DELIVERY row (``display_kind=
"async_delegation_complete"``, read by TUI/desktop pollers). Failures RAISE (after bounded retries
on transient errors) so callers can rewind cursors / retry instead of silently losing the event."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A wake self-post runs the whole agent turn synchronously (stream=false); generous ceiling so long
# tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff between retries on transient failures. The API server has no per-session lock (concurrent
# turns are last-writer-wins) but DOES enforce a global max_concurrent_runs cap via HTTP 429, which
# is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends. Reads
    ``supports_async_delivery`` off the adapter class rather than the request-scoped contextvar
    (background watchers run outside any bound session context). Adapters that don't declare
    the flag are push-capable."""
    return bool(getattr(adapter, "supports_async_delivery", True))


class WakeNotAccepted(RuntimeError):
    """No adapter admission: retry without treating a healthy chat as dead."""


def session_owned_by_profile(config: Any, profile: Optional[str], session_id: Any) -> bool:
    """True when a stateless (``api_server``) destination's raw session id is canonically owned by
    served *profile*'s own session store.

    A shared-listener mirror platform has no chat/thread/guild anchor a ``profile_routes`` entry
    could match, so the session store itself is the ownership proof: the row must exist in that
    profile's ``state.db`` under its own home and carry that profile's stamp (a NULL legacy stamp
    belongs to the store's own profile — the same rule the dashboard's session routes apply). An
    unserved profile, a missing row, a row stamped for another profile, or an unreadable store all
    fail closed. Shared by the Kanban notifier and the background-process wake path.
    """
    import contextlib
    from pathlib import Path
    if not session_id or not profile:
        return False
    profile = str(profile)
    try:
        from gateway.run import _multiplex_profile_homes
        home = dict(_multiplex_profile_homes(config)).get(profile)
        if home is None:
            return False
        from hermes_state import SessionDB
        db = SessionDB(Path(home) / "state.db", read_only=True)
    except Exception as exc:
        logger.debug("wake: session ownership check unavailable for %s/%s: %s", profile, session_id, exc)
        return False
    try:
        row = db.get_session(str(session_id))
    except Exception as exc:
        logger.debug("wake: session ownership lookup failed for %s/%s: %s", profile, session_id, exc)
        return False
    finally:
        with contextlib.suppress(Exception):
            db.close()
    return bool(row) and (row.get("profile_name") or profile) == profile


async def admit_internal_event(adapter: Any, event: Any) -> None:
    """Require a concrete adapter admission, not merely a handler returning None.

    The public handler return stays unchanged. This receipt means scheduled/queued,
    not model execution, authorization of a later turn, or successful outbound delivery.
    """
    event._gateway_accepted = False
    await adapter.handle_message(event)
    if event._gateway_accepted is not True:
        raise WakeNotAccepted("internal wake not accepted by adapter")


async def deliver_wake(adapter: Any, *, text: str, session_id: str = "", source: Any = None,
                       notification_category: str = "result", profile: Optional[str] = None,
                       idempotency_key: Optional[str] = None,
                       require_persist_ack: bool = False) -> None:
    """Deliver a wake turn to the session behind ``adapter``. ``session_id`` is the RAW session id
    (``X-Hermes-Session-Id`` / state.db key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` for the synthetic event — required for push-capable adapters. ``profile``
    names the served profile that canonically owns a non-push destination; a non-default value is
    delivered in-process under the caller's profile scope (see ``_self_post_chat_completion``).

    ``idempotency_key`` (keyword-only, defaulted): when set, sent to the non-push self-post as the
    ``Idempotency-Key`` (HTTP header on the default-profile path, de-dup key on the in-process
    secondary path) so a same-key redelivery does not double-execute a turn that already persisted.
    ``require_persist_ack`` (keyword-only, defaulted False): when True the non-push self-post treats
    a delivery as success ONLY if the turn actually committed — the default-profile HTTP path gates
    on the server-generated ``X-Hermes-Turn-Persisted: true`` response header, the secondary
    in-process path gates on ``run_internal_session_turn``'s persist return — so a durable caller
    can rewind and redeliver an unpersisted turn instead of losing it.

    Raises on failure (bad arguments, exhausted retries, HTTP error, or — only under
    ``require_persist_ack`` — a self-post that does not confirm persistence) so the caller can
    rewind/retry."""
    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError("deliver_wake: push-capable adapter requires a SessionSource")
        from gateway.platforms.event import MessageEvent, MessageType
        synth_event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=True,
                                   metadata={"notification_category": notification_category})
        await admit_internal_event(adapter, synth_event)
        return
    if not session_id:
        raise ValueError("deliver_wake: non-push adapter (supports_async_delivery=False) "
                         "requires the raw session id to self-post the wake turn")
    extra: dict = {}
    if profile:
        extra["profile"] = profile
    if notification_category == "diagnostic":
        extra["notification_category"] = notification_category
    if idempotency_key is not None:
        extra["idempotency_key"] = idempotency_key
    if require_persist_ack:
        extra["require_persist_ack"] = require_persist_ack
    await _self_post_chat_completion(adapter, text=text, session_id=session_id, **extra)


def _delegation_display_metadata(evt: dict) -> dict:
    """Display-only metadata for a persisted delegation delivery row. Mirrors
    ``tui_gateway.server._async_delegation_display_metadata`` (same ``display_kind`` consumer
    contract) without importing the TUI stack."""
    raw_results = evt.get("results")
    results = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(1 for r in results if r.get("status") in {"completed", "success"})
    failed_count = sum(1 for r in results if r.get("status") in {"failed", "error"})
    metadata = {"delegation_id": str(evt.get("delegation_id") or ""), "task_count": task_count,
                "completed_count": completed_count or task_count - failed_count,
                "failed_count": failed_count}
    if evt.get("task_failure_notice"):
        metadata["delivery_notice"] = f"task_failure:{results[0].get('task_index', '') if results else ''}"
        metadata["notification_category"] = "diagnostic"
        from gateway.warning_notifications import warning_notifications_enabled
        if not warning_notifications_enabled("api_server"):
            metadata["presentation_suppressed"] = True
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


async def persist_delegation_delivery(adapter: Any, *, text: str, session_id: str, evt: Optional[dict] = None) -> None:
    """Persist an async-delegation completion as a durable DELIVERY row (see module docstring)
    WITHOUT running any agent turn. Raises on failure so the caller can release the durable claim
    and retry.

    85957: on stateless api_server sessions the client owns the turn after ``event.complete`` — a completion
    must never become a new ``role=user`` prompt via the self-post (that starts an unauthorized agent turn
    and can cross a pending human-confirmation gate). Instead, append the completion to the session
    transcript as a timeline bookkeeping row (``display_kind="async_delegation_complete"`` + display
    metadata — the exact shape the TUI/desktop delivery path persists), WITHOUT running any agent turn.
    Clients polling ``GET /api/sessions/{id}/messages`` see it immediately; the pre-request repair belt
    folds it into the next real client turn as context. See #85957.
    """
    if not session_id:
        raise ValueError("persist_delegation_delivery: raw session id required to persist "
                         "the completion on the api_server session transcript")
    ensure = getattr(adapter, "_ensure_session_db", None)
    db: Any = await asyncio.to_thread(ensure) if callable(ensure) else None
    if db is None:
        raise RuntimeError("persist_delegation_delivery: api_server SessionDB unavailable — "
                           f"cannot persist completion for session {session_id}")
    # #98619: the parent run may have compressed/rotated between dispatch and this detached
    # completion — the captured origin id is then a closed parent and the append below is
    # rejected with CompressionSessionClosedError forever (the watcher retries the same stale
    # id). Adopt the live continuation tip first, the same canonical resolution
    # /api/sessions/{id}/messages reads use, so the delivery row lands where the next run and
    # the messages endpoint both resolve. Fails open to the original id.
    resolver = getattr(db, "resolve_resume_session_id", None)
    if callable(resolver):
        try:
            resolved = await asyncio.to_thread(resolver, session_id)
            if resolved:
                session_id = str(resolved)
        except Exception:
            logger.debug("delegation delivery continuation resolve failed for %s", session_id, exc_info=True)
    await asyncio.to_thread(
        db.append_delegation_delivery, session_id, text, _delegation_display_metadata(evt or {}),
    )
    logger.info(
        "async delegation completion persisted as delivery row for api_server session %s (no wake turn)", session_id
    )


async def _self_post_chat_completion(adapter: Any, *, text: str, session_id: str,
                                      notification_category: str = "result",
                                      profile: Optional[str] = None,
                                      idempotency_key: Optional[str] = None,
                                      require_persist_ack: bool = False) -> None:
    """POST the wake text to the in-pod API server as a normal session turn, using the adapter's
    own bind host/port/key. Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY``, so a missing key is a hard error rather than a wake in a fresh session
    nobody watches.

    ``profile`` (a served secondary under ``gateway.multiplex_profiles``) takes the IN-PROCESS
    route instead of the HTTP one: the shared listener's ``/p/<profile>/`` mirror authenticates
    with that profile's own ``API_SERVER_KEY`` — which a route-only profile legitimately does not
    have — and an unprefixed self-post would resume the session in the DEFAULT profile's store.
    The caller already holds the owner profile's runtime scope, so the turn lands in that profile's
    own session. A non-default profile whose adapter cannot run in-process fails closed.

    When ``require_persist_ack`` is set a 2xx alone is NOT success: the DEFAULT-profile HTTP path
    requires the response to carry ``X-Hermes-Turn-Persisted: true`` (a server-generated ack that
    the turn actually committed) and the SECONDARY in-process path requires
    ``run_internal_session_turn`` to return a persisted result; a missing / false ack RAISES so a
    durable caller does not treat an unpersisted turn as delivered. ``idempotency_key`` is sent as
    the ``Idempotency-Key`` header (HTTP path) / de-dup key (in-process path) so a redelivery under
    the same key does not double-execute a turn that already persisted. Both default to the prior
    2xx-is-success, no-idempotency behaviour.
    """
    if profile and str(profile) != "default":
        in_process: Any = getattr(adapter, "run_internal_session_turn", None)
        if not callable(in_process):
            raise RuntimeError(
                f"wake self-post for served profile {profile!r} requires in-process session "
                "delivery; refusing to self-post as the default profile")
        # The in-process route runs the wake turn under the owner profile's scope and returns the
        # turn's persistence result (True/False/None). A durable caller (require_persist_ack) must
        # NOT treat an unpersisted turn as delivered — raise so it rewinds its cursor; the opt-in
        # idempotency_key lets a redelivery de-dup a turn that already persisted (see
        # api_server_runs.run_internal_session_turn). A non-durable caller ignores the result.
        persisted = await in_process(session_id=session_id, text=text, profile=str(profile),
                                     notification_category=notification_category,
                                     idempotency_key=idempotency_key)
        if require_persist_ack and persisted is not True:
            raise RuntimeError(
                f"in-process wake for served profile {profile!r} session {session_id} did not "
                f"confirm persistence (turn_persisted={persisted!r}); treating as undelivered")
        return
    import aiohttp
    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"  # wildcard bind — connect over loopback
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError("wake self-post requires API_SERVER_KEY: session continuation via "
                           "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
                           "server, so the wake cannot reach the target session")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "X-Hermes-Session-Id": session_id}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if require_persist_ack:
        # Opt-in signal to the api_server idempotency cache: keep an UNPERSISTED turn OUT of the
        # cache so this durable caller's same-key retry re-runs until the turn commits. Absent this
        # header the server keeps the upstream default (cache any completed result). Distinct
        # INBOUND header; the response-only X-Hermes-Turn-Persisted ack is untouched.
        headers["X-Hermes-Require-Persist"] = "1"
    payload = {"model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
               "messages": [{"role": "user", "content": text}], "stream": False}
    if notification_category == "diagnostic":
        payload["hermes_notification_category"] = "diagnostic"
    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:  # concurrency cap — transient; back off and retry
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) for session {session_id}"
                        )
                        logger.warning("%s; attempt %d/%d", last_err, attempt + 1, attempts)
                        continue
                    if resp.status >= 400:  # non-transient (auth/validation): fail immediately
                        body = (await resp.text())[:300]
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    # A 2xx does not prove the turn committed: the handler 200s even when the
                    # session-db write failed. A durable caller (require_persist_ack) requires the
                    # server-generated persist ack; a missing / malformed / false value is a
                    # delivery failure (the caller must not advance its cursor). NOT retried inline
                    # — the turn already ran; the durable notifier re-delivers later under the same
                    # Idempotency-Key so the api_server dedups it. Non-durable callers keep the
                    # prior 2xx-is-success behaviour.
                    if require_persist_ack:
                        persisted = str(
                            resp.headers.get("X-Hermes-Turn-Persisted", "")
                        ).strip().lower()
                        if persisted != "true":
                            raise RuntimeError(
                                f"wake self-post for session {session_id} did not confirm "
                                f"persistence (X-Hermes-Turn-Persisted={persisted!r}); "
                                f"treating as undelivered")
                    logger.info("wake self-post delivered for session %s (attempt %d)", session_id, attempt + 1)
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning("wake self-post transient failure for session %s (attempt %d/%d): %s",
                           session_id, attempt + 1, attempts, exc)
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after {attempts} attempts: {last_err}"
    ) from last_err
