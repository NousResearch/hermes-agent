"""Gateway admission while the turn's provider is at a usage limit (``agent.usage_hold``).

A turn routed to a provider whose plan limit is known to be closed would only burn a request
and fail the same way. With a fallback chain configured the turn runs as usual and the normal
fallback takes over; without one the gateway answers at once, naming the local reset time, and
never calls the provider.

When the user opted in to reruns at the reset (``desktop.quota_resume.enabled``, the same switch
that arms the TUI rerun), a request a usage limit refused (held here) or killed (the provider
answered with a reported reset) is queued durably per profile in
``<HERMES_HOME>/usage_hold_reruns.json``. The gateway is the session's owner for every route it
serves, messaging chats and local Desktop chats alike, so this one queue covers both. At the reset
the exact stored request runs again as one producer turn (``agent.quota_resume.resume_note``) in
the same session, as the sender who wrote it. The rules the queue keeps:

* At most once: an entry is claimed (written ``claimed``) before it is dispatched, and a claimed
  entry is never dispatched again, so a crash mid-dispatch loses the rerun instead of doubling it.
* Only the request it stored: the entry carries the request text and its sender. At the reset the
  session must still be the same session id, idle, and its latest human turn must still be that
  text; any newer request, ``/new`` or a running turn drops the rerun.
* Only text: a turn with native images or another attachment the re-sent text cannot carry is
  told the reset time and never promised a rerun.
* Bounded: one pending rerun per session (a newer refused request replaces it), ``max_attempts``
  consecutive reruns, and nothing queued past ``agent.quota_resume.DEFAULT_MAX_WAIT_SECONDS``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

QUEUE_FILE = "usage_hold_reruns.json"
QUEUED_NOTICE = "This message is queued and runs automatically once the usage limit resets."
TICK_SECONDS = 20.0
PRODUCER = "usage-limit-rerun"
# A claimed entry only remembers the attempt count for the rerun it started.
_CLAIMED_TTL_SECONDS = 3600.0
_LOCK = threading.Lock()


def _load(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _rerun_settings(user_config: Any) -> tuple[bool, int]:
    from agent.quota_resume import RERUN_ENABLED_DEFAULT, RERUN_MAX_ATTEMPTS_DEFAULT
    from utils import is_truthy_value

    desktop = user_config.get("desktop") if isinstance(user_config, dict) else None
    cfg = desktop.get("quota_resume") if isinstance(desktop, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    try:
        attempts = max(0, int(cfg.get("max_attempts", RERUN_MAX_ATTEMPTS_DEFAULT)))
    except (TypeError, ValueError):
        attempts = RERUN_MAX_ATTEMPTS_DEFAULT
    return is_truthy_value(cfg.get("enabled"), default=RERUN_ENABLED_DEFAULT), attempts


def _request_text(ctx: Any) -> Any:
    """The user row this turn persists: what the rerun re-sends and recognizes at the reset."""
    return ctx.persist_user_message if ctx.persist_user_message is not None else ctx.message


def _has_native_images(runner: Any, session_key: str) -> bool:
    """Images the native-vision path buffered for this turn: the stored text cannot carry them."""
    peek = getattr(runner, "_peek_session_state", None)
    state = peek(session_key) if callable(peek) else None
    return bool(state is not None and getattr(state.persistent, "native_image_paths", None))


def _local_owner_live(runner: Any, ctx: Any) -> bool:
    """A local chat reruns only through the session authority that owns it (Desktop, ``serve``);
    a one-shot or ``chat -q`` route has no owner to admit the rerun into."""
    from gateway.session_authorities import active_authority

    authority = active_authority(runner)
    live = authority.sessions.get(ctx.source.chat_id) if authority is not None else None
    return live is not None and live.route == ctx.session_key


def _is_own_rerun(ctx: Any) -> bool:
    """This internal turn is the rerun this queue claimed for the session: it may spend the rest
    of the stored request's attempt budget. Any other internal delivery stays unqueueable."""
    from agent.quota_resume import resume_note
    from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND

    if ctx.persist_user_display_kind != INTERNAL_NOTIFICATION_DISPLAY_KIND:
        return False
    with _LOCK:
        prior = _load(get_hermes_home() / QUEUE_FILE).get(ctx.session_key) or {}
    return (prior.get("state") == "claimed" and isinstance(prior.get("text"), str)
            and _request_text(ctx) == resume_note(prior["text"]))


def _rerunnable(runner: Any, ctx: Any) -> bool:
    """A person's plain-text turn in a chat the gateway can admit a turn into later."""
    from gateway.config import Platform

    platform = getattr(ctx.source, "platform", None)
    if not (ctx.session_key and ctx.session_id and ctx.message and platform is not None
            and not ctx.scheduled_heartbeat and (not ctx.persist_user_display_kind or _is_own_rerun(ctx))
            and not getattr(ctx, "voice_turn", False) and isinstance(_request_text(ctx), str)
            and callable(getattr(ctx.source, "to_dict", None))):
        return False
    if _has_native_images(runner, ctx.session_key):
        return False
    return platform != Platform.LOCAL or _local_owner_live(runner, ctx)


def queue_rerun(runner: Any, ctx: Any, hold: dict[str, Any], *, now: Optional[float] = None) -> Optional[float]:
    """Queue this refused request to run once at the reset. Returns the rerun time, or None when
    the user has not opted in or a bound refuses it."""
    from agent.quota_resume import DEFAULT_MAX_WAIT_SECONDS

    enabled, max_attempts = _rerun_settings(ctx.user_config)
    if not enabled or not _rerunnable(runner, ctx):
        return None
    now = time.time() if now is None else now
    # The hold's ``until`` is the quota-resume plan's time, which already adds the reset grace.
    resume_at = float(hold["until"])
    if resume_at - now > DEFAULT_MAX_WAIT_SECONDS:
        return None
    path = get_hermes_home() / QUEUE_FILE
    try:
        with _LOCK:
            data = _load(path)
            prior = data.get(ctx.session_key) or {}
            # A held rerun is the same request held again: count it, so a wall that keeps moving ends.
            attempt = int(prior.get("attempt", 0)) + 1 if prior.get("state") == "claimed" else 1
            if attempt > max_attempts:
                data.pop(ctx.session_key, None)
                atomic_json_write(path, data)
                return None
            data[ctx.session_key] = {
                "session_key": ctx.session_key, "session_id": ctx.session_id, "state": "pending",
                "resume_at": resume_at, "queued_at": now, "attempt": attempt,
                "provider": str(hold.get("provider") or ""),
                # The rerun's text and its author, never re-derived from the session at the reset.
                "text": _rerun_request(prior, ctx), "sender": ctx.source.to_dict(),
            }
            atomic_json_write(path, data)
    except OSError:
        # A full disk or unwritable home must not replace the turn's own failure and reset time.
        logger.warning("Usage limit: could not queue the rerun for %s", ctx.session_key, exc_info=True)
        return None
    logger.info("Usage limit: queued session %s to rerun at %s (attempt %d/%d)", ctx.session_key,
                time.strftime("%H:%M", time.localtime(resume_at)), attempt, max_attempts)
    return resume_at


def _rerun_request(prior: dict[str, Any], ctx: Any) -> str:
    """The original request: a rerun refused again re-queues what it was rerunning, not its note."""
    from agent.quota_resume import resume_note

    text = _request_text(ctx)
    if prior.get("state") == "claimed" and prior.get("text") and text == resume_note(prior["text"]):
        return prior["text"]
    return text


def _plan_payload(hold_until: float, provider: Any, source: str, rerun_at: Optional[float]) -> dict[str, Any]:
    out = {"eligible": True, "resume_at": hold_until, "source": source, "scheduled": rerun_at is not None}
    if provider:
        out["provider"] = str(provider)
    return out


def held_turn_result(runner: Any, ctx: Any, runtime_kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The turn's answer while its provider is held, or None to run the turn normally."""
    from agent.usage_hold import active_hold

    hold = active_hold(runtime_kwargs.get("provider"), runtime_kwargs.get("base_url") or "")
    if hold is None or runner._refresh_fallback_model():
        return None
    from agent.turn_failure_copy import limit_reset_copy, provider_label_for, stamp_failure

    rerun_at = queue_rerun(runner, ctx, hold)
    until = float(hold["until"])
    lead = (f"{provider_label_for(hold.get('provider')) or 'This provider'} is at its usage limit, so this "
            f"message was not sent to it. {limit_reset_copy(until)}").strip()
    result = {
        "final_response": lead, "messages": [], "api_calls": 0, "tools": [], "failed": True,
        "completed": False, "error": "provider usage limit hold", "failure_resets_at": until,
        # The agent never ran, so nothing flushed the user row: the gateway writes it.
        "agent_persisted": False,
        # The terminal frame's limit card reads this (``completion_payload``).
        "quota_resume": _plan_payload(until, hold.get("provider"), "usage_hold", rerun_at),
    }
    if rerun_at is not None:
        result["failed_turn_notice"] = QUEUED_NOTICE
    logger.info("Usage limit: held turn for session %s (%s until %s), rerun=%s", ctx.session_key,
                hold.get("provider"), time.strftime("%H:%M", time.localtime(until)), rerun_at is not None)
    return stamp_failure(result, "rate_limit", True)


def _sent_text_only(result: dict[str, Any]) -> bool:
    """The provider saw this request as plain text (no image parts the stored text would drop)."""
    users = [m for m in result.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return not users or isinstance(users[-1].get("content"), str)


def schedule_failed_turn(runner: Any, ctx: Any, result: Any) -> None:
    """A turn the provider refused with a reported reset: queue its rerun like a held one and tell
    the user (the failed-turn notice and the terminal frame's ``quota_resume.scheduled``)."""
    plan = result.get("quota_resume") if isinstance(result, dict) and result.get("failed") else None
    if not isinstance(plan, dict) or not plan.get("eligible") or not isinstance(plan.get("resume_at"), (int, float)):
        return
    rerun_at = None
    if _sent_text_only(result):
        rerun_at = queue_rerun(runner, ctx, {"until": plan["resume_at"], "provider": plan.get("provider")})
    plan["scheduled"] = rerun_at is not None
    if rerun_at is not None:
        result["failed_turn_notice"] = QUEUED_NOTICE


# ----------------------------------------------------------------- rerun at the reset


def _session_entry(runner: Any, session_key: str) -> Any:
    store = runner.session_store
    with store._lock:
        store._ensure_loaded_locked()
        return store._entries.get(session_key)


def _latest_request(transcript: list[dict[str, Any]]) -> Any:
    """Content of the session's latest human turn (bookkeeping and producer rows skipped)."""
    from agent.context_compressor import user_originated_turn_view

    for row in reversed(transcript):
        view = user_originated_turn_view(row)
        if view is not None:
            return view.get("content")
    return None


def _sender_source(runner: Any, session: Any, entry: dict[str, Any]) -> Any:
    """The stored sender, re-pinned to the session's receiving bot (no live provenance yet)."""
    from gateway.session import SessionSource
    from gateway.session_identity import restore_identity

    sender = entry.get("sender")
    if not isinstance(sender, dict):
        return None
    try:
        source = SessionSource.from_dict(sender)
    except (KeyError, TypeError, ValueError):
        return None
    restore_identity(source, runner=runner, transport_profile=getattr(session, "transport_profile", None))
    return source


async def _rerun_target(runner: Any, entry: dict[str, Any]) -> tuple[str, Any, Any]:
    """``(verdict, adapter, source)``: ``run``, ``wait`` (adapter offline: keep it pending) or a
    drop reason (the request identity no longer holds). Every identity check runs after the last
    await, so nothing can start a newer turn between the checks and the dispatch."""
    from gateway.config import Platform

    key = entry["session_key"]
    session = _session_entry(runner, key)
    source = _sender_source(runner, session, entry) if session is not None else None
    if source is None:
        return "no session or sender to rerun as", None, None
    async with runner._async_profile_scope_for_source(source):
        transcript = await runner.async_session_store.load_transcript(session.session_id)
    session = _session_entry(runner, key)
    if session is None or session.session_id != entry.get("session_id"):
        return "session changed", None, None
    if runner._is_session_running(key):
        return "a newer turn is running", None, None
    # The request itself, or (a rerun held again) the rerun turn that carried it.
    from agent.quota_resume import resume_note
    if _latest_request(transcript or []) not in (entry.get("text"), resume_note(str(entry.get("text") or ""))):
        return "the conversation moved on", None, None
    if runner._session_key_for_source(source) != key:
        # The sender's message would land in another session (per-delivery routes, a re-keyed
        # thread): it could never continue this one.
        return "the sender no longer routes to this session", None, None
    if source.platform == Platform.LOCAL:
        return "run", None, source  # the owning authority's local adapter admits it
    adapter = runner._delivery_adapter_for(source)
    if adapter is None:
        return "wait", None, None
    if not runner._resume_owner_authorized(key, source):
        return "sender no longer authorized", None, None
    return "run", adapter, source


def _claim(path: Path, key: str, queued_at: Any) -> bool:
    """Flip one pending entry to claimed on disk; False when it changed or is gone (at most once)."""
    with _LOCK:
        data = _load(path)
        entry = data.get(key)
        if not entry or entry.get("state") != "pending" or entry.get("queued_at") != queued_at:
            return False
        entry.update(state="claimed", claimed_at=time.time())
        atomic_json_write(path, data)
    return True


def _drop(path: Path, key: str, queued_at: Any) -> None:
    with _LOCK:
        data = _load(path)
        if (data.get(key) or {}).get("queued_at") == queued_at:
            data.pop(key, None)
            atomic_json_write(path, data)


def _due(path: Path, now: float) -> list[dict[str, Any]]:
    """Pending entries at or past their rerun time; expired bookkeeping is pruned on the way."""
    from agent.quota_resume import DEFAULT_MAX_WAIT_SECONDS

    with _LOCK:
        data = _load(path)
        live = {
            k: v for k, v in data.items()
            if (v.get("state") == "claimed" and now - float(v.get("claimed_at") or 0) < _CLAIMED_TTL_SECONDS)
            or (v.get("state") == "pending" and now - float(v.get("queued_at") or 0) < DEFAULT_MAX_WAIT_SECONDS * 2)
        }
        if live != data:
            atomic_json_write(path, live)
    return [v for v in live.values() if v.get("state") == "pending" and float(v.get("resume_at") or 0) <= now]


def _committed_sender_matches(authority: Any, session: Any, source: Any) -> bool:
    """The authority borrows the route's last committed human envelope for a producer turn: it
    must be the stored sender's, or the rerun would speak (and be authorized) as someone else."""
    from gateway.session_automation import committed_route_envelope

    envelope = committed_route_envelope(authority, session)
    sender = (envelope or {}).get("source") or {}
    return bool(envelope) and sender.get("user_id") == source.user_id and sender.get("chat_id") == source.chat_id


async def _admit_owned(runner: Any, authority: Any, adapter: Any, source: Any, entry: dict[str, Any]) -> bool:
    """Admit the rerun into the session authority's FIFO as a producer turn; False when refused."""
    from agent.quota_resume import resume_note
    from gateway.config import Platform
    from gateway.session_automation import admit_route_wake

    text = resume_note(entry["text"])
    if source.platform == Platform.LOCAL:
        from gateway.session_local_recovery import local_adapter_map, restore_local_session

        restore_local_session(authority, source.chat_id)  # a chat a restart left closed
        live = authority.sessions.get(source.chat_id)
        if live is None:
            return False
        return await admit_route_wake(runner, local_adapter_map(authority)[Platform.LOCAL], live.source, text, PRODUCER)
    if not _committed_sender_matches(authority, _session_entry(runner, entry["session_key"]), source):
        return False
    return await admit_route_wake(runner, adapter, source, text, PRODUCER)


async def _dispatch(runner: Any, adapter: Any, source: Any, entry: dict[str, Any]) -> bool:
    """Run the stored request once more in its session, as its sender."""
    from agent.quota_resume import resume_note
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_authorities import active_authority

    async with runner._async_profile_scope_for_source(source):
        authority = active_authority(runner)
        if authority is not None:
            return await _admit_owned(runner, authority, adapter, source, entry)
        if adapter is None:
            return False
        # No authority (a bare runner): the adapter's inbound path, on the receiving bot's transport
        # so delivery and provenance resolve as they did for the sender's own message.
        source._transport_adapter_ref = weakref.ref(adapter)
        await adapter.handle_message(MessageEvent(
            text=resume_note(entry["text"]), message_type=MessageType.TEXT, source=source, internal=True,
            allow_gateway_control=False, metadata={"gateway_session_key": entry["session_key"]}))
        return True


async def rerun_due(runner: Any, homes: list[Path], *, now: Optional[float] = None) -> int:
    """One pass over every served profile's queue: rerun each due, still-valid request."""
    sent = 0
    for home in homes:
        path = Path(home) / QUEUE_FILE
        if not path.exists():
            continue
        for entry in _due(path, time.time() if now is None else now):
            verdict, adapter, source = await _rerun_target(runner, entry)
            if verdict == "wait":
                continue
            if verdict != "run":
                logger.info("Usage limit: dropping the queued rerun for %s: %s", entry["session_key"], verdict)
                _drop(path, entry["session_key"], entry.get("queued_at"))
                continue
            if not _claim(path, entry["session_key"], entry.get("queued_at")):
                continue
            logger.info("Usage limit reset: rerunning the held request for %s", entry["session_key"])
            try:
                admitted = await _dispatch(runner, adapter, source, entry)
            except Exception:
                logger.warning("Usage limit: the rerun for %s was refused", entry["session_key"], exc_info=True)
                admitted = False
            if not admitted:
                logger.info("Usage limit: the session owner refused the rerun for %s", entry["session_key"])
                continue
            sent += 1
    return sent


async def rerun_watcher(runner: Any) -> None:
    """Supervised loop: reruns held requests at their reset across every served profile."""
    from gateway.run_idle_exit import _served_homes

    while not runner._shutdown_event.is_set():
        try:
            await rerun_due(runner, _served_homes(runner))
        except Exception:
            logger.warning("Usage-limit rerun pass failed", exc_info=True)
        try:
            await asyncio.wait_for(runner._shutdown_event.wait(), timeout=TICK_SECONDS)
        except TimeoutError:
            pass
