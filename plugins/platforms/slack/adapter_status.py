"""Slack typing indicator: the "is thinking…" status line set while a turn runs.

A threaded reply shows it on the thread (``agents.sessions.setStatus``, legacy
``assistant.threads.setStatus`` fallback). A flat reply (``reply_in_thread: false``, no real
thread) shows it on the inbound message itself through the sessions API: Slack attaches an
``agent_session`` to a top-level ts without creating a thread, while the legacy endpoint on a
top-level ts opens an assistant thread, so a flat target never uses it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from agent.i18n import t
from gateway.platforms.base_status_anchor import STATUS_ANCHOR_MESSAGE_ID, STATUS_OWNER_KEY

logger = logging.getLogger(__name__)

# Slack's setStatus (agents.sessions / assistant.threads) is Tier 3 (~50+/min per workspace); the
# base ``_keep_typing`` heartbeat ticks every 2s, and a few concurrent turns blow through that —
# the resulting throttling can also reject the final clear, leaving "Working…" stuck on the
# message until Slack's own ~90 min status expiry. A `processing` session status
# persists on its own for that long, so only a cheap periodic keep-alive and the legacy free-text
# elapsed label (which actually changes) need to re-send at all; see ``_status_send_due``.
_STATUS_KEEPALIVE_INTERVAL = 60.0
_STATUS_LABEL_THROTTLE_INTERVAL = 30.0
# A failed or abandoned (``_keep_typing``'s wait_for timeout) set is retried after this, not after
# the keep-alive interval, without hammering an endpoint that is rate-limiting us.
_STATUS_RETRY_BACKOFF = 5.0
# Heartbeat state untouched by any tick for this long is no live ``_keep_typing`` loop's (those tick
# every few seconds), so it is the only state pruned once the map outgrows its cap.
_STATUS_HEARTBEAT_STALE = 300.0
# One ``stop_typing`` clear (every target, both endpoints, rate-limit retries) never holds turn
# completion longer than this, and a repeat clear of the same target inside it is a no-op.
_STATUS_CLEAR_BUDGET = 10.0
_STATUS_CLEAR_RETRY_DEFAULT_DELAY = 1.0

_SESSIONS, _LEGACY = "agents.sessions", "assistant.threads"


def _status_send_due(payload: str, state: dict[str, Any], now: float, *,
                     is_elapsed_label: bool) -> bool:
    """Whether ``send_typing`` should actually call Slack's API for ``payload`` on this target,
    given its heartbeat ``state``: the first set ever, a keep-alive at
    ``_STATUS_KEEPALIVE_INTERVAL`` after the last DELIVERED set, or a changed payload. A changed
    ``is_elapsed_label`` (the auto-generated "still working… (2m03s)" fallback — the only thing
    that changes on its own every tick) is throttled to ``_STATUS_LABEL_THROTTLE_INTERVAL``; any
    other changed payload (a live per-tool status, ``typing_status_text``) sends immediately. An
    undelivered last attempt only holds off the next one for ``_STATUS_RETRY_BACKOFF``."""
    failed_at = state.get("failed_at")
    if failed_at is not None and now - failed_at < _STATUS_RETRY_BACKOFF:
        return False
    sent_at = state.get("sent_at")
    if sent_at is None:
        return True
    elapsed = now - sent_at
    if elapsed >= _STATUS_KEEPALIVE_INTERVAL:
        return True
    if payload == state.get("sent_payload"):
        return False
    return elapsed >= _STATUS_LABEL_THROTTLE_INTERVAL if is_elapsed_label else True


def _is_ratelimit_error(exc: BaseException) -> bool:
    """429 / ``ratelimited`` only — narrower than ``_is_retryable_upload_error`` (which also
    treats 5xx as transient): the clear retry exists for Slack's Tier-3 ``setStatus`` limit."""
    if getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    body = str(exc).lower()
    return "ratelimited" in body or "rate_limited" in body or "429" in body


def _target_keys(mapping: dict, team_id: str, chat_id: str, thread_ts: str) -> list:
    """``(team_id, channel_id, ts)`` keys of ``mapping`` naming this status target. A team-less
    side matches any workspace (a target first seen before its workspace was known); two known,
    different workspaces sharing a Slack Connect channel id are different targets."""
    return [k for k in mapping if k[1] == str(chat_id) and k[2] == str(thread_ts)
            and (not team_id or not k[0] or k[0] == str(team_id))]


def _facade():
    """The ``adapter`` facade module, which owns the sessions-capability probe (late import: the
    facade imports this mixin at module load)."""
    try:
        from . import adapter
    except ImportError:  # pragma: no cover - plugin loaded outside package context
        import adapter  # type: ignore
    return adapter


class SlackTypingStatusMixin:
    """``send_typing`` / ``stop_typing`` for ``SlackAdapter`` and the ``setStatus`` call itself."""

    def _flat_status_anchor(self, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """The inbound message's own ts when the turn replies flat (``reply_in_thread: false`` and no
        real thread), else None. ``STATUS_ANCHOR_MESSAGE_ID`` is the gateway's explicit anchor; a session
        thread_id equal to ``message_id`` is the synthetic per-message key, i.e. the message itself."""
        if self.config.extra.get("reply_in_thread", True) or not metadata:
            return None
        thread = str(metadata.get("thread_id") or metadata.get("thread_ts") or "")
        anchor = str(metadata.get(STATUS_ANCHOR_MESSAGE_ID) or "")
        if thread and thread == str(metadata.get("message_id") or ""):
            anchor = anchor or thread
        return anchor if anchor and thread in ("", anchor) else None

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show the status line on the reply's thread, or on the inbound message for a flat reply.
        Needs assistant:write or chat:write scope; auto-clears on a threaded reply.

        Throttled per target: the base ``_keep_typing`` heartbeat ticks every 2s, but
        this only actually calls Slack's ``setStatus`` when ``_status_send_due`` says so, comparing
        against the endpoint and payload Slack last ACCEPTED for the target. ``stop_typing``'s
        clear is never throttled."""
        if self._suppressed_ignored(chat_id, "typing/status in", level=logging.DEBUG):
            return
        if not self._app:
            return
        thread_ts = flat_anchor = self._flat_status_anchor(metadata)
        if metadata and not thread_ts:
            # Same synthetic-thread guard as sending: with reply_in_thread=false thread_id is the
            # message's own ts, and setStatus on it would open an assistant thread prematurely.
            thread_ts = self._resolve_thread_ts(
                reply_to=metadata.get("message_id"), metadata=metadata)
        if not thread_ts:
            return  # Can only set status in a thread context
        team_id = self._metadata_team_id(metadata) or self._channel_team.get(chat_id, "")
        status_key = self._workspace_thread_key(team_id, chat_id, str(thread_ts))
        # Heartbeat (#45702): preserve the first refresh's start time across _keep_typing refreshes so a
        # long turn surfaces elapsed time ("still working… (2m03s)") instead of a static "is
        # thinking..." that reads as stuck — which is what provokes mid-turn "you there?" pings. Stored
        # inside the tracked status entry so it shares the existing bounds/eviction and is dropped by
        # stop_typing with the rest of the status state.
        _prev_entry = self._active_status_threads.get(status_key) if status_key else None
        _prev_entry = _prev_entry if isinstance(_prev_entry, dict) else {}
        _status_started = _prev_entry.get("started")
        if not isinstance(_status_started, (int, float)):
            _status_started = time.monotonic()
        # May lack assistant:write scope or assistant context; reactions still work.
        _explicit_status = getattr(self, "_status_text", {}).get(str(chat_id)) or getattr(
            self.config, "typing_status_text", None)
        _status = _explicit_status or self._default_status_text(_status_started)
        now = time.monotonic()
        state = self._status_heartbeat_state(status_key, now)
        payload = self._status_payload_preview(chat_id, team_id, _status, state.get("endpoint"))
        send_now = _status_send_due(payload, state, now, is_elapsed_label=_explicit_status is None)
        if status_key:
            self._active_status_threads[status_key] = {
                "thread_ts": str(thread_ts), "team_id": str(team_id) if team_id else "",
                "started": _status_started, "flat": bool(flat_anchor),
                "owner": str((metadata or {}).get(STATUS_OWNER_KEY) or _prev_entry.get("owner") or "")}
            # Evict oldest-thread-first (key[2] is the thread ts) so the newest survives.
            self._evict_oldest_by_ts(
                self._active_status_threads, self._ACTIVE_STATUS_THREADS_MAX, lambda k: k[2])
        if not send_now:
            return
        state["failed_at"] = now  # stays set if the call fails or _keep_typing's timeout cancels it
        # Before the call: a set cancelled or lost in flight may still have landed on Slack, so the
        # next clear of this target must really be sent, not deduped against the previous one.
        for key in _target_keys(self._status_clears, team_id, chat_id, thread_ts):
            del self._status_clears[key]
        accepted = await self._set_thread_status(
            chat_id, team_id, thread_ts, _status, "failed", flat=bool(flat_anchor))
        if accepted is None:
            return
        endpoint, sent_payload = accepted
        state.update(failed_at=None, sent_at=now, sent_payload=sent_payload, endpoint=endpoint)

    def _status_heartbeat_state(self, status_key: Optional[tuple], now: float) -> dict[str, Any]:
        """Throttle state for ``status_key`` (a throwaway dict when there is no key). Kept apart from
        ``_active_status_threads`` because that map evicts at its cap even while the target's
        ``_keep_typing`` loop is alive; this one is only dropped by ``stop_typing``, and pruned
        past the same cap only of entries no live loop has ticked in ``_STATUS_HEARTBEAT_STALE``."""
        if not status_key:
            return {}
        state = self._status_heartbeats.setdefault(status_key, {})
        state["seen"] = now
        if len(self._status_heartbeats) > self._ACTIVE_STATUS_THREADS_MAX:
            for key in [k for k, s in self._status_heartbeats.items()
                        if now - s.get("seen", now) >= _STATUS_HEARTBEAT_STALE]:
                del self._status_heartbeats[key]
        return state

    def _status_payload_preview(
            self, chat_id: str, team_id: str, status: str, endpoint: Optional[str]) -> str:
        """The payload ``_set_thread_status`` would send for ``status`` without making the call.
        A target whose sessions call was rejected and the legacy fallback accepted (a session-less
        thread) is compared on the legacy free text, so its label changes are not hidden behind a
        constant ``processing``."""
        if endpoint == _LEGACY:
            return status
        _, using_sessions = _facade()._session_status_method(self._get_client(chat_id, team_id=team_id))
        return ("processing" if status else "active") if using_sessions else status

    @staticmethod
    def _default_status_text(started: Optional[float]) -> str:
        """Fallback status label: after 30s show elapsed progress so long turns don't read
        as stuck (live-status phrases and ``typing_status_text`` always win over this)."""
        elapsed = int(time.monotonic() - started) if started is not None else 0
        if elapsed < 30:
            return t("platform.slack.status.thinking")
        mins, secs = divmod(elapsed, 60)
        return t("platform.slack.status.still_working",
                 elapsed=f"{mins}m{secs:02d}s" if mins else f"{secs}s")

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        """Clear the status line set by ``send_typing`` (thread or flat message anchor)."""
        if self._suppressed_ignored(chat_id, "status clear in", level=logging.DEBUG):
            self._active_status_threads.pop(chat_id, None)
            return
        if not self._app:
            return
        flat_anchor = self._flat_status_anchor(metadata)
        requested_thread_ts = flat_anchor or ""
        if metadata and not requested_thread_ts:
            requested_thread_ts = str(metadata.get("thread_id") or metadata.get("thread_ts") or "")
        requested_team_id = self._metadata_team_id(metadata)
        active = None
        ambiguous_tracked = False
        if requested_thread_ts and requested_team_id:
            active_key = self._workspace_thread_key(requested_team_id, chat_id, requested_thread_ts)
            if active_key:
                active = self._active_status_threads.pop(active_key, None)
        else:
            # Slack Connect workspaces can share a channel ID, so a team-less clear
            # only pops a UNIQUE tracked match for this channel (+ thread when given).
            matching_keys = [
                key
                for key in self._active_status_threads
                if key[1] == str(chat_id)
                and (not requested_thread_ts or key[2] == requested_thread_ts)]
            if len(matching_keys) == 1:
                active = self._active_status_threads.pop(matching_keys[0], None)
            ambiguous_tracked = bool(requested_thread_ts) and len(matching_keys) > 1
        active = active or {}
        thread_ts = active.get("thread_ts", "")
        team_id = requested_team_id or active.get("team_id", "")
        if not thread_ts and requested_thread_ts and not ambiguous_tracked:
            # Untracked (restart/eviction) but the caller named the exact thread: clear anyway so
            # a stuck status is always dismissable; skipped when several workspaces track it.
            thread_ts = requested_thread_ts
        if not thread_ts:
            return
        flat = bool(flat_anchor or active.get("flat"))
        deadline = time.monotonic() + _STATUS_CLEAR_BUDGET
        await self._clear_status_target(chat_id, team_id, thread_ts, flat, deadline)
        if flat:
            owner = str((metadata or {}).get(STATUS_OWNER_KEY) or active.get("owner") or "")
            for key in self._session_flat_status_keys(chat_id, team_id, thread_ts, owner):
                if key[2] == thread_ts:
                    # Same ts as the clear just sent above (e.g. a team-less duplicate
                    # registration of the opener itself): already cleared, just drop tracking.
                    self._active_status_threads.pop(key, None)
                    continue
                entry = self._active_status_threads[key]
                attempted = await self._clear_status_target(
                    chat_id, entry.get("team_id") or team_id, key[2], True, deadline)
                if attempted:
                    self._active_status_threads.pop(key, None)
                # else: the shared clear budget ran out before this queued anchor's turn, so its
                # clear was never attempted — keep it tracked so the next stop_typing's cleanup
                # (this same loop, run again) actually attempts it instead of leaving it to
                # Slack's own status expiry.

    async def _clear_status_target(
            self, chat_id: str, team_id: str, thread_ts: str, flat: bool, deadline: float) -> bool:
        """Send one target's clear — never throttled, rate-limit retries bounded by ``deadline``.
        A turn end clears the same target several times (the ``_keep_typing`` finally plus
        ``_stop_typing_refresh``'s attempts), so a repeat inside ``_STATUS_CLEAR_BUDGET`` of a clear
        already sent or in flight is skipped, unless a later ``send_typing`` re-set the status.

        Returns whether the clear was attempted — sent to at least one endpoint just now, or
        already attempted by a concurrent call within the dedup window — as opposed to skipped
        outright because the shared ``deadline`` had already passed. The caller uses this to
        decide whether it is safe to stop tracking the target: a clear that was never attempted
        must stay tracked so the next ``stop_typing`` retries it."""
        for key in _target_keys(self._status_heartbeats, team_id, chat_id, thread_ts):
            del self._status_heartbeats[key]
        now = time.monotonic()
        for key in [k for k, until in self._status_clears.items() if until <= now]:
            del self._status_clears[key]
        if _target_keys(self._status_clears, team_id, chat_id, thread_ts):
            logger.debug("[Slack] status clear on %s already sent or in flight; skipped", thread_ts)
            return True
        if now >= deadline:
            # No dedup marker: the clear was never attempted, so a later stop must still send it.
            logger.debug("[Slack] status clear on %s skipped: clear budget spent", thread_ts)
            return False
        clear_key = (str(team_id or ""), str(chat_id), str(thread_ts))
        self._status_clears[clear_key] = now + _STATUS_CLEAR_BUDGET
        accepted = await self._set_thread_status(
            chat_id, team_id, thread_ts, "", "clear failed", flat=flat, deadline=deadline)
        if accepted is None:
            # Neither endpoint accepted it (e.g. a transient non-429 failure on both): don't hold
            # the dedup marker for the full budget, or the next _stop_typing_refresh attempt (and
            # the next turn's stop) would skip the retry and leave `processing` stuck on Slack's
            # side until its own ~90 min status expiry.
            self._status_clears.pop(clear_key, None)
        return accepted is not None

    def _session_flat_status_keys(self, chat_id: str, team_id: str, anchor_ts: str, owner: str) -> list:
        """Tracked flat statuses in this workspace channel that belong to the finishing turn chain.

        A ``/queue`` follow-up's progress lane anchors its status on the queued message, but the turn
        chain ends with one clear on the message that opened it, so that clear takes along every flat
        status owned by the same session (``STATUS_OWNER_KEY``), plus a team-less entry for the opener
        itself, registered before the workspace was known. A concurrent session's status in the same
        channel (another user's group session, another top-level DM session) is left alone."""
        return [
            key for key, entry in self._active_status_threads.items()
            if key[1] == str(chat_id) and key[0] in (team_id, "") and isinstance(entry, dict)
            and entry.get("flat") and (key[2] == anchor_ts or (owner and entry.get("owner") == owner))]

    async def _set_thread_status(
        self, chat_id: str, team_id: str, thread_ts: str, status: str, fail_label: str, *,
        flat: bool = False, deadline: Optional[float] = None) -> Optional[tuple[str, str]]:
        """Set/clear the thread status line; returns ``(endpoint, payload)`` Slack accepted, or
        None when every attempt failed (failures are debug-logged, never raised).

        Prefers ``agents.sessions.setStatus`` (slack-sdk >= 3.44.0) with lifecycle
        enum values — `processing` while working, `active` to clear — which is the
        only API that supports Slack session channels. Free-text status strings from
        ``send_typing``/``_keep_typing`` are mapped: any non-empty text becomes
        ``processing``; an empty (clear) becomes ``active``. When the sessions API
        is unavailable or rejects the call (older SDK, no session scope), retry once
        on the legacy ``assistant.threads.setStatus`` with the original free text /
        empty-string clear. A ``flat`` target is a top-level message ts:
        sessions API only, since the legacy endpoint on it opens an assistant thread.

        ``deadline`` (``stop_typing``'s clear only): a 429/``ratelimited`` response is retried
        after ``Retry-After`` so a rate-limited clear doesn't leave "Working…" stuck until Slack's
        own ~90 min status expiry, with ONE budget across both endpoints: no retry sleep and no
        legacy fallback starts past it.
        """
        client = self._get_client(chat_id, team_id=team_id)
        set_status, using_sessions = _facade()._session_status_method(client)
        if flat and not using_sessions:
            logger.debug("[Slack] status %s on flat message %s: no agents.sessions API", fail_label, thread_ts)
            return None
        # A user-configured label that merely spells an enum is still ongoing work
        # from send_typing, so any non-empty status maps to `processing`.
        payload = ("processing" if status else "active") if using_sessions else status
        try:
            await self._call_status_setter(
                set_status, deadline, channel_id=chat_id, thread_ts=thread_ts, status=payload)
            return (_SESSIONS if using_sessions else _LEGACY), payload
        except Exception as e:
            if not using_sessions or flat:
                logger.debug("[Slack] %s.setStatus %s: %s", _SESSIONS if flat else _LEGACY,
                             fail_label, e, exc_info=True)
                return None
            if deadline is not None and time.monotonic() >= deadline:
                logger.debug("[Slack] agents.sessions.setStatus %s: %s; clear budget spent, no "
                             "legacy fallback", fail_label, e, exc_info=True)
                return None
            # Session channels lack the legacy API, but older SDKs and session-less
            # channels still accept it with the original free text / empty clear.
            try:
                await self._call_status_setter(
                    client.assistant_threads_setStatus, deadline,
                    channel_id=chat_id, thread_ts=thread_ts, status=status)
                return _LEGACY, status
            except Exception as e2:
                logger.debug("[Slack] assistant.threads.setStatus %s: %s (%s)", fail_label, e2, e,
                             exc_info=True)
                return None

    async def _call_status_setter(
            self, method: Callable[..., Any], deadline: Optional[float], **kwargs: Any) -> None:
        """Call a status setter; with a ``deadline`` every request is cut off at it, and a
        429/``ratelimited`` error is retried ONCE after ``Retry-After``, the wait cut to what is
        left of the budget (no retry once it is spent). Without one (the throttled
        ``send_typing`` heartbeat) any failure just raises."""
        try:
            await self._status_request_within(method, deadline, kwargs)
        except Exception as e:
            if deadline is None or not _is_ratelimit_error(e):
                raise
            delay = min(self._retry_after_from_exc(e) or _STATUS_CLEAR_RETRY_DEFAULT_DELAY,
                        deadline - time.monotonic())
            if delay <= 0:
                raise
            logger.debug("[Slack] status clear rate-limited; retrying in %.1fs", delay)
            await asyncio.sleep(delay)
            if time.monotonic() >= deadline:
                raise
            await self._status_request_within(method, deadline, kwargs)

    @staticmethod
    async def _status_request_within(
            method: Callable[..., Any], deadline: Optional[float], kwargs: dict[str, Any]) -> None:
        if deadline is None:
            await method(**kwargs)
        else:
            await asyncio.wait_for(method(**kwargs), timeout=deadline - time.monotonic())
