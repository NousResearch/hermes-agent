"""Kanban notifier: claim terminal task events per subscription and deliver them.

``GatewayKanbanWatchersMixin._kanban_notifier_watcher`` owns the loop and
the GC cadence; the per-tick claim (``_notifier_collect``) and the
per-subscription delivery (``_KanbanNotification``) live here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from agent.i18n import t

from gateway.kanban_watchers_common import _list_boards, _to_thread_process_service, logger


def _kbc():
    from hermes_cli import kanban_db_connect
    return kanban_db_connect


def _kbn():
    from hermes_cli import kanban_db_notify
    return kanban_db_notify

# "status" covers dashboard drag-drop and `_set_status_direct()`.
# ``review_requested`` wakes the origin like a block but is not one;
# the task is not archived so later review cycles keep notifying.
TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked", "block_loop_detected", "review_requested", "changes_requested")
# Kinds that hand a decision back to the origin, which must take a turn.
# status/archived/unblocked are bookkeeping.
_WAKE_KINDS = ("completed", "gave_up", "crashed", "timed_out", "blocked", "review_requested", "changes_requested", "block_loop_detected")
# Consecutive send failures (adapter raised OR reported SendResult(success=False))
# before a sub is dropped as a dead chat. 12 ≈ 60s at the 5s cadence: a transient
# API outage must not permanently unsubscribe a live review-gate channel.
# Subscriptions are removed only when the task reaches the irreversible archived status. ``done`` is
# reversible in review/controller flows, so removing its subscription would silence a later reopen. We used
# to also unsub on any terminal event kind (gave_up / crashed / timed_out / blocked), but that silently
# dropped the user out of the loop whenever the dispatcher respawned the task: a worker that crashes, gets
# reclaimed, runs again, and crashes a second time would only notify on the first crash because the
# subscription was deleted after the first event. Same shape as the reblock-after-unblock cycle that PR
# #22941 fixed for `blocked`. Keeping the subscription alive until the task is archived lets the cursor
# (advanced atomically by claim_unseen_events_for_sub) handle dedup, and any retry-loop event reaches the
# user. Per-subscription send-failure counter. Adapter.send raising means the chat is dead (deleted, bot
# kicked, etc.) — after N consecutive send failures the sub is dropped so we don't spin against a dead chat
# every 5 seconds forever. A genuinely dead chat still drops, just ~60s later — a fine trade for an
# unattended gate where a false drop means silent work pileup.
MAX_SEND_FAILURES = 12

_LOCAL_PATH_RE = re.compile(r"(?<![\w:/])(?:/(?:Users|home|private|tmp|var|etc|workspace)/[^\s,;]+|" r"[A-Za-z]:\\[^\s,;]+)")


def _safe_review_reason(value: Any, limit: int = 160) -> str:
    """Return a mobile-friendly review reason safe for external delivery."""
    from agent.redact import redact_sensitive_text

    reason = redact_sensitive_text("" if value is None else str(value), force=True, redact_url_credentials=True)
    reason = " ".join(_LOCAL_PATH_RE.sub("[local path]", reason).split())
    if len(reason) > limit:
        reason = reason[: limit - 1].rstrip() + "…"
    return reason


def _wake_scope_id(adapter: Any, sub: dict) -> Optional[str]:
    """Return the tenant scope (Slack workspace) a subscription's wake keys to.

    ``build_session_key()`` includes ``scope_id`` on multi-tenant platforms,
    so the wake must carry the same scope as inbound messages. Persisted
    ``delivery_metadata`` wins (it records the creating scope); the adapter's
    live chat → scope map only covers rows without metadata. ``None`` means
    unscoped, matching an unscoped platform's key.
    """
    delivery_meta = sub.get("delivery_metadata")
    if isinstance(delivery_meta, dict):
        for key in ("scope_id", "slack_team_id", "team_id"):
            value = delivery_meta.get(key)
            if value:
                return str(value)
    resolver = getattr(adapter, "scope_id_for_chat", None)
    if not callable(resolver):
        return None
    try:
        resolved = resolver(str(sub.get("chat_id") or ""))
    except Exception as exc:
        # An adapter-side lookup failure yields no scope, never an error.
        logger.debug("kanban notifier: scope lookup failed for chat %s: %s", sub.get("chat_id"), exc, exc_info=True)
        return None
    return str(resolved) if resolved else None


def _platform_names(mapping: Any) -> set[str]:
    """Lower-cased platform names of an adapters mapping (Platform enums or strings)."""
    return {getattr(platform, "value", str(platform)).lower() for platform in mapping}


# --- Collection (runs in a worker thread) ---


class _Collector:
    """One tick's claim state: which profiles/platforms this gateway serves and the GC gate."""

    def __init__(self, runner: Any, kb: Any, *, notifier_profile: Optional[str], gc_due: bool, gc_retention_days: int) -> None:
        self.kb = kb
        self.notifier_profile = notifier_profile
        self.gc_due = gc_due
        self.gc_retention_days = gc_retention_days
        self.deliveries: list[dict] = []
        self.include_unowned = runner._owns_kanban_dispatcher_lock()
        self.profile_adapters = getattr(runner, "_profile_adapters", {})
        self.notifier_profiles = {notifier_profile}
        self.notifier_profiles.update(str(p).strip() for p in self.profile_adapters if str(p).strip())
        # Include every platform any secondary profile has live. This is only a
        # coarse pre-filter; the precise per-profile check (_authorization_adapter,
        # no default fallback) runs at delivery and rewinds the claim if it
        # resolves to None. An unclaimed event never retries, so dropping a
        # secondary-profile sub here would lose it.
        self.active_platforms = _platform_names(runner.adapters).union(
            *(_platform_names(m) for m in self.profile_adapters.values()))

    def collect(self) -> list[dict]:
        if not self.active_platforms:
            logger.debug("kanban notifier: no connected adapters; skipping tick")
            return self.deliveries
        # Poll each resolved DB path once: several slugs can map to one DB when
        # HERMES_KANBAN_DB pins the board path.
        kb = self.kb
        seen_db_paths: set[str] = set()
        for board_meta in _list_boards(kb):
            slug = board_meta.get("slug") or kb.DEFAULT_BOARD
            db_path = board_meta.get("db_path")
            try:
                resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(kb.kanban_db_path(slug).resolve())
            except Exception:
                resolved_db_path = f"slug:{slug}"
            if resolved_db_path in seen_db_paths:
                logger.debug("kanban notifier: skipping duplicate board slug %s for DB %s", slug, resolved_db_path)
                continue
            seen_db_paths.add(resolved_db_path)
            self.collect_board(slug)
        return self.deliveries

    def _board_has_subs(self, slug: str) -> bool:
        """Cheap read-only probe before the writable connect() (schema init, WAL
        sidecars, checkpoints); a probe failure falls back to the writable open."""
        try:
            count = _kbn().count_notify_subs(
                board=slug, notifier_profiles=self.notifier_profiles, include_unowned=self.include_unowned)
        except Exception as exc:
            logger.debug("kanban notifier: read-only subscription probe failed "
                         "for board %s (%s); falling back to writable open", slug, exc)
            return True
        if count == 0:
            logger.debug("kanban notifier: board %s has no subscriptions owned by %s; skipping open",
                         slug, sorted(self.notifier_profiles))
        return count != 0

    def _gc_stale_subs(self, conn: Any, slug: str) -> None:
        """Best-effort stale-sub sweep: a failed sweep never blocks delivery; the next hourly gate retries."""
        try:
            _purged = _kbn().purge_stale_done_notify_subs(conn, max_age_days=self.gc_retention_days)
            if _purged:
                logger.info("kanban notifier: purged %d stale done/blocked-task subscription(s) on board %s (retention %dd)",
                            _purged, slug, self.gc_retention_days)
        except Exception as _gc_exc:
            logger.debug("kanban notifier: stale-sub GC failed for board %s: %s", slug, _gc_exc)

    def _claim_for_sub(self, conn: Any, slug: str, sub: dict) -> Optional[dict]:
        """Claim one subscription's unseen events; None when skipped or nothing new."""
        owner_profile = sub.get("notifier_profile") or None
        if owner_profile and owner_profile != self.notifier_profile and not self.profile_adapters.get(owner_profile):
            logger.debug("kanban notifier: subscription for %s owned by profile %s; current profile %s has no adapter for it, skipping",
                         sub.get("task_id"), owner_profile, self.notifier_profile)
            return None
        platform = (sub.get("platform") or "").lower()
        if platform not in self.active_platforms:
            logger.debug("kanban notifier: subscription for %s on %s skipped; adapter not connected",
                         sub.get("task_id"), platform or "<missing>")
            return None
        old_cursor, cursor, events = _kbn().claim_unseen_events_for_sub(
            conn, task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "", kinds=TERMINAL_KINDS,
        )
        if not events:
            return None
        task = self.kb.get_task(conn, sub["task_id"])
        logger.debug("kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                     len(events), sub["task_id"], slug, old_cursor, cursor)
        return {"sub": sub, "old_cursor": old_cursor, "cursor": cursor, "events": events, "task": task, "board": slug}

    def collect_board(self, slug: str) -> None:
        """Claim events on one board, appending delivery dicts to ``deliveries``."""
        if not self._board_has_subs(slug):
            return
        kb = self.kb
        try:
            conn = _kbc().connect(board=slug)
        except Exception as exc:
            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
            return
        try:
            if self.gc_due:
                self._gc_stale_subs(conn, slug)
            # No explicit init_db(): connect() already runs the migration once per
            # process, and init_db() would re-run it on a second connection racing
            # the first.
            subs = _kbn().list_notify_subs(conn, notifier_profiles=self.notifier_profiles, include_unowned=self.include_unowned)
            if not subs:
                logger.debug("kanban notifier: board %s has no subscriptions", slug)
            for sub in subs:
                try:
                    claimed = self._claim_for_sub(conn, slug, sub)
                    if claimed is not None:
                        self.deliveries.append(claimed)
                except Exception as sub_exc:
                    # One bad subscription must not block the rest of the tick.
                    logger.warning("kanban notifier: subscription for %s on board %s failed: %s",
                                   sub.get("task_id"), slug, sub_exc)
        finally:
            conn.close()


def _notifier_collect(runner: Any, kb: Any, *, notifier_profile: Optional[str], gc_due: bool, gc_retention_days: int) -> list[dict]:
    """Claim unseen terminal events for every owned subscription on every board.

    Each gateway polls only subscriptions owned by profiles whose adapters it
    hosts; legacy rows without a profile stamp are visible only to the process
    holding the singleton dispatcher lock.
    """
    return _Collector(
        runner, kb, notifier_profile=notifier_profile, gc_due=gc_due, gc_retention_days=gc_retention_days,
    ).collect()


# --- Per-event message formatting: kind -> (msg, wake_handoff, wake_review_detail) ---
# ``None`` for handoff / review_detail leaves the accumulated wake value untouched.


def _payload(ev: Any, key: str) -> Any:
    """Shared "payload present and truthy" read."""
    return ev.payload.get(key) if ev.payload and ev.payload.get(key) else None


def _clip(ev: Any, key: str, fmt: str, limit: int) -> str:
    """``fmt`` applied to the truncated payload value, or ``""`` when absent."""
    value = _payload(ev, key)
    return fmt.format(str(value)[:limit]) if value else ""


_NL = "\n{}"


def _first_line(text: str, limit: int) -> str:
    lines = text.strip().splitlines()
    return lines[0][:limit] if lines else text[:limit]


def _fmt_completed(ev, n) -> tuple:
    # Prefer the run summary from the event payload; fall back to task.result for legacy rows.
    wake_handoff = None
    payload_summary = _payload(ev, "summary")
    if payload_summary:
        wake_handoff = _first_line(str(payload_summary), 200)
    elif n.task and n.task.result:
        wake_handoff = _first_line(n.task.result, 160)
    handoff = f"\n{wake_handoff}" if wake_handoff is not None else ""
    return f"✔ {n.head} done — {n.title}{handoff}", wake_handoff, None


def _fmt_review_requested(ev, n) -> tuple:
    # Implementation done; task moved to the review lane. Carry the handoff
    # into the wake turn like ``completed`` so the reviewer needn't re-read the board.
    handoff = ""
    wake_handoff = None
    summary = _payload(ev, "summary")
    if summary:
        summary = str(summary)
        handoff = f"\n{summary[:200]}"
        wake_handoff = _first_line(summary, 200)
    return f"👀 {n.head} ready for review — {n.title}{handoff}", wake_handoff, None


def _fmt_changes_requested(ev, n) -> tuple:
    payload = ev.payload or {}
    reason = _safe_review_reason(payload.get("reason"))
    reviewer = _safe_review_reason(payload.get("reviewer"), 48)
    implementer = _safe_review_reason(payload.get("implementer"), 48)
    reason_text = reason or "reviewer feedback requires changes"
    provenance = f" — reviewer @{reviewer}" if reviewer else ""
    if implementer:
        provenance += f" → implementer @{implementer}"
    msg = f"🛑 {n.board_tag}Kanban {n.task_id} review requested changes/BLOCK: {reason_text}{provenance}"
    return msg, None, reason_text


# archived / unblocked are claimed (so the cursor advances past them) but
# intentionally silent (no formatter), and excluded from _WAKE_KINDS so they
# never wake the creator.
_EVENT_FORMATTERS: dict[str, Callable[[Any, "_KanbanNotification"], tuple]] = {
    "completed": _fmt_completed,
    "blocked": lambda ev, n: (f"⏸ {n.head} blocked{_clip(ev, 'reason', ': {}', 160)}", None, None),
    "gave_up": lambda ev, n: (
        f"✖ {n.head} gave up after repeated spawn failures{_clip(ev, 'error', _NL, 200)}", None, None,
    ),
    "crashed": lambda ev, n: (f"✖ {n.head} worker crashed (pid gone); dispatcher will retry", None, None),
    "timed_out": lambda ev, n: (
        f"⏱ {n.head} timed out (max_runtime={int(_payload(ev, 'limit_seconds') or 0)}s); will retry", None, None,
    ),
    "status": lambda ev, n: (f"🔄 {n.head} → {_payload(ev, 'status') or ''}", None, None),
    "review_requested": _fmt_review_requested,
    "changes_requested": _fmt_changes_requested,
    # Re-blocked for the same cause past the limit and routed to `triage` for a
    # human. It emits no blocked/status event, so ping loudly here.
    "block_loop_detected": lambda ev, n: (
        f"🛑 {n.head} routed to TRIAGE — needs a human decision"
        f"{_clip(ev, 'recurrences', ' (blocked {}x for the same cause)', 200)}{_clip(ev, 'reason', ': {}', 160)}",
        None, None,
    ),
}


# --- Delivery of one claimed batch (one subscription, N events) ---


class _KanbanNotification:
    """Deliver one subscription's claimed events, then settle the cursor.

    Cursor advance ordering by adapter class:
    * push + notify: the text send WAS the delivery → advance now; wake
      injection stays best-effort.
    * non-push or wake-only: the wake IS the delivery → it runs FIRST and the
      cursor advances only after it succeeds; failure rewinds like a failed
      send(). An unknown platform advances the cursor so it can't replay forever.
    """

    def __init__(self, runner: Any, d: dict, *, platform_cls: Any, sub_fail_counts: dict) -> None:
        self.runner = runner
        self.d = d
        self.platform_cls = platform_cls
        self.sub_fail_counts = sub_fail_counts
        self.sub = sub = d["sub"]
        self.task = task = d["task"]
        self.board_slug = d.get("board")
        self.platform_str = (sub["platform"] or "").lower()
        self.task_id = sub["task_id"]
        self.sub_profile = sub.get("notifier_profile") or ""
        self.title = (task.title if task else sub["task_id"])[:120]
        self.board_tag = f"[{self.board_slug}] " if self.board_slug else ""
        # Attribute the ping to the worker that did the work.
        tag = f"@{task.assignee} " if task and task.assignee else ""
        self.head = f"{self.board_tag}{tag}Kanban {self.task_id}"
        # The wake self-post path needs the key even when every event was skipped.
        self.sub_key = (sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or "")
        mode = sub.get("delivery_mode") or "notify"
        self.wake_agent = mode in ("notify+wake", "wake")
        self.send_passive = mode != "wake"
        # Worker handoff carried into the synthetic wake turn so the woken
        # creator doesn't re-decompose work already on the board.
        self.wake_handoff = self.wake_review_detail = self.session_key = self.synth = ""
        self.plat: Any = None
        self.adapter: Any = None
        self.is_push_adapter = True
        self.wake_kinds: set = set()

    # -- cursor / subscription ops (blocking, run in a fresh-context thread) --

    async def rewind(self) -> None:
        await _to_thread_process_service(
            self.runner._kanban_rewind, self.sub, self.d["cursor"], self.d.get("old_cursor", 0), self.board_slug,
        )

    async def advance(self) -> None:
        await _to_thread_process_service(self.runner._kanban_advance, self.sub, self.d["cursor"], self.board_slug)

    async def unsub(self) -> None:
        await _to_thread_process_service(self.runner._kanban_unsub, self.sub, self.board_slug)

    def clear_failures(self) -> None:
        self.sub_fail_counts.pop(self.sub_key, None)

    async def delivery_failed(self, fmt: str, prefix: tuple, drop_fmt: str, exc: Exception, exc_info: bool) -> None:
        """Bump the failure counter; drop the sub past the limit, else rewind the claim so the next tick retries."""
        fails = self.sub_fail_counts.get(self.sub_key, 0) + 1
        self.sub_fail_counts[self.sub_key] = fails
        logger.warning(fmt, *prefix, fails, MAX_SEND_FAILURES, exc, exc_info=exc_info)
        if fails >= MAX_SEND_FAILURES:
            logger.warning(drop_fmt, self.task_id, self.platform_str, fails)
            await self.unsub()
            self.clear_failures()
        else:
            await self.rewind()

    async def _wake_failed(self, fmt: str, exc: Exception) -> None:
        drop_fmt = "kanban notifier: dropping subscription %s on %s after %d consecutive wake failures"
        await self.delivery_failed(fmt, (self.task_id,), drop_fmt, exc, True)

    # -- formatting --

    def format_event(self, ev: Any) -> Optional[str]:
        """Render one event; accumulates wake handoff/review detail. None → silent kind."""
        formatter = _EVENT_FORMATTERS.get(ev.kind)
        if formatter is None:
            return None
        msg, handoff, review_detail = formatter(ev, self)
        if handoff is not None:
            self.wake_handoff = handoff
        if review_detail is not None:
            self.wake_review_detail = review_detail
        return msg

    def build_wake_text(self) -> None:
        """Set ``wake_kinds`` / ``session_key`` / ``synth`` for the wake paths."""
        task, sub = self.task, self.sub
        self.wake_kinds = {ev.kind for ev in self.d["events"] if ev.kind in _WAKE_KINDS} if self.wake_agent else set()
        if not self.wake_kinds:
            return
        if self.is_push_adapter:
            self.session_key = getattr(task, "session_id", None) or ""
        else:
            # Non-push wakes target sub["chat_id"] (the raw session id the
            # subscriber registered). task.session_id may be a WORKER session
            # for child tasks; use it only for legacy rows.
            self.session_key = sub["chat_id"] or getattr(task, "session_id", None) or ""
        # i18n keys: gateway.kanban.wake.<kind> for each _WAKE_KINDS entry.
        _parts = [t(f"gateway.kanban.wake.{k}") for k in _WAKE_KINDS if k in self.wake_kinds]
        _status = t("gateway.kanban.wake.status_joiner").join(_parts) or t("gateway.kanban.wake.status_default")
        synth = t(
            "gateway.kanban.wake.message",
            task_id=sub["task_id"], status=_status, title=self.title,
            assignee=task.assignee if task else "", board=self.board_slug,
        )
        # Label as an automatic notification and carry the handoff so the
        # creator inspects the board instead of re-decomposing.
        if self.wake_handoff:
            synth += "\n" + t("gateway.kanban.wake.handoff", summary=self.wake_handoff)
        if self.wake_review_detail:
            synth += "\n" + t("gateway.kanban.wake.review_detail", reason=self.wake_review_detail)
        self.synth = synth + "\n\n" + t("gateway.kanban.wake.guidance")

    def _log_woke(self) -> None:
        logger.info("kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                    self.task_id, self.platform_str, self.sub["chat_id"], self.sub_profile or "default", self.wake_kinds)

    async def wake(self) -> None:
        """Wake the creator session (raises on failure): push adapters get a full SessionSource, non-push a raw self-post."""
        from gateway.wake import deliver_wake
        sub = self.sub
        if not self.is_push_adapter:
            await deliver_wake(self.adapter, text=self.synth, session_id=self.session_key)
            self._log_woke()
            return
        from gateway.session import SessionSource
        # Rebuild the creator's real session scope from the persisted chat_type:
        # build_session_key() keys DMs differently from group/thread, so a
        # hardcoded "group" mis-routed DM/thread creators into a fresh session.
        # Legacy rows may carry chat_type in delivery_metadata; last resort is
        # "group". A mismatch only degrades to a fresh session.
        # Legacy rows written before the column existed may still carry chat_type in delivery_metadata
        # (#60600 rows) — fall back to that, then to "group" (the historical default that suits the
        # dashboard/group flows). handle_message() get_or_create_session's the target, so a mismatch only
        # ever degrades to a fresh session, never an exception.
        _chat_type = str(sub.get("chat_type") or "").strip()
        if not _chat_type:
            _delivery_meta = sub.get("delivery_metadata")
            if isinstance(_delivery_meta, dict):
                _chat_type = str(_delivery_meta.get("chat_type") or "").strip()
        _source = SessionSource(
            platform=self.plat, chat_id=sub["chat_id"], chat_type=_chat_type or "group",
            thread_id=sub.get("thread_id") or None, user_id=sub.get("user_id"), user_id_alt=sub.get("user_id_alt"),
            profile=self.sub_profile or None, scope_id=_wake_scope_id(self.adapter, sub),
        )
        await deliver_wake(self.adapter, text=self.synth, session_id=self.session_key, source=_source)
        self._log_woke()

    async def _send_event(self, ev: Any, msg: str) -> None:
        """Send one text ping; raises on adapter exception or SendResult(success=False)."""
        sub, adapter = self.sub, self.adapter
        delivery_metadata = sub.get("delivery_metadata")
        metadata: dict[str, Any] = dict(delivery_metadata) if isinstance(delivery_metadata, dict) else {}
        if sub.get("thread_id") and not metadata.get("thread_id"):
            metadata["thread_id"] = sub["thread_id"]
        _send_res = await adapter.send(sub["chat_id"], msg, metadata=metadata)
        # SendResult(success=False) without an exception is a FAILED delivery
        # (else the event is lost); None / non-SendResult keeps the
        # "no exception == delivered" contract.
        if getattr(_send_res, "success", True) is False:
            raise RuntimeError(f"adapter send() reported failure: {getattr(_send_res, 'error', None) or 'unknown error'}")
        logger.debug("kanban notifier: delivered %s event for %s to %s/%s on board %s",
                     ev.kind, self.task_id, self.platform_str, sub["chat_id"], self.board_slug)
        # Upload artifact paths from the completion payload / legacy result as
        # native files. Only on ``completed`` so retries never spam attachments.
        if ev.kind == "completed":
            try:
                await self.runner._deliver_kanban_artifacts(
                    adapter=adapter, chat_id=sub["chat_id"], metadata=metadata,
                    event_payload=getattr(ev, "payload", None), task=self.task,
                )
            except Exception as art_exc:
                logger.debug("kanban notifier: artifact delivery for %s failed: %s", self.task_id, art_exc)

    async def _send_pings(self) -> bool:
        """Send every text ping; False when a send failed (claim already rewound/dropped)."""
        for ev in self.d["events"]:
            msg = self.format_event(ev)
            if msg is None:
                continue
            # Non-push adapters (api_server) always report SendResult(success=False)
            # from send(); treating that as failure would drop the sub forever and
            # make the wake path unreachable. Skip the doomed send; the self-post
            # IS the delivery and resolves the failure counter.
            if not self.is_push_adapter and self.wake_agent:
                logger.debug(
                    "kanban notifier: adapter %s has no push channel; skipping text ping for %s, relying "
                    "on wake self-post instead", self.platform_str, self.task_id,
                )
                continue
            if not self.send_passive:
                # Wake-only: the wake path is the sole delivery and resolves the counter.
                continue
            try:
                await self._send_event(ev, msg)
                self.clear_failures()
            except Exception as exc:
                await self.delivery_failed(
                    "kanban notifier: send failed for %s on %s (attempt %d/%d): %s", (self.task_id, self.platform_str),
                    "kanban notifier: dropping subscription %s on %s after %d consecutive send failures", exc, False,
                )
                return False
        return True

    async def deliver(self) -> None:
        try:
            self.plat = self.platform_cls(self.platform_str)
        except ValueError:
            await self.advance()
            return
        # Same chokepoint as authorization: a stamped profile is served by ITS
        # same-platform adapter and never falls back to the default profile's
        # bot (cross-profile mis-delivery). None only when the profile (or
        # default) has no adapter.
        adapter = self.runner._authorization_adapter(self.plat, self.sub_profile or None)
        if adapter is None:
            logger.debug("kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                         self.platform_str, self.task_id)
            await self.rewind()
            return
        self.adapter = adapter
        from gateway.wake import adapter_supports_push
        self.is_push_adapter = adapter_supports_push(adapter)

        if not await self._send_pings():
            return
        # All text pings delivered (or skipped for non-push / wake-only).
        self.build_wake_text()
        wake_kinds, is_push = self.wake_kinds, self.is_push_adapter

        # Non-push self-post, or wake-only push sub: the wake IS the delivery
        # and must succeed BEFORE the cursor advances.
        if wake_kinds and (not self.send_passive if is_push else bool(self.session_key)):
            try:
                await self.wake()
                self.clear_failures()
            except Exception as _wk_err:
                await self._wake_failed(
                    "kanban notifier: wake-only delivery failed for %s (attempt %d/%d): %s" if is_push
                    else "kanban notifier: wake self-post failed for %s (attempt %d/%d): %s",
                    _wk_err,
                )
                return

        # Delivery complete: advance the cursor (the dedup mechanism).
        await self.advance()
        if not is_push:
            self.clear_failures()
        if is_push and self.send_passive and wake_kinds:
            # notify+wake: text ping was the delivery and the cursor has
            # advanced; the wake stays best-effort, but log at WARNING so a
            # persistently failing wake is visible.
            try:
                await self.wake()
            except Exception as _wk_err:
                logger.warning("kanban notifier: wakeup injection failed for %s: %s", self.task_id, _wk_err, exc_info=True)
        # Unsubscribe only on archive; ``done`` is reversible.
        if self.task and self.task.status == "archived":
            await self.unsub()
