"""Kanban notifier: claim terminal task events per subscription and deliver them.

``GatewayKanbanWatchersMixin._kanban_notifier_watcher`` owns the loop and
the GC cadence; the per-tick claim (``_notifier_collect``) and the
per-subscription delivery (``_KanbanNotification``) live here.
"""

from __future__ import annotations

import re
import json
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
TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked", "block_loop_detected", "review_requested", "changes_requested") + ("dependency_wait", "review_handoff_required", "delivery_phase_completed", "delivery_changes_requested", "delivery_accepted", "delivery_review_hold")
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

_DEFAULT_WAKE_KINDS = ("completed", "crashed", "review_requested", "review_handoff_required", "changes_requested")
_ALL_WAKE_KINDS = _WAKE_KINDS + ("review_handoff_required",)

def _configured_wake_kinds() -> tuple:
    """Event kinds allowed to wake the subscribed agent (``kanban.wake_events``).

    Reads the read-only config on every call (cheap dict lookup; the loader
    caches) so an operator edit takes effect at the next notifier tick without
    a restart. Unknown kinds are ignored; an unset or invalid key falls back to
    ``_DEFAULT_WAKE_KINDS``.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("wake_events")
    except Exception:
        raw = None
    if isinstance(raw, (list, tuple)):
        kinds = tuple(str(k).strip() for k in raw if str(k).strip() in _ALL_WAKE_KINDS)
        if kinds:
            return kinds
    return _DEFAULT_WAKE_KINDS


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
        self.runner = runner
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
            seen = getattr(self.runner, "_kanban_undeliverable_subs", set())
            key = (sub["task_id"], platform, sub["chat_id"], sub.get("thread_id"))
            if key not in seen:
                logger.warning("kanban notifier: undeliverable subscription for %s on %s", sub["task_id"], platform)
                seen.add(key)
                self.runner._kanban_undeliverable_subs = seen
            return None
        _kbn().stage_unseen_notify_deliveries_for_sub(
            conn, task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
            thread_id=sub.get("thread_id") or "", kinds=TERMINAL_KINDS,
        )
        rows = _kbn().list_notify_deliveries(conn, task_id=sub["task_id"], platform=sub["platform"],
            chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or "",
            states=("pending", "sending", "ambiguous", "reconciling", "parked"))
        if not rows:
            return None
        task = self.kb.get_task(conn, sub["task_id"])
        return {"sub": sub, "old_cursor": sub.get("last_event_id", 0), "cursor": sub.get("last_event_id", 0),
                "events": [r["event"] for r in rows], "delivery_rows": rows,
                "pending_parents": self.kb.pending_parents(conn, sub["task_id"]),
                "task": task, "board": slug}

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


_NOTICE_REASON_LIMIT = 400
_SELF_ADVANCING_STATUSES = ("ready", "running", "todo", "review", "in_review")

def _ready_line(task: Any) -> str:
    """Answer "will the board move this on its own?" — the Ready yes/no."""
    status = str(getattr(task, "status", "") or "unknown")
    if status in _SELF_ADVANCING_STATUSES:
        return f"yes ({status} — the board will pick this up on its own)"
    return f"no ({status} — nothing moves until a human acts)"


def _stop_notice(
    *,
    icon: str,
    board_tag: str,
    who_tag: str,
    task_id: str,
    headline: str,
    title: str,
    reason: str,
    ready: str,
    next_action: str,
) -> str:
    """Render one durable, actionable stop notice.

    Every stop carries the same four answers, because an operator reading a
    ping in a busy thread needs all four to act without opening the board:
    which card, what exactly stopped it, whether the board will recover on its
    own, and who does what next.
    """
    lines = [f"{icon} {board_tag}{who_tag}Kanban {task_id} {headline} — {title}"]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append(f"Ready: {ready}")
    lines.append(f"Next: {next_action}")
    return "\n".join(lines)


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
def _notice_completed(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    handoff = ''
    payload_summary = None
    if ev.payload and ev.payload.get('summary'):
        payload_summary = str(ev.payload['summary'])
    if payload_summary:
        lines = payload_summary.strip().splitlines()
        h = lines[0][:200] if lines else payload_summary[:200]
        handoff = f'\n{h}'
        wake_handoff = h
    elif task and task.result:
        lines = task.result.strip().splitlines()
        r = lines[0][:160] if lines else task.result[:160]
        handoff = f'\n{r}'
        wake_handoff = r
    completion_kind = str((ev.payload or {}).get('completion_kind') or 'final')
    review_handoff = (ev.payload or {}).get('review_handoff')
    if completion_kind == 'delivery_review_result':
        implementation_task = _safe_review_reason(review_handoff.get('implementation_task_id'), 48) if isinstance(review_handoff, dict) else ''
        review_owner = _safe_review_reason(review_handoff.get('owner'), 48) if isinstance(review_handoff, dict) else ''
        msg = _stop_notice(icon='🧾', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='REVIEW RESULT RECORDED', title=title, reason=_safe_review_reason(payload_summary or 'independent review run completed', _NOTICE_REASON_LIMIT), ready='no (the review result is intermediate; the delivery controller has not accepted it)', next_action=f"@{review_owner or 'review owner'} — evaluate this result through the canonical delivery controller" + (f' for implementation task {implementation_task}.' if implementation_task else '.'))
    elif completion_kind == 'implementation_ready_for_review' and isinstance(review_handoff, dict):
        review_task_id = _safe_review_reason(review_handoff.get('review_task_id'), 48)
        review_owner = _safe_review_reason(review_handoff.get('owner'), 48)
        msg = _stop_notice(icon='👀', board_tag=board_tag, who_tag='', task_id=sub['task_id'], headline='REVIEW HANDOFF', title=title, reason=_safe_review_reason(payload_summary or 'implementation ready', _NOTICE_REASON_LIMIT), ready='no (implementation ready for independent review; acceptance is pending)', next_action=f"@{review_owner or 'review owner'} — ensure {review_task_id or 'the canonical review task'} is independently reviewed. This implementation handoff is not final acceptance.")
    else:
        msg = f"✔ {board_tag}{tag}Kanban {sub['task_id']} done — {title}{handoff}"
    return msg, wake_handoff, wake_review_detail


def _notice_delivery_phase_completed(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    payload = ev.payload or {}
    implementation_task = _safe_review_reason(payload.get('implementation_task'), 48) or sub['task_id']
    review_task = _safe_review_reason(payload.get('review_task'), 48)
    head = _safe_review_reason(payload.get('head'), 40)
    pair = f'head {head}'
    if review_task:
        pair += f'; review task {review_task}'
    review_requirement = getattr(task, 'review_requirement', None) if task else None
    if isinstance(review_requirement, str):
        try:
            review_requirement = json.loads(review_requirement)
        except (TypeError, ValueError):
            review_requirement = {}
    if not isinstance(review_requirement, dict):
        review_requirement = {}
    contract_owner = _safe_review_reason(review_requirement.get('owner'), 48)
    if kind == 'delivery_phase_completed':
        evidence = payload.get('evidence')
        evidence = evidence if isinstance(evidence, dict) else {}
        artifact = _safe_review_reason(evidence.get('artifact'), 120)
        checks = evidence.get('checks')
        check_count = len(checks) if isinstance(checks, list) else 0
        evidence_text = f'; evidence {artifact}, {check_count} focused check(s)' if artifact else f'; {check_count} focused check(s) recorded'
        msg = _stop_notice(icon='👀', board_tag=board_tag, who_tag='', task_id=implementation_task, headline='PHASE COMPLETE', title=title, reason=f'{pair}{evidence_text}; acceptance pending', ready='no (independent review and acceptance are pending)', next_action=f"review task {review_task or 'named in the contract'} — independently review exact head {head}.")
    elif kind == 'delivery_changes_requested':
        findings = _safe_review_reason(payload.get('findings'), _NOTICE_REASON_LIMIT)
        msg = _stop_notice(icon='🛑', board_tag=board_tag, who_tag='', task_id=implementation_task, headline='CHANGES REQUESTED', title=title, reason=findings or 'reviewer requested changes', ready='no (this generation is not accepted)', next_action=f'{owner} — address the findings and submit a new exact-head generation; prior {pair}.')
        wake_review_detail = findings
    elif kind == 'delivery_accepted':
        accepted = payload.get('acceptance') == 'ready'
        msg = _stop_notice(icon='✅', board_tag=board_tag, who_tag='', task_id=implementation_task, headline='DELIVERY ACCEPTED', title=title, reason=f'canonical independent review accepted {pair}' if accepted else f'acceptance metadata is incomplete for {pair}', ready='yes (canonical acceptance is ready)' if accepted else 'no (canonical acceptance is not ready)', next_action='origin owner — delivery is accepted; proceed with the separately authorized release or activation step.' if accepted else 'origin owner — inspect the malformed acceptance event.')
    else:
        reason = _safe_review_reason(payload.get('reason'), _NOTICE_REASON_LIMIT)
        msg = _stop_notice(icon='⏸', board_tag=board_tag, who_tag='', task_id=implementation_task, headline='DELIVERY HOLD', title=title, reason=reason or 'delivery review is held', ready='no (delivery acceptance is held)', next_action=f"@{contract_owner or 'review owner'} — resolve the hold for {pair}, then submit the required evidence through the same canonical pair.")
    return msg, wake_handoff, wake_review_detail


def _notice_blocked(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    payload = ev.payload or {}
    raw_reason = payload.get('reason')
    block_kind = str(payload.get('kind') or '').strip()
    ready = _ready_line(task)
    if kind == 'dependency_wait':
        pending = d.get('pending_parents') or []
        if pending:
            waiting_on = ', '.join((f"{parent['id']} “{str(parent.get('title') or '')[:60]}” ({parent.get('status')})" for parent in pending[:5]))
            next_action = f'nothing for you yet — this card resumes when {waiting_on} finishes.'
        else:
            next_action = f'{owner} — this card declared a dependency but has no unsatisfied dependency on the board, so it will be re-dispatched and repeat the same work. Link the card it is really waiting on, or re-block it as a real blocker (`--kind needs_input`).'
        msg = _stop_notice(icon='⏳', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='is waiting on a dependency', title=title, reason=_safe_review_reason(raw_reason, _NOTICE_REASON_LIMIT), ready=ready, next_action=next_action)
    else:
        if payload.get('routed_from') == 'dependency':
            next_action = f"{owner} — this arrived as a dependency wait with nothing to wait for, so it was held here instead of being re-dispatched. Resolve it, then `hermes kanban unblock {sub['task_id']}`."
        elif block_kind == 'capability':
            next_action = f"{owner} — grant the missing tool/permission above, then `hermes kanban unblock {sub['task_id']}`."
        elif block_kind == 'needs_input':
            next_action = f"{owner} — answer the blocker above, then `hermes kanban unblock {sub['task_id']}`."
        else:
            next_action = f"{owner} — resolve the blocker above, then `hermes kanban unblock {sub['task_id']}`."
        msg = _stop_notice(icon='⏸', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='blocked', title=title, reason=_safe_review_reason(raw_reason, _NOTICE_REASON_LIMIT), ready=ready, next_action=next_action)
    return msg, wake_handoff, wake_review_detail


def _notice_gave_up(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    msg = _stop_notice(icon='✖', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='gave up after repeated spawn failures', title=title, reason=_safe_review_reason((ev.payload or {}).get('error'), _NOTICE_REASON_LIMIT), ready=_ready_line(task), next_action=f'{owner} — fix the spawn failure above (profile venv, PATH, credentials, quota), then requeue the card. The dispatcher has stopped retrying it.')
    return msg, wake_handoff, wake_review_detail


def _notice_crashed(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    msg = _stop_notice(icon='✖', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='worker crashed (pid gone)', title=title, reason=_safe_review_reason((ev.payload or {}).get('error') or 'the worker exited without a terminal lifecycle call', _NOTICE_REASON_LIMIT), ready=_ready_line(task), next_action=f'{owner} — operator decision required: requeue this card or block it with the real reason. Confirm the outcome yourself rather than assuming a retry.')
    return msg, wake_handoff, wake_review_detail


def _notice_timed_out(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    limit = 0
    if ev.payload and ev.payload.get('limit_seconds'):
        limit = int(ev.payload['limit_seconds'])
    msg = _stop_notice(icon='⏱', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline='timed out', title=title, reason=f'exceeded max_runtime={limit}s', ready=_ready_line(task), next_action=f'{owner} — decide whether to raise max_runtime, split the card, or block it. Do not assume the retry will finish any faster.')
    return msg, wake_handoff, wake_review_detail


def _notice_status(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    new_status = ''
    if ev.payload and ev.payload.get('status'):
        new_status = str(ev.payload['status'])
    msg = f"🔄 {board_tag}{tag}Kanban {sub['task_id']} → {new_status}"
    return msg, wake_handoff, wake_review_detail


def _notice_review_requested(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    payload = ev.payload or {}
    summary = str(payload.get('summary') or '').strip()
    if summary:
        lines = summary.strip().splitlines()
        wake_handoff = lines[0][:200] if lines else summary[:200]
    reviewer = _safe_review_reason(payload.get('reviewer'), 48)
    implementer = _safe_review_reason(payload.get('implementer'), 48)
    contract_owner = _safe_review_reason(payload.get('owner'), 48)
    if kind == 'review_handoff_required':
        review_task_id = _safe_review_reason(payload.get('review_task_id'), 48)
        if review_task_id:
            ready = 'no (canonical review generation awaits handoff)'
            next_action = f"@{contract_owner or 'review owner'} — hand exact generation evidence to review task {review_task_id}; acceptance remains pending."
        else:
            ready = 'no (canonical review task is missing)'
            next_action = f"@{contract_owner or 'review owner'} — create or assign exactly one independent review task, link it as a dependency child, attach its id to review_requirement, then complete this implementation handoff once to release it."
    elif reviewer:
        ready = 'yes (canonical reviewer is assigned)'
        next_action = f'@{reviewer} — independently review this same card; approve with complete or return it with request-changes. An OPEN/non-Draft PR is not review or acceptance proof.'
    else:
        ready = 'no (canonical reviewer is unassigned)'
        origin = f'@{implementer}' if implementer else 'origin owner'
        next_action = f'origin owner ({origin}) — assign exactly one independent reviewer to this same card. Do not create or reuse an unrelated child.'
    msg = _stop_notice(icon='👀', board_tag=board_tag, who_tag='', task_id=sub['task_id'], headline='REVIEW HANDOFF', title=title, reason=_safe_review_reason(summary or 'implementation entered structured review', _NOTICE_REASON_LIMIT), ready=ready, next_action=next_action)
    return msg, wake_handoff, wake_review_detail


def _notice_changes_requested(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    payload = ev.payload or {}
    reason = _safe_review_reason(payload.get('reason'))
    reviewer = _safe_review_reason(payload.get('reviewer'), 48)
    implementer = _safe_review_reason(payload.get('implementer'), 48)
    reason_text = reason or 'reviewer feedback requires changes'
    provenance = ''
    if reviewer:
        provenance += f' — reviewer @{reviewer}'
    if implementer:
        provenance += f' → implementer @{implementer}'
    msg = f"🛑 {board_tag}Kanban {sub['task_id']} review requested changes/BLOCK: {reason_text}{provenance}"
    wake_review_detail = reason_text
    return msg, wake_handoff, wake_review_detail


def _notice_block_loop_detected(ev, n):
    task, sub, d = n.task, n.sub, n.d
    title, board_tag = n.title, n.board_tag
    tag = f"@{task.assignee} " if task and task.assignee else ""
    owner = f"@{task.assignee}" if task and task.assignee else "origin owner"
    kind = ev.kind
    wake_handoff = wake_review_detail = None
    payload = ev.payload or {}
    recurrences = payload.get('recurrences')
    rc = f' (blocked {recurrences}x for the same cause)' if recurrences else ''
    msg = _stop_notice(icon='🛑', board_tag=board_tag, who_tag=tag, task_id=sub['task_id'], headline=f'routed to TRIAGE{rc}', title=title, reason=_safe_review_reason(payload.get('reason'), _NOTICE_REASON_LIMIT), ready=_ready_line(task), next_action=f'{owner} — the unblock loop was broken deliberately; a human has to decide what changes before this card runs again. Unblocking it as-is will loop.')
    return msg, wake_handoff, wake_review_detail

_EVENT_FORMATTERS = {
    'completed': _notice_completed,
    'delivery_phase_completed': _notice_delivery_phase_completed,
    'delivery_changes_requested': _notice_delivery_phase_completed,
    'delivery_accepted': _notice_delivery_phase_completed,
    'delivery_review_hold': _notice_delivery_phase_completed,
    'blocked': _notice_blocked,
    'dependency_wait': _notice_blocked,
    'gave_up': _notice_gave_up,
    'crashed': _notice_crashed,
    'timed_out': _notice_timed_out,
    'status': _notice_status,
    'review_requested': _notice_review_requested,
    'review_handoff_required': _notice_review_requested,
    'changes_requested': _notice_changes_requested,
    'block_loop_detected': _notice_block_loop_detected,
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
        """Ledger rows retain unsettled work without rewinding delivered events."""
        return None
    async def advance(self) -> None:
        """Per-event ledger acknowledgement owns cursor progress."""
        return None
    async def unsub(self) -> None:
        await _to_thread_process_service(self.runner._kanban_unsub, self.sub, self.board_slug)

    def clear_failures(self) -> None:
        self.sub_fail_counts.pop(self.sub_key, None)

    async def delivery_failed(self, fmt: str, prefix: tuple, drop_fmt: str, exc: Exception, exc_info: bool) -> None:
        """Bump the failure counter; drop the sub past the limit, else rewind the claim so the next tick retries."""
        fails = self.sub_fail_counts.get(self.sub_key, 0) + 1
        self.sub_fail_counts[self.sub_key] = fails
        logger.warning(fmt, *prefix, fails, MAX_SEND_FAILURES, exc, exc_info=exc_info)
        # A durable pending/ambiguous row survives transport outages and restart.
        if fails >= MAX_SEND_FAILURES:
            logger.error("kanban notification held for %s: %s", self.task_id, exc)

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
        self.wake_kinds = {ev.kind for ev in self.wake_events if ev.kind in _configured_wake_kinds()} if self.wake_agent else set()
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

    async def ledger(self, operation: str, ev: Any, **kwargs):
        bridges = {"acknowledge_notify_delivery": "_kanban_ack_delivery",
                   "begin_notify_delivery": "_kanban_begin_delivery",
                   "retry_notify_delivery": "_kanban_retry_delivery"}
        if operation in bridges:
            arguments = {
                "acknowledge_notify_delivery": (kwargs.get("claim_token"), kwargs.get("message_id"), self.board_slug),
                "begin_notify_delivery": (self.board_slug,),
                "retry_notify_delivery": (kwargs.get("claim_token"), kwargs.get("error"), self.board_slug),
            }
            return await _to_thread_process_service(
                getattr(self.runner, bridges[operation]), self.sub, ev.id, *arguments[operation])
        def call():
            with _kbc().connect_closing(board=self.board_slug) as conn:
                return getattr(_kbn(), operation)(conn, task_id=self.task_id,
                    platform=self.sub["platform"], chat_id=self.sub["chat_id"],
                    thread_id=self.sub.get("thread_id") or "", event_id=ev.id, **kwargs)
        return await _to_thread_process_service(call)

    async def reconcile_row(self, row: dict) -> bool:
        ev = row["event"]
        claim = await self.ledger("claim_notify_reconciliation", ev)
        if not claim:
            return False
        reconcile = getattr(self.adapter, "reconcile_delivery", None)
        if not callable(reconcile) or row.get("delivery_identity") != "content_marker_v1":
            logger.error("kanban notifier: operator decision required for %s ambiguous delivery", self.task_id)
            await self.ledger("park_notify_delivery", ev, claim_token=claim,
                              error="ambiguous delivery has no durable reconciliation identity")
            return False
        metadata = dict(self.sub.get("delivery_metadata") or {})
        metadata.update(thread_id=self.sub.get("thread_id") or None,
                        delivery_identity="content_marker_v1", delivery_created_at=row["created_at"])
        try:
            result = await reconcile(self.sub["chat_id"], row["delivery_token"], metadata=metadata)
        except Exception as exc:
            await self.ledger("release_notify_reconciliation", ev, claim_token=claim, error=str(exc))
            return False
        if getattr(result, "success", False) is True:
            try:
                await self.ledger("acknowledge_notify_delivery", ev, claim_token=claim,
                                  message_id=getattr(result, "message_id", None))
            except Exception as exc:
                await self.ledger("release_notify_reconciliation", ev, claim_token=claim, error=str(exc))
            return False
        if not (getattr(result, "raw_response", None) or {}).get("delivery_unknown"):
            await self.ledger("retry_notify_delivery", ev, claim_token=claim,
                              error="confirmed absent in durable transport history")
            return False
        logger.error("kanban notifier: operator decision required for %s ambiguous delivery", self.task_id)
        await self.ledger("park_notify_delivery", ev, claim_token=claim,
                          error=getattr(result, "error", None) or "delivery acceptance cannot be established")
        return False

    async def _send_event(self, ev: Any, msg: str) -> None:
        """Send one text ping; raises on adapter exception or SendResult(success=False)."""
        sub, adapter = self.sub, self.adapter
        delivery_metadata = sub.get("delivery_metadata")
        metadata: dict[str, Any] = dict(delivery_metadata) if isinstance(delivery_metadata, dict) else {}
        if sub.get("thread_id") and not metadata.get("thread_id"):
            metadata["thread_id"] = sub["thread_id"]
        claim, row = self.active_claim, self.active_row
        metadata.update(delivery_token=row["delivery_token"], delivery_identity="content_marker_v1")
        msg += f"\n[kanban-delivery:{row['delivery_token']}]"
        try:
            _send_res = await adapter.send(sub["chat_id"], msg, metadata=metadata)
        except Exception as exc:
            await self.ledger("mark_notify_delivery_ambiguous", ev, claim_token=claim, error=str(exc))
            raise
        # SendResult(success=False) without an exception is a FAILED delivery
        # (else the event is lost); None / non-SendResult keeps the
        # "no exception == delivered" contract.
        if getattr(_send_res, "success", True) is False:
            raw = getattr(_send_res, "raw_response", None) or {}
            operation = "mark_notify_delivery_ambiguous" if raw.get("delivery_ambiguous") else "retry_notify_delivery"
            await self.ledger(operation, ev, claim_token=claim, error=str(getattr(_send_res, "error", "send failed")))
            raise RuntimeError("adapter send() reported failure")
        try:
            await self.ledger("acknowledge_notify_delivery", ev, claim_token=claim,
                              message_id=getattr(_send_res, "message_id", None))
        except Exception as exc:
            await self.ledger("mark_notify_delivery_ambiguous", ev, claim_token=claim, error=str(exc))
            raise
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
        """Each event is separately fenced; earlier successes never replay."""
        self.deferred = []
        self.wake_events = []
        successors = {(ev.task_id, ev.run_id) for ev in self.d["events"]
                      if ev.kind == "delivery_phase_completed" and ev.run_id is not None}
        for row in self.d.get("delivery_rows", []):
            ev = row["event"]
            if row["state"] != "pending":
                if row["state"] == "parked" or not await self.reconcile_row(row):
                    continue
            claim = await self.ledger("begin_notify_delivery", ev)
            if not claim:
                continue
            self.active_claim, self.active_row = claim, row
            precursor = ev.kind == "review_handoff_required" and (ev.task_id, ev.run_id) in successors
            precursor = precursor or (ev.kind == "blocked" and (ev.payload or {}).get("operator_held") is True)
            msg = None if precursor else self.format_event(ev)
            if msg is None:
                try:
                    await self.ledger("acknowledge_notify_delivery", ev, claim_token=claim)
                except Exception as exc:
                    await self.ledger("retry_notify_delivery", ev, claim_token=claim, error=str(exc))
                    return False
                continue
            if not self.send_passive or (not self.is_push_adapter and self.wake_agent):
                self.deferred.append((ev, claim))
                self.wake_events.append(ev)
                continue
            try:
                await self._send_event(ev, msg)
                self.wake_events.append(ev)
                self.clear_failures()
            except Exception as exc:
                await self.delivery_failed(
                    "kanban notifier: send failed for %s on %s (attempt %d/%d): %s",
                    (self.task_id, self.platform_str), "", exc, False)
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
                for ev, claim in self.deferred:
                    await self.ledger("acknowledge_notify_delivery", ev, claim_token=claim)
            except Exception as _wk_err:
                for ev, claim in self.deferred:
                    await self.ledger("mark_notify_delivery_ambiguous", ev, claim_token=claim, error=str(_wk_err))
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
