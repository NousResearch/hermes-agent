"""Subscription-scoped canonical snapshots and durable delivery authority.

Only the existing notifier calls collect_surface/offer_surface. The extension owns
rendering and scheduling; this service never polls or changes canonical tasks.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import dataclass
import hashlib
import logging
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

from gateway.live_todo import DeliveryOutcome, DeliveryStatus, TodoRegistration
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_surface as receipts

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _ScopeDeclined:
    manager: object
    registration: object


@dataclass(frozen=True)
class TaskCardSnapshot:
    profile: str
    board: str
    task_id: str
    incarnation: int
    revision: int
    status: str
    title: str
    assignee: str
    updated_at: int
    presentation: dict | None = None
    steps: tuple = ()
    published_at: int | None = None
    publication_stale: bool = True


def register_task_cards(ctx, factory, *, scope):
    if not callable(factory):
        raise ValueError("task card factory must be callable")
    from gateway.surface_scope import parse_surface_scope
    parsed_scope = parse_surface_scope(scope, require_tasks=False)
    manager = ctx._manager
    if not parsed_scope.task_resources:
        work = getattr(manager, "_work_presentation_registration", None)
        if (work is None or not work.active or work.plugin_id != ctx.plugin_id
                or work.scope != parsed_scope):
            raise ValueError(
                "task cards require at least one exact task resource unless matching work "
                "presentation is active"
            )
    current = getattr(manager, "_task_card_registration", None)
    if current is not None and current.active:
        if current.plugin_id == ctx.plugin_id:
            if current.scope != parsed_scope:
                raise ValueError("task card scope is already bound for this profile")
            return current
        raise ValueError("a task card consumer already owns this profile")
    reg = CardRegistration(factory, manager.home_path, parsed_scope)
    reg.plugin_id = ctx.plugin_id
    manager._task_card_registration = reg
    ctx.on_unload(reg.close)
    return reg


class CardRegistration(TodoRegistration):
    """Reuse the accepted synchronous admission fence and bounded drain order."""
    def __init__(self, factory, home, scope):
        super().__init__(factory, home, scope)
        self.lanes = {}


def subscription_registration(runner, sub, board, *, distinguish_scope_decline=False):
    # Legacy unowned subscriptions remain on the ordinary notifier path.
    if (sub.get("platform") != "telegram" or not sub.get("notifier_profile")
            or sub.get("delivery_mode") == "wake"):
        return None
    from gateway.config import Platform
    from gateway.session import SessionSource
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.plugins import get_plugin_manager
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=sub["chat_id"],
                           thread_id=sub.get("thread_id") or None, profile=sub["notifier_profile"])
    home = runner._resolve_profile_home_for_source(source)
    if home is None:
        return None
    token = set_hermes_home_override(home)
    try:
        manager = get_plugin_manager()
        reg = getattr(manager, "_task_card_registration", None)
        if reg is None or not reg.active:
            return None
        work = getattr(manager, "_work_presentation_registration", None)
        if reg.scope.allows_card(
            sub["notifier_profile"], sub["platform"], sub["chat_id"],
            sub.get("thread_id") or None, board, sub["task_id"],
        ) or (work is not None and work.active and work.plugin_id == reg.plugin_id
              and reg.scope.allows_route(sub["notifier_profile"], sub["platform"],
                                         sub["chat_id"], sub.get("thread_id") or None)):
            return reg
        return _ScopeDeclined(manager, reg) if distinguish_scope_decline else None
    finally:
        reset_hermes_home_override(token)


def _quiet(home):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.config_effective import load_user_config_effective
    from gateway.display_config import resolve_tool_progress
    token = set_hermes_home_override(home)
    try:
        mode, explicit = resolve_tool_progress(load_user_config_effective() or {}, "telegram")
        return explicit and mode == "off"
    finally:
        reset_hermes_home_override(token)


def collect_surface(conn, board, sub, registration):
    """Read one committed snapshot and persist its desire in the same transaction."""
    with kb.write_txn(conn, allow_nested=bool(conn.in_transaction)):
        source = receipts.get_task_source(conn, sub["task_id"])
        task = kb.get_task(conn, sub["task_id"])
        event = conn.execute("SELECT created_at FROM task_events WHERE id=?", (source.current_revision,)).fetchone()
        from hermes_cli.kanban_publication import project_task_publication
        publication = project_task_publication(conn, task.id, source.current_revision)
        legacy = registration.scope.allows_card(
            sub["notifier_profile"], sub["platform"], sub["chat_id"],
            sub.get("thread_id") or None, board, sub["task_id"])
        audience = publication["audience"] if publication else None
        dynamic = bool(audience and registration.scope.allows_route(
            audience.profile, audience.platform, audience.chat_id, audience.thread_id)
            and audience.profile == sub["notifier_profile"]
            and audience.chat_id == str(sub["chat_id"])
            and audience.thread_id == (str(sub.get("thread_id")) if sub.get("thread_id") else None))
        if not legacy and not dynamic:
            return None
        receipt = receipts.ensure_delivery_receipt(
            conn, task_id=task.id, task_incarnation=source.task_incarnation,
            desired_revision=source.current_revision, platform=sub["platform"],
            chat_id=sub["chat_id"], thread_id=sub.get("thread_id"),
            notifier_profile=sub["notifier_profile"], renderer_version="task-card-1")
        # Binding token is source authority, not presentation state. Old attempts
        # retain their identity and may settle, but cannot acquire new authority.
        conn.execute("UPDATE kanban_delivery_receipts SET binding_token=? WHERE id=?",
                     (sub["binding_token"], receipt.id))
        snapshot = TaskCardSnapshot(sub["notifier_profile"], board, task.id,
                                    source.task_incarnation, source.current_revision,
                                    task.status, task.title, task.assignee or "", int(event[0]),
                                    publication["presentation"] if publication else None,
                                    tuple(publication["steps"]) if publication else (),
                                    publication["published_at"] if publication else None,
                                    publication["publication_stale"] if publication else True)
        path = conn.execute("PRAGMA database_list").fetchone()[2]
    return dict(snapshot=snapshot, receipt_id=receipt.id, db_path=str(Path(path).resolve()),
                binding_token=sub["binding_token"],
                publication_audience=publication["audience"] if publication else None)


async def offer_surface(runner, delivery, adapter):
    """Return ownership of passive notification, not a claim of remote delivery.

    Once a tick chose the card handoff it never falls back to ordinary sends.
    Disabled next ticks use the untouched normal cursor path. Wake stays canonical.
    """
    data = delivery.get("surface")
    if data is None:
        return False
    sub = delivery["sub"]
    snapshot = data["snapshot"]
    reg = subscription_registration(
        runner, sub, snapshot.board, distinguish_scope_decline=True,
    )
    if isinstance(reg, _ScopeDeclined):
        # Collection belonged to an older registration. A live replacement that
        # explicitly excludes this destination returns passive delivery to the
        # notifier; unload/unavailable states below retain the no-fallback fence.
        declined = reg
        reg = declined.registration
        with reg.lock:
            if (not reg.active
                    or getattr(declined.manager, "_task_card_registration", None) is not reg
                    or _quiet(reg.profile_home)):
                return True
            return False
    if reg is None:
        return True
    if _quiet(reg.profile_home):
        return True
    if getattr(adapter, "live_todo_transport", None) != 1:
        logger.warning("Task cards require fenced single-message transport capability 1; no card sent")
        return True
    if adapter._bot is None or getattr(adapter, "_send_path_degraded", False):
        return True
    key = (data["db_path"], data["receipt_id"])
    with reg.lock:
        if not reg.active:
            return True
        previous = reg.lanes.get(key)
        if previous is not None:
            if previous.current_binding(adapter, data):
                previous.consumer.publish(data["snapshot"])
                return True
            previous.close()
            if previous.task is not None and not previous.task.done():
                return True
        if len(reg.sources) >= 1024:
            logger.warning("Task card active surface capacity reached; deferred to notifier tick")
            return True
        if getattr(adapter, "_live_todo_client", None) is not adapter._bot:
            adapter._live_todo_epoch = uuid4().hex
            adapter._live_todo_client = adapter._bot
        try:
            source = CardSource(reg, runner, adapter, sub, data)
        except Exception:
            logger.exception("Optional task card factory failed; durable intent preserved")
            return True
        reg.lanes[key] = source
        reg.sources.add(source)
        adapter._live_todo_sources.add(source)
        source.consumer.publish(data["snapshot"])
        source.task = asyncio.create_task(source.run())
    return True


class CardHandle:
    """Public extension handle: no database, route override, SDK or settlement API."""
    __slots__ = ("__source",)

    def __init__(self, source):
        self.__source = source

    def admitted(self):
        return self.__source.admitted()

    def controls(self, snapshot):
        service = getattr(self.__source.registration, "decisions", None)
        return service.controls(self.__source, snapshot) if service else ()

    async def deliver(self, snapshot, text, *, controls=(), links=()):
        return await self.__source.deliver(snapshot, text, controls=controls, links=links)


class CardSource:
    durable_card = True

    def __init__(self, reg, runner, adapter, sub, data):
        self.registration, self.runner, self.adapter = reg, runner, adapter
        self.sub, self.data = dict(sub), dict(data)
        self.client, self.epoch = adapter._bot, adapter._live_todo_epoch
        self.binding = SimpleNamespace(chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or None)
        self.lock = threading.RLock()
        self.loop = asyncio.get_running_loop()
        self.active, self.inflight, self.unknown = True, False, False
        self.task = self.lease = self.message_id = self.last_outcome = None
        self.reply_markup = None
        service = getattr(reg, "decisions", None)
        if service is not None:
            service.observe(self)
        try:
            self.consumer = reg.factory(CardHandle(self))
        except BaseException:
            self.active = False
            raise

    def current_binding(self, adapter, data):
        return (self.active and self.adapter is adapter and self.client is adapter._bot
                and self.epoch == adapter._live_todo_epoch
                and self.data["binding_token"] == data["binding_token"])

    def _subscription_current(self, conn):
        row = conn.execute("SELECT binding_token, notifier_profile FROM kanban_notify_subs "
                           "WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?",
                           (self.sub["task_id"], self.sub["platform"], self.sub["chat_id"], self.sub.get("thread_id") or "")).fetchone()
        return (row is not None and row[0] == self.data["binding_token"]
                and row[1] == self.sub["notifier_profile"])

    def admitted(self):
        if not self.active or not self.registration.active or self.unknown:
            return False
        with self.registration.lock, self.lock:
            if (not self.active or not self.registration.active or self.adapter._bot is not self.client
                    or self.adapter._live_todo_epoch != self.epoch or _quiet(self.registration.profile_home)):
                return False
            snapshot = self.data["snapshot"]
            audience = self.data.get("publication_audience")
            dynamic = bool(audience and audience.bot_id == getattr(self.client, "id", None)
                           and self.registration.scope.allows_route(
                               audience.profile, audience.platform, audience.chat_id, audience.thread_id)
                           and audience.profile == self.sub["notifier_profile"]
                           and audience.chat_id == str(self.sub["chat_id"])
                           and audience.thread_id == (str(self.sub.get("thread_id"))
                                                      if self.sub.get("thread_id") else None))
            if not dynamic and not self.registration.scope.allows_card(
                self.sub["notifier_profile"], self.sub["platform"], self.sub["chat_id"],
                self.sub.get("thread_id") or None, snapshot.board, snapshot.task_id,
            ):
                return False
            from gateway.kanban_watchers_notifier import _adapter_for_subscription
            from gateway.config import Platform
            if _adapter_for_subscription(self.runner, Platform(self.sub["platform"]), self.sub,
                                         self.sub["notifier_profile"]) is not self.adapter:
                return False
            with closing(kbc.connect(Path(self.data["db_path"]))) as conn:
                if not self._subscription_current(conn):
                    return False
                canonical = receipts.get_task_source(conn, self.sub["task_id"])
                if canonical.task_incarnation != self.data["snapshot"].incarnation:
                    return False
                if self.lease is not None:
                    r = receipts.get_delivery_receipt(conn, self.data["receipt_id"])
                    return (r.state == "pending" and r.attempt_id == self.lease.attempt_id
                            and r.owner_id == self.registration.token
                            and r.lease_expires_at > time.time())
            return True

    async def deliver(self, snapshot, text, *, controls=(), links=()):
        if not self.admitted():
            return DeliveryOutcome(DeliveryStatus.REJECTED, reason="stopped card source")
        # Validate snapshot revision before claiming; do not send old text as a newer attempt.
        with closing(kbc.connect(Path(self.data["db_path"]))) as conn:
            r = receipts.get_delivery_receipt(conn, self.data["receipt_id"])
            if r is None or snapshot.revision != r.desired_revision:
                return DeliveryOutcome(DeliveryStatus.SKIPPED, reason="coalesced newer demand")
            if type(links) not in {tuple, list} or len(links) > 4:
                return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid links")
            clean_links = []
            for link in links:
                from urllib.parse import urlsplit
                try:
                    parsed = urlsplit(link.get("url")) if isinstance(link, dict) else None
                except (TypeError, ValueError):
                    parsed = None
                if (type(link) is not dict or set(link) != {"label", "url"}
                        or not isinstance(link["label"], str) or not 1 <= len(link["label"]) <= 80
                        or not isinstance(link["url"], str) or len(link["url"]) > 2048
                        or parsed is None or parsed.scheme != "https" or not parsed.hostname
                        or parsed.username is not None or parsed.password is not None
                        or any(c.isspace() for c in link["url"])):
                    return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid links")
                clean_links.append(dict(text=link["label"], url=link["url"]))
            service = getattr(self.registration, "decisions", None)
            if controls:
                if service is None or tuple((c[0], c[1]) for c in controls) != service.controls(self, snapshot):
                    return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid controls")
            # Preserve the read-only wire shape. Explicit empty markup is needed
            # only on a known message that may have received decision controls.
            had_controls = bool(r.destination_message_id and conn.execute(
                "SELECT 1 FROM kanban_action_records WHERE origin_message_id=? "
                "AND task_id=? AND json_extract(action_payload, '$.receipt_id')=? LIMIT 1",
                (r.destination_message_id, snapshot.task_id, r.id)).fetchone())
            clear_controls = bool(r.renderer_hash and not controls
                                  and (service is None or not service.controls(self, snapshot)))
            self.reply_markup = [] if had_controls else None
            rows = []
            if clean_links:
                rows.append(clean_links)
            if controls and service is not None:
                rows.append([dict(text=c[2], callback_data=service.callback_prefix + c[1]) for c in controls])
            if rows:
                self.reply_markup = rows
            from gateway.live_todo import rendered_payload_hash
            renderer_key = rendered_payload_hash(text, rows)
            if (r.destination_message_id and r.delivered_revision is not None
                    and r.renderer_hash == renderer_key and r.state == "sent"):
                try:
                    receipts.confirm_equivalent_delivery(
                        conn, r.id, desired_revision=snapshot.revision,
                        renderer_hash=renderer_key,
                        message_id=r.destination_message_id,
                    )
                except receipts.DeliveryReceiptError:
                    pass
                else:
                    self.message_id = r.destination_message_id
                    return DeliveryOutcome(
                        DeliveryStatus.SKIPPED, reason="confirmed card payload unchanged")
            try:
                if r.state == "deleted" and r.replacement_budget > 0:
                    receipts.authorize_delivery_replacement(conn, r.id, desired_revision=r.desired_revision)
                self.lease = receipts.claim_delivery_receipt(
                    conn, r.id, owner_id=self.registration.token, lease_seconds=30,
                    expected_revision=snapshot.revision,
                    refresh=bool(controls) or bool(clean_links) or clear_controls)
            except receipts.DeliveryReceiptError as exc:
                return DeliveryOutcome(DeliveryStatus.SKIPPED, reason=type(exc).__name__)
            self.message_id = self.lease.receipt.destination_message_id
        lease = self.lease
        # A reused source may still hold the preceding create's success. Clear
        # evidence before the adapter's first await (including chat lock/slot).
        self.last_outcome = None
        try:
            outcome = await self.adapter.deliver_live_todo(self, text)
        except asyncio.CancelledError:
            outcome = self.last_outcome or DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="cancelled")
            self._settle(lease, outcome)
            raise
        except Exception:
            self._settle(lease, DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="transport exception"))
            raise
        else:
            control_key = (hashlib.sha256(controls[0][1].encode()).hexdigest()
                           if len(controls) == 1 else None)
            self._settle(lease, outcome, renderer_key=renderer_key, control_key=control_key)
            return outcome
        finally:
            self.lease = None

    def _settle(self, lease, outcome, *, renderer_key=None, control_key=None):
        # Host-owned evidence path intentionally survives source/registration stop.
        status = outcome.status
        state = {DeliveryStatus.DELIVERED: "sent", DeliveryStatus.UNKNOWN: "unknown"}.get(status, "failed")
        if outcome.reason == "known message deleted":
            state = "deleted"
        delay = outcome.retry_after
        safe = (status in {DeliveryStatus.SKIPPED, DeliveryStatus.REJECTED}
                and ((delay is not None and math.isfinite(delay) and 0 <= delay <= 30)
                     or outcome.reason in {"operation failed before request write", "stale binding",
                                           "stale or disconnected binding"}))
        with closing(kbc.connect(Path(self.data["db_path"]))) as conn, kb.write_txn(conn):
            current = receipts.get_delivery_receipt(conn, lease.receipt.id)
            if current is None:
                raise receipts.DeliveryReceiptNotFound("receipt disappeared before settlement")
            expired = (current.owner_id != lease.owner_id or current.lease_expires_at is None
                       or current.lease_expires_at <= time.time())
            disposition = "safe_retry" if safe else "exhausted"
            receipts.reconcile_delivery_outcome(
                conn, lease.receipt.id, owner_id=lease.owner_id, owner_epoch=lease.owner_epoch,
                attempt_id=lease.attempt_id, desired_revision=lease.desired_revision, state=state,
                message_id=outcome.message_id if state == "sent" else lease.receipt.destination_message_id,
                destination_profile=self.sub["notifier_profile"] if state == "sent" else None,
                delivered_revision=lease.desired_revision if state == "sent" else None,
                retry_disposition=disposition if state == "failed" else None, error=outcome.reason)
            conn.execute("UPDATE kanban_delivery_receipts SET retry_at=? WHERE id=?",
                         (math.ceil(time.time() + max(0.5, delay or 0)) if safe else 0, lease.receipt.id))
            if state == "sent":
                # Payload equivalence and action authority are separate facts.
                # Both become trusted only in the exact confirmed settlement.
                conn.execute(
                    "UPDATE kanban_delivery_receipts SET renderer_hash=?, control_hash=? WHERE id=?",
                    (renderer_key, control_key, lease.receipt.id),
                )
            else:
                # A token that was not positively delivered must never become
                # authority on a later retry or equivalent-payload shortcut.
                conn.execute(
                    "DELETE FROM kanban_action_records WHERE task_id=? AND expected_revision=? "
                    "AND state='pending' AND json_extract(action_payload, '$.receipt_id')=?",
                    (self.sub["task_id"], lease.desired_revision, lease.receipt.id),
                )
        self.last_outcome = outcome
        if expired:
            # Late evidence settles the receipt, never renews this writer's lease.
            self.close()

    def close(self):
        with self.lock:
            self.active = False
            lease = self.lease
        if lease is not None:
            with closing(kbc.connect(Path(self.data["db_path"]))) as conn:
                receipts.quarantine_delivery(conn, lease)
        try:
            self.consumer.close()
        except Exception:
            logger.exception("Optional task card close failed after fencing")

    async def run(self):
        try:
            await self.consumer.run()
        except Exception:
            logger.exception("Optional task card consumer failed; intent preserved")
        finally:
            self.close()
            with self.registration.lock:
                self.registration.sources.discard(self)
                key = (self.data["db_path"], self.data["receipt_id"])
                if self.registration.lanes.get(key) is self:
                    self.registration.lanes.pop(key, None)
                self.adapter._live_todo_sources.discard(self)

    async def finish(self):
        self.close()
        task = self.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            done, _ = await asyncio.wait({task}, timeout=2)
            if not done:
                task.cancel()
                await asyncio.wait({task}, timeout=0.1)
