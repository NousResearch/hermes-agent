"""Host-owned, default-off task decisions. No poller or additional task store."""
from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import time

from hermes_cli import kanban_db as kb, kanban_db_actions as actions
from hermes_cli import kanban_db_connect as kbc, kanban_db_surface as receipts


class TaskDecisions:
    """Plugin-facing service accepts a native update, never caller-claimed authority."""
    def __init__(self, registration, *, callback_prefix):
        if not isinstance(callback_prefix, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,15}:[a-z:]{0,8}", callback_prefix):
            raise ValueError("task callback prefix must be a bounded plugin namespace")
        self.callback_prefix = callback_prefix
        self.registration = registration
        self.bindings = {}
        self.mutation_lock = asyncio.Lock()
        self.policy_digest = None
        self.policy_epoch = None

    def _grants(self):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli import managed_scope
        scope = set_hermes_home_override(self.registration.profile_home)
        try:
            try:
                # Decision authority is fail-closed even though ordinary config readers preserve
                # last-known-good behavior for torn user edits. A broken current policy must revoke
                # both minting and execution until a fresh valid policy is observed.
                config = load_user_config_effective(fail_closed=True, side_effect_free=True) or {}
            except Exception:
                config = {}
            kanban = config.get("kanban", {})
            grants = kanban.get("decision_grants", []) if isinstance(kanban, dict) else []
            grants = grants if isinstance(grants, list) else []
            plugins = config.get("plugins", {}) if isinstance(config.get("plugins", {}), dict) else {}
            settings = plugins.get("entries", {}).get(self.registration.plugin_id, {}).get("settings", {})
            automatic = bool(
                self.registration.plugin_id in plugins.get("enabled", [])
                and self.registration.plugin_id not in plugins.get("disabled", [])
                and isinstance(settings, dict) and settings.get("enabled") is True
                and settings.get("decisions") is True and settings.get("work_briefs") is True)
            policy_paths = [Path(self.registration.profile_home) / "config.yaml"]
            managed_dir = managed_scope.get_managed_dir()
            if managed_dir is not None:
                policy_paths.append(managed_dir / "config.yaml")
            file_identity = []
            for policy_path in policy_paths:
                try:
                    stat = policy_path.stat()
                    file_identity.append((str(policy_path), stat.st_dev, stat.st_ino, stat.st_size,
                                          stat.st_mtime_ns, stat.st_ctime_ns))
                except OSError:
                    file_identity.append((str(policy_path), None))
            digest = hashlib.sha256(json.dumps(
                [grants, automatic, file_identity], sort_keys=True, default=str).encode()).hexdigest()
            if digest != self.policy_digest:
                self.policy_digest = digest
                # Stable while the same current file/policy is in force, but a malformed edit
                # and its later repair have new file identity and cannot revive an old token.
                self.policy_epoch = digest
            return grants, automatic
        finally:
            reset_hermes_home_override(scope)

    def _grant(self, source):
        snapshot = source.data["snapshot"]
        required = dict(profile=snapshot.profile, board=snapshot.board, task_id=snapshot.task_id,
                        actions=["unblock_needs_input"])
        configured, automatic = self._grants()
        grants = [g for g in configured if isinstance(g, dict)
                  and all(g.get(k) == v for k, v in required.items())
                  and isinstance(g.get("id"), str) and g["id"]
                  and type(g.get("actor")) is int and g["actor"] > 0]
        if len(grants) == 1:
            return grants[0]
        if grants or not automatic or snapshot.publication_stale:
            return None
        audience = source.data.get("publication_audience")
        if (audience is None or audience.profile != snapshot.profile
                or audience.platform != "telegram" or audience.bot_id != getattr(source.client, "id", None)
                or audience.chat_id != str(source.sub["chat_id"])
                or audience.thread_id != (str(source.sub.get("thread_id")) if source.sub.get("thread_id") else None)
                or type(audience.actor_id) is not int or audience.actor_id <= 0):
            return None
        # Same internal shape as a configured grant so execution keeps one
        # exact-principal, exact-revision authorization path.
        return dict(id="published-task", actor=audience.actor_id,
                    profile=snapshot.profile, board=snapshot.board, task_id=snapshot.task_id,
                    actions=["unblock_needs_input"])

    def observe(self, source):
        key = (source.data["db_path"], source.data["receipt_id"])
        if key in self.bindings or len(self.bindings) < 4096:
            self.bindings[key] = source

    @staticmethod
    def _scope_allows(source):
        snapshot = source.data["snapshot"]
        audience = source.data.get("publication_audience")
        dynamic = bool(audience and audience.bot_id == getattr(source.client, "id", None)
                       and source.registration.scope.allows_route(
                           audience.profile, audience.platform, audience.chat_id, audience.thread_id)
                       and audience.profile == source.sub["notifier_profile"]
                       and audience.chat_id == str(source.sub["chat_id"])
                       and audience.thread_id == (str(source.sub.get("thread_id"))
                                                  if source.sub.get("thread_id") else None))
        return dynamic or source.registration.scope.allows_card(
            source.sub["notifier_profile"], source.sub["platform"], source.sub["chat_id"],
            source.sub.get("thread_id") or None, snapshot.board, snapshot.task_id)

    def _binding_current(self, source, conn):
        from gateway.kanban_surfaces import _quiet
        from gateway.kanban_watchers_notifier import _adapter_for_subscription
        from gateway.config import Platform
        snapshot = source.data["snapshot"]
        return (self.registration.active and not _quiet(self.registration.profile_home)
                and source.registration is self.registration
                and self._scope_allows(source)
                and source.adapter._bot is source.client
                and source.adapter._live_todo_epoch == source.epoch
                and source._subscription_current(conn)
                and _adapter_for_subscription(source.runner, Platform.TELEGRAM, source.sub,
                                              source.sub["notifier_profile"]) is source.adapter)

    def controls(self, source, snapshot):
        """Mint only for a confirmed current card; tokens remain inert until its edit settles."""
        grant = self._grant(source)
        if (grant is None or not source.admitted()
                or self.bindings.get((source.data["db_path"], source.data["receipt_id"])) is not source):
            return ()
        with closing(kbc.connect(Path(source.data["db_path"]))) as conn, kb.write_txn(conn):
            # Acquiring the writer lock can block while policy changes.
            grant = self._grant(source)
            if grant is None:
                return ()
            r = receipts.get_delivery_receipt(conn, source.data["receipt_id"])
            if (r is None or not self._binding_current(source, conn) or r.state != "sent"
                    or not r.destination_message_id or r.delivered_revision != snapshot.revision
                    or r.desired_revision != snapshot.revision
                    or not actions.blocker_choice_applicable(conn, snapshot.task_id)):
                return ()
            payload = dict(grant=grant, generation=self.registration.token, epoch=source.epoch,
                           policy_epoch=self.policy_epoch,
                           binding=source.data["binding_token"], receipt_id=r.id)
            key = hashlib.sha256(json.dumps([payload, snapshot.revision, snapshot.incarnation,
                                            r.destination_message_id], sort_keys=True).encode()).hexdigest()
            row = conn.execute("SELECT token, created_at, expires_at, state FROM kanban_action_records WHERE idempotency_key=?",
                               (key,)).fetchone()
            if row is None:
                record = actions.issue_action(
                    conn, task_id=snapshot.task_id, task_incarnation=snapshot.incarnation,
                    expected_revision=snapshot.revision, expected_task_status="blocked",
                    board_identity=snapshot.board, profile=snapshot.profile,
                    telegram_principal=grant["actor"], origin_chat_id=source.sub["chat_id"],
                    origin_thread_id=source.sub.get("thread_id"),
                    origin_message_id=r.destination_message_id, action_kind="unblock_needs_input",
                    action_payload=payload, expires_at=int(time.time()) + 600,
                    idempotency_key=key, conflict_key=f"{snapshot.task_id}:{snapshot.incarnation}:{snapshot.revision}")
                token = record.token
            else:
                if not row[1] <= time.time() < row[2] or row[3] != "pending":
                    return ()
                token = row[0]
            return (("unblock_needs_input", token),)

    def _principal(self, update, source):
        from telegram import Update, Message
        from gateway.session import SessionSource
        from gateway.config import Platform
        from gateway.session_identity import canonical_identity
        if not isinstance(update, Update) or update.callback_query is None:
            raise actions.ActionAuthorizationError("native callback required")
        q = update.callback_query
        m = q.message
        if (not isinstance(m, Message) or m.date.timestamp() <= 0 or q.inline_message_id
                or m.forward_origin or not q.from_user or q.from_user.is_bot
                or not m.from_user or m.from_user.id != source.client.id):
            raise actions.ActionAuthorizationError("verified original message required")
        s = SessionSource(platform=Platform.TELEGRAM, chat_id=str(m.chat.id),
                          chat_type=source.adapter._normalize_chat_type(m.chat.type, is_forum=m.message_thread_id is not None),
                          thread_id=str(m.message_thread_id) if m.message_thread_id is not None else None,
                          user_id=str(q.from_user.id))
        identity = canonical_identity(s, runner=source.runner, adapter=source.adapter)
        if (identity is None or identity.adapter() is not source.adapter
                or identity.runtime_profile != source.sub["notifier_profile"]
                or not self._scope_allows(source)
                or Path(identity.runtime_home).resolve() != Path(self.registration.profile_home).resolve()
                or source.adapter._is_sender_authorized(s.user_id, s.chat_type, s.chat_id,
                                                       thread_id=s.thread_id) is not True):
            raise actions.ActionAuthorizationError("principal is not admitted")
        return dict(profile=identity.runtime_profile, board_identity=source.data["snapshot"].board,
                    telegram_principal=q.from_user.id, origin_chat_id=s.chat_id,
                    origin_thread_id=s.thread_id, origin_message_id=str(m.message_id))

    async def execute(self, update, adapter):
        """Called by a scoped plugin handler on the existing native dispatcher."""
        data = getattr(getattr(update, "callback_query", None), "data", None)
        if not isinstance(data, str) or not re.fullmatch(re.escape(self.callback_prefix) + r"[A-Za-z0-9_-]{24}", data):
            return {"ok": False}
        token = data[len(self.callback_prefix):]
        # The wait carries no cached authority; every check happens again after admission.
        async with self.mutation_lock:
            with self.registration.lock:
                if not self.registration.active:
                    return {"ok": False}
                for source in tuple(self.bindings.values()):
                    if source.adapter is not adapter:
                        continue
                    with closing(kbc.connect(Path(source.data["db_path"]))) as conn:
                        candidate = conn.execute(
                            "SELECT action_payload FROM kanban_action_records WHERE token=? AND task_id=?",
                            (token, source.sub["task_id"]),
                        ).fetchone()
                        if (candidate is None or json.loads(candidate[0]).get("receipt_id")
                                != source.data["receipt_id"]):
                            continue
                        try:
                            route = self._principal(update, source)
                            def authorize(record, transaction):
                                # SQLite writer lock has been acquired; no await until COMMIT.
                                if (self._principal(update, source) != route
                                        or not self._binding_current(source, transaction)
                                        or record.action_payload.get("grant") != self._grant(source)
                                        or record.action_payload.get("receipt_id") != source.data["receipt_id"]
                                        or (record.state != "completed" and (
                                            record.action_payload.get("policy_epoch") != self.policy_epoch
                                            or record.action_payload.get("generation") != self.registration.token
                                            or record.action_payload.get("epoch") != source.epoch))
                                        or record.action_payload.get("binding") != source.data["binding_token"]):
                                    raise actions.ActionAuthorizationError("current grant/binding required")
                                r = receipts.get_delivery_receipt(transaction, source.data["receipt_id"])
                                if (r is None or r.destination_message_id != record.origin_message_id
                                        or r.state != "sent" or r.task_incarnation != record.task_incarnation
                                        or (record.state != "completed" and r.delivered_revision != record.expected_revision)):
                                    raise actions.ActionAuthorizationError("confirmed card required")
                                # The keyboard and text must have a verified exact settlement.
                                if record.state != "completed" and r.control_hash != hashlib.sha256(record.token.encode()).hexdigest():
                                    raise actions.ActionAuthorizationError("control delivery not confirmed")
                            return actions.execute_blocker_choice(conn, token, authorize=authorize, **route)
                        except (actions.ActionRecordError, receipts.DeliveryReceiptError):
                            return {"ok": False}
        return {"ok": False}
