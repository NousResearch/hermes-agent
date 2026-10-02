"""Publish a Bot's shared files with its turn, then retire the private copy exactly once.

Every output gets one durable obligation row before any byte moves: ``ack``
when the turn's member message carries the files, ``discard`` when it does not
(the turn failed, was stopped, deferred or superseded). A failed step stays
``pending`` with a bounded backoff, or becomes ``blocked`` when retrying cannot
help; both show in the room status. While a terminal still waits on its files
its whole thread is held out of Policy selection, so replies in that thread
stay in order; other discussions go on in FIFO order.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    output_store_exists,
    scope_has_output,
    validate_terminal_artifact_manifest,
)
from gateway.hosted_room_driver import OUTPUT_OBLIGATIONS_TABLE as OBLIGATIONS
from gateway.hosted_room_output_fence import message_event_id, upload_id
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)

MAX_RETRY_DELAY_SECONDS = 60.0
# The source outbox keeps unacknowledged bytes for 30 days; stop retrying before.
RETRY_HORIZON_SECONDS = 29 * 24 * 60 * 60
# A completed row is an idempotency receipt; it goes with its task row, in bounded batches.
COMPLETED_PRUNE_BATCH = 256
_TERMINAL_STATUSES = ("deferred", "settled", "failed", "cancelled")
_RETRYABLE_REASONS = frozenset({"storage_unavailable", "runtime_draining"})


def output_retryable(error) -> bool:
    """Transient storage or transport faults retry; integrity and authority faults block."""
    if isinstance(error, hosted_rooms.EventCursorConflictError):
        return True
    if isinstance(error, sqlite3.Error):
        return getattr(error, "sqlite_errorcode", None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    if isinstance(error, RuntimeStoreError):
        return error.reason in _RETRYABLE_REASONS
    return (getattr(error, "retryable", False) is True
            or isinstance(error, (ConnectionError, OSError, TimeoutError)))


def _reason(error) -> str:
    if isinstance(error, RuntimeStoreError):
        return error.reason
    if isinstance(error, ValueError):
        return "verification_failed"
    return "output_failed"


def ensure_obligations(conn) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {OBLIGATIONS} (
        room_id TEXT NOT NULL, task_id TEXT NOT NULL, execution_generation INTEGER NOT NULL,
        member_id TEXT NOT NULL,
        operation TEXT NOT NULL CHECK (operation IN ('ack', 'discard')),
        state TEXT NOT NULL CHECK (state IN ('pending', 'blocked', 'completed')),
        reason_code TEXT NOT NULL, attempts INTEGER NOT NULL, next_attempt_at REAL NOT NULL,
        identity_json TEXT NOT NULL, scope_json TEXT NOT NULL, manifest_json TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY (room_id, task_id, execution_generation))""")


def obligations_exist(conn) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (OBLIGATIONS,)).fetchone() is not None


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class _TransportedOutputSource:
    """A named Bot's outbox, reached through the owner transport to its profile."""

    def __init__(self, service, identity, scope: RoomArtifactScope, manifest, home: Path):
        from gateway.session_hosted_transport import HostedRoomOwnerRPC

        self.scope, self.manifest = scope, manifest
        self.rpc = HostedRoomOwnerRPC(home=home, source_home=service.authority.profile_id,
                                      room_id=scope.room_id, member_id=scope.member_id,
                                      profile=scope.target_profile)
        self.params = {"task": asdict(identity), "execution_generation": scope.execution_generation,
                       "artifact_scope": scope.as_mapping(), "manifest_digest": manifest["manifest_digest"]}

    def read(self, scope, artifact_id):
        from gateway.session_hosted_output_owner import read_exported_item

        if scope != self.scope:
            raise RoomArtifactError("Group Chat output scope changed")
        return read_exported_item(self.rpc, self.params, self.manifest, artifact_id)

    def acknowledge(self, scope, artifact_ids, *, message_event_id):
        if scope != self.scope:
            raise RoomArtifactError("Group Chat output scope changed")
        result = self.rpc.output_ack(**self.params, artifact_ids=list(artifact_ids),
                                     message_event_id=message_event_id)
        if (type(result) is not dict or set(result) != {"acknowledged", "changed"}
                or result["acknowledged"] is not True or type(result["changed"]) is not int):
            raise RoomArtifactError("Group Chat output acknowledgement was not confirmed")
        return result["changed"]

    def discard_durably(self, scope):
        if scope != self.scope:
            raise RoomArtifactError("Group Chat output scope changed")
        result = self.rpc.output_discard(**self.params)
        if (type(result) is not dict or set(result) != {"discarded", "removed"}
                or result["discarded"] is not True or type(result["removed"]) is not int):
            raise RoomArtifactError("Group Chat output discard was not confirmed")
        return result["removed"]


class CanonicalHostedOutput:
    """Output publication and retirement for the canonical hosted room service."""

    _output_clock = staticmethod(time.time)

    def start(self):
        # The gateway watcher ensures start every second; an already-live owner is unchanged.
        if self.runtime.status().get('running'):
            return super().start()
        self._peer_output_generation = getattr(self, '_peer_output_generation', 0) + 1
        self._peer_output_stopping = False
        return super().start()

    def stop(self, *, timeout=5.0):
        self._peer_output_stopping = True
        self._peer_output_generation = getattr(self, '_peer_output_generation', 0) + 1
        # These are only transfer observations. Durable obligations remain untouched.
        with self._policy_lock:
            getattr(self, '_peer_output_io', {}).clear()
        return super().stop(timeout=timeout)

    # ------------------------------------------------------------------ obligations
    def _output_write(self, operation):
        def write(conn):
            _epoch(conn, self.authority.epoch)
            ensure_obligations(conn)
            return operation(conn)
        return self.authority.db._execute_write(write)

    @staticmethod
    def _load_obligation(conn, room_id, task_id, generation):
        if not obligations_exist(conn):
            return None
        row = conn.execute(
            f"SELECT * FROM {OBLIGATIONS} WHERE room_id=? AND task_id=? AND execution_generation=?",
            (room_id, task_id, generation)).fetchone()
        return dict(row) if row is not None else None

    def _obligation(self, room_id, task_id, generation):
        with self.authority.db._read_ctx() as conn:
            return self._load_obligation(conn, room_id, task_id, generation)

    def _record_intent(self, identity, scope: RoomArtifactScope, manifest, operation):
        """Record the disposition before any byte moves; only an unpublished ACK may become a discard."""
        now = float(self._output_clock())
        identity_json = _canonical(asdict(identity))
        scope_json = _canonical(scope.as_mapping())
        manifest_json = _canonical(manifest) if manifest is not None else None
        key = (scope.room_id, scope.task_id, scope.execution_generation)

        def write(conn):
            row = self._load_obligation(conn, *key)
            if row is None:
                conn.execute(
                    f"""INSERT INTO {OBLIGATIONS} (room_id, task_id, execution_generation, member_id,
                        operation, state, reason_code, attempts, next_attempt_at, identity_json,
                        scope_json, manifest_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'pending', 'pending', 0, 0, ?, ?, ?, ?, ?)""",
                    (*key, scope.member_id, operation, identity_json, scope_json, manifest_json, now, now))
                return self._load_obligation(conn, *key)
            if row["state"] == "completed":
                return row
            if (row["scope_json"] != scope_json or row["identity_json"] != identity_json
                    or (manifest_json is not None and row["manifest_json"] not in {None, manifest_json})):
                conn.execute(f"""UPDATE {OBLIGATIONS} SET state='blocked', reason_code='scope_changed',
                    updated_at=? WHERE room_id=? AND task_id=? AND execution_generation=?""", (now, *key))
            elif row["operation"] != operation:
                if row["operation"] == "ack" and operation == "discard":
                    # Superseded before its member message existed: nothing was published.
                    conn.execute(f"""UPDATE {OBLIGATIONS} SET operation='discard', state='pending',
                        reason_code='superseded', attempts=0, next_attempt_at=0, updated_at=?
                        WHERE room_id=? AND task_id=? AND execution_generation=?""", (now, *key))
                else:
                    conn.execute(f"""UPDATE {OBLIGATIONS} SET state='blocked',
                        reason_code='disposition_changed', updated_at=?
                        WHERE room_id=? AND task_id=? AND execution_generation=?""", (now, *key))
            return self._load_obligation(conn, *key)
        return self._output_write(write)

    def _record_outcome(self, scope: RoomArtifactScope, *, error=None, completed="completed"):
        now = float(self._output_clock())
        key = (scope.room_id, scope.task_id, scope.execution_generation)

        def write(conn):
            row = self._load_obligation(conn, *key)
            if row is None or row["state"] == "completed":
                return row
            attempts = min(2_147_483_647, int(row["attempts"]) + 1)
            if error is None:
                state, reason, next_at = "completed", completed, 0.0
            elif output_retryable(error) and now - float(row["created_at"]) < RETRY_HORIZON_SECONDS:
                state, reason = "pending", "transient"
                next_at = now + min(MAX_RETRY_DELAY_SECONDS, 2.0 ** min(attempts - 1, 16))
            else:
                state, next_at = "blocked", 0.0
                reason = "expired" if output_retryable(error) else _reason(error)
            conn.execute(f"""UPDATE {OBLIGATIONS} SET state=?, reason_code=?, attempts=?,
                next_attempt_at=?, updated_at=? WHERE room_id=? AND task_id=? AND execution_generation=?""",
                         (state, reason, attempts, next_at, now, *key))
            return self._load_obligation(conn, *key)
        saved = self._output_write(write)
        if error is not None:
            message = f"room {scope.room_id}: Group Chat output for {scope.task_id} failed: {error}"
            logger.warning(message)
            self.runtime._record_error(message)
        return saved

    def _due(self, row) -> bool:
        return row is None or (row["state"] == "pending"
                               and float(self._output_clock()) >= float(row["next_attempt_at"]))

    def _waiting(self, row) -> bool:
        """A pending step backs off. A blocked one is still re-planned: a newer request in
        its thread supersedes the reply, and its files are then discarded instead."""
        return row is not None and row["state"] == "pending" and not self._due(row)

    def _run_obligation(self, identity, scope, manifest, operation, action) -> bool:
        """Run one recorded step; True only when the obligation is complete."""
        row = self._record_intent(identity, scope, manifest, operation)
        if row["state"] == "completed":
            return True
        if row["operation"] != operation or not self._due(row):
            return False
        try:
            action()
        except Exception as exc:
            self._record_outcome(scope, error=exc)
            return False
        self._record_outcome(scope)
        return True

    # ------------------------------------------------------------------ sources
    def _task_output(self, task):
        """The validated output a settled receipt reported for this exact attempt."""
        from gateway.session_hosted_output import output_receipt_fields

        fields = output_receipt_fields(task.get("result"))
        if not fields:
            return None
        scope = RoomArtifactScope.from_mapping(fields["artifact_scope"])
        payload = task["payload"]
        if ((scope.room_id, scope.task_id, scope.execution_generation)
                != (task["identity"].room_id, task["identity"].task_id, int(task["execution_generation"]))
                or scope.member_id != payload.get("target_member_id", payload.get("target_profile"))
                or scope.target_profile != payload.get("target_profile")):
            logger.warning("Ignoring Group Chat output reported for another attempt of %s",
                           task["identity"].task_id)
            return None
        return scope, fields["artifacts"]

    def _local_scope(self, room, task):
        """The scope this gateway's own profile would have used for the attempt."""
        payload = task["payload"]
        profile = payload.get("target_profile")
        member_id = payload.get("target_member_id", profile)
        member = next((m for m in room["members"] if m.get("member_id") == member_id), None)
        home = self.profile_homes().get(profile)
        if (member is None or member.get("target", {}).get("kind", "local") != "local"
                or home is None or Path(home).resolve() != Path(self.authority.profile_id).resolve()):
            return None
        gateway, epoch = str(room["authority_gateway_id"]), int(room["authority_epoch"])
        return RoomArtifactScope.from_mapping(dict(
            room_id=task["identity"].room_id, task_id=task["identity"].task_id,
            execution_generation=int(task["execution_generation"]), member_id=member_id,
            target_profile=profile, home_install_id=gateway, target_install_id=gateway,
            authority_gateway_id=gateway, authority_epoch=epoch))

    def _unreported_output(self, room, task):
        """Open bytes of this gateway's own profile that no receipt reported (stopped, failed)."""
        payload = task['payload']
        member = next((member for member in room['members'] if member['member_id'] == payload.get('target_member_id', payload.get('target_profile'))), None)
        if member is not None and member.get('target', {}).get('kind') == 'peer':
            from tui_gateway.hosted_room_peer_output import stored_consent
            target = member['target']
            scope = RoomArtifactScope.from_mapping(dict(room_id=room['room_id'], task_id=task['identity'].task_id,
                execution_generation=task['execution_generation'], member_id=member['member_id'], target_profile=target['profile'],
                home_install_id=room['authority_gateway_id'], target_install_id=target['installation_id'],
                authority_gateway_id=room['authority_gateway_id'], authority_epoch=room['authority_epoch']))
            consent = stored_consent(self.db_path, scope.as_mapping())
            from gateway.hosted_room_driver import is_proven_nonadmission
            if is_proven_nonadmission(task) or (task.get('result') or {}).get('peer_output_empty') == scope.as_mapping():
                return None
            if task['status'] == 'cancelled' and consent is not None and (consent.get('dispatched') is False
                    or consent.get('unreceived_cancel_generation') == task['cancel_generation']):
                return None
            return (scope, None) if consent is not None and consent['contract'] is not None else None
        with self.authority.db._read_ctx() as conn:
            if not output_store_exists(conn):
                return None  # no Bot ever shared a file here: nothing to look up
        scope = self._local_scope(room, task)
        if scope is None:
            return None
        with self.authority.db._read_ctx() as conn:
            return (scope, None) if scope_has_output(conn, scope) else None

    def _peer_output_consent(self, scope):
        from tui_gateway.hosted_room_peer_output import stored_consent
        consent = stored_consent(self.db_path, scope.as_mapping())
        return consent is not None and consent['contract'] is not None

    def _output_source(self, identity, scope: RoomArtifactScope, manifest):
        from tui_gateway.hosted_room_peer_output import PeerOutputSource, stored_consent
        consent = stored_consent(self.db_path, scope.as_mapping())
        member = next((m for m in self._room(scope.room_id)['members'] if m['member_id'] == scope.member_id), {})
        if (member.get('target', {}).get('kind') == 'peer' or
                (consent is not None and consent['contract'] is not None)):
            return PeerOutputSource(self, scope, manifest)
        if (not member or member.get('target', {}).get('kind', 'local') != 'local'
                or scope.target_install_id != scope.home_install_id):
            raise RoomArtifactError('Group Chat output source installation changed')
        home = self.profile_homes().get(scope.target_profile)
        if home is None:
            raise RuntimeStoreError("permission_denied")
        if Path(home).resolve() == Path(self.authority.profile_id).resolve():
            return RoomArtifactOutbox(self.db_path)
        if manifest is None:
            raise RoomArtifactError("Group Chat output manifest is unavailable")
        return _TransportedOutputSource(self, identity, scope, manifest, Path(home))

    # ------------------------------------------------------------------ publication
    def _publish_terminal_tasks(self, room) -> bool:
        """True when the room log or its held threads may have changed, so Policy is read again."""
        changed, held, room_id, local_profiles = False, False, str(room["room_id"]), self.local_profiles()
        cursor = int(room["latest_seq"])
        for task in self._list_tasks(room_id, _TERMINAL_STATUSES):
            status, generation, identity = task["status"], int(task["execution_generation"]), task["identity"]
            reported = self._task_output(task)
            if self.policy_checkpoint.publication_exists(
                    room_id=room_id, task_id=identity.task_id, status=status, execution_generation=generation):
                if reported is not None:
                    self._finish_published_output(identity, *reported)
                continue
            output = reported or (self._unreported_output(room, task) if generation > 0 else None)
            if output is not None and self._waiting(self._obligation(room_id, identity.task_id, generation)):
                held = True
                continue
            task_events = self.policy_checkpoint.events_for_task(
                room_id=room_id, source_event_seq=int(task["payload"]["source_event_seq"]),
                input_context=task["payload"].get("input_context"), task_id=identity.task_id)
            plan = discussion.reconstruct_task_plan(room, task_events, task, local_profiles=local_profiles)
            message_id = f"dmessage:{identity.task_id.removeprefix('dtask:')}"
            existing_message = next((event for event in task_events if event.get("event_id") == message_id
                                     and event["kind"] == "message.member"), None)
            if existing_message is not None:
                # Finish an already visible immutable reply even if a later request arrived.
                task_events = [event for event in task_events if event["kind"] != "message.user"
                               or int(event["seq"]) <= int(task["payload"]["source_event_seq"])]
            deferred_generation = generation if status == "deferred" else None
            result, expected = task.get("result"), None
            if output is not None:
                scope, manifest = output
                initial = discussion.plan_publication(
                    room, task_events, plan, status=status, result=result,
                    execution_generation=deferred_generation, local_profiles=local_profiles)
                if manifest is not None and initial.terminal_kind == "turn.settled":
                    staged = self._stage_for_publication(room, identity, scope, manifest, existing_message)
                    if staged is None:
                        held = True
                        continue
                    result = {**result, **staged}
                    expected = dict(scope=scope.as_mapping(), manifest=manifest,
                                    cancel_generation=task["cancel_generation"])
                elif not self._discard_output(identity, scope, manifest):
                    held = True
                    continue  # the private copy is retired before the terminal becomes visible
            publication = discussion.plan_publication(
                room, task_events, plan, status=status, result=result,
                execution_generation=deferred_generation, local_profiles=local_profiles)
            try:
                for event in publication.events:
                    appended = hosted_rooms.append_event(
                        self.db_path, **event.append_kwargs(room_id), expected_latest_seq=cursor,
                        **({"expected_output": expected} if expected is not None else {}))
                    cursor = max(cursor, int(appended["seq"]))
            except Exception as exc:
                if expected is None:
                    raise
                self._abort_staged(room_id, message_id)
                if isinstance(exc, hosted_rooms.EventCursorConflictError):
                    raise
                self._record_outcome(scope, error=exc)
                held = True
                continue
            changed = True
            if expected is not None:
                self._finish_published_output(identity, scope, manifest)
        self._retire_unowned_output(room_id)
        return changed or held

    def _discard_output(self, identity, scope, manifest) -> bool:
        return self._run_obligation(identity, scope, manifest, "discard", lambda: self._output_source(
            identity, scope, manifest).discard_durably(scope))

    def _stage_for_publication(self, room, identity, scope, manifest, existing_message):
        """Copy the verified bytes into the room store, held for the one member message."""
        row = self._record_intent(identity, scope, manifest, "ack")
        if row["state"] != "pending" or row["operation"] != "ack" or not self._due(row):
            return None
        recipients = [member["member_id"] for member in room["members"]]
        try:
            if existing_message is not None:
                attachments = existing_message["payload"].get("attachments", [])
            else:
                attachments = self._stage_output(identity, scope, manifest, recipients)
        except Exception as exc:
            self._abort_staged(scope.room_id, message_event_id(scope))
            self._record_outcome(scope, error=exc)
            return None
        return {"attachments": attachments, "recipient_member_ids": recipients}

    def _stage_output(self, identity, scope, manifest, recipients):
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore

        items = validate_terminal_artifact_manifest(manifest)
        source = self._output_source(identity, scope, manifest)
        store = HostedRoomAttachmentStore(self.db_path)
        canonical = []
        for item in items:
            metadata, data = source.read(scope, item["artifact_id"])
            if (metadata != item or not isinstance(data, bytes) or len(data) != item["size"]
                    or hashlib.sha256(data).hexdigest() != item["sha256"]):
                raise RoomArtifactError("Group Chat output bytes do not match the terminal manifest")
            saved = store.put(room_id=scope.room_id, upload_id=upload_id(scope, item),
                              name=item["name"], kind=item["kind"], mime=item["mime"], data=data)
            canonical.append({key: saved[key] for key in ("attachment_id", "kind", "name", "size", "mime")})
        return store.commit_message(
            room_id=scope.room_id, event_id=message_event_id(scope), manifest=canonical,
            recipient_member_ids=recipients, viewer_access=True, hold_until_event=True)

    def _abort_staged(self, room_id, event_id):
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore
        try:
            HostedRoomAttachmentStore(self.db_path).abort_unpublished_event(room_id=room_id, event_id=event_id)
        except Exception:
            logger.debug("Could not release staged Group Chat output for %s", event_id, exc_info=True)

    def _published_files(self, scope, manifest):
        """The member message carrying exactly these files, or None when it was never published."""
        with self.authority.db._read_ctx() as conn:
            event = conn.execute(
                "SELECT kind, actor_json, payload_json, authority_epoch FROM hosted_room_events "
                "WHERE room_id=? AND event_id=?", (scope.room_id, message_event_id(scope))).fetchone()
            if event is None:
                settled = conn.execute(
                    "SELECT 1 FROM hosted_room_policy_publications WHERE room_id=? AND task_id=? "
                    "AND kind='turn.settled'", (scope.room_id, scope.task_id)).fetchone()
                if settled is not None:
                    raise RoomArtifactError("published Group Chat output message is unavailable")
                return None
            if event["kind"] != "message.member":
                raise RoomArtifactError("Group Chat output publication kind changed")
            payload, actor = json.loads(event["payload_json"]), json.loads(event["actor_json"])
            attachments = payload.get("attachments")
            if type(attachments) is not list or not attachments:
                raise RoomArtifactError("Group Chat output publication attachments changed")
            items = validate_terminal_artifact_manifest(manifest)
            if (event["authority_epoch"] != scope.authority_epoch or actor.get("id") != scope.member_id
                    or actor.get("profile") != scope.target_profile or payload.get("task_id") != scope.task_id
                    or len(attachments) != len(items)):
                raise RoomArtifactError("Group Chat output publication changed")
            for item, attachment in zip(items, attachments):
                row = conn.execute(
                    "SELECT upload_id FROM hosted_room_attachments WHERE room_id=? AND attachment_id=?",
                    (scope.room_id, attachment.get("attachment_id"))).fetchone()
                if row is None or row["upload_id"] != upload_id(scope, item):
                    raise RoomArtifactError("Group Chat output publication source changed")
        return items, attachments

    def _verify_published_bytes(self, scope, items, attachments):
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore

        store = HostedRoomAttachmentStore(self.db_path)
        for item, attachment in zip(items, attachments):
            saved = store.read_viewer(
                room_id=scope.room_id, event_id=message_event_id(scope),
                attachment_id=attachment["attachment_id"], authority_gateway_id=scope.authority_gateway_id,
                authority_epoch=scope.authority_epoch)
            if (any(saved.attachment[key] != item[key] for key in ("kind", "name", "size", "mime"))
                    or hashlib.sha256(saved.data).hexdigest() != item["sha256"]):
                raise RoomArtifactError("published Group Chat output bytes changed")

    def _acknowledge_source(self, identity, scope, manifest, items):
        source = self._output_source(identity, scope, manifest)
        if isinstance(source, RoomArtifactOutbox) and source.retirement_complete(scope):
            return  # ACK receipts expired after an earlier, unrecorded completion
        source.acknowledge(scope, [item["artifact_id"] for item in items],
                           message_event_id=message_event_id(scope))

    def _finish_published_output(self, identity, scope, manifest):
        """After the terminal is visible: ACK the source when its files were published, else discard it."""
        row = self._obligation(scope.room_id, scope.task_id, scope.execution_generation)
        if row is not None and (row["state"] == "completed" or not self._due(row)):
            return
        try:
            published = self._published_files(scope, manifest)
        except Exception as exc:
            self._record_intent(identity, scope, manifest, row["operation"] if row else "ack")
            self._record_outcome(scope, error=exc)
            return
        if published is None:
            self._discard_output(identity, scope, manifest)
            return
        items, attachments = published

        def acknowledge():
            self._verify_published_bytes(scope, items, attachments)
            self._acknowledge_source(identity, scope, manifest, items)
        self._run_obligation(identity, scope, manifest, "ack", acknowledge)

    def _open_obligations(self, room_id, *, where="", params=()):
        with self.authority.db._read_ctx() as conn:
            if not obligations_exist(conn):
                return []
            return [dict(row) for row in conn.execute(
                f"SELECT * FROM {OBLIGATIONS} WHERE room_id=? AND state IN ('pending', 'blocked') {where} "
                "ORDER BY created_at", (room_id, *params))]

    @staticmethod
    def _obligation_parts(row):
        from gateway.hosted_room_driver import TaskIdentity

        identity = TaskIdentity(**json.loads(row["identity_json"]))
        scope = RoomArtifactScope.from_mapping(json.loads(row["scope_json"]))
        manifest = json.loads(row["manifest_json"]) if row["manifest_json"] else None
        return identity, scope, manifest

    def _retire_unowned_output(self, room_id):
        """Discards no current terminal carries (late receipts, older attempts), and old rows."""
        rows = self._open_obligations(room_id, where=(
            "AND operation='discard' AND NOT EXISTS (SELECT 1 FROM hosted_room_driver_tasks t "
            f"WHERE t.room_id={OBLIGATIONS}.room_id AND t.task_id={OBLIGATIONS}.task_id "
            f"AND t.execution_generation={OBLIGATIONS}.execution_generation "
            "AND t.status IN ('deferred', 'settled', 'failed', 'cancelled'))"))
        for row in rows[:32]:
            if self._due(row):
                self._discard_output(*self._obligation_parts(row))

        prunable = f"""SELECT o.rowid FROM {OBLIGATIONS} o WHERE o.room_id=? AND o.state='completed'
            AND NOT EXISTS (SELECT 1 FROM hosted_room_driver_tasks t
                            WHERE t.room_id=o.room_id AND t.task_id=o.task_id) LIMIT ?"""
        with self.authority.db._read_ctx() as conn:
            if not obligations_exist(conn) or conn.execute(prunable, (room_id, 1)).fetchone() is None:
                return
        # A completed row is only an idempotency receipt once its task is gone.
        self._output_write(lambda conn: conn.execute(
            f"DELETE FROM {OBLIGATIONS} WHERE rowid IN ({prunable})", (room_id, COMPLETED_PRUNE_BATCH)))

    # ------------------------------------------------------------------ driver hooks
    def retire_stale_output(self, binding, task, generation, result):
        """A late receipt reports files for an attempt that will never publish them."""
        try:
            from gateway.session_hosted_output import output_receipt_fields

            fields = output_receipt_fields(dict(result))
            if not fields:
                return
            scope = RoomArtifactScope.from_mapping(fields["artifact_scope"])
            if ((scope.room_id, scope.task_id, scope.execution_generation)
                    != (task["identity"].room_id, task["identity"].task_id, int(generation))):
                return
            same_attempt = int(task["execution_generation"]) == int(generation)
            if same_attempt and task["status"] in {"running", "stopping", "queued", "indeterminate"}:
                return  # still resolvable: a later harvest may settle and publish it
            current = self._task_output(task) if same_attempt else None
            if current is not None and current[0] == scope:
                return  # the same receipt was settled: publication owns it
            self._record_intent(task["identity"], scope, fields["artifacts"], "discard")
        except Exception:
            logger.warning("Could not retire late Group Chat output", exc_info=True)
            return
        self.runtime.wakeup()

    def _policy_snapshot(self, room):
        room_id = str(room["room_id"])
        return self.policy_checkpoint.snapshot(
            room_id=room_id, latest_seq=int(room["latest_seq"]),
            held_output_threads=lambda conn: self._held_output_threads(conn, room_id))

    @staticmethod
    def _held_output_threads(conn, room_id) -> frozenset[str]:
        """Threads whose terminal still waits on its files, read in Policy's own transaction.

        A held thread is left out of selection as a whole: no later turn in it can overtake
        the waiting reply, and a newer request there still supersedes that reply.
        """
        if not obligations_exist(conn) or conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_tasks'").fetchone() is None:
            return frozenset()
        rows = conn.execute(
            f"""SELECT t.thread_id FROM {OBLIGATIONS} o
                JOIN hosted_room_driver_tasks t ON t.room_id=o.room_id AND t.task_id=o.task_id
                 AND t.execution_generation=o.execution_generation
                WHERE o.room_id=? AND o.state IN ('pending', 'blocked')
                  AND NOT EXISTS (SELECT 1 FROM hosted_room_policy_publications p
                      WHERE p.room_id=o.room_id AND p.task_id=o.task_id AND (
                        (t.status='deferred' AND p.kind='turn.deferred'
                         AND p.execution_generation=o.execution_generation)
                        OR (t.status!='deferred'
                            AND p.kind IN ('turn.settled', 'turn.failed', 'turn.cancelled'))))""",
            (room_id,)).fetchall()
        return frozenset(str(row["thread_id"]) for row in rows)

    # ------------------------------------------------------------------ room lifecycle
    def stop_room(self, room_id, *, cancel_id, require_acknowledged=False):
        result = super().stop_room(room_id, cancel_id=cancel_id, require_acknowledged=require_acknowledged)
        if require_acknowledged:
            self._retire_room_output(room_id)
        return result

    def _retire_room_output(self, room_id):
        """Before a room is deleted every output is published-and-ACKed or discarded."""
        with self._policy_lock:
            room = self._room(room_id)
            self._policy_snapshot(room)  # publication reads the synced checkpoint
            try:
                self._publish_terminal_tasks(room)
            except hosted_rooms.EventCursorConflictError:
                pass
            for row in self._open_obligations(room_id):
                identity, scope, manifest = self._obligation_parts(row)
                try:
                    published = (self._published_files(scope, manifest)
                                 if manifest is not None and row["operation"] == "ack" else None)
                except (RoomArtifactError, ValueError) as exc:
                    # Damaged publication evidence cannot prove the files were unpublished.
                    self._record_outcome(scope, error=exc)
                    continue
                if published is not None:
                    def acknowledge():
                        self._verify_published_bytes(scope, *published)
                        self._acknowledge_source(identity, scope, manifest, published[0])
                    self._force_obligation(identity, scope, manifest, "ack", acknowledge)
                elif (scope.target_install_id == scope.home_install_id and not self._peer_output_consent(scope)
                      and any(m['member_id'] == scope.member_id and m['profile'] == scope.target_profile
                              and m.get('target', {}).get('kind', 'local') == 'local' for m in room['members'])
                      and self.profile_homes().get(scope.target_profile) is None):
                    # No longer served here: that profile's own outbox expiry retires the bytes.
                    self._force_obligation(identity, scope, manifest, "discard", lambda: None,
                                           reason="source_unavailable")
                else:
                    self._force_obligation(identity, scope, manifest, "discard", lambda: self._output_source(
                        identity, scope, manifest).discard_durably(scope))
            remaining = self._open_obligations(room_id)
            if not remaining:
                # Every output is retired; the room's receipts would otherwise outlive it.
                with self.authority.db._read_ctx() as conn:
                    known = obligations_exist(conn)
                if known:
                    self._output_write(lambda conn: conn.execute(
                        f"DELETE FROM {OBLIGATIONS} WHERE room_id=? AND state='completed'", (room_id,)))
        if remaining:
            raise RuntimeError("File cleanup is still pending. Try ending the group chat again after it finishes.")

    def _force_obligation(self, identity, scope, manifest, operation, action, *, reason="room_disbanded"):
        """Disband: settle one obligation now, overriding its backoff or block."""
        now = float(self._output_clock())
        key = (scope.room_id, scope.task_id, scope.execution_generation)

        def reopen(conn):
            conn.execute(f"""UPDATE {OBLIGATIONS} SET operation=?, state='pending', next_attempt_at=0,
                reason_code=?, updated_at=? WHERE room_id=? AND task_id=?
                AND execution_generation=? AND state!='completed'""", (operation, reason, now, *key))
        self._output_write(reopen)
        try:
            action()
        except Exception as exc:
            self._record_outcome(scope, error=exc)
            return False
        self._record_outcome(scope, completed=reason)
        return True

    # ------------------------------------------------------------------ status
    def status(self, room_id=None):
        result = super().status(room_id)
        if room_id is None:
            return result
        # Informational only: these retire files, they never re-run a Bot turn.
        actions = [dict(kind="output_retry", task_id=row["task_id"],
                        execution_generation=row["execution_generation"], member_id=row["member_id"],
                        operation=row["operation"], blocked=row["state"] == "blocked",
                        reason_code=row["reason_code"], attempts=row["attempts"],
                        next_attempt_at=row["next_attempt_at"]) for row in self._open_obligations(room_id)]
        return {**result, "pending_actions": [*result["pending_actions"], *actions]}
