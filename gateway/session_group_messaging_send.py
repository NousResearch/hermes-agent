"""Explicit native-owner consent for one private canonical Group Send.

The stored right is deliberately separate from inventory/detail consent.  A
messaging actor remains independently authenticated with ``session:read``; the
exact ``_MessagingRoomSend`` value is the only delegated mutation authority.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
import uuid
from typing import Any

from hermes_state_errors import StateDbReplacedError
from hermes_state_runtime import RuntimeStoreError, _epoch
from gateway.session_group_messaging_identity import trusted_person
from gateway.session_group_messaging_read import (
    _binding,
    _json,
    _load,
    _recipient,
    _room_binding,
    _room_service,
    _attest_room_read,
)

SEND_BINDING_METHODS = {
    "groups.messaging.room.send.grant": "session:operator",
    "groups.messaging.room.send.revoke": "session:operator",
}
SEND_BINDING_FIELDS = {
    "groups.messaging.room.send.grant": {
        "request_id",
        "recipient",
        "room_id",
        "room_read_binding_id",
        "room_read_generation",
        "expected_generation",
    },
    "groups.messaging.room.send.revoke": {
        "request_id",
        "recipient",
        "room_id",
        "room_read_binding_id",
        "room_read_generation",
        "expected_generation",
        "binding_id",
    },
}
_PREFIX = "gateway.messaging.send.v1."
_SCOPE = "groups.send"
_MAX_GENERATION = 2**63 - 2
_MAX_BINDINGS = 4096
_MAX_REQUESTS = 16384
_SEND_FIELDS = frozenset(
    {
        "version",
        "recipient",
        "profile_id",
        "owner",
        "room_id",
        "scope",
        "inventory_binding_id",
        "room_read_binding_id",
        "room_read_generation",
        "room_ref",
        "binding_id",
        "generation",
        "active",
    }
)
_CLIENT_EVENT_RE = re.compile(r"msg:([0-9a-f]{64}):([0-9a-f]{32})")


def _text(value, maximum):
    if (
        type(value) is not str
        or not 1 <= len(value) <= maximum
        or value != value.strip()
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise RuntimeStoreError("invalid_params")
    return value


def _binding_id(value):
    if type(value) is not str or re.fullmatch(r"mrs-[0-9a-f]{32}", value) is None:
        raise RuntimeStoreError("invalid_params")
    return value


def _recipient_digest(recipient):
    return hashlib.sha256(_json(recipient).encode()).hexdigest()


def _binding_key(recipient, room_id):
    room_digest = hashlib.sha256(room_id.encode()).hexdigest()
    return _PREFIX + "binding." + _recipient_digest(recipient) + "." + room_digest


def _request_key(owner, request_id):
    return _PREFIX + "request." + hashlib.sha256(
        _json([owner, request_id]).encode()
    ).hexdigest()


def _record(value, recipient, profile_id, room_id=None):
    if (
        type(value) is not dict
        or set(value) != _SEND_FIELDS
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["recipient"] != recipient
        or value["profile_id"] != profile_id
        or value["scope"] != _SCOPE
        or type(value["owner"]) is not str
        or not 1 <= len(value["owner"]) <= 1024
        or type(value["room_id"]) is not str
        or not 1 <= len(value["room_id"]) <= 128
        or (room_id is not None and value["room_id"] != room_id)
        or type(value["room_read_generation"]) is not int
        or not 1 <= value["room_read_generation"] <= _MAX_GENERATION
        or type(value["room_ref"]) is not int
        or not 1 <= value["room_ref"] <= 2**63 - 2
        or type(value["generation"]) is not int
        or not 1 <= value["generation"] <= _MAX_GENERATION
        or type(value["active"]) is not bool
    ):
        raise RuntimeStoreError("permission_denied")
    try:
        from gateway.session_group_messaging_read import _binding_id as read_inventory_id
        from gateway.session_group_messaging_read import _room_binding_id

        read_inventory_id(value["inventory_binding_id"])
        _room_binding_id(value["room_read_binding_id"])
        _binding_id(value["binding_id"])
    except RuntimeStoreError as exc:
        raise RuntimeStoreError("permission_denied") from exc
    return value


def _send_binding(conn, recipient, profile_id, room_id):
    value = _load(conn, _binding_key(recipient, room_id))
    return None if value is None else _record(value, recipient, profile_id, room_id)


def _response(record):
    return {
        field: record[field]
        for field in ("binding_id", "generation", "active")
    }


def _require_submit(operation):
    operation.require_current()
    if "session:submit" not in operation.actor.capabilities:
        raise RuntimeStoreError("permission_denied")


@dataclass(frozen=True)
class _PreparedSendBinding:
    operation: object
    service: object
    checker: object
    request_key: str
    intent_json: str


def prepare_native_send_binding(connection, method, params):
    """Freeze one native mutation before dispatching it to the writer thread."""
    from gateway.session_group_peers import _native_owner
    from gateway.session_group_messaging_read import _room_binding_id

    if method not in SEND_BINDING_FIELDS or set(params) != SEND_BINDING_FIELDS[method]:
        raise RuntimeStoreError("invalid_params")
    operation = _native_owner(connection)
    _require_submit(operation)
    recipient = _recipient(params["recipient"])
    request_id = _text(params["request_id"], 128)
    room_id = _text(params["room_id"], 128)
    read_binding_id = _room_binding_id(params["room_read_binding_id"])
    read_generation = params["room_read_generation"]
    generation = params["expected_generation"]
    if (
        type(read_generation) is not int
        or not 1 <= read_generation <= _MAX_GENERATION
        or type(generation) is not int
        or not 0 <= generation < _MAX_GENERATION
    ):
        raise RuntimeStoreError("invalid_params")
    binding_id = _binding_id(params["binding_id"]) if method.endswith(".revoke") else None
    owner = _text(operation.actor.subject, 1024)
    service, checker = _room_service(operation)

    operation.require_current()
    _require_submit(operation)
    _room_service(operation, service, checker)
    with operation.db._read_ctx() as conn:
        _epoch(conn, operation.epoch)
        row = conn.execute(
            "SELECT instance_id FROM runtime_epoch WHERE singleton=1"
        ).fetchone()
        if row is None or row[0] != operation.instance_id:
            raise RuntimeStoreError("stale_epoch")
        checker(owner, room_id, conn=conn)
        inventory = _binding(conn, recipient, operation.profile_id)
        read = _room_binding(conn, recipient, operation.profile_id, room_id)
        if (
            inventory is None
            or not inventory["active"]
            or inventory["owner"] != owner
            or read is None
            or not read["active"]
            or read["owner"] != owner
            or read["inventory_binding_id"] != inventory["binding_id"]
            or read["binding_id"] != read_binding_id
            or read["generation"] != read_generation
        ):
            raise RuntimeStoreError("permission_denied")
    operation.require_current()
    _require_submit(operation)
    _room_service(operation, service, checker)
    intent = {
        "method": method,
        "recipient": recipient,
        "room_id": room_id,
        "room_read_binding_id": read_binding_id,
        "room_read_generation": read_generation,
        "expected_generation": generation,
        "binding_id": binding_id,
        "owner": owner,
        "profile_id": operation.profile_id,
    }
    return _PreparedSendBinding(
        operation,
        service,
        checker,
        _request_key(owner, request_id),
        _json(intent),
    )


def commit_native_send_binding(prepared):
    operation = prepared.operation
    intent = json.loads(prepared.intent_json)
    recipient = intent["recipient"]
    owner = intent["owner"]
    room_id = intent["room_id"]
    key = _binding_key(recipient, room_id)

    def require_writer(conn):
        operation.require_current(conn)
        if "session:submit" not in operation.actor.capabilities:
            raise RuntimeStoreError("permission_denied")
        _room_service(operation, prepared.service, prepared.checker)
        inventory = _binding(conn, recipient, operation.profile_id)
        read = _room_binding(conn, recipient, operation.profile_id, room_id)
        if (
            inventory is None
            or not inventory["active"]
            or inventory["owner"] != owner
            or read is None
            or not read["active"]
            or read["owner"] != owner
            or read["inventory_binding_id"] != inventory["binding_id"]
            or read["binding_id"] != intent["room_read_binding_id"]
            or read["generation"] != intent["room_read_generation"]
        ):
            raise RuntimeStoreError("permission_denied")
        prepared.checker(owner, room_id, conn=conn)
        operation.require_current(conn)
        return inventory, read

    def write(conn):
        inventory, read = require_writer(conn)
        current = _send_binding(conn, recipient, operation.profile_id, room_id)
        prior = _load(conn, prepared.request_key)
        if prior is not None:
            if set(prior) != {"intent", "state"}:
                raise RuntimeStoreError("permission_denied")
            if prior["intent"] != intent:
                raise RuntimeStoreError("admission_conflict")
            if current is None or prior["state"] != current:
                raise RuntimeStoreError("messaging_room_send_stale")
            return _response(current)
        if current is not None and current["owner"] != owner:
            raise RuntimeStoreError("permission_denied")
        if (current["generation"] if current else 0) != intent["expected_generation"]:
            raise RuntimeStoreError("messaging_room_send_stale")
        granting = intent["method"].endswith(".grant")
        same_read = current is not None and (
            current["inventory_binding_id"] == inventory["binding_id"]
            and current["room_read_binding_id"] == read["binding_id"]
            and current["room_read_generation"] == read["generation"]
            and current["room_ref"] == read["room_ref"]
        )
        if granting:
            if current is not None and current["active"] and same_read:
                raise RuntimeStoreError("admission_conflict")
        elif (
            current is None
            or not current["active"]
            or current["binding_id"] != intent["binding_id"]
            or not same_read
        ):
            raise RuntimeStoreError("messaging_room_send_stale")

        binding_count = conn.execute(
            "SELECT COUNT(*) FROM state_meta WHERE key LIKE ?",
            (_PREFIX + "binding.%",),
        ).fetchone()[0]
        request_count = conn.execute(
            "SELECT COUNT(*) FROM state_meta WHERE key LIKE ?",
            (_PREFIX + "request.%",),
        ).fetchone()[0]
        if (
            (current is None and binding_count >= _MAX_BINDINGS)
            or request_count >= _MAX_REQUESTS
            or (granting and request_count + binding_count + 2 > _MAX_REQUESTS)
        ):
            raise RuntimeStoreError("messaging_room_send_capacity")

        record = {
            "version": 1,
            "recipient": recipient,
            "profile_id": operation.profile_id,
            "owner": owner,
            "room_id": room_id,
            "scope": _SCOPE,
            "inventory_binding_id": inventory["binding_id"],
            "room_read_binding_id": read["binding_id"],
            "room_read_generation": read["generation"],
            "room_ref": read["room_ref"],
            "binding_id": "mrs-" + uuid.uuid4().hex if granting else current["binding_id"],
            "generation": intent["expected_generation"] + 1,
            "active": granting,
        }
        require_writer(conn)
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, _json(record)),
        )
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)",
            (prepared.request_key, _json({"intent": intent, "state": record})),
        )
        require_writer(conn)
        return _response(record)

    return operation.db._execute_write(write)


def _stable_message_identity(event, recipient):
    """Adapt M73's stable messaging-event identity to the current MessageEvent/source contract."""
    if (
        event is None
        or getattr(event, "internal", False) is True
        or getattr(event, "allow_gateway_control", None) is not True
        or not trusted_person(event)
        or any(
            getattr(event, field, None)
            for field in ("media_urls", "media_types", "media_text_inlined")
        )
    ):
        raise RuntimeStoreError("permission_denied")
    message_id = _text(getattr(event, "message_id", None), 512)
    source_id = getattr(event.source, "message_id", None)
    if source_id is not None and str(source_id) != message_id:
        raise RuntimeStoreError("permission_denied")
    coordinates = {
        "version": 1,
        "recipient": recipient,
        "message_id": message_id,
    }
    return _json(coordinates), hashlib.sha256(_json(coordinates).encode()).hexdigest()


@dataclass(frozen=True)
class _MessagingRoomSend:
    """One exact Send operation; the messaging principal itself gains no mutation capability."""

    room_read: Any
    owner: str
    binding_id: str
    generation: int
    active: bool
    source_identity_json: str
    source_digest: str
    client_event_id: str
    payload_json: str
    event_text: str
    runtime: Any
    runtime_generation: str

    @property
    def runner(self):
        return self.room_read.runner

    @property
    def authority(self):
        return self.room_read.authority

    @property
    def actor(self):
        return self.room_read.actor

    @property
    def recipient_json(self):
        return self.room_read.recipient_json

    @property
    def room_id(self):
        return self.room_read.room_id

    @property
    def room_ref(self):
        return self.room_read.room_ref

    @property
    def source_prefix(self):
        return f"msg:{self.source_digest}:"

    def _request(self, method, params):
        if type(self) is not _MessagingRoomSend or method != "groups.send":
            raise RuntimeStoreError("permission_denied")
        if type(params) is not dict or set(params) != {"room_id", "event_id", "payload"}:
            raise RuntimeStoreError("invalid_params")
        if (
            params["room_id"] != self.room_id
            or params["event_id"] != self.client_event_id
            or type(params["payload"]) is not dict
            or set(params["payload"]) != {"text", "thread_id"}
            or params["payload"]["thread_id"] != self.client_event_id
            or _json(params["payload"]) != self.payload_json
        ):
            raise RuntimeStoreError("permission_denied")
        return params

    def _current_lineage(self, conn, recipient, *, require_active):
        inventory = _binding(conn, recipient, self.room_read.inventory.profile_id)
        read = _room_binding(
            conn,
            recipient,
            self.room_read.inventory.profile_id,
            self.room_id,
        )
        send = _send_binding(
            conn,
            recipient,
            self.room_read.inventory.profile_id,
            self.room_id,
        )
        if (
            inventory is None
            or not inventory["active"]
            or inventory["owner"] != self.owner
            or read is None
            or not read["active"]
            or read["owner"] != self.owner
            or read["inventory_binding_id"] != inventory["binding_id"]
            or read["binding_id"] != self.room_read.binding_id
            or read["generation"] != self.room_read.generation
            or read["room_ref"] != self.room_ref
            or send is None
            or send["owner"] != self.owner
            or send["inventory_binding_id"] != inventory["binding_id"]
            or send["room_read_binding_id"] != read["binding_id"]
            or send["room_read_generation"] != read["generation"]
            or send["room_ref"] != read["room_ref"]
            or send["binding_id"] != self.binding_id
            or (require_active and (not send["active"] or send["generation"] != self.generation))
            or (not require_active and not (
                (send["active"] == self.active and send["generation"] == self.generation)
                or (self.active and not send["active"]
                    and send["generation"] == self.generation + 1)
            ))
        ):
            raise RuntimeStoreError("messaging_room_send_stale")
        return send

    def _require_event(self):
        inventory = self.room_read.inventory
        recipient = inventory._require_context()
        identity_json, digest = _stable_message_identity(inventory.event, recipient)
        if (
            identity_json != self.source_identity_json
            or digest != self.source_digest
            or inventory.event.text != self.event_text
        ):
            raise RuntimeStoreError("permission_denied")
        return recipient

    def _require_runtime(self):
        inventory = self.room_read.inventory
        if (
            inventory.service.runtime is not self.runtime
            or self.runtime.process_generation != self.runtime_generation
        ):
            raise RuntimeStoreError("runtime_coordination_required")
        return self.runtime

    def require_current(self, method=None, params=None):
        if method is not None:
            self._request(method, params)
        recipient = self._require_event()
        self._require_runtime()
        self.room_read.require_current(room_id=self.room_id)
        with self.room_read.inventory.db._read_ctx() as conn:
            self.room_read.inventory._require_context()
            self._require_runtime()
            self._current_lineage(conn, recipient, require_active=False)
            self.room_read.inventory.checker(self.owner, self.room_id, conn=conn)
            self.room_read.inventory._require_context()
            self._require_runtime()
        return True

    @contextmanager
    def new_event_lifetime(self):
        """Hold runtime then owner-generation lifetime through external commit."""

        inventory = self.room_read.inventory
        self._require_event()
        inventory._require_context()
        runtime = self._require_runtime()
        body_started = False
        try:
            with runtime.new_event_admission():
                with inventory.db.live_external_writer_lifetime() as require_external:
                    inventory._require_context()
                    self._require_runtime()

                    def require_current_writer(conn):
                        # These exceptions have a known owner-guard origin and
                        # occur before the canonical transaction commits.
                        try:
                            require_external(conn)
                        except (sqlite3.Error, StateDbReplacedError) as exc:
                            raise RuntimeStoreError("runtime_coordination_required") from exc

                    body_started = True
                    yield require_current_writer
                    inventory._require_context()
                    self._require_runtime()
        except (sqlite3.Error, StateDbReplacedError) as exc:
            # Exceptions thrown through yield belong to the canonical writer,
            # whose commit may already have settled. Never turn that uncertainty
            # into a definitive pre-admission denial. Guard-exit errors also
            # retain their type for the service's post-commit classification.
            if body_started:
                raise
            raise RuntimeStoreError("runtime_coordination_required") from exc

    def _authorize_new_writer(self, conn, *, source_conflict):
        recipient = self._require_event()
        inventory = self.room_read.inventory
        if (
            conn is None
            or not conn.in_transaction
            or Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
            != inventory.home / "state.db"
        ):
            raise RuntimeStoreError("profile_mismatch")
        _epoch(conn, inventory.epoch)
        row = conn.execute(
            "SELECT instance_id FROM runtime_epoch WHERE singleton=1"
        ).fetchone()
        if row is None or row[0] != inventory.instance_id:
            raise RuntimeStoreError("stale_epoch")
        inventory._require_context()
        self._current_lineage(conn, recipient, require_active=True)
        inventory.checker(self.owner, self.room_id, conn=conn)
        runtime = self._require_runtime()
        status = runtime.status()
        if not status.get("running") or status.get("stopping"):
            raise RuntimeStoreError("runtime_coordination_required")
        if source_conflict:
            upper = self.source_prefix + "g"
            prior = conn.execute(
                "SELECT room_id,event_id FROM hosted_room_events "
                "INDEXED BY idx_hosted_room_events_message_thread "
                "WHERE kind='message.user' "
                "AND json_extract(payload_json,'$.thread_id')>=? "
                "AND json_extract(payload_json,'$.thread_id')<? LIMIT 1",
                (self.source_prefix, upper),
            ).fetchone()
            if prior is not None:
                raise RuntimeStoreError("admission_conflict")
        inventory._require_context()
        self._require_runtime()
        return True

    def authorize_new_event(self, conn):
        """Fence NEW insertion on its distinct immediate writer connection."""

        return self._authorize_new_writer(conn, source_conflict=True)

    def authorize_new_commit(self, conn):
        """Recheck mutable authority at the last point before writer commit."""

        return self._authorize_new_writer(conn, source_conflict=False)


def attest_room_send(runner, event, room_ref, text):
    """Capture one stable private user intent and its current/historical Send lineage."""
    from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES

    if (type(text) is not str or not text.strip()
            or len(text.encode("utf-8")) > MAX_USER_TEXT_BYTES):
        raise RuntimeStoreError("invalid_params")
    room_read = _attest_room_read(runner, event, room_ref)
    inventory = room_read.inventory
    recipient = inventory._require_context()
    identity_json, source_digest = _stable_message_identity(event, recipient)
    with inventory.db._read_ctx() as conn:
        room_read._require_current_on_connection(conn, recipient, held_writer=False)
        send = _send_binding(conn, recipient, inventory.profile_id, room_read.room_id)
        if (
            send is None
            or send["owner"] != room_read.owner
            or send["inventory_binding_id"]
            != json.loads(inventory.state_json)["binding_id"]
            or send["room_read_binding_id"] != room_read.binding_id
            or send["room_read_generation"] != room_read.generation
            or send["room_ref"] != room_read.room_ref
        ):
            raise RuntimeStoreError("permission_denied")
        inventory.checker(room_read.owner, room_read.room_id, conn=conn)
        inventory._require_context()
    client_event_id = f"msg:{source_digest}:{send['binding_id'][4:]}"
    if _CLIENT_EVENT_RE.fullmatch(client_event_id) is None:
        raise RuntimeStoreError("permission_denied")
    payload_json = _json({"text": text, "thread_id": client_event_id})
    context = _MessagingRoomSend(
        room_read,
        room_read.owner,
        send["binding_id"],
        send["generation"],
        send["active"],
        identity_json,
        source_digest,
        client_event_id,
        payload_json,
        event.text,
        inventory.service.runtime,
        inventory.service.runtime.process_generation,
    )
    context.require_current()
    return context
