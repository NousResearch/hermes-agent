"""Ordered forensic snapshots of accepted gateway input at shutdown."""

from __future__ import annotations

from gateway.platforms.base_pending import pending_dispatch_needs_snapshot

from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import json
import logging
import math
from pathlib import Path
import time
from typing import Any
import uuid

from gateway.platforms.base_pending import _PendingDispatchReservation, pending_dispatch_records, ingress_order, pending_dispatch_needs_snapshot
from gateway.platforms.event import MessageEvent

logger = logging.getLogger(__name__)
PENDING_SCHEMA = "hermes.gateway.pending"
PENDING_VERSION = 1


@dataclass(frozen=True)
class PendingQueueSnapshot:
    session_key: str
    runtime_home: str
    events: list[dict[str, Any]]
    ts: float

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PENDING_SCHEMA, "version": PENDING_VERSION,
            "session_key": self.session_key, "runtime_home": self.runtime_home,
            "events": self.events, "ts": self.ts,
        }
        json.dumps(payload, allow_nan=False)
        return payload

    @classmethod
    def capture(cls, session_key: str, events: list[MessageEvent]) -> PendingQueueSnapshot:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home().resolve())
        records = [_capture_event(event) for event in events]
        snapshot = cls(session_key, home, records, time.time())
        return cls.from_payload(snapshot.to_payload())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PendingQueueSnapshot:
        if payload.get("schema") != PENDING_SCHEMA or payload.get("version") != PENDING_VERSION:
            raise ValueError("unsupported pending snapshot schema or version")
        session_key, home, events = payload.get("session_key"), payload.get("runtime_home"), payload.get("events")
        if not isinstance(session_key, str) or not session_key or not isinstance(home, str) or not home:
            raise ValueError("pending snapshot requires a session key and runtime home")
        if not isinstance(events, list) or not events:
            raise ValueError("pending snapshot events must be a non-empty list")
        seen: set[str] = set()
        for record in events:
            if not isinstance(record, dict):
                raise ValueError("pending snapshot record must be an object")
            uid, event = record.get("uid"), record.get("event")
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("pending snapshot record identity is missing or duplicated")
            seen.add(uid)
            timestamp = record.get("timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError("pending event timestamp must be finite")
            routing = record.get("routing")
            if routing is not None and (not isinstance(routing, dict) or routing.get("runtime_home") != home):
                raise ValueError("pending event routing differs from its snapshot home")
            if not isinstance(event, dict) or not isinstance(event.get("text"), str):
                raise ValueError("pending snapshot event requires text")
            for field_name in ("media_urls", "media_types"):
                values = event.get(field_name)
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise ValueError(f"pending event {field_name} must be a string list")
            flags = event.get("media_text_inlined")
            if not isinstance(flags, list) or any(value is not None and not isinstance(value, bool) for value in flags):
                raise ValueError("pending event media_text_inlined must be a boolean list")
            if not isinstance(event.get("internal"), bool) or not isinstance(event.get("allow_gateway_control"), bool):
                raise ValueError("pending event control permissions must be boolean")
            source = event.get("source")
            if not isinstance(source, dict) or not source.get("platform") or not source.get("chat_id"):
                raise ValueError("pending snapshot event requires its source")
        ts = payload.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError("pending snapshot requires a finite timestamp")
        snapshot = cls(session_key, home, events, ts)
        snapshot.to_payload()
        return snapshot


def _capture_event(event: MessageEvent) -> dict[str, Any]:
    from gateway.session_identity import identity_of
    if not isinstance(event, MessageEvent) or event.source is None:
        raise ValueError("pending snapshot requires a complete MessageEvent")
    body = {
        item.name: getattr(event, item.name)
        for item in fields(event)
        if item.init
        and not item.name.startswith("_")
        and item.name not in {"raw_message", "source"}
    }
    body["message_type"] = event.message_type.value
    body["timestamp"] = event.timestamp.isoformat()
    body["source"] = {item.name: getattr(event.source, item.name) for item in fields(event.source)}
    body["source"]["platform"] = event.source.platform.value
    uid = getattr(event, "_pending_snapshot_uid", None)
    if uid is None:
        uid = uuid.uuid4().hex
        setattr(event, "_pending_snapshot_uid", uid)
    identity = identity_of(event.source)
    routing = None
    if identity is not None:
        routing = {
            "transport_profile": identity.transport_profile, "runtime_profile": identity.runtime_profile,
            "authorization_home": str(identity.authorization_home), "runtime_home": str(identity.runtime_home),
            "multiplexed": identity.multiplexed,
        }
    record = {
        "uid": uid,
        "event": body,
        "routing": routing,
        "timestamp": event.timestamp.timestamp(),
    }
    if event._quoted_media_dependencies or event._inbound_context_dependencies:
        record["context"] = {
            "quoted_media": [
                asdict(dependency) for dependency in event._quoted_media_dependencies
            ],
            "snapshots": [
                snapshot.pending_state(event)
                for snapshot in event._inbound_context_dependencies
            ],
        }
    from gateway.shutdown_pending_codec import capture_pending_provenance
    record.update(capture_pending_provenance(event))
    return record


def flush_adapter_pending(adapter: Any, reservations: dict[str, list[_PendingDispatchReservation]]) -> set[str]:
    """Snapshot each adapter-owned session before pending stores are cleared."""
    runner = getattr(adapter, "gateway_runner", None)
    for key in getattr(adapter, "_pending_dispatch_reservations", {}):
        recorded = reservations.setdefault(key, [])
        for reserved in pending_dispatch_records(adapter, key):
            if not any(previous is reserved for previous in recorded):
                recorded.append(reserved)
    pending = getattr(adapter, "_pending_messages", {})
    buffered = adapter._text_debounce_store()
    ingress: dict[str, list[MessageEvent]] = {}
    for attr in ("_pending_text_batches", "_pending_photo_batches", "_media_group_events"):
        for event in getattr(adapter, attr, {}).values():
            ingress.setdefault(adapter._event_session_key(event), []).append(event)
    for event in getattr(adapter, "_held_inbound_events", ()):
        ingress.setdefault(adapter._event_session_key(event), []).append(event)
    keys = set(pending) | set(buffered) | set(reservations) | set(ingress)
    delivery = getattr(runner, "_delivery_adapter_for", None)
    for key, tail in getattr(runner, "_queued_events", {}).items():
        if tail and callable(delivery) and delivery(tail[0].source) is adapter:
            keys.add(key)
    written: set[str] = set()
    for key in keys:
        events = [reserved.event for reserved in reservations.get(key, []) if pending_dispatch_needs_snapshot(adapter, reserved)]
        events.extend(ingress.get(key, ()))
        if pending.get(key) is not None:
            events.append(pending[key])
        queue = getattr(runner, "_overflow_queue", None)
        overflow = queue(key) if callable(queue) else []
        events.extend(overflow or [])
        state = buffered.get(key)
        if state is not None:
            events.extend([*state.earlier_events, state.event])
        events = list({id(event): event for event in events}.values())
        events.sort(key=ingress_order)
        if not events:
            written.add(key)
            continue
        try:
            _write_snapshot(runner, key, events)
            if overflow:
                overflow.clear()
            written.add(key)
        except Exception:
            state_for = getattr(runner, "_session_state", None)
            if callable(state_for):
                state_for(key).conversation.queued_events[:] = events
            logger.warning("Could not preserve pending input for %s at shutdown", key, exc_info=True)
    return written


def discard_persisted_ingress(adapter: Any, session_keys: set[str]) -> None:
    for attr, tasks_attr in (
        ("_pending_text_batches", "_pending_text_batch_tasks"),
        ("_pending_photo_batches", "_pending_photo_batch_tasks"),
        ("_media_group_events", "_media_group_tasks"),
    ):
        store = getattr(adapter, attr, {})
        tasks = getattr(adapter, tasks_attr, {})
        for key, event in list(store.items()):
            if adapter._event_session_key(event) not in session_keys:
                continue
            store.pop(key, None)
            task = tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
    held = getattr(adapter, "_held_inbound_events", None)
    if held is not None:
        held[:] = [event for event in held if adapter._event_session_key(event) not in session_keys]


def project_pending_snapshot(path: Path, payload: dict[str, Any], *, session_resolver: Any) -> int:
    """Append an inspection projection once and retain the complete event records."""
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write
    from gateway.shutdown_pending_lock import pending_snapshot_lock

    with pending_snapshot_lock(path):
        if json.loads(path.read_bytes()) != payload:
            raise ValueError("pending snapshot changed before inspection projection")
        snapshot = PendingQueueSnapshot.from_payload(payload)
        if Path(snapshot.runtime_home).resolve() != get_hermes_home().resolve():
            raise ValueError("pending snapshot belongs to another profile home")
        if payload.get("projection") is not None:
            projection = payload["projection"]
            if not isinstance(projection, dict) or projection.get("message_uids") != [record["uid"] for record in snapshot.events]:
                raise ValueError("invalid pending snapshot projection marker")
            return 0
        if session_resolver is None:
            return 0
        resolved = session_resolver(snapshot.session_key, not_after=snapshot.ts)
        if not resolved or resolved[1] is None:
            return 0
        session_id, target = resolved
        db_path = getattr(target, "db_path", None)
        if db_path is None or Path(db_path).resolve().parent != Path(snapshot.runtime_home).resolve():
            raise ValueError("resolved pending session store belongs to another profile")
        writer = getattr(type(target), "append_recovered_messages_batch", None)
        if not callable(writer):
            raise ValueError("pending snapshot projection requires the canonical batch writer")
        messages = []
        for record in snapshot.events:
            event = record["event"]
            content = "[Pending input preserved at gateway shutdown; not executed]\n" + event["text"]
            for index, media in enumerate(event.get("media_urls") or []):
                media_types = event.get("media_types") or []
                kind = media_types[index] if index < len(media_types) else "attachment"
                content += f"\n[Attachment: {kind} {media}]"
            messages.append({"role": "user", "content": content, "message_uid": record["uid"],
                             "timestamp": record["timestamp"]})
        inserted = writer(target, session_id, messages)
        if isinstance(inserted, bool) or not isinstance(inserted, int):
            raise ValueError("invalid pending snapshot projection count")
        atomic_json_write(path, {**payload, "projection": {
            "session_id": session_id, "message_uids": [record["uid"] for record in snapshot.events],
        }}, mode=0o600)
        return inserted


def _write_snapshot(runner: Any, session_key: str, events: list[MessageEvent]) -> None:
    from gateway.shutdown_flush import _get_flush_dir, _write_payload
    scope = getattr(runner, "_profile_scope_for_source", None)
    with scope(events[0].source) if callable(scope) else nullcontext():
        snapshot = PendingQueueSnapshot.capture(session_key, events)
        _write_payload(_get_flush_dir(), snapshot.to_payload())


def flush_runner_pending(runner: Any) -> None:
    """Retry adapter snapshots and preserve orphaned tails under their source scope."""
    queued = getattr(runner, "_queued_events", {})
    startup = getattr(runner, "_startup_restore_queue", [])
    pending = {key: list(events) for key, events in queued.items() if events}
    for event in startup:
        key = runner._session_key_for_source(event.source)
        pending.setdefault(key, []).append(event)
    for key, events in pending.items():
        try:
            _write_snapshot(runner, key, events)
        except Exception:
            logger.warning("Could not preserve remaining pending input for %s at shutdown", key, exc_info=True)
            continue
        if key in queued:
            queued[key].clear()
        identities = {id(event) for event in events}
        startup[:] = [event for event in startup if id(event) not in identities]
