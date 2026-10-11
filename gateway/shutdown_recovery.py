"""Recover profile-owned shutdown spools at gateway startup."""

import logging
import json
from pathlib import Path

from gateway.input_owner import recorded_gateway_input_owner
from gateway.session import SessionStore
from gateway.session_transcript import TranscriptReadError
from gateway.shutdown_pending_lock import pending_snapshot_lock
from gateway.shutdown_pending import PENDING_SCHEMA, PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_source

logger = logging.getLogger("gateway.run")


def consume_executed_pending(store: SessionStore) -> int:
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    home = get_hermes_home().resolve()
    consumed = 0
    for path in (home / "pending_messages").glob("*.json"):
        try:
            with pending_snapshot_lock(path):
                original = path.read_bytes()
                payload = json.loads(original)
                if (
                    not isinstance(payload, dict)
                    or payload.get("schema") != PENDING_SCHEMA
                ):
                    continue
                snapshot = PendingQueueSnapshot.from_payload(payload)
                if Path(snapshot.runtime_home).resolve() != home:
                    continue
                resolved = store.resolve_session_id_for_key(
                    snapshot.session_key, not_after=snapshot.ts
                )
                if resolved is None:
                    continue
                session_id, db = resolved
                if Path(db.db_path).resolve().parent != home:
                    raise ValueError(
                        f"pending execution store {db.db_path} belongs outside profile home {home}"
                    )
                remaining = []
                for record in snapshot.events:
                    if "input_owner" not in record:
                        remaining.append(record)
                        continue
                    source = decode_pending_source(record)
                    if store._generate_session_key(source) != snapshot.session_key:
                        raise ValueError(
                            "pending owner differs from its session namespace"
                        )
                    owner = recorded_gateway_input_owner(
                        source, record["uid"], record["input_owner"]
                    )
                    if not store.has_input_owner(session_id, owner):
                        remaining.append(record)
                count = len(snapshot.events) - len(remaining)
                if not count or path.read_bytes() != original:
                    continue
                if remaining:
                    atomic_json_write(
                        path, {**payload, "events": remaining}, mode=0o600
                    )
                else:
                    path.unlink()
                consumed += count
        except (OSError, ValueError, TypeError, KeyError, TranscriptReadError):
            logger.warning(
                "Could not verify pending execution in %s; preserving its records",
                path,
                exc_info=True,
            )
    return consumed
