"""Trusted display lineage on copied rows; canonical payloads are never inspected."""
from collections import defaultdict
import json

from agent.message_metadata import message_uid_or_none


def inherit_display_provenance(store, conn, session_id, messages, *, historical=False):
    """Fill missing sidecars from unambiguous same-session/UID/role donors.

    Explicit current values win. Rewind rows cannot donate. Historical projections
    use only older rows and change dictionaries, never SQLite. One query per 900
    requested identities; the all-generation UID index bounds each lookup.
    """
    targets = [m for m in messages if message_uid_or_none(m) and
               (m.get("display_kind") is None or m.get("display_metadata") is None)]
    uids = list(dict.fromkeys(m["message_uid"] for m in targets))
    donors = defaultdict(list)
    for start in range(0, len(uids), 900):
        chunk = uids[start:start + 900]
        rows = conn.execute(
            "SELECT id, message_uid, role, display_kind, display_metadata FROM messages "
            f"WHERE session_id = ? AND message_uid IN ({','.join('?' for _ in chunk)}) "
            "AND (active = 1 OR compacted = 1) "
            "AND (display_kind IS NOT NULL OR display_metadata IS NOT NULL)",
            (session_id, *chunk))
        for row in rows:
            donors[row["message_uid"]].append(dict(row))
    for target in targets:
        candidates = [d for d in donors[target["message_uid"]]
                      if d["role"] == target.get("role") and
                      (not historical or d["id"] < target["id"])]
        if not candidates:
            continue
        # Missing sidecars are not contradictory values. An existing partial
        # copy may donate its known half without vetoing a complete older copy.
        kinds = {d["display_kind"] for d in candidates if d["display_kind"] is not None}
        metadata_values = {}
        for candidate in candidates:
            metadata = store._decode_display_metadata(candidate["display_metadata"])
            if metadata is not None:
                metadata_values[json.dumps(metadata, sort_keys=True)] = metadata
        if len(kinds) > 1 or len(metadata_values) > 1:
            continue
        donor_kind = next(iter(kinds), None)
        donor_metadata = next(iter(metadata_values.values()), None)
        current_metadata = store._decode_display_metadata(target.get("display_metadata"))
        if current_metadata is not None and donor_metadata is not None and current_metadata != donor_metadata:
            continue
        if target.get("display_kind") not in (None, donor_kind):
            continue
        if target.get("display_kind") is None and donor_kind is not None:
            target["display_kind"] = donor_kind
        if target.get("display_metadata") is None and donor_metadata is not None:
            target["display_metadata"] = donor_metadata


def recover_display_rows(store, rows, session_id):
    """Recover sidecars on selected display rows, preserving row identity/order/payload."""
    projected = [dict(row) for row in rows]
    groups = defaultdict(list)
    for row in projected:
        groups[row.get("session_id", session_id)].append(row)
    with store._read_ctx() as conn:
        for sid, messages in groups.items():
            inherit_display_provenance(store, conn, sid, messages, historical=True)
    return projected
