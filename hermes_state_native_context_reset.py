"""Empty native context rotation preserves the accepted conversation's FIFO identity."""

from dataclasses import dataclass
import json
import time

from hermes_state_runtime import RuntimeStoreError, _admission, _canonical_chain, _epoch

PREFIX = "gateway.native_context_reset.v1."


@dataclass(frozen=True)
class ContextResetClaim:
    epoch: int
    owner_id: str
    admission_id: str
    generation: int
    parent_id: str
    profile_id: str


def _require_claim(conn, claim):
    from hermes_state_runtime_workers import worker_states

    _epoch(conn, claim.epoch)
    row = _admission(conn, claim.admission_id)
    if row["status"] == "unknown":
        raise RuntimeStoreError("unknown_execution")
    if (
        row["target_session_id"] != claim.owner_id
        or row["status"] != "started"
        or row["owner_epoch"] != claim.epoch
        or "native_text_v1" not in json.loads(row["payload_json"])
    ):
        raise RuntimeStoreError("admission_conflict")
    if row["generation"] != claim.generation:
        raise RuntimeStoreError("stale_generation")
    owner = conn.execute(
        "SELECT runtime_generation FROM sessions WHERE id=?", (claim.owner_id,)
    ).fetchone()
    if owner is None or owner[0] != claim.generation:
        raise RuntimeStoreError("stale_generation")
    if conn.execute(
        "SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'",
        (claim.owner_id,),
    ).fetchone():
        raise RuntimeStoreError("unknown_execution")
    if "unknown" in worker_states(conn, claim.owner_id):
        raise RuntimeStoreError("unknown_execution")


def _require_parent(conn, claim, entry):
    from hermes_state_mutation_binding import same_history_owner

    if _canonical_chain(conn, claim.owner_id)[-1] != claim.parent_id:
        raise RuntimeStoreError("admission_conflict")
    parent = conn.execute(
        "SELECT * FROM sessions WHERE id=?", (claim.parent_id,)
    ).fetchone()
    origin = entry["origin"]
    if (
        not same_history_owner(conn, claim.owner_id, claim.parent_id)
        or parent["ended_at"] is not None
        or parent["source"] in {"local", "api_server", "bot_room", "cli", "tui", "gui"}
        or parent["source"] != entry["platform"]
        or parent["session_key"] != entry["session_key"]
        or parent["user_id"] != origin["user_id"]
        or parent["chat_id"] != origin["chat_id"]
        or parent["thread_id"] != origin.get("thread_id")
    ):
        raise RuntimeStoreError("permission_denied")
    if conn.execute(
        "SELECT 1 FROM compression_locks WHERE session_id=? AND expires_at>?",
        (claim.parent_id, time.time()),
    ).fetchone():
        raise RuntimeStoreError("session_busy")
    return parent


def _publish_empty_child(db, conn, claim, entry, parent):
    child = entry["session_id"]
    # This is an intentional empty context boundary. The ordinary compressor's
    # nonempty handoff contract is separate and remains unchanged.
    db._publish_child_session_row(
        conn,
        parent,
        parent_session_id=claim.parent_id,
        child_session_id=child,
        source=parent["source"],
        model=None,
        model_config=None,
        system_prompt=None,
        cwd=parent["cwd"],
        profile_name=parent["profile_name"],
    )
    conn.execute(
        "UPDATE sessions SET tool_names=NULL,transport_profile=?,cwd=? WHERE id=?",
        (parent["transport_profile"], parent["cwd"] or None, child),
    )
    changed = conn.execute(
        "UPDATE sessions SET ended_at=?,end_reason='compression' WHERE id=? AND ended_at IS NULL",
        (time.time(), claim.parent_id),
    )
    if changed.rowcount != 1:
        raise RuntimeStoreError("admission_conflict")
    # Invalidate old viewer mutation snapshots without invalidating A's running
    # admission generation, which normal settlement still owns.
    conn.execute(
        "UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?",
        (claim.owner_id,),
    )
    if parent["archived"]:
        db._unarchive_auto_archived_lineage(conn, child)


def publish_context_reset(db, claim, entry, route_guard):
    """One writer owns the empty child and its exact recovery projection."""

    def write(conn):
        _require_claim(conn, claim)
        route_guard()
        parent = _require_parent(conn, claim, entry)
        _publish_empty_child(db, conn, claim, entry, parent)
        receipt = {
            "owner_id": claim.owner_id,
            "parent_id": claim.parent_id,
            "admission_id": claim.admission_id,
            "generation": claim.generation,
            "epoch": claim.epoch,
            "profile_id": claim.profile_id,
            "queued_through_seq": conn.execute(
                "SELECT MAX(seq) FROM session_admissions WHERE target_session_id=?",
                (claim.owner_id,),
            ).fetchone()[0],
            "entry": entry,
        }
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)",
            (PREFIX + entry["session_id"], json.dumps(receipt)),
        )
        return entry

    return db._execute_write(write)


def require_context_target(conn, expected_id):
    """A new completion may follow compression, but not an empty-context reset."""
    for child in _canonical_chain(conn, expected_id)[1:]:
        if conn.execute(
            "SELECT 1 FROM state_meta WHERE key=?", (PREFIX + child,)
        ).fetchone():
            raise RuntimeStoreError("admission_conflict")


def predates_context_reset(db, owner_id, sequence):
    """Only accepted rows present at the reset commit belong to the old context."""
    with db._read_ctx() as conn:
        for child in _canonical_chain(conn, owner_id)[1:]:
            saved = conn.execute(
                "SELECT value FROM state_meta WHERE key=?", (PREFIX + child,)
            ).fetchone()
            if (
                saved is not None
                and sequence <= json.loads(saved[0])["queued_through_seq"]
            ):
                return True
    return False


def _projection_receipt(conn, chain, route):
    from hermes_state_mutation_binding import same_history_owner

    for child in reversed(chain[1:]):
        saved = conn.execute(
            "SELECT value FROM state_meta WHERE key=?", (PREFIX + child,)
        ).fetchone()
        if saved is None:
            continue
        value = json.loads(saved[0])
        entry = value["entry"]
        index = chain.index(child)
        if (
            entry["session_id"] != child
            or entry["session_key"] != route
            or value["parent_id"] != chain[index - 1]
            or value["owner_id"] != chain[0]
            or not same_history_owner(conn, value["owner_id"], child)
        ):
            raise RuntimeStoreError("admission_conflict")
        return entry
    return None


def reset_projection(db, previous_id, target_id, route):
    """Read exact fresh state only when crossing a verified recorded reset edge."""
    from hermes_state_compression import _CHAIN_CAP, _CHAIN_STEP_SQL

    try:
        with db._read_ctx() as conn:
            root, seen = target_id, {target_id}
            for _ in range(_CHAIN_CAP):
                row = conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id=?", (root,)
                ).fetchone()
                if row is None:
                    raise RuntimeStoreError("not_found")
                parent = row[0]
                step = (
                    conn.execute(_CHAIN_STEP_SQL, (parent,)).fetchone()
                    if parent
                    else None
                )
                if step is None or step["id"] != root:
                    break
                if parent in seen:
                    raise RuntimeStoreError("admission_conflict")
                root = parent
                seen.add(root)
            else:
                raise RuntimeStoreError("admission_conflict")
            chain = _canonical_chain(conn, root)
            if chain[-1] != target_id:
                raise RuntimeStoreError("admission_conflict")
            if previous_id is not None and previous_id not in chain:
                # Generic stale-route recovery also sees explicit reset/branch
                # successors. Their existing recovery path owns those edges.
                return None
            entry = _projection_receipt(conn, chain, route)
            if entry is None or (
                previous_id is not None
                and chain.index(previous_id) >= chain.index(entry["session_id"])
            ):
                return None
            entry = dict(entry, session_id=target_id)
            active = conn.execute(
                "SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (target_id,)
            ).fetchone()
            if active is not None:
                entry["is_fresh_reset"] = False
            return entry
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeStoreError("storage_unavailable") from exc
