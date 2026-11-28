"""A profile transfer binds copied rows, membership and guarded donor retirement."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from persistence.core.errors import AuthorityRefusal
from .._state.accounting_markers import pending_usage, require_legacy_accounting_clear
from .._state.lineage_fence import DONOR_TRANSFER_PREFIX, session_lineage_guard
from ..authority._runtime_guards import require_idle
from ..conversation._store.sessions import _end_and_bump, _retire_undrained_queue_rows_conn, get_session
from ._adoption_records import AdoptionConflict
from ._adoption_witness import _adoption_accounting_covers, _adoption_history_covers, _copy_adoption_portable_state
from ._portability import _import_normalized, _validate_import_payload

logger = logging.getLogger("hermes_state")


def _import_entry(segment):
    return {**segment.session.as_dict(), "messages": [message.as_wire_row() for message in segment.messages]}


def _result_dict(result):
    values = {"ok": result.ok, "imported": result.imported, "skipped": result.skipped, "detached": result.detached}
    if result.ok:
        values.update(imported_ids=list(result.imported_ids), skipped_ids=list(result.skipped_ids))
    values["errors"] = list(result.errors)
    return values


def _retire_member(db, conn, sid):
    row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id=?", (sid,)).fetchone()
    if row is None:
        return False
    require_idle(db, conn, [sid])
    _retire_undrained_queue_rows_conn(conn, sid)
    reason = "adopted_by_profile"
    _end_and_bump(db, conn,
                  "UPDATE sessions SET ended_at=coalesce(ended_at,?), end_reason=? WHERE id=? AND end_reason IS NOT ?",
                  (time.time(), reason, sid, reason), sid, reason)
    return True


class LineageTransfer:
    def __init__(self, target, donor):
        self.db = target._store.holder()
        self.donor = donor

    def adopt(self, session_id: str, *, retire_donor: bool, pending_only: bool) -> dict[str, Any]:
        """Copy a complete lineage, durably retaining membership for retirement retries.

        Copies and immutable membership commit together. Same-home member locks remain
        held through donor retirement; standalone pairs retain their existing contract.
        Existing rows are skipped and donor-ahead copies are never retired.
        """
        donor_db = self.donor._store.holder()
        require_legacy_accounting_clear(self.db)
        require_legacy_accounting_clear(donor_db)
        # A queued writer may need the same lineage lock. Drain before acquiring it;
        # a timeout cannot authorize moving a stale accounting snapshot.
        if not donor_db.flush_token_counts() or not self.db.flush_token_counts():
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "error": "Queued accounting is busy; retry adoption after it settles"}
        pair_error = self._validate_adoption_store_pair(donor_db.db_path)
        if pair_error:
            return {"ok": False, "adopted": False, "donor_retired": False, "error": pair_error}
        try:
            for _attempt in range(3):
                outcome = self._attempt_lineage_adoption(donor_db, session_id, retire_donor=retire_donor, pending_only=pending_only)
                if outcome is not None:
                    return outcome
        except AdoptionConflict as exc:
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "conflict": True, "error": str(exc)}
        return {"ok": False, "adopted": False, "donor_retired": False,
                "error": "donor membership kept changing; retry after its lineage stabilizes"}

    def _adoption_segments(self, session_id: str, record) -> list[dict[str, Any]]:
        if record is None:
            payload = self.donor.export_session_lineage(session_id, include_inactive=True)
            return [_import_entry(segment) for segment in payload.segments] if payload else []
        segments = [self.donor.export_session(mid, include_inactive=True) for mid in record["member_ids"]]
        if any(segment is None for segment in segments):
            raise AdoptionConflict("Saved adoption donor member is missing; recover the complete transfer")
        return [_import_entry(segment) for segment in segments]

    def _completed_result(self, donor_db, session_id, record, retire_donor):
        """Replay an already-completed transfer without reopening or retiring rows."""
        from ._adoption_records import validate_members

        validate_members(self.db, donor_db, record)
        with donor_db._read_ctx() as source, self.db._read_ctx() as target:
            retired = True
            for member in record["manifest"]["members"]:
                sid = member["id"]
                state = source.execute("SELECT end_reason, archived FROM sessions WHERE id=?", (sid,)).fetchone()
                donor_count = source.execute("SELECT count(*) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
                local_count = target.execute("SELECT count(*) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
                if donor_count > local_count:
                    logger.warning("adoption divergence: donor segment %s has %d messages, local copy has %d — donor will NOT be retired",
                                   sid, donor_count, local_count)
                retired &= bool(state and tuple(state) == ("adopted_by_profile", 1)
                                and donor_count <= local_count
                                and _adoption_history_covers(source, target, sid, retired_witness=member)
                                and _adoption_accounting_covers(source, target, sid))
        return {"ok": True, "imported": 0, "skipped": 1, "detached": 0,
                "imported_ids": [], "skipped_ids": [session_id], "errors": [],
                "adopted": True, "donor_retired": bool(retire_donor and retired)}

    def _attempt_lineage_adoption(self, donor_db: Any, session_id: str, *, retire_donor: bool, pending_only: bool):
        from ._adoption_records import find, ownership, validate_fresh, validate_members, validate_handles
        validate_handles(self.db, donor_db)
        record = find(self.db, session_id, donor_db)
        if record is not None and record["status"] == "complete" and not pending_only:
            return self._completed_result(donor_db, session_id, record, retire_donor)
        if pending_only and (record is None or record["status"] != "pending"):
            return {"ok": False, "adopted": False, "donor_retired": False, "retry_skipped": True}
        segments = self._adoption_segments(session_id, record)
        if not segments:
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "error": f"session {session_id!r} not found in donor store"}
        ids = tuple(segment["id"] for segment in segments)
        authority = None
        if record is None:
            validate_fresh(self.db, segments)
        else:
            validate_members(self.db, donor_db, record)
            authority = ownership(record, donor_db.db_path, self.db.db_path)
        with session_lineage_guard(self.db.db_path, ids, transfer_from=donor_db.db_path,
                                   adoption=authority, timeout_s=self.db._WRITE_PATIENCE_S) as snapshot:
            # Never wait under this guard for a writer that may be waiting for it.
            require_legacy_accounting_clear(self.db)
            require_legacy_accounting_clear(donor_db)
            if pending_usage(donor_db) or pending_usage(self.db):
                return {"ok": False, "adopted": False, "donor_retired": False,
                        "error": "Queued accounting is busy; retry adoption after it settles"}
            # Reload authority under the complete lock set. Returning releases it before
            # the bounded outer loop discovers/reacquires any expanded membership.
            validate_handles(self.db, donor_db)
            current = find(self.db, session_id, donor_db)
            if pending_only and (current is None or current["status"] != "pending"):
                return {"ok": False, "adopted": False, "donor_retired": False, "retry_skipped": True}
            if (current is None) != (record is None):
                return None
            if current is not None and current["manifest"] != record["manifest"]:
                return None
            refreshed = self._adoption_segments(session_id, current)
            if tuple(segment["id"] for segment in refreshed) != ids:
                return None
            pair_error = self._validate_adoption_store_pair(donor_db.db_path)
            if pair_error:
                return {"ok": False, "adopted": False, "donor_retired": False, "error": pair_error}
            self._validate_adoption_inventory(snapshot, donor_db.db_path, ids)
            if current is None:
                validate_fresh(self.db, refreshed)
            else:
                validate_members(self.db, donor_db, current)
            return self._adopt_lineage_segments_locked(
                donor_db, session_id, refreshed, retire_donor=retire_donor, record=current, pending_only=pending_only)

    def _validate_adoption_inventory(self, snapshot, donor_path, ids) -> None:
        if snapshot is None:
            return  # both resolutions positively classified the standalone compatibility path
        from .._state.lineage_fence import SessionLineageAmbiguousError, SessionLineageInventoryError, _same_file
        for session_id in ids:
            rows = snapshot.rows_by_id.get(session_id, ())
            if not any(_same_file(row.store_path, Path(donor_path)) for row in rows):
                raise SessionLineageInventoryError("Cannot inventory the adoption donor; use a managed profile store")
            if any(not _same_file(row.store_path, Path(donor_path))
                   and not _same_file(row.store_path, self.db.db_path) for row in rows):
                raise SessionLineageAmbiguousError("Adoption has a third store copy; adjudicate its ownership first")

    def _adoption_import_and_record(self, donor_db, segments, record, *, retire_donor, pending_only):
        from ._adoption_records import AdoptionConflict, build_manifest, find_in, insert, set_retirement_intent, validate_handles
        normalized, errors = _validate_import_payload(segments)
        if errors:
            return {"ok": False, "imported": 0, "skipped": 0, "detached": 0, "errors": errors}, record
        first_id = segments[0]["id"]

        def write(conn):
            validate_handles(self.db, donor_db)
            current = find_in(conn, first_id, donor_db, self.db)
            if pending_only and (current is None or current["status"] != "pending"):
                return {"ok": False, "imported": 0, "skipped": 0, "retry_skipped": True}, current
            if (current is None) != (record is None):
                raise AdoptionConflict("Adoption record appeared or disappeared; retry with its complete membership")
            if current is not None and current["manifest"] != record["manifest"]:
                raise AdoptionConflict("Adoption membership changed; inspect the saved transfer")
            # Refuse the whole transfer before imports or the adoption journal when an
            # existing target does not preserve the donor transcript as a prefix.
            for segment in segments:
                sid = segment["id"]
                if conn.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
                    with donor_db._read_ctx() as donor_conn:
                        if not _adoption_history_covers(donor_conn, conn, sid):
                            member = next((m for m in current["manifest"]["members"] if m["id"] == sid), None) if current else None
                            if member is None or not _adoption_history_covers(donor_conn, conn, sid, retired_witness=member):
                                raise AdoptionConflict("Target transcript differs from donor; recover the history before adoption")
                        try:
                            portable = _adoption_accounting_covers(donor_conn, conn, sid)
                        except ValueError as exc:
                            raise AdoptionConflict(str(exc)) from exc
                        if not portable:
                            raise AdoptionConflict("Target accounting or context differs from donor; inspect before adoption")
            result = _result_dict(_import_normalized(self.db, conn, normalized, [segment['id'] for segment in segments]))
            if result.get("ok"):
                # A fresh import can normalize or omit legacy sidecars; do not commit
                # a copy that cannot prove it preserves the donor's complete history.
                with donor_db._read_ctx() as donor_conn:
                    try:
                        for sid in result["imported_ids"]:
                            _copy_adoption_portable_state(self.db, donor_conn, conn, sid)
                        members = {m["id"]: m for m in current["manifest"]["members"]} if current else {}
                        if any(not (_adoption_history_covers(donor_conn, conn, segment["id"])
                                    or (segment["id"] in members and _adoption_history_covers(
                                        donor_conn, conn, segment["id"], retired_witness=members[segment["id"]])))
                               or not _adoption_accounting_covers(donor_conn, conn, segment["id"])
                               for segment in segments):
                            raise AdoptionConflict("Imported history or accounting differs from donor; inspect before adoption")
                    except ValueError as exc:
                        raise AdoptionConflict(str(exc)) from exc
                if current is None:
                    insert(conn, build_manifest(self.db, donor_db, segments, conn), retire_donor=retire_donor)
                else:
                    set_retirement_intent(conn, current, retire_donor)
                current = find_in(conn, first_id, donor_db, self.db)
            return result, current

        return self.db._execute_write(write)

    def _adopt_lineage_segments_locked(self, donor_db: Any, session_id: str, segments: list[dict[str, Any]],
                                       *, retire_donor: bool, record=None, pending_only: bool = False) -> dict[str, Any]:
        from ._adoption_records import mark_complete
        members = [segment["id"] for segment in segments]
        result, saved = self._adoption_import_and_record(donor_db, segments, record, retire_donor=retire_donor, pending_only=pending_only)
        adopted = result.get("ok", False) and (result.get("imported", 0) + result.get("skipped", 0)) == len(members)
        donor_retired = False
        pending = list(members)
        if adopted and retire_donor:
            pending = []
            for order, member in enumerate(members):
                existing = get_session(donor_db, member)
                if existing and existing.get("end_reason") == "adopted_by_profile" and existing.get("archived"):
                    continue
                if not self._retire_donor_segment(donor_db, member):
                    pending = members[order:]
                    break
            for member in members:
                row = get_session(donor_db, member)
                if (row is None or row.get("end_reason") != "adopted_by_profile" or not row.get("archived")) and member not in pending:
                    pending.append(member)
            donor_retired = not pending
            if donor_retired:
                mark_complete(self.db, saved, donor_db)
        return {**result, "adopted": adopted, "donor_retired": donor_retired}

    def _retire_donor_segment(self, donor_db: Any, seg_id: str) -> bool:
        """Retire a copied member atomically while its full history remains on target."""
        from ._adoption_records import find_in

        def retire(conn):
            with self.db._read_ctx() as target_conn:
                record = find_in(target_conn, seg_id, donor_db, self.db)
                member = next((m for m in record["manifest"]["members"] if m["id"] == seg_id), None) if record else None
                if member is None:
                    raise AdoptionConflict("Donor retirement requires the committed transfer member record")
                covered = _adoption_history_covers(conn, target_conn, seg_id)
                if not covered:
                    covered = _adoption_history_covers(conn, target_conn, seg_id, retired_witness=member)
                if not covered or not _adoption_accounting_covers(conn, target_conn, seg_id):
                    logger.warning("adoption divergence at retirement for segment %s; leaving donor unretired", seg_id)
                    return False
            if not _retire_member(donor_db, conn, seg_id):
                return False
            key = DONOR_TRANSFER_PREFIX + seg_id
            marker = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
            if marker is not None and marker[0] != record["transfer_id"]:
                raise AdoptionConflict("Donor retirement conflicts with an existing transfer receipt")
            conn.execute("INSERT OR IGNORE INTO state_meta(key,value) VALUES (?,?)", (key, record["transfer_id"]))
            conn.execute("UPDATE sessions SET archived=1, auto_archived=0 WHERE id=?", (seg_id,))
            return True

        # Re-read the donor on the same write connection that commits its retirement.
        try:
            return bool(donor_db._execute_write(retire))
        except (AdoptionConflict, AuthorityRefusal, sqlite3.Error, OSError):
            logger.exception("failed to retire donor segment %s after adoption", seg_id)
            return False

    def _validate_adoption_store_pair(self, donor_path: Path) -> str | None:
        from .._state.lineage_fence import _managed_root_and_current, _same_file
        # Resolution errors are unverifiable stores, never standalone permission.
        donor = _managed_root_and_current(donor_path)
        target = _managed_root_and_current(self.db.db_path)
        if _same_file(Path(donor_path), self.db.db_path):
            return "Donor and target are the same database; choose distinct stores"
        if (donor is None) != (target is None):
            standalone = "donor" if donor is None else "target"
            return f"Adoption refuses a mixed managed/standalone pair (standalone {standalone}); use one managed home"
        if donor is not None and donor[0] != target[0]:
            return "Adoption requires both stores in the same managed home"
        return None
