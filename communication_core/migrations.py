"""Read-only legacy migrations into Communication Core."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import DatabaseMissingError, ScopeViolationError
from .repository import CommunicationRepository, json_text, new_id, stable_id, utc_now


def _open_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise DatabaseMissingError("legacy Facebook database does not exist")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


class FacebookMigrationBridge:
    """Migrate verified local Facebook CRM rows without opening a browser."""

    source_system = "facebook_crm_v2"

    def __init__(self, repository: CommunicationRepository) -> None:
        self.repository = repository

    @staticmethod
    def source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def migrate(self, source_path: str | Path, connected_account_id: str) -> dict[str, Any]:
        source_path = Path(source_path)
        digest = self.source_hash(source_path)
        account = self.repository.get_account(connected_account_id)
        if account is None or account["provider"] != "facebook":
            raise ScopeViolationError("legacy Facebook source requires one exact Facebook account")
        run_id = stable_id("migration", self.source_system, connected_account_id, digest)
        with self.repository.read_connection() as target:
            existing = target.execute(
                """SELECT * FROM migration_runs WHERE id = ?
                   AND connected_account_id = ? AND status = 'succeeded'""",
                (run_id, connected_account_id),
            ).fetchone()
        if existing is not None:
            return {
                "run_id": run_id,
                "status": "succeeded",
                "idempotent_replay": True,
                "counts": json.loads(existing["counts_json"]),
                "reconciliation": json.loads(existing["reconciliation_json"]),
            }

        started_at = utc_now()
        with self.repository.transaction() as target:
            target.execute(
                """INSERT INTO migration_runs(
                       id, source_system, connected_account_id, source_hash,
                       status, started_at
                   ) VALUES (?, ?, ?, ?, 'running', ?)
                   ON CONFLICT(source_system, source_hash, connected_account_id) DO UPDATE SET
                       status = 'running', counts_json = '{}',
                       reconciliation_json = '{}', started_at = excluded.started_at,
                       finished_at = NULL""",
                (run_id, self.source_system, connected_account_id, digest, started_at),
            )

        counts = {"people": 0, "identities": 0, "conversations": 0, "messages": 0, "events": 0, "legacy_records": 0}
        source_counts = {"friends": 0, "messages": 0, "interactions": 0, "legacy_records": 0}
        try:
            source = _open_readonly(source_path)
            try:
                friends = source.execute(
                    """SELECT id, canonical_key, name, profile_url, thread_url,
                              added_at, updated_at FROM friends ORDER BY id"""
                ).fetchall()
                source_counts["friends"] = len(friends)
                for friend in friends:
                    keys = self._migrate_friend(
                        friend, connected_account_id, digest
                    )
                    counts["people"] += int(keys["person_created"])
                    counts["identities"] += 1
                    counts["conversations"] += 1
                messages = source.execute(
                    """SELECT id, friend_id, message_key, message_id, sender_name,
                              COALESCE(message_text, text, '') AS body,
                              COALESCE(sent_at, timestamp, created_at) AS sent_at,
                              direction, source_system, source_record_id, created_at
                       FROM messages ORDER BY friend_id, sent_at, id"""
                ).fetchall()
                source_counts["messages"] = len(messages)
                for message in messages:
                    if self._migrate_message(message, connected_account_id, digest):
                        counts["messages"] += 1
                interactions = source.execute(
                    """SELECT id, friend_id, interaction_key, type, details,
                              interacted_at, source_system, source_record_id
                       FROM interactions ORDER BY friend_id, interacted_at, id"""
                ).fetchall()
                source_counts["interactions"] = len(interactions)
                for event in interactions:
                    self._migrate_interaction(event, connected_account_id, digest)
                    counts["events"] += 1
                legacy_count = self._archive_legacy_state(
                    source, connected_account_id, digest
                )
                source_counts["legacy_records"] = legacy_count
                counts["legacy_records"] = legacy_count
            finally:
                source.close()
            reconciliation = self.reconcile(connected_account_id, digest, source_counts)
            with self.repository.transaction() as target:
                target.execute(
                    """UPDATE migration_runs SET status = 'succeeded', counts_json = ?,
                           reconciliation_json = ?, finished_at = ? WHERE id = ?""",
                    (json_text(counts), json_text(reconciliation), utc_now(), run_id),
                )
            return {
                "run_id": run_id,
                "status": "succeeded",
                "idempotent_replay": False,
                "counts": counts,
                "reconciliation": reconciliation,
            }
        except BaseException:
            with self.repository.transaction() as target:
                target.execute(
                    "UPDATE migration_runs SET status = 'failed', finished_at = ? WHERE id = ?",
                    (utc_now(), run_id),
                )
            raise

    def _mapping(
        self,
        account_id: str,
        entity_type: str,
        legacy_id: str,
        canonical_id: str,
        digest: str,
    ) -> None:
        with self.repository.transaction() as target:
            target.execute(
                """INSERT INTO legacy_id_mappings VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_system, connected_account_id, entity_type, legacy_id)
                   DO UPDATE SET
                       source_hash = excluded.source_hash,
                       migrated_at = excluded.migrated_at""",
                (
                    self.source_system,
                    account_id,
                    entity_type,
                    legacy_id,
                    canonical_id,
                    digest,
                    utc_now(),
                ),
            )

    def _migrate_friend(
        self, friend: sqlite3.Row, account_id: str, digest: str
    ) -> dict[str, Any]:
        legacy_id = str(friend["id"])
        external_id = str(friend["canonical_key"] or friend["profile_url"] or legacy_id)
        person_id = stable_id("person", self.source_system, account_id, legacy_id)
        person_created = self.repository.get_person(person_id) is None
        if person_created:
            self.repository.create_person(friend["name"], person_id=person_id)
        identity, endpoint = self.repository.upsert_identity(
            connected_account_id=account_id,
            external_id=external_id,
            display_name=friend["name"],
            profile_ref=friend["profile_url"],
            person_id=person_id,
            provenance={"source": self.source_system, "legacy_id": legacy_id, "source_hash": digest},
            observed_at=friend["updated_at"] or friend["added_at"] or utc_now(),
        )
        conversation_external = str(friend["thread_url"] or f"friend:{legacy_id}")
        conversation = self.repository.upsert_conversation(
            connected_account_id=account_id,
            endpoint_id=endpoint["id"],
            external_id=conversation_external,
            kind="direct",
            title=friend["name"],
            provenance={"source": self.source_system, "legacy_friend_id": legacy_id, "source_hash": digest},
            observed_at=friend["updated_at"] or utc_now(),
        )
        for entity_type, canonical_id in (
            ("person", person_id),
            ("identity", identity["id"]),
            ("endpoint", endpoint["id"]),
            ("conversation", conversation["id"]),
        ):
            self._mapping(account_id, entity_type, legacy_id, canonical_id, digest)
        return {"person_created": person_created, "identity": identity, "endpoint": endpoint, "conversation": conversation}

    def _mapped(self, account_id: str, entity_type: str, legacy_id: str) -> str:
        with self.repository.read_connection() as target:
            row = target.execute(
                """SELECT canonical_id FROM legacy_id_mappings
                   WHERE source_system = ? AND connected_account_id = ?
                   AND entity_type = ? AND legacy_id = ?""",
                (self.source_system, account_id, entity_type, legacy_id),
            ).fetchone()
        if row is None:
            raise ScopeViolationError("legacy row references an unmapped Facebook contact")
        return str(row[0])

    def _migrate_message(self, message: sqlite3.Row, account_id: str, digest: str) -> bool:
        friend_id = str(message["friend_id"])
        identity_id = self._mapped(account_id, "identity", friend_id)
        endpoint_id = self._mapped(account_id, "endpoint", friend_id)
        conversation_id = self._mapped(account_id, "conversation", friend_id)
        external_id = str(message["message_id"] or message["message_key"] or message["id"])
        row, inserted = self.repository.upsert_message(
            connected_account_id=account_id,
            endpoint_id=endpoint_id,
            conversation_id=conversation_id,
            external_id=external_id,
            direction={"sent": "outgoing", "received": "incoming"}.get(message["direction"], "system"),
            body=message["body"],
            sent_at=message["sent_at"] or message["created_at"] or utc_now(),
            sender_identity_id=identity_id,
            provenance={
                "source": self.source_system,
                "legacy_id": str(message["id"]),
                "legacy_source_system": message["source_system"],
                "legacy_source_record_id": message["source_record_id"],
                "source_hash": digest,
            },
            observed_at=utc_now(),
        )
        self._mapping(account_id, "message", str(message["id"]), row["id"], digest)
        return inserted

    def _migrate_interaction(self, event: sqlite3.Row, account_id: str, digest: str) -> None:
        friend_id = str(event["friend_id"])
        person_id = self._mapped(account_id, "person", friend_id)
        endpoint_id = self._mapped(account_id, "endpoint", friend_id)
        row = self.repository.add_contact_event(
            person_id=person_id,
            connected_account_id=account_id,
            endpoint_id=endpoint_id,
            event_type=f"facebook.{event['type']}",
            external_id=str(event["interaction_key"] or event["id"]),
            happened_at=event["interacted_at"],
            data={"details": event["details"]},
            provenance={
                "source": self.source_system,
                "legacy_id": str(event["id"]),
                "legacy_source_system": event["source_system"],
                "legacy_source_record_id": event["source_record_id"],
                "source_hash": digest,
            },
        )
        self._mapping(account_id, "event", str(event["id"]), row["id"], digest)

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone() is not None

    def _archive_legacy_state(
        self, source: sqlite3.Connection, account_id: str, digest: str
    ) -> int:
        """Preserve legacy state inertly; old approvals never become active."""
        tables = (
            "facebook_settings",
            "online_activity_log",
            "outreach_campaigns",
            "outreach_tasks",
            "birthday_wishes",
            "write_approvals",
            "facebook_write_outbox",
            "facebook_write_outbox_events",
            "migration_runs",
            "migration_reconciliation",
        )
        archived = 0
        for table in tables:
            if not self._table_exists(source, table):
                continue
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            for index, row in enumerate(rows):
                payload = dict(row)
                legacy_value = (
                    payload.get("id")
                    or payload.get("token_hash")
                    or payload.get("key")
                    or f"row:{index}"
                )
                legacy_id = str(legacy_value)
                person_id = None
                friend_id = payload.get("friend_id")
                if friend_id is not None:
                    try:
                        person_id = self._mapped(account_id, "person", str(friend_id))
                    except ScopeViolationError:
                        person_id = None
                if table == "birthday_wishes" and payload.get("profile_url"):
                    friend = source.execute(
                        """SELECT id FROM friends
                           WHERE lower(rtrim(trim(profile_url), '/')) =
                                 lower(rtrim(trim(?), '/'))""",
                        (payload["profile_url"],),
                    ).fetchall()
                    if len(friend) == 1:
                        person_id = self._mapped(
                            account_id, "person", str(friend[0]["id"])
                        )
                record_id = stable_id(
                    "legacyrecord", self.source_system, account_id, table, legacy_id
                )
                with self.repository.transaction() as target:
                    target.execute(
                        """INSERT INTO legacy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(source_system, connected_account_id,
                                       entity_type, legacy_id) DO UPDATE SET
                               person_id = excluded.person_id,
                               payload_json = excluded.payload_json,
                               source_hash = excluded.source_hash,
                               migrated_at = excluded.migrated_at""",
                        (
                            record_id,
                            self.source_system,
                            account_id,
                            table,
                            legacy_id,
                            person_id,
                            json_text(payload),
                            digest,
                            utc_now(),
                        ),
                    )
                archived += 1
        return archived

    def reconcile(
        self, account_id: str, digest: str, source_counts: dict[str, int]
    ) -> dict[str, Any]:
        expected = {
            "person": source_counts["friends"],
            "identity": source_counts["friends"],
            "endpoint": source_counts["friends"],
            "conversation": source_counts["friends"],
            "message": source_counts["messages"],
            "event": source_counts["interactions"],
        }
        with self.repository.read_connection() as target:
            actual = {
                row["entity_type"]: row["count"]
                for row in target.execute(
                    """SELECT entity_type, COUNT(*) AS count FROM legacy_id_mappings
                       WHERE source_system = ? AND connected_account_id = ?
                       AND source_hash = ? GROUP BY entity_type""",
                    (self.source_system, account_id, digest),
                ).fetchall()
            }
        mismatches = {
            name: {"expected": count, "actual": actual.get(name, 0)}
            for name, count in expected.items()
            if actual.get(name, 0) != count
        }
        with self.repository.read_connection() as target:
            archived = target.execute(
                """SELECT COUNT(*) FROM legacy_records WHERE source_system = ?
                   AND connected_account_id = ? AND source_hash = ?""",
                (self.source_system, account_id, digest),
            ).fetchone()[0]
        if archived != source_counts["legacy_records"]:
            mismatches["legacy_records"] = {
                "expected": source_counts["legacy_records"],
                "actual": archived,
            }
        return {
            "expected": {**expected, "legacy_records": source_counts["legacy_records"]},
            "actual": {**actual, "legacy_records": archived},
            "mismatches": mismatches,
            "ok": not mismatches,
        }

    def rollback(self, run_id: str) -> dict[str, Any]:
        with self.repository.transaction() as target:
            run = target.execute(
                "SELECT * FROM migration_runs WHERE id = ? AND source_system = ?",
                (run_id, self.source_system),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            digest = run["source_hash"]
            account_id = run["connected_account_id"]
            removed_legacy = target.execute(
                """DELETE FROM legacy_records WHERE source_system = ?
                   AND connected_account_id = ? AND source_hash = ?""",
                (self.source_system, account_id, digest),
            ).rowcount
            mappings = target.execute(
                """SELECT entity_type, canonical_id FROM legacy_id_mappings
                   WHERE source_system = ? AND connected_account_id = ?
                   AND source_hash = ?""",
                (self.source_system, account_id, digest),
            ).fetchall()
            by_type: dict[str, list[str]] = {}
            for mapping in mappings:
                by_type.setdefault(mapping["entity_type"], []).append(mapping["canonical_id"])
            table_order = (
                ("event", "contact_events"),
                ("message", "messages"),
                ("conversation", "conversations"),
                ("endpoint", "contact_endpoints"),
                ("identity", "platform_identities"),
            )
            removed: dict[str, int] = {}
            removed["legacy_records"] = removed_legacy
            for entity_type, table in table_order:
                ids = by_type.get(entity_type, [])
                count = 0
                for canonical_id in ids:
                    count += target.execute(
                        f"DELETE FROM {table} WHERE id = ?", (canonical_id,)
                    ).rowcount
                removed[entity_type] = count
            person_ids = by_type.get("person", [])
            for person_id in person_ids:
                target.execute(
                    "DELETE FROM communication_journeys WHERE person_id = ?", (person_id,)
                )
            removed["person"] = sum(
                target.execute("DELETE FROM persons WHERE id = ?", (person_id,)).rowcount
                for person_id in person_ids
            )
            target.execute(
                """DELETE FROM legacy_id_mappings WHERE source_system = ?
                   AND connected_account_id = ? AND source_hash = ?""",
                (self.source_system, account_id, digest),
            )
            target.execute(
                "UPDATE migration_runs SET status = 'rolled_back', finished_at = ? WHERE id = ?",
                (utc_now(), run_id),
            )
        return {"run_id": run_id, "status": "rolled_back", "removed": removed}
