"""Local persistence for Hermes OS architecture artifacts."""

import json
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class LocalRepository:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def save(self, collection: str, record_id: str, record: Any):
        directory = os.path.join(self.root_path, ".hermes", "records", collection)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _safe_name(record_id) + ".json")
        payload = _to_jsonable(record)
        if isinstance(payload, dict):
            payload.setdefault("id", record_id)
            payload.setdefault("stored_at", datetime.now(timezone.utc).isoformat())
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def get(self, collection: str, record_id: str):
        path = os.path.join(self.root_path, ".hermes", "records", collection, _safe_name(record_id) + ".json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list(self, collection: str):
        directory = os.path.join(self.root_path, ".hermes", "records", collection)
        if not os.path.isdir(directory):
            return []
        records = []
        for file_name in sorted(os.listdir(directory)):
            if not file_name.endswith(".json"):
                continue
            with open(os.path.join(directory, file_name), "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        return records

    def latest(self, collection: str):
        records = self.list(collection)
        if not records:
            return None
        return sorted(records, key=lambda item: item.get("stored_at", ""))[-1]


class SQLiteRepository:
    """SQLite-backed repository for Hermes OS source-of-truth records.

    ``root_or_db_path`` may be either a project directory or an explicit
    ``.sqlite``/``.db`` path. Directory input stores records under
    ``<root>/.hermes/hermes-os.sqlite3``.
    """

    def __init__(self, root_or_db_path: str):
        self.db_path = _resolve_db_path(root_or_db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema()

    def save(self, collection: str, record_id: str, record: Any):
        now = datetime.now(timezone.utc).isoformat()
        payload = _to_jsonable(record)
        if isinstance(payload, dict):
            payload.setdefault("id", record_id)
            payload.setdefault("stored_at", now)
        body = json.dumps(payload, indent=2, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                insert into hermes_os_records(collection, record_id, stored_at, payload)
                values (?, ?, ?, ?)
                on conflict(collection, record_id) do update set
                    stored_at = excluded.stored_at,
                    payload = excluded.payload
                """,
                (collection, _safe_name(record_id), now, body),
            )
        return self.db_path

    def get(self, collection: str, record_id: str):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                select payload from hermes_os_records
                where collection = ? and record_id = ?
                """,
                (collection, _safe_name(record_id)),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                select payload from hermes_os_records
                where collection = ?
                order by stored_at asc, record_id asc
                """,
                (collection,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest(self, collection: str):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                select payload from hermes_os_records
                where collection = ?
                order by stored_at desc, record_id desc
                limit 1
                """,
                (collection,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("pragma journal_mode=wal")
            conn.execute(
                """
                create table if not exists hermes_os_meta (
                    key text primary key,
                    value text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists hermes_os_records (
                    collection text not null,
                    record_id text not null,
                    stored_at text not null,
                    payload text not null,
                    primary key (collection, record_id)
                )
                """
            )
            conn.execute(
                """
                create index if not exists idx_hermes_os_records_collection_stored_at
                on hermes_os_records(collection, stored_at)
                """
            )
            conn.execute(
                """
                insert into hermes_os_meta(key, value)
                values ('schema_version', '1')
                on conflict(key) do nothing
                """
            )

    def schema_version(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "select value from hermes_os_meta where key = 'schema_version'"
            ).fetchone()
        return int(row[0]) if row else 0

    def migrate(self, target_version: int = 1):
        current = self.schema_version()
        if current > target_version:
            raise ValueError("Database schema is newer than this Hermes OS build")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                insert into hermes_os_meta(key, value)
                values ('schema_version', ?)
                on conflict(key) do update set value = excluded.value
                """,
                (str(target_version),),
            )
        return {"from": current, "to": target_version, "db_path": self.db_path}

    def import_from_local(self, local: LocalRepository):
        imported = 0
        records_root = os.path.join(local.root_path, ".hermes", "records")
        if not os.path.isdir(records_root):
            return {"imported": 0}
        for collection in sorted(os.listdir(records_root)):
            for record in local.list(collection):
                record_id = str(record.get("id") or record.get("project_id") or imported)
                self.save(collection, record_id, record)
                imported += 1
        return {"imported": imported}

    def export_bundle(self, path: str):
        collections = {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "select collection, payload from hermes_os_records order by collection, record_id"
            ).fetchall()
        for collection, payload in rows:
            collections.setdefault(collection, []).append(json.loads(payload))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": self.schema_version(),
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "collections": collections,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return path

    def integrity_check(self):
        issues = []
        seen = set()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "select collection, record_id, payload from hermes_os_records"
            ).fetchall()
        for collection, record_id, payload in rows:
            key = (collection, record_id)
            if key in seen:
                issues.append({"type": "duplicate-id", "collection": collection, "record_id": record_id})
            seen.add(key)
            try:
                json.loads(payload)
            except json.JSONDecodeError as exc:
                issues.append({"type": "malformed-payload", "collection": collection, "record_id": record_id, "error": str(exc)})
        return {"ok": not issues, "issues": issues, "checked": len(rows)}


def persist_review_report(repository: LocalRepository, report):
    return repository.save("review-reports", report.project_id, report)


def persist_grill_me_session(repository: LocalRepository, session):
    return repository.save("grill-me-sessions", session.project_id, session)


def persist_decision(repository: LocalRepository, decision_id: str, decision: Dict[str, Any]):
    return repository.save("decisions", decision_id, decision)


def persist_approval(repository: LocalRepository, approval_id: str, approval: Dict[str, Any]):
    return repository.save("approvals", approval_id, approval)


def persist_agent_artifact(repository: LocalRepository, artifact_id: str, artifact: Dict[str, Any]):
    return repository.save("agent-artifacts", artifact_id, artifact)


def persist_work_graph(repository: LocalRepository, project_id: str, graph: Any):
    return repository.save("work-graphs", project_id, graph)


def persist_score_history(repository: LocalRepository, record_id: str, score_record: Dict[str, Any]):
    return repository.save("score-history", record_id, score_record)


def persist_runtime_usage(repository: LocalRepository, usage_id: str, usage_record: Dict[str, Any]):
    return repository.save("runtime-usage", usage_id, usage_record)


def persist_runtime_audit(repository: LocalRepository, audit_id: str, audit_record: Dict[str, Any]):
    return repository.save("runtime-audits", audit_id, audit_record)


def persist_task_definitions(repository: LocalRepository, project_id: str, tasks: Any):
    return repository.save("tasks", project_id, {"project_id": project_id, "tasks": _to_jsonable(tasks)})


def _to_jsonable(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _safe_name(value: str):
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in str(value))


def _resolve_db_path(root_or_db_path: str):
    value = root_or_db_path or os.getcwd()
    if value.endswith((".sqlite", ".sqlite3", ".db")):
        return os.path.abspath(value)
    return os.path.join(os.path.abspath(value), ".hermes", "hermes-os.sqlite3")
