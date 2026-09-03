"""Profile-scoped crash-safe SQLite state for Collective Wisdom."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home


SCHEMA_VERSION = 9


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def filesystem_identity(path: Path) -> str | None:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}" if stat.st_ino else None


class WisdomStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (get_hermes_home() / "wisdom")
        self.path = self.root / "wisdom.db"
        self._lock = threading.RLock()
        self._prepare()

    def _prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_skill (
                  id TEXT PRIMARY KEY,
                  canonical_path TEXT NOT NULL,
                  fs_identity TEXT,
                  current_hash TEXT,
                  source_kind TEXT NOT NULL,
                  deleted_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS local_skill_fs_identity
                  ON local_skill(fs_identity) WHERE fs_identity IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS local_skill_active_path
                  ON local_skill(canonical_path) WHERE deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS snapshot (
                  skill_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  captured_at TEXT NOT NULL,
                  tree_json TEXT NOT NULL DEFAULT '{}',
                  skill_text TEXT,
                  PRIMARY KEY(skill_id, content_hash),
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS candidate (
                  skill_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  qualification TEXT NOT NULL,
                  state TEXT NOT NULL,
                  suggested_at TEXT,
                  dismissed_at TEXT,
                  PRIMARY KEY(skill_id, content_hash, qualification),
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS local_draft (
                  id TEXT PRIMARY KEY,
                  skill_id TEXT NOT NULL,
                  source_hash TEXT NOT NULL,
                  overlay_path TEXT NOT NULL,
                  draft_commit TEXT,
                  server_revision TEXT,
                  state TEXT NOT NULL,
                  description TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  description_hash TEXT NOT NULL,
                  manifest_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS review_receipt (
                  id TEXT PRIMARY KEY,
                  draft_id TEXT NOT NULL UNIQUE,
                  server_revision TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  description_hash TEXT NOT NULL,
                  manifest_hash TEXT NOT NULL,
                  reviewed_at TEXT NOT NULL,
                  consumed_at TEXT,
                  FOREIGN KEY(draft_id) REFERENCES local_draft(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS installation_identity (
                  singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                  installation_id TEXT NOT NULL,
                  verified_org_id TEXT,
                  verified_at TEXT,
                  disclosure_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS managed_install (
                  skill_id TEXT PRIMARY KEY,
                  org_id TEXT NOT NULL,
                  slug TEXT NOT NULL,
                  version INTEGER NOT NULL CHECK(version > 0),
                  content_hash TEXT NOT NULL,
                  baseline_json TEXT NOT NULL,
                  target_path TEXT NOT NULL UNIQUE,
                  update_mode TEXT NOT NULL,
                  state TEXT NOT NULL,
                  installed_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operation_journal (
                  id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  phase TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  state TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(kind, entity_id, state)
                );
                CREATE TABLE IF NOT EXISTS usage_day (
                  skill_id TEXT NOT NULL,
                  day_local TEXT NOT NULL,
                  timezone_name TEXT NOT NULL,
                  use_count INTEGER NOT NULL CHECK(use_count >= 0),
                  PRIMARY KEY(skill_id, timezone_name, day_local),
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS refinement (
                  skill_id TEXT NOT NULL,
                  from_hash TEXT NOT NULL,
                  to_hash TEXT NOT NULL,
                  classification TEXT NOT NULL,
                  structural_json TEXT NOT NULL,
                  recorded_at TEXT NOT NULL,
                  PRIMARY KEY(skill_id, to_hash),
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS stability_job (
                  skill_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  due_at TEXT NOT NULL,
                  state TEXT NOT NULL,
                  evaluated_at TEXT,
                  session_id TEXT,
                  task_id TEXT,
                  PRIMARY KEY(skill_id, content_hash),
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS local_event (
                  id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  session_id TEXT,
                  task_id TEXT,
                  skill_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  qualification TEXT NOT NULL,
                  organization_id TEXT,
                  qualification_sequence INTEGER,
                  payload_json TEXT NOT NULL,
                  state TEXT NOT NULL,
                  telegram_delivered_at TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS local_event_delivery (
                  event_id TEXT NOT NULL,
                  surface TEXT NOT NULL,
                  delivered_at TEXT NOT NULL,
                  PRIMARY KEY(event_id, surface),
                  FOREIGN KEY(event_id) REFERENCES local_event(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS wisdom_organization (
                  organization_id TEXT PRIMARY KEY,
                  display_name TEXT,
                  resolved INTEGER NOT NULL DEFAULT 0,
                  checked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feed_state (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                  cursor TEXT,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feed_event (
                  event_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  skill_id TEXT NOT NULL,
                  version INTEGER,
                  installation_id TEXT,
                  update_mode TEXT,
                  payload_json TEXT NOT NULL,
                  cadence TEXT NOT NULL,
                  due_at TEXT NOT NULL,
                  local_seen_at TEXT,
                  telegram_delivered_at TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feed_event_delivery (
                  event_id TEXT NOT NULL,
                  surface TEXT NOT NULL,
                  delivered_at TEXT NOT NULL,
                  PRIMARY KEY(event_id, surface),
                  FOREIGN KEY(event_id) REFERENCES feed_event(event_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS operation_lock (
                  entity_id TEXT PRIMARY KEY,
                  owner TEXT NOT NULL,
                  expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS professionalism_review (
                  id TEXT PRIMARY KEY,
                  skill_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  author_description_hash TEXT NOT NULL,
                  package_json TEXT NOT NULL,
                  author_description TEXT NOT NULL,
                  state TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  available_at TEXT NOT NULL,
                  lease_owner TEXT,
                  lease_expires_at TEXT,
                  result_json TEXT,
                  last_error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(skill_id, content_hash, author_description_hash)
                );
                CREATE INDEX IF NOT EXISTS professionalism_review_queue
                  ON professionalism_review(state, available_at);
                """
            )
            columns = {
                str(row[1])
                for row in db.execute("PRAGMA table_info(snapshot)").fetchall()
            }
            if "tree_json" not in columns:
                db.execute(
                    "ALTER TABLE snapshot ADD COLUMN tree_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "skill_text" not in columns:
                db.execute("ALTER TABLE snapshot ADD COLUMN skill_text TEXT")
            stability_columns = {
                str(row[1])
                for row in db.execute("PRAGMA table_info(stability_job)").fetchall()
            }
            if "session_id" not in stability_columns:
                db.execute("ALTER TABLE stability_job ADD COLUMN session_id TEXT")
            if "task_id" not in stability_columns:
                db.execute("ALTER TABLE stability_job ADD COLUMN task_id TEXT")
            usage_columns = {
                str(row[1])
                for row in db.execute("PRAGMA table_info(usage_day)").fetchall()
            }
            if "day_local" not in usage_columns:
                # V4 recorded UTC calendar dates. Preserve those private
                # counters in a distinct UTC bucket so a configured local
                # timezone never combines them with business-day evidence.
                db.executescript(
                    """
                    ALTER TABLE usage_day RENAME TO usage_day_v4;
                    CREATE TABLE usage_day (
                      skill_id TEXT NOT NULL,
                      day_local TEXT NOT NULL,
                      timezone_name TEXT NOT NULL,
                      use_count INTEGER NOT NULL CHECK(use_count >= 0),
                      PRIMARY KEY(skill_id, timezone_name, day_local),
                      FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                    );
                    INSERT INTO usage_day(skill_id,day_local,timezone_name,use_count)
                      SELECT skill_id,day_utc,'UTC',use_count FROM usage_day_v4;
                    DROP TABLE usage_day_v4;
                    """
                )
            candidate_sql = str(
                db.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidate'"
                ).fetchone()[0]
            ).replace("\n", " ")
            if (
                "PRIMARY KEY(skill_id, content_hash, qualification)"
                not in candidate_sql
            ):
                db.executescript(
                    """
                    ALTER TABLE candidate RENAME TO candidate_v1;
                    CREATE TABLE candidate (
                      skill_id TEXT NOT NULL,
                      content_hash TEXT NOT NULL,
                      qualification TEXT NOT NULL,
                      state TEXT NOT NULL,
                      suggested_at TEXT,
                      dismissed_at TEXT,
                      PRIMARY KEY(skill_id, content_hash, qualification),
                      FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                    );
                    INSERT INTO candidate
                      SELECT skill_id,content_hash,qualification,state,suggested_at,dismissed_at
                      FROM candidate_v1;
                    DROP TABLE candidate_v1;
                    """
                )
            event_columns = {
                str(row[1])
                for row in db.execute("PRAGMA table_info(local_event)").fetchall()
            }
            if "qualification" not in event_columns:
                db.executescript(
                    """
                    ALTER TABLE local_event RENAME TO local_event_v2;
                    CREATE TABLE local_event (
                      id TEXT PRIMARY KEY,
                      kind TEXT NOT NULL,
                      session_id TEXT,
                      task_id TEXT,
                      skill_id TEXT NOT NULL,
                      content_hash TEXT NOT NULL,
                      qualification TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      state TEXT NOT NULL,
                      telegram_delivered_at TEXT,
                      created_at TEXT NOT NULL,
                      UNIQUE(kind, skill_id, content_hash, qualification),
                      FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                    );
                    INSERT INTO local_event
                      SELECT id,kind,session_id,task_id,skill_id,content_hash,
                        COALESCE(json_extract(payload_json,'$.qualification'),'legacy'),
                        payload_json,state,NULL,created_at
                      FROM local_event_v2;
                    DROP TABLE local_event_v2;
                    """
                )
                event_columns.add("telegram_delivered_at")
            if "telegram_delivered_at" not in event_columns:
                db.execute(
                    "ALTER TABLE local_event ADD COLUMN telegram_delivered_at TEXT"
                )
            if "organization_id" not in event_columns:
                db.execute("ALTER TABLE local_event ADD COLUMN organization_id TEXT")
            if "qualification_sequence" not in event_columns:
                db.execute(
                    "ALTER TABLE local_event ADD COLUMN qualification_sequence INTEGER"
                )
            organization_columns = {
                str(row[1])
                for row in db.execute(
                    "PRAGMA table_info(wisdom_organization)"
                ).fetchall()
            }
            if "resolved" not in organization_columns:
                db.execute(
                    "ALTER TABLE wisdom_organization "
                    "ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0"
                )
            active_org = db.execute(
                "SELECT verified_org_id FROM installation_identity WHERE singleton=1"
            ).fetchone()
            active_org_id = str(active_org[0]) if active_org and active_org[0] else None
            if active_org_id:
                sequence_row = db.execute(
                    "SELECT COALESCE(MAX(qualification_sequence),0) FROM local_event "
                    "WHERE kind='wisdom.candidate' AND organization_id=?",
                    (active_org_id,),
                ).fetchone()
                sequence = int(sequence_row[0]) if sequence_row else 0
                legacy_events = db.execute(
                    "SELECT id FROM local_event WHERE kind='wisdom.candidate' "
                    "AND qualification_sequence IS NULL ORDER BY created_at,id"
                ).fetchall()
                for event in legacy_events:
                    sequence += 1
                    db.execute(
                        "UPDATE local_event SET organization_id=?,qualification_sequence=? "
                        "WHERE id=?",
                        (active_org_id, sequence, event["id"]),
                    )
            event_sql = str(
                db.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' "
                    "AND name='local_event'"
                ).fetchone()[0]
            ).replace("\n", " ")
            if "UNIQUE(kind, skill_id, content_hash, qualification)" in event_sql:
                db.executescript(
                    """
                    CREATE TEMP TABLE local_event_delivery_v9 AS
                      SELECT * FROM local_event_delivery;
                    CREATE TABLE local_event_v9 (
                      id TEXT PRIMARY KEY,
                      kind TEXT NOT NULL,
                      session_id TEXT,
                      task_id TEXT,
                      skill_id TEXT NOT NULL,
                      content_hash TEXT NOT NULL,
                      qualification TEXT NOT NULL,
                      organization_id TEXT,
                      qualification_sequence INTEGER,
                      payload_json TEXT NOT NULL,
                      state TEXT NOT NULL,
                      telegram_delivered_at TEXT,
                      created_at TEXT NOT NULL,
                      FOREIGN KEY(skill_id) REFERENCES local_skill(id) ON DELETE CASCADE
                    );
                    INSERT INTO local_event_v9
                      SELECT id,kind,session_id,task_id,skill_id,content_hash,
                        qualification,organization_id,qualification_sequence,
                        payload_json,state,telegram_delivered_at,created_at
                      FROM local_event;
                    DROP TABLE local_event;
                    ALTER TABLE local_event_v9 RENAME TO local_event;
                    INSERT OR IGNORE INTO local_event_delivery
                      SELECT * FROM local_event_delivery_v9;
                    DROP TABLE local_event_delivery_v9;
                    """
                )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS local_event_org_identity "
                "ON local_event(kind,COALESCE(organization_id,''),skill_id,"
                "content_hash,qualification)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS local_event_org_qualification_sequence "
                "ON local_event(organization_id,qualification_sequence) "
                "WHERE kind='wisdom.candidate' AND qualification_sequence IS NOT NULL"
            )
            # Seed the transport-neutral delivery ledger from pre-v7 Telegram
            # timestamps. Keeping the legacy columns during the migration
            # window preserves older readers while Slack gains independent
            # delivery state.
            db.execute(
                "INSERT OR IGNORE INTO local_event_delivery(event_id,surface,delivered_at) "
                "SELECT id,'telegram',telegram_delivered_at FROM local_event "
                "WHERE telegram_delivered_at IS NOT NULL"
            )
            db.execute(
                "INSERT OR IGNORE INTO feed_event_delivery(event_id,surface,delivered_at) "
                "SELECT event_id,'telegram',telegram_delivered_at FROM feed_event "
                "WHERE telegram_delivered_at IS NOT NULL"
            )
            journal_sql = str(
                db.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='operation_journal'"
                ).fetchone()[0]
            ).replace("\n", " ")
            if "UNIQUE(kind, entity_id, state)" in journal_sql:
                db.executescript(
                    """
                    ALTER TABLE operation_journal RENAME TO operation_journal_v2;
                    CREATE TABLE operation_journal (
                      id TEXT PRIMARY KEY,
                      kind TEXT NOT NULL,
                      entity_id TEXT NOT NULL,
                      phase TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      state TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    INSERT INTO operation_journal SELECT * FROM operation_journal_v2;
                    DROP TABLE operation_journal_v2;
                    """
                )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS operation_journal_one_pending "
                "ON operation_journal(kind,entity_id) WHERE state='pending'"
            )
            db.execute(
                "INSERT INTO schema_meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
        self._secure_database_files()

    def _secure_database_files(self) -> None:
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            self._secure_database_files()
            try:
                yield db
                if db.in_transaction:
                    db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
            finally:
                db.close()
                self._secure_database_files()

    def register_skill(
        self,
        path: Path,
        *,
        content_hash: str | None,
        source_kind: str,
        tree: dict[str, str] | None = None,
        snapshot_text: str | None = None,
    ) -> str:
        resolved = path.resolve()
        fs_identity = filesystem_identity(resolved)
        now = utc_now()
        with self.transaction() as db:
            row = db.execute(
                "SELECT id FROM local_skill WHERE canonical_path=? AND deleted_at IS NULL",
                (str(resolved),),
            ).fetchone()
            if row is None and fs_identity:
                matches = db.execute(
                    "SELECT id FROM local_skill WHERE fs_identity=? AND deleted_at IS NULL",
                    (fs_identity,),
                ).fetchall()
                row = matches[0] if len(matches) == 1 else None
            if row is None and content_hash:
                matches = db.execute(
                    "SELECT id FROM local_skill WHERE current_hash=? AND canonical_path<>? AND deleted_at IS NOT NULL",
                    (content_hash, str(resolved)),
                ).fetchall()
                row = matches[0] if len(matches) == 1 else None
            skill_id = str(row["id"]) if row else str(uuid.uuid4())
            if row:
                db.execute(
                    "UPDATE local_skill SET canonical_path=?,fs_identity=?,current_hash=?,"
                    "source_kind=?,deleted_at=NULL,updated_at=? WHERE id=?",
                    (
                        str(resolved),
                        fs_identity,
                        content_hash,
                        source_kind,
                        now,
                        skill_id,
                    ),
                )
            else:
                db.execute(
                    "INSERT INTO local_skill VALUES(?,?,?,?,?,?,?,?)",
                    (
                        skill_id,
                        str(resolved),
                        fs_identity,
                        content_hash,
                        source_kind,
                        None,
                        now,
                        now,
                    ),
                )
            if content_hash:
                db.execute(
                    "INSERT INTO snapshot(skill_id,content_hash,captured_at,tree_json,skill_text) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(skill_id,content_hash) DO UPDATE SET "
                    "tree_json=excluded.tree_json,skill_text=COALESCE(snapshot.skill_text,excluded.skill_text)",
                    (
                        skill_id,
                        content_hash,
                        now,
                        json.dumps(tree or {}, sort_keys=True),
                        snapshot_text,
                    ),
                )
        return skill_id

    def local_skill(self, skill_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM local_skill WHERE id=?", (skill_id,)
            ).fetchone()
            return dict(row) if row else None

    def latest_snapshot(self, skill_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM snapshot WHERE skill_id=? ORDER BY captured_at DESC LIMIT 1",
                (skill_id,),
            ).fetchone()
            if not row:
                return None
            value = dict(row)
            value["tree"] = json.loads(value.pop("tree_json"))
            return value

    def record_usage_day(
        self,
        skill_id: str,
        day_local: str,
        *,
        timezone_name: str,
        retain_after: str,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO usage_day VALUES(?,?,?,1) "
                "ON CONFLICT(skill_id,timezone_name,day_local) "
                "DO UPDATE SET use_count=MIN(use_count+1,2147483647)",
                (skill_id, day_local, timezone_name),
            )
            db.execute(
                "DELETE FROM usage_day WHERE skill_id=? AND day_local<?",
                (skill_id, retain_after),
            )

    def usage_days(self, skill_id: str, *, since: str, timezone_name: str) -> list[str]:
        with self.transaction() as db:
            return [
                str(row[0])
                for row in db.execute(
                    "SELECT day_local FROM usage_day WHERE skill_id=? "
                    "AND timezone_name=? AND day_local>=? ORDER BY day_local",
                    (skill_id, timezone_name, since),
                ).fetchall()
            ]

    def record_refinement(
        self,
        skill_id: str,
        *,
        from_hash: str,
        to_hash: str,
        classification: str,
        structural: dict[str, Any],
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO refinement VALUES(?,?,?,?,?,?)",
                (
                    skill_id,
                    from_hash,
                    to_hash,
                    classification,
                    json.dumps(structural, sort_keys=True),
                    utc_now(),
                ),
            )

    def meaningful_refinement_count(self, skill_id: str, *, since: str) -> int:
        with self.transaction() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM refinement WHERE skill_id=? AND classification='meaningful' "
                "AND recorded_at>=?",
                (skill_id, since),
            ).fetchone()
            return int(row[0]) if row else 0

    def schedule_stability(
        self,
        skill_id: str,
        content_hash: str,
        due_at: str,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO stability_job(skill_id,content_hash,due_at,state,evaluated_at,session_id,task_id) "
                "VALUES(?,?,?,'pending',NULL,?,?) ON CONFLICT(skill_id,content_hash) "
                "DO UPDATE SET due_at=excluded.due_at,state='pending',evaluated_at=NULL,"
                "session_id=excluded.session_id,task_id=excluded.task_id",
                (skill_id, content_hash, due_at, session_id, task_id),
            )

    def due_stability_jobs(self, now: str) -> list[dict[str, Any]]:
        with self.transaction() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM stability_job WHERE state='pending' AND due_at<=? ORDER BY due_at",
                    (now,),
                ).fetchall()
            ]

    def finish_stability_job(self, skill_id: str, content_hash: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE stability_job SET state='done',evaluated_at=? WHERE skill_id=? AND content_hash=?",
                (utc_now(), skill_id, content_hash),
            )

    def emit_local_event(
        self,
        *,
        kind: str,
        skill_id: str,
        content_hash: str,
        payload: dict[str, Any],
        session_id: str | None,
        task_id: str | None,
        qualification: str,
    ) -> str | None:
        event_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as db:
            organization_id: str | None = None
            if kind == "wisdom.candidate":
                organization = db.execute(
                    "SELECT verified_org_id FROM installation_identity WHERE singleton=1"
                ).fetchone()
                organization_id = (
                    str(organization[0])
                    if organization is not None and organization[0]
                    else None
                )
            existing = db.execute(
                "SELECT 1 FROM local_event WHERE kind=? AND skill_id=? "
                "AND content_hash=? AND qualification=? "
                "AND COALESCE(organization_id,'')=COALESCE(?,'')",
                (kind, skill_id, content_hash, qualification, organization_id),
            ).fetchone()
            if existing is not None:
                return None
            cursor = db.execute(
                "INSERT OR IGNORE INTO candidate VALUES(?,?,?,'suggested',?,NULL)",
                (skill_id, content_hash, qualification, now),
            )
            if cursor.rowcount == 0:
                row = db.execute(
                    "SELECT state FROM candidate WHERE skill_id=? AND content_hash=? AND qualification=?",
                    (skill_id, content_hash, qualification),
                ).fetchone()
                if row and row["state"] == "dismissed":
                    return None
            qualification_sequence: int | None = None
            if kind == "wisdom.candidate":
                if organization_id is None:
                    sequence_row = db.execute(
                        "SELECT COALESCE(MAX(qualification_sequence),0) FROM local_event "
                        "WHERE kind='wisdom.candidate' AND organization_id IS NULL"
                    ).fetchone()
                else:
                    sequence_row = db.execute(
                        "SELECT COALESCE(MAX(qualification_sequence),0) FROM local_event "
                        "WHERE kind='wisdom.candidate' AND organization_id=?",
                        (organization_id,),
                    ).fetchone()
                qualification_sequence = int(sequence_row[0]) + 1
            cursor = db.execute(
                "INSERT OR IGNORE INTO local_event("
                "id,kind,session_id,task_id,skill_id,content_hash,qualification,"
                "organization_id,qualification_sequence,payload_json,state,"
                "telegram_delivered_at,created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,'unread',NULL,?)",
                (
                    event_id,
                    kind,
                    session_id,
                    task_id,
                    skill_id,
                    content_hash,
                    qualification,
                    organization_id,
                    qualification_sequence,
                    json.dumps(payload, sort_keys=True),
                    now,
                ),
            )
            return event_id if cursor.rowcount else None

    def local_events(
        self, *, kind: str | None = None, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT e.* FROM local_event e WHERE e.state='unread' "
            "AND NOT EXISTS ("
            "SELECT 1 FROM local_draft d "
            "WHERE d.skill_id=e.skill_id AND d.source_hash=e.content_hash "
            "AND d.state IN ('pending_moderation','published')"
            ")"
        )
        params: list[str] = []
        if kind:
            query += " AND e.kind=?"
            params.append(kind)
        if kind == "wisdom.candidate":
            active_org_id = self.active_org_id()
            if active_org_id is None:
                query += " AND e.organization_id IS NULL"
            else:
                query += " AND e.organization_id=?"
                params.append(active_org_id)
        if session_id:
            query += " AND e.session_id=?"
            params.append(session_id)
        query += " ORDER BY e.created_at DESC"
        with self.transaction() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

    def local_event(self, event_id: str) -> dict[str, Any] | None:
        """Return one local event without changing its delivery or consent state."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM local_event WHERE id=?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        return value

    def enqueue_professionalism_review(
        self,
        *,
        skill_id: str,
        content_hash: str,
        author_description_hash: str,
        package: list[dict[str, str]],
        author_description: str,
    ) -> dict[str, Any]:
        """Persist one exact hash-bound review job without resetting completed work."""

        now = utc_now()
        review_id = str(uuid.uuid4())
        package_json = json.dumps(package, sort_keys=True, separators=(",", ":"))
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO professionalism_review("
                "id,skill_id,content_hash,author_description_hash,package_json,"
                "author_description,state,attempts,available_at,lease_owner,"
                "lease_expires_at,result_json,last_error,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,'pending',0,?,NULL,NULL,NULL,NULL,?,?)",
                (
                    review_id,
                    skill_id,
                    content_hash,
                    author_description_hash,
                    package_json,
                    author_description,
                    now,
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM professionalism_review WHERE skill_id=? "
                "AND content_hash=? AND author_description_hash=?",
                (skill_id, content_hash, author_description_hash),
            ).fetchone()
        if row is None:  # pragma: no cover - insert/select invariant
            raise RuntimeError("professionalism review was not persisted")
        return self._decode_professionalism_review(dict(row))

    def professionalism_review(
        self,
        *,
        skill_id: str,
        content_hash: str,
        author_description_hash: str,
    ) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM professionalism_review WHERE skill_id=? "
                "AND content_hash=? AND author_description_hash=?",
                (skill_id, content_hash, author_description_hash),
            ).fetchone()
        return self._decode_professionalism_review(dict(row)) if row else None

    def claim_professionalism_review(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        review_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim pending work or recover an expired lease atomically."""

        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as db:
            query = (
                "SELECT * FROM professionalism_review WHERE "
                "((state IN ('pending','retry') AND available_at<=?) OR "
                "(state='running' AND lease_expires_at<=?))"
            )
            params: list[str] = [now_text, now_text]
            if review_id:
                query += " AND id=?"
                params.append(review_id)
            query += " ORDER BY available_at,created_at LIMIT 1"
            row = db.execute(query, params).fetchone()
            if row is None:
                return None
            cursor = db.execute(
                "UPDATE professionalism_review SET state='running',attempts=attempts+1,"
                "lease_owner=?,lease_expires_at=?,updated_at=? WHERE id=? AND "
                "((state IN ('pending','retry') AND available_at<=?) OR "
                "(state='running' AND lease_expires_at<=?))",
                (
                    worker_id,
                    lease_expires_at,
                    now_text,
                    row["id"],
                    now_text,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = db.execute(
                "SELECT * FROM professionalism_review WHERE id=?", (row["id"],)
            ).fetchone()
        return self._decode_professionalism_review(dict(claimed)) if claimed else None

    def expedite_professionalism_review(self, review_id: str) -> None:
        """Make a queued retry immediately claimable before an explicit submission."""

        with self.transaction() as db:
            db.execute(
                "UPDATE professionalism_review SET available_at=?,updated_at=? "
                "WHERE id=? AND state IN ('pending','retry')",
                (utc_now(), utc_now(), review_id),
            )

    def complete_professionalism_review(
        self, review_id: str, *, worker_id: str, result: dict[str, Any]
    ) -> bool:
        now = utc_now()
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE professionalism_review SET state='complete',result_json=?,"
                "lease_owner=NULL,lease_expires_at=NULL,last_error=NULL,updated_at=? "
                "WHERE id=? AND state='running' AND lease_owner=?",
                (
                    json.dumps(result, sort_keys=True, separators=(",", ":")),
                    now,
                    review_id,
                    worker_id,
                ),
            )
            return cursor.rowcount == 1

    def retry_professionalism_review(
        self,
        review_id: str,
        *,
        worker_id: str,
        error: str,
        unavailable_result: dict[str, Any],
        max_attempts: int = 2,
        retry_delay_seconds: int = 5,
    ) -> str | None:
        """Release a failed claim or make its bounded unavailable result terminal."""

        now = datetime.now(timezone.utc)
        with self.transaction() as db:
            row = db.execute(
                "SELECT attempts FROM professionalism_review WHERE id=? "
                "AND state='running' AND lease_owner=?",
                (review_id, worker_id),
            ).fetchone()
            if row is None:
                return None
            terminal = int(row["attempts"]) >= max_attempts
            state = "complete" if terminal else "retry"
            result_json = (
                json.dumps(unavailable_result, sort_keys=True, separators=(",", ":"))
                if terminal
                else None
            )
            available_at = (
                now if terminal else now + timedelta(seconds=retry_delay_seconds)
            ).isoformat()
            db.execute(
                "UPDATE professionalism_review SET state=?,available_at=?,"
                "lease_owner=NULL,lease_expires_at=NULL,result_json=?,last_error=?,"
                "updated_at=? WHERE id=? AND state='running' AND lease_owner=?",
                (
                    state,
                    available_at,
                    result_json,
                    error[:512],
                    now.isoformat(),
                    review_id,
                    worker_id,
                ),
            )
        return state

    @staticmethod
    def _decode_professionalism_review(value: dict[str, Any]) -> dict[str, Any]:
        value["package"] = json.loads(value.pop("package_json"))
        raw_result = value.pop("result_json")
        value["result"] = json.loads(raw_result) if raw_result else None
        return value

    def pending_telegram_events(
        self, *, kind: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Return unread session events not yet surfaced in Telegram.

        Telegram delivery is deliberately independent from ``state``. The same
        candidate must remain available to Desktop and Dashboard until the
        owner dismisses it or advances its exact bytes into publication.
        """
        return self.pending_surface_events(
            kind=kind, session_id=session_id, surface="telegram"
        )

    def pending_surface_events(
        self, *, kind: str, session_id: str, surface: str
    ) -> list[dict[str, Any]]:
        """Return unread session events not yet delivered to one surface."""
        with self.transaction() as db:
            organization = db.execute(
                "SELECT verified_org_id FROM installation_identity WHERE singleton=1"
            ).fetchone()
            active_org_id = (
                str(organization[0])
                if organization is not None and organization[0]
                else None
            )
            organization_clause = (
                "AND e.organization_id=? "
                if active_org_id is not None and kind == "wisdom.candidate"
                else "AND e.organization_id IS NULL "
                if kind == "wisdom.candidate"
                else ""
            )
            params: list[str] = [kind, session_id]
            if active_org_id is not None and kind == "wisdom.candidate":
                params.append(active_org_id)
            params.append(surface)
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT e.* FROM local_event e WHERE e.kind=? AND e.session_id=? "
                    + organization_clause
                    + "AND e.state='unread' AND NOT EXISTS ("
                    "SELECT 1 FROM local_event_delivery d "
                    "WHERE d.event_id=e.id AND d.surface=?"
                    ") ORDER BY e.created_at",
                    params,
                ).fetchall()
            ]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

    def mark_telegram_delivered(self, event_ids: list[str]) -> None:
        self.mark_surface_delivered(event_ids, surface="telegram")
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self.transaction() as db:
            db.execute(
                f"UPDATE local_event SET telegram_delivered_at=? "
                f"WHERE id IN ({placeholders}) AND telegram_delivered_at IS NULL",
                (utc_now(), *event_ids),
            )

    def mark_surface_delivered(self, event_ids: list[str], *, surface: str) -> None:
        if not event_ids:
            return
        delivered_at = utc_now()
        with self.transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO local_event_delivery "
                "(event_id,surface,delivered_at) VALUES(?,?,?)",
                [(event_id, surface, delivered_at) for event_id in event_ids],
            )

    def dismiss_candidate(self, skill_id: str, content_hash: str) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute(
                "UPDATE candidate SET state='dismissed',dismissed_at=? WHERE skill_id=? AND content_hash=?",
                (now, skill_id, content_hash),
            )
            db.execute(
                "UPDATE local_event SET state='dismissed' WHERE skill_id=? AND content_hash=?",
                (skill_id, content_hash),
            )

    def complete_contribution(self, draft_id: str, state: str) -> None:
        """Retire a candidate event after its owner-approved contribution succeeds."""

        now = utc_now()
        with self.transaction() as db:
            draft = db.execute(
                "SELECT skill_id,source_hash FROM local_draft WHERE id=?",
                (draft_id,),
            ).fetchone()
            if draft is None:
                return
            skill_id = str(draft["skill_id"])
            content_hash = str(draft["source_hash"])
            db.execute(
                "UPDATE local_draft SET state=?,updated_at=? WHERE id=?",
                (state, now, draft_id),
            )
            db.execute(
                "UPDATE candidate SET state='contributed',dismissed_at=NULL "
                "WHERE skill_id=? AND content_hash=?",
                (skill_id, content_hash),
            )
            db.execute(
                "UPDATE local_event SET state='handled' "
                "WHERE skill_id=? AND content_hash=?",
                (skill_id, content_hash),
            )

    def mark_missing_skills(self, seen_paths: set[str]) -> None:
        """Tombstone disappeared paths so delete/recreate gets a new identity."""
        now = utc_now()
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id,canonical_path FROM local_skill WHERE deleted_at IS NULL"
            ).fetchall()
            for row in rows:
                if str(row["canonical_path"]) not in seen_paths:
                    db.execute(
                        "UPDATE local_skill SET deleted_at=?,updated_at=? WHERE id=?",
                        (now, now, row["id"]),
                    )

    def existing_installation_identity(self) -> str | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT installation_id FROM installation_identity WHERE singleton=1"
            ).fetchone()
            return str(row["installation_id"]) if row else None

    def installation_identity(self) -> str:
        with self.transaction() as db:
            row = db.execute(
                "SELECT installation_id FROM installation_identity WHERE singleton=1"
            ).fetchone()
            if row:
                return str(row["installation_id"])
            installation_id = "hwi_" + uuid.uuid4().hex
            db.execute(
                "INSERT INTO installation_identity VALUES(1,?,?,?,?)",
                (installation_id, None, None, utc_now()),
            )
            return installation_id

    def activate_installation_identity(self, installation_id: str, org_id: str) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute(
                "UPDATE managed_install SET state='inactive',updated_at=? "
                "WHERE org_id<>? AND state='active'",
                (now, org_id),
            )
            db.execute(
                """INSERT INTO installation_identity(
                     singleton,installation_id,verified_org_id,verified_at,disclosure_at
                   ) VALUES(1,?,?,?,?)
                   ON CONFLICT(singleton) DO UPDATE SET
                     installation_id=excluded.installation_id,
                     verified_org_id=excluded.verified_org_id,
                     verified_at=excluded.verified_at""",
                (installation_id, org_id, now, now),
            )

    def verify_installation_identity(self, org_id: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE managed_install SET state='inactive',updated_at=? "
                "WHERE org_id<>? AND state='active'",
                (utc_now(), org_id),
            )
            db.execute(
                "UPDATE installation_identity SET verified_org_id=?,verified_at=? WHERE singleton=1",
                (org_id, utc_now()),
            )

    def active_org_id(self) -> str | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT verified_org_id FROM installation_identity WHERE singleton=1"
            ).fetchone()
            return str(row[0]) if row and row[0] else None

    def organization_display_name(self, organization_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT display_name,resolved,checked_at FROM wisdom_organization "
                "WHERE organization_id=?",
                (organization_id,),
            ).fetchone()
        return dict(row) if row else None

    def record_organization_display_name_check(
        self, organization_id: str, display_name: str | None
    ) -> None:
        normalized = display_name.strip()[:160] if display_name else None
        with self.transaction() as db:
            db.execute(
                "INSERT INTO wisdom_organization(organization_id,display_name,resolved,checked_at) "
                "VALUES(?,?,?,?) ON CONFLICT(organization_id) DO UPDATE SET "
                "display_name=COALESCE(excluded.display_name,wisdom_organization.display_name),"
                "resolved=excluded.resolved,"
                "checked_at=excluded.checked_at",
                (organization_id, normalized or None, int(bool(normalized)), utc_now()),
            )

    def record_draft(self, values: dict[str, str]) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute(
                """INSERT INTO local_draft(
                   id,skill_id,source_hash,overlay_path,draft_commit,server_revision,state,
                   description,content_hash,description_hash,manifest_hash,created_at,updated_at
                 ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(id) DO UPDATE SET draft_commit=excluded.draft_commit,
                   overlay_path=excluded.overlay_path,
                   server_revision=excluded.server_revision,state=excluded.state,
                   description=excluded.description,content_hash=excluded.content_hash,
                   description_hash=excluded.description_hash,manifest_hash=excluded.manifest_hash,
                   updated_at=excluded.updated_at""",
                (
                    values["id"],
                    values["skill_id"],
                    values["source_hash"],
                    values["overlay_path"],
                    values.get("draft_commit"),
                    values.get("server_revision"),
                    values["state"],
                    values["description"],
                    values["content_hash"],
                    values["description_hash"],
                    values["manifest_hash"],
                    now,
                    now,
                ),
            )

    def draft(self, draft_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM local_draft WHERE id=?", (draft_id,)
            ).fetchone()
            return dict(row) if row else None

    def prepared_draft(self, skill_id: str, source_hash: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM local_draft WHERE skill_id=? AND source_hash=? "
                "AND state='prepared' ORDER BY updated_at DESC LIMIT 1",
                (skill_id, source_hash),
            ).fetchone()
            return dict(row) if row else None

    def latest_draft_for_source(
        self, skill_id: str, source_hash: str
    ) -> dict[str, Any] | None:
        """Return the newest contribution state for one exact local source version."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM local_draft WHERE skill_id=? AND source_hash=? "
                "ORDER BY updated_at DESC, CASE WHEN id LIKE 'local:%' THEN 1 ELSE 0 END, id DESC "
                "LIMIT 1",
                (skill_id, source_hash),
            ).fetchone()
            return dict(row) if row else None

    def set_draft_state(self, draft_id: str, state: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE local_draft SET state=?,updated_at=? WHERE id=?",
                (state, utc_now(), draft_id),
            )

    def save_receipt(
        self,
        *,
        draft_id: str,
        server_revision: str,
        content_hash: str,
        description_hash: str,
        manifest_hash: str,
    ) -> str:
        receipt_id = str(uuid.uuid4())
        with self.transaction() as db:
            db.execute("DELETE FROM review_receipt WHERE draft_id=?", (draft_id,))
            db.execute(
                "INSERT INTO review_receipt VALUES(?,?,?,?,?,?,?,NULL)",
                (
                    receipt_id,
                    draft_id,
                    server_revision,
                    content_hash,
                    description_hash,
                    manifest_hash,
                    utc_now(),
                ),
            )
        return receipt_id

    def receipt(self, draft_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM review_receipt WHERE draft_id=? AND consumed_at IS NULL",
                (draft_id,),
            ).fetchone()
            return dict(row) if row else None

    def consume_receipt(self, draft_id: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE review_receipt SET consumed_at=? WHERE draft_id=?",
                (utc_now(), draft_id),
            )

    def journal(
        self, kind: str, entity_id: str, phase: str, payload: dict[str, Any]
    ) -> str:
        operation_id = str(uuid.uuid4())
        with self.transaction() as db:
            existing = db.execute(
                "SELECT id FROM operation_journal WHERE kind=? AND entity_id=? AND state='pending'",
                (kind, entity_id),
            ).fetchone()
            if existing:
                return str(existing["id"])
            db.execute(
                "INSERT INTO operation_journal VALUES(?,?,?,?,?,'pending',?,?)",
                (
                    operation_id,
                    kind,
                    entity_id,
                    phase,
                    json.dumps(payload, sort_keys=True),
                    utc_now(),
                    utc_now(),
                ),
            )
        return operation_id

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM operation_journal WHERE id=?", (operation_id,)
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload_json"])
        return value

    def replace_operation_payload(
        self, operation_id: str, payload: dict[str, Any]
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE operation_journal SET payload_json=?,updated_at=? WHERE id=?",
                (json.dumps(payload, sort_keys=True), utc_now(), operation_id),
            )

    def advance(self, operation_id: str, phase: str, *, done: bool = False) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE operation_journal SET phase=?,state=?,updated_at=? WHERE id=?",
                (phase, "done" if done else "pending", utc_now(), operation_id),
            )

    def pending_operations(self) -> list[dict[str, Any]]:
        with self.transaction() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM operation_journal WHERE state='pending' ORDER BY created_at"
                ).fetchall()
            ]

    def acquire_operation_lock(
        self, entity_id: str, *, ttl_seconds: int = 900
    ) -> str | None:
        owner = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)
        with self.transaction() as db:
            db.execute(
                "DELETE FROM operation_lock WHERE expires_at<=?", (now.isoformat(),)
            )
            cursor = db.execute(
                "INSERT OR IGNORE INTO operation_lock VALUES(?,?,?)",
                (entity_id, owner, expires.isoformat()),
            )
            return owner if cursor.rowcount else None

    def release_operation_lock(self, entity_id: str, owner: str) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM operation_lock WHERE entity_id=? AND owner=?",
                (entity_id, owner),
            )

    def record_install(self, values: dict[str, Any]) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute(
                """INSERT INTO managed_install VALUES(?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(skill_id) DO UPDATE SET org_id=excluded.org_id,
                   slug=excluded.slug,version=excluded.version,content_hash=excluded.content_hash,
                   baseline_json=excluded.baseline_json,target_path=excluded.target_path,
                   update_mode=excluded.update_mode,state=excluded.state,updated_at=excluded.updated_at""",
                (
                    values["skill_id"],
                    values["org_id"],
                    values["slug"],
                    int(values["version"]),
                    values["content_hash"],
                    json.dumps(values["baseline"], sort_keys=True),
                    values["target_path"],
                    values["update_mode"],
                    values.get("state", "active"),
                    now,
                    now,
                ),
            )

    def installations(self) -> list[dict[str, Any]]:
        with self.transaction() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM managed_install ORDER BY slug"
                ).fetchall()
            ]
        for row in rows:
            row["baseline"] = json.loads(row.pop("baseline_json"))
        return rows

    def installation(self, skill_id: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM managed_install WHERE skill_id=?", (skill_id,)
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        value["baseline"] = json.loads(value.pop("baseline_json"))
        return value

    def deactivate_install(self, skill_id: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE managed_install SET state='inactive',updated_at=? WHERE skill_id=?",
                (utc_now(), skill_id),
            )

    def feed_cursor(self) -> str | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT cursor FROM feed_state WHERE singleton=1"
            ).fetchone()
            return str(row[0]) if row and row[0] else None

    def persist_feed_page(
        self,
        events: list[dict[str, Any]],
        *,
        next_cursor: str,
        cadences: dict[str, str],
        now: str,
    ) -> int:
        current = datetime.fromisoformat(now).astimezone(timezone.utc)

        def due_at(cadence: str) -> str:
            if cadence == "daily":
                due = (current + timedelta(days=1)).replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
            elif cadence == "weekly":
                days = 7 - current.weekday()
                due = (current + timedelta(days=days)).replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
            else:
                due = current
            return due.isoformat()

        inserted = 0
        with self.transaction() as db:
            for event in events:
                kind = str(event["kind"])
                cadence = cadences.get(
                    str(event["event_id"]), cadences.get(kind, "immediate")
                )
                cursor = db.execute(
                    "INSERT OR IGNORE INTO feed_event VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(event["event_id"]),
                        kind,
                        str(event["skill_id"]),
                        event.get("version"),
                        event.get("installation_id"),
                        event.get("update_mode"),
                        json.dumps(event, sort_keys=True),
                        cadence,
                        due_at(cadence),
                        None,
                        None,
                        str(event.get("occurred_at") or now),
                    ),
                )
                inserted += max(cursor.rowcount, 0)
            db.execute(
                "INSERT INTO feed_state VALUES(1,?,?) ON CONFLICT(singleton) "
                "DO UPDATE SET cursor=excluded.cursor,updated_at=excluded.updated_at",
                (next_cursor, now),
            )
        return inserted

    def feed_events(
        self,
        *,
        unseen_only: bool = False,
        telegram_due_at: str | None = None,
        surface: str | None = None,
        surface_due_at: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT e.* FROM feed_event e WHERE 1=1"
        params: list[str] = []
        if unseen_only:
            query += " AND e.local_seen_at IS NULL AND e.cadence!='off' AND e.due_at<=?"
            params.append(utc_now())
        if telegram_due_at is not None:
            query += (
                " AND NOT EXISTS (SELECT 1 FROM feed_event_delivery d "
                "WHERE d.event_id=e.event_id AND d.surface='telegram') "
                "AND e.cadence!='off' AND e.due_at<=?"
            )
            params.append(telegram_due_at)
        if surface is not None and surface_due_at is not None:
            query += (
                " AND NOT EXISTS (SELECT 1 FROM feed_event_delivery d "
                "WHERE d.event_id=e.event_id AND d.surface=?) "
                "AND e.cadence!='off' AND e.due_at<=?"
            )
            params.extend([surface, surface_due_at])
        query += " ORDER BY e.created_at,e.event_id"
        with self.transaction() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

    def persist_local_notice(
        self,
        *,
        event_id: str,
        kind: str,
        skill_id: str,
        payload: dict[str, Any],
        cadence: str = "immediate",
    ) -> bool:
        now = utc_now()
        current = datetime.fromisoformat(now)
        if cadence == "daily":
            due = (
                (current + timedelta(days=1))
                .replace(hour=9, minute=0, second=0, microsecond=0)
                .isoformat()
            )
        elif cadence == "weekly":
            due = (
                (current + timedelta(days=7 - current.weekday()))
                .replace(hour=9, minute=0, second=0, microsecond=0)
                .isoformat()
            )
        else:
            due = now
        with self.transaction() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO feed_event VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    kind,
                    skill_id,
                    payload.get("version"),
                    None,
                    None,
                    json.dumps(payload, sort_keys=True),
                    cadence,
                    due,
                    None,
                    None,
                    now,
                ),
            )
            return bool(cursor.rowcount)

    def mark_feed_local_seen(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self.transaction() as db:
            db.execute(
                f"UPDATE feed_event SET local_seen_at=? WHERE event_id IN ({placeholders})",
                [utc_now(), *event_ids],
            )

    def mark_feed_telegram_delivered(self, event_ids: list[str]) -> None:
        self.mark_feed_surface_delivered(event_ids, surface="telegram")
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self.transaction() as db:
            db.execute(
                f"UPDATE feed_event SET telegram_delivered_at=? WHERE event_id IN ({placeholders})",
                [utc_now(), *event_ids],
            )

    def mark_feed_surface_delivered(
        self, event_ids: list[str], *, surface: str
    ) -> None:
        if not event_ids:
            return
        delivered_at = utc_now()
        with self.transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO feed_event_delivery "
                "(event_id,surface,delivered_at) VALUES(?,?,?)",
                [(event_id, surface, delivered_at) for event_id in event_ids],
            )
