"""Producer custody for explicit classic Group Chat output; no room authority."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox

PENDING_TTL = 24 * 3600
MAX_EXPORTS = 1024
MAX_FILES = 4096


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identifier(value) -> str:
    if not isinstance(value, str) or not value or len(value) > 160 or any(ord(c) < 32 for c in value):
        raise RoomArtifactError("Invalid classic export identity")
    return value


@dataclass(frozen=True)
class ClassicExportScope:
    export_id: str
    execution_generation: int

    def as_mapping(self):
        return {"kind": "classic", "export_id": self.export_id,
                "execution_generation": self.execution_generation}

    @property
    def key(self):
        return hashlib.sha256(encoded(self.as_mapping()).encode()).hexdigest()

    @property
    def lineage_json(self):
        return encoded({"kind": "classic", "export_id": self.export_id})

    @property
    def lineage_key(self):
        return hashlib.sha256(self.lineage_json.encode()).hexdigest()


def validate_write(conn, scope):
    row = conn.execute("SELECT state, expires FROM classic_output_exports WHERE export_id=? AND generation=?",
                       (scope.export_id, scope.execution_generation)).fetchone()
    if row is None or row["state"] != "running" or row["expires"] <= time.time():
        raise RoomArtifactError("Classic export is not an active admitted turn")


class ClassicExports:
    def __init__(self, home: Path | str):
        self.outbox = RoomArtifactOutbox(Path(home) / "state.db")
        with self.outbox._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS classic_output_exports (
                export_id TEXT PRIMARY KEY, profile_home TEXT NOT NULL,
                session_key TEXT NOT NULL, request_id TEXT NOT NULL, generation INTEGER NOT NULL,
                binding TEXT NOT NULL, state TEXT NOT NULL, expires REAL NOT NULL,
                text TEXT NOT NULL DEFAULT '', UNIQUE(profile_home, session_key, request_id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS classic_output_expiry ON classic_output_exports(state, expires)")
            conn.execute("CREATE TABLE IF NOT EXISTS classic_retired_groups (profile_home TEXT, group_id TEXT, PRIMARY KEY(profile_home,group_id))")
        self.home = str(Path(home).resolve())
        self.prune()

    def prune(self):
        with self.outbox._connect() as conn:
            rows = conn.execute("SELECT * FROM classic_output_exports WHERE state!='published' AND expires<? LIMIT 32",
                                (time.time(),)).fetchall()
        for row in rows:
            with self.outbox._connect() as conn:
                changed = conn.execute("UPDATE classic_output_exports SET state='retired' WHERE export_id=? AND state!='published' AND expires<?",
                                       (row["export_id"], time.time())).rowcount
                conn.commit()
            if not changed:
                continue
            self.outbox.discard(self.scope(row))
            with self.outbox._connect() as conn:
                conn.execute("DELETE FROM classic_output_exports WHERE export_id=? AND state='retired' AND expires<?",
                             (row["export_id"], time.time()))

    def lookup(self, export_id):
        with self.outbox._connect() as conn:
            row = conn.execute("SELECT * FROM classic_output_exports WHERE export_id=? AND profile_home=?",
                               (identifier(export_id), self.home)).fetchone()
        if row is None:
            raise RoomArtifactError("Classic export not found in this profile")
        return dict(row)

    def prior(self, session_key, request_id):
        with self.outbox._connect() as conn:
            row = conn.execute("SELECT * FROM classic_output_exports WHERE profile_home=? AND session_key=? AND request_id=?",
                               (self.home, session_key, request_id)).fetchone()
        return dict(row) if row else None

    def admit(self, session_key, request, text):
        if not isinstance(request, dict) or set(request) != {"request_id", "group_id", "thread_id", "recipients", "issued_at"}:
            raise RoomArtifactError("Invalid classic export admission")
        request_id, group, thread = (identifier(request[k]) for k in ("request_id", "group_id", "thread_id"))
        recipients = request["recipients"]
        if not isinstance(recipients, list) or not 1 <= len(recipients) <= 6:
            raise RoomArtifactError("Invalid classic export recipients")
        for recipient in recipients:
            if not isinstance(recipient, dict) or set(recipient) != {"installation", "profile"}:
                raise RoomArtifactError("Invalid classic export recipient")
            for value in recipient.values():
                identifier(value)
        binding = encoded({"group_id": group, "thread_id": thread, "recipients": recipients,
                           "issued_at": request["issued_at"],
                           "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})
        with self.outbox._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM classic_output_exports WHERE profile_home=? AND session_key=? AND request_id=?",
                               (self.home, session_key, request_id)).fetchone()
            if old:
                if old["binding"] != binding:
                    raise RoomArtifactError("Classic export replay changed")
                return dict(old), False
            issued = request["issued_at"]
            if isinstance(issued, bool) or not isinstance(issued, (int, float)) or not time.time() - PENDING_TTL < issued <= time.time() + 300:
                raise RoomArtifactError("Classic export request expired; use a new explicit turn")
            if conn.execute("SELECT 1 FROM classic_retired_groups WHERE profile_home=? AND group_id=?", (self.home, group)).fetchone():
                raise RoomArtifactError("Classic group was retired")
            if conn.execute("SELECT COUNT(*) FROM classic_output_exports").fetchone()[0] >= MAX_EXPORTS:
                raise RoomArtifactError("Classic export quota exceeded; retire existing files first")
            generation = conn.execute("SELECT COALESCE(MAX(generation),0)+1 FROM classic_output_exports WHERE profile_home=? AND session_key=?",
                                      (self.home, session_key)).fetchone()[0]
            export_id = "ce_" + hashlib.sha256(encoded([self.home, session_key, request_id]).encode()).hexdigest()
            conn.execute("INSERT INTO classic_output_exports VALUES (?,?,?,?,?,?, 'running',?,'')",
                         (export_id, self.home, session_key, request_id, generation, binding, max(time.time(), issued) + PENDING_TTL))
            conn.commit()
        return self.lookup(export_id), True

    @staticmethod
    def scope(row):
        return ClassicExportScope(row["export_id"], row["generation"])

    def settle(self, export_id, text, success):
        row = self.lookup(export_id)
        if row["state"] in {"published", "settled"}:
            if success and row["text"] != str(text)[:64000]:
                raise RoomArtifactError("Classic terminal receipt changed")
            return
        if not success:
            self.retire(export_id)
            return
        with self.outbox._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            validate_write(conn, self.scope(row))
            has_files = conn.execute("SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=? LIMIT 1",
                                     (self.scope(row).key,)).fetchone()
            conn.execute("UPDATE classic_output_exports SET state=?,text=? WHERE export_id=?",
                         ('published' if has_files else 'settled', str(text)[:64000], export_id))
            conn.commit()

    def status(self, export_id):
        row = self.lookup(export_id)
        binding = json.loads(row["binding"])
        return {"export_id": export_id, "generation": row["generation"], "state": row["state"],
                "group_id": binding["group_id"], "thread_id": binding["thread_id"],
                "recipients": binding["recipients"], "text": row["text"],
                "items": self.outbox.list(self.scope(row)) if row["state"] == "published" else []}

    def read(self, export_id, artifact_id):
        row = self.lookup(export_id)
        if row["state"] != "published":
            raise RoomArtifactError("Classic export is not published")
        result = self.outbox.read(self.scope(row), identifier(artifact_id))
        if self.lookup(export_id)["state"] != "published":
            raise RoomArtifactError("Classic export was retired")
        return result

    def retire(self, export_id):
        row = self.lookup(export_id)
        with self.outbox._connect() as conn:
            conn.execute("UPDATE classic_output_exports SET state='retired' WHERE export_id=?", (export_id,))
            conn.execute("UPDATE hosted_room_output_artifacts SET cleanup_required_at=? WHERE scope_key=?",
                         (time.time(), self.scope(row).key))
            conn.commit()
        self.outbox.discard(self.scope(row))

    def retire_group(self, group_id):
        with self.outbox._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT 1 FROM classic_retired_groups WHERE profile_home=? AND group_id=?", (self.home, group_id)).fetchone()
            if not existing and conn.execute("SELECT COUNT(*) FROM classic_retired_groups").fetchone()[0] >= MAX_EXPORTS:
                raise RoomArtifactError("Classic retirement metadata quota exceeded")
            conn.execute("INSERT OR IGNORE INTO classic_retired_groups VALUES (?,?)", (self.home, identifier(group_id)))
            rows = conn.execute("SELECT export_id, generation FROM classic_output_exports WHERE profile_home=? AND json_extract(binding,'$.group_id')=?",
                                (self.home, identifier(group_id))).fetchall()
            conn.execute("UPDATE classic_output_exports SET state='retired' WHERE profile_home=? AND json_extract(binding,'$.group_id')=?",
                         (self.home, group_id))
            for row in rows:
                conn.execute("UPDATE hosted_room_output_artifacts SET cleanup_required_at=? WHERE scope_key=?",
                             (time.time(), self.scope(row).key))
            conn.commit()
        for row in rows:
            self.outbox.discard(self.scope(row))
