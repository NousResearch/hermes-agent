from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CreationsStore:
    def __init__(self, home: Path):
        self.root = Path(home) / "creations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.media_dir = self.root / "media"
        self.media_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, action TEXT NOT NULL, body_hash TEXT NOT NULL, body_json TEXT NOT NULL, state TEXT NOT NULL, run_id TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS plans(plan_id TEXT PRIMARY KEY, json TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, json TEXT NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS assets(url TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS library(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, node_id TEXT, source_url TEXT, sha256 TEXT, filename TEXT, mime_type TEXT, bytes INTEGER, kind TEXT, model TEXT, provider TEXT, storage_state TEXT, created_at REAL NOT NULL, UNIQUE(run_id,node_id,source_url));
        """)
        self.db.commit()

    def reserve_request(self, request_id: str, action: str, body: dict) -> dict:
        digest = hashlib.sha256(_canonical(body).encode()).hexdigest()
        row = self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        if row:
            if row["body_hash"] != digest or row["action"] != action:
                raise ValueError("request_conflict: request ID already has a different body")
            return dict(row)
        self.db.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,?)", (request_id, action, digest, _canonical(body), "pending", None, time.time()))
        self.db.commit()
        return dict(self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone())

    def save_plan(self, plan: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO plans VALUES(?,?,?)", (plan["planId"], _canonical(plan), time.time())); self.db.commit()

    def get_plan(self, plan_id: str) -> dict | None:
        row = self.db.execute("SELECT json FROM plans WHERE plan_id=?", (plan_id,)).fetchone(); return json.loads(row[0]) if row else None

    def owns_plan(self, plan_id: str) -> bool: return self.get_plan(plan_id) is not None

    def attach_run(self, request_id: str, run: dict) -> None:
        now = time.time()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?)", (run["runId"], _canonical(run), now))
            self.db.execute("UPDATE requests SET state=?,run_id=? WHERE request_id=?", (run.get("state", "running"), run["runId"], request_id))

    def get_run(self, run_id: str) -> dict | None:
        row = self.db.execute("SELECT json FROM runs WHERE run_id=?", (run_id,)).fetchone(); return json.loads(row[0]) if row else None

    def list_runs(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT json FROM runs ORDER BY updated_at DESC")]

    def update_run(self, run: dict) -> None:
        self.db.execute("UPDATE runs SET json=?,updated_at=? WHERE run_id=?", (_canonical(run), time.time(), run["runId"])); self.db.commit()

    def pending_requests(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM requests WHERE state IN ('pending','reconciling') ORDER BY created_at")]

    def register_asset(self, url: str) -> None: self.db.execute("INSERT OR IGNORE INTO assets VALUES(?)", (url,)); self.db.commit()
    def owns_asset(self, url: str) -> bool: return self.db.execute("SELECT 1 FROM assets WHERE url=?", (url,)).fetchone() is not None

    def upsert_library(self, item: dict) -> None:
        self.db.execute("""INSERT INTO library(run_id,node_id,source_url,sha256,filename,mime_type,bytes,kind,model,provider,storage_state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,node_id,source_url) DO UPDATE SET sha256=excluded.sha256,filename=excluded.filename,mime_type=excluded.mime_type,bytes=excluded.bytes,storage_state=excluded.storage_state""", (item.get("runId"), item.get("nodeId"), item.get("sourceUrl"), item.get("sha256"), item.get("filename"), item.get("mimeType"), item.get("bytes"), item.get("kind"), item.get("model"), item.get("provider"), item.get("storageState", "copying"), time.time())); self.db.commit()

    def list_library(self, kind: str | None = None, cursor: str | None = None) -> dict:
        args: list[Any] = []; where = ""
        if kind:
            where = " WHERE kind=?"; args.append(kind)
        rows = self.db.execute(f"SELECT * FROM library{where} ORDER BY created_at DESC,id DESC LIMIT 31", args).fetchall()
        items = [dict(r) for r in rows[:30]]
        for item in items:
            item["mimeType"] = item.pop("mime_type"); item["sourceUrl"] = item.pop("source_url"); item["storageState"] = item.pop("storage_state"); item["createdAt"] = item.pop("created_at")
        next_cursor = base64.urlsafe_b64encode(str(rows[29]["id"]).encode()).decode() if len(rows) > 30 else None
        return {"items": items, "nextCursor": next_cursor}

    def library_item(self, item_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM library WHERE id=?", (item_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["mimeType"] = item.pop("mime_type")
        item["sourceUrl"] = item.pop("source_url")
        item["storageState"] = item.pop("storage_state")
        item["createdAt"] = item.pop("created_at")
        item["path"] = str(self.media_dir / item["filename"]) if item.get("filename") else None
        item["id"] = int(item["id"])
        return item

    def close(self) -> None: self.db.close()
