from __future__ import annotations

import aiosqlite

from .config import DB_PATH

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,
    email       TEXT NOT NULL,
    intended_use TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    revoked     INTEGER NOT NULL DEFAULT 0,
    first_parse_at TEXT
);

CREATE TABLE IF NOT EXISTS parse_jobs (
    id               TEXT PRIMARY KEY,
    api_key_id       TEXT NOT NULL REFERENCES api_keys(id),
    status           TEXT NOT NULL DEFAULT 'QUEUED',
    filename_original TEXT,
    filename_stored  TEXT,
    dp_file_id       TEXT,
    dp_folder_id     TEXT,
    result_json      TEXT,
    error_detail     TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at     TEXT
);

CREATE TABLE IF NOT EXISTS request_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id TEXT,
    endpoint   TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    ts         TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def init_db() -> None:
    import os
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


def connect() -> aiosqlite.Connection:
    """Return a new connection; use as `async with connect() as db:`."""
    return aiosqlite.connect(DB_PATH)


async def fetchone(conn: aiosqlite.Connection, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
    cursor = await conn.execute(sql, params)
    return await cursor.fetchone()
