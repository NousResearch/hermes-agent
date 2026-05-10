"""SQLite schema for the Recall memory provider."""

SCHEMA_VERSION = "1"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    project_path TEXT,
    user_text TEXT,
    assistant_text TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    trust_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    content TEXT NOT NULL,
    redacted_content TEXT NOT NULL,
    source_session_id TEXT,
    project_path TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    supersedes TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    content,
    redacted_content,
    type,
    scope,
    content='observations',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
  INSERT INTO observations_fts(rowid, content, redacted_content, type, scope)
  VALUES (new.rowid, new.content, new.redacted_content, new.type, new.scope);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, content, redacted_content, type, scope)
  VALUES ('delete', old.rowid, old.content, old.redacted_content, old.type, old.scope);
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, content, redacted_content, type, scope)
  VALUES ('delete', old.rowid, old.content, old.redacted_content, old.type, old.scope);
  INSERT INTO observations_fts(rowid, content, redacted_content, type, scope)
  VALUES (new.rowid, new.content, new.redacted_content, new.type, new.scope);
END;

CREATE INDEX IF NOT EXISTS idx_observations_status_expires_order
ON observations(status, expires_at, importance DESC, confidence DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_scope_project_status_expires_order
ON observations(scope, project_path, status, expires_at, importance DESC, confidence DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_supersedes_status_expires
ON observations(supersedes, status, expires_at);

CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    phase TEXT NOT NULL,
    operation TEXT NOT NULL,
    target TEXT NOT NULL,
    content_preview TEXT,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""
