-- Layer 5 schema: SQLite + FTS5 index over existing markdown.
-- This database is a REGENERABLE CACHE. Markdown is the source of truth;
-- rebuilding from the same inputs yields an identical result (deterministic).
-- The file (index.db) is gitignored.

-- Each logical source gets a content table + a matching FTS5 virtual table.
-- We keep all indexable markdown in one `notes` table for simplicity, plus
-- the supporting tables named in the architecture for forward compatibility.

PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(content, tags, content='conversations', content_rowid='id');

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS projects_fts USING fts5(content, tags, content='projects', content_rowid='id');

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(content, tags, content='decisions', content_rowid='id');

CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(content, tags, content='files', content_rowid='id');

CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(content, tags, content='prompts', content_rowid='id');

-- Primary indexable table for Phase 1 (all markdown chunks land here).
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(content, tags, content='notes', content_rowid='id');

CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    memory_layer TEXT NOT NULL,
    content     TEXT,
    tags        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    extra       TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS tags_fts USING fts5(content, tags, content='tags', content_rowid='id');

CREATE INDEX IF NOT EXISTS idx_notes_source ON notes(source_file);
CREATE INDEX IF NOT EXISTS idx_notes_layer ON notes(memory_layer);

-- Phase 3: archive lifecycle pending queue. Tracks sessions whose raw
-- transcript has been enqueued for (re)indexing but not yet flushed into
-- `notes`. Mirrors the same DDL in MemoryIndex._base_schema_only() so the
-- fallback (no-FTS5) path also has it.
CREATE TABLE IF NOT EXISTS index_pending (
    source_file  TEXT PRIMARY KEY,
    enqueued_at  TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    last_attempt TEXT,
    status       TEXT NOT NULL DEFAULT 'pending'  -- 'pending' | 'failed' | 'done'
);
