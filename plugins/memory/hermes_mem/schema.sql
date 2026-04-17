-- Hermes Mem initial SQLite schema
--
-- Purpose:
--   Store operational memory observations extracted from Hermes sessions,
--   plus session-level summaries and a small audit trail of recalled context.
--
-- Notes:
--   - This is a design-first schema for the initial hermes_mem MVP.
--   - It intentionally uses SQLite + FTS5 only; embeddings are deferred.
--   - The raw session transcript remains in Hermes's existing SessionDB.
--   - hermes_mem stores reusable observations, not the full conversation log.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    parent_id        INTEGER REFERENCES observations(id) ON DELETE SET NULL,
    project          TEXT,
    workspace        TEXT,
    profile          TEXT,
    kind             TEXT NOT NULL,
    title            TEXT NOT NULL,
    summary          TEXT NOT NULL,
    detail           TEXT DEFAULT '',
    source_type      TEXT NOT NULL,
    source_ref       TEXT,
    importance       REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 0.5,
    token_cost_hint  INTEGER DEFAULT 0,
    turn_number      INTEGER,
    delegated        INTEGER NOT NULL DEFAULT 0,
    pinned           INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at       TEXT,
    CHECK (kind IN (
        'fact',
        'decision',
        'bugfix',
        'investigation',
        'tool_result',
        'file_change',
        'user_preference',
        'delegation_result',
        'session_summary'
    )),
    CHECK (source_type IN (
        'turn',
        'tool',
        'delegation',
        'compression',
        'session_end',
        'manual'
    )),
    CHECK (importance >= 0.0 AND importance <= 1.0),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (delegated IN (0, 1)),
    CHECK (pinned IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_observations_session_created
    ON observations(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_kind_created
    ON observations(kind, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_project_created
    ON observations(project, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_profile_created
    ON observations(profile, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_workspace_created
    ON observations(workspace, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_importance
    ON observations(importance DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_active
    ON observations(deleted_at, created_at DESC);

CREATE TABLE IF NOT EXISTS observation_tags (
    observation_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    tag              TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (observation_id, tag)
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    title,
    summary,
    detail,
    tags,
    content='observations',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, title, summary, detail, tags)
    VALUES (new.id, new.title, new.summary, new.detail, COALESCE((SELECT GROUP_CONCAT(tag, ' ')
      FROM observation_tags WHERE observation_id = new.id), ''));
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail, tags)
    VALUES ('delete', old.id, old.title, old.summary, old.detail, COALESCE((SELECT GROUP_CONCAT(tag, ' ')
      FROM observation_tags WHERE observation_id = old.id), ''));
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail, tags)
    VALUES ('delete', old.id, old.title, old.summary, old.detail, COALESCE((SELECT GROUP_CONCAT(tag, ' ')
      FROM observation_tags WHERE observation_id = old.id), ''));
    INSERT INTO observations_fts(rowid, title, summary, detail, tags)
    VALUES (new.id, new.title, new.summary, new.detail, COALESCE((SELECT GROUP_CONCAT(tag, ' ')
      FROM observation_tags WHERE observation_id = new.id), ''));
END;

CREATE INDEX IF NOT EXISTS idx_observation_tags_tag
    ON observation_tags(tag);

CREATE TRIGGER IF NOT EXISTS observation_tags_ai AFTER INSERT ON observation_tags BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail, tags)
    VALUES ('delete', new.observation_id,
        COALESCE((SELECT title FROM observations WHERE id = new.observation_id), ''),
        COALESCE((SELECT summary FROM observations WHERE id = new.observation_id), ''),
        COALESCE((SELECT detail FROM observations WHERE id = new.observation_id), ''),
        COALESCE((SELECT GROUP_CONCAT(tag, ' ') FROM observation_tags WHERE observation_id = new.observation_id), '')
    );
    INSERT INTO observations_fts(rowid, title, summary, detail, tags)
    SELECT o.id, o.title, o.summary, o.detail,
           COALESCE((SELECT GROUP_CONCAT(tag, ' ') FROM observation_tags WHERE observation_id = o.id), '')
      FROM observations o
     WHERE o.id = new.observation_id;
END;

CREATE TRIGGER IF NOT EXISTS observation_tags_ad AFTER DELETE ON observation_tags BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail, tags)
    VALUES ('delete', old.observation_id,
        COALESCE((SELECT title FROM observations WHERE id = old.observation_id), ''),
        COALESCE((SELECT summary FROM observations WHERE id = old.observation_id), ''),
        COALESCE((SELECT detail FROM observations WHERE id = old.observation_id), ''),
        COALESCE((SELECT GROUP_CONCAT(tag, ' ') FROM observation_tags WHERE observation_id = old.observation_id), '')
    );
    INSERT INTO observations_fts(rowid, title, summary, detail, tags)
    SELECT o.id, o.title, o.summary, o.detail,
           COALESCE((SELECT GROUP_CONCAT(tag, ' ') FROM observation_tags WHERE observation_id = o.id), '')
      FROM observations o
     WHERE o.id = old.observation_id;
END;

CREATE TABLE IF NOT EXISTS observation_links (
    from_observation_id  INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    to_observation_id    INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    link_type            TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_observation_id, to_observation_id, link_type),
    CHECK (link_type IN ('related', 'caused_by', 'supersedes', 'supports', 'contradicts', 'follows'))
);

CREATE INDEX IF NOT EXISTS idx_observation_links_to
    ON observation_links(to_observation_id, link_type);

CREATE TABLE IF NOT EXISTS session_summaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL UNIQUE,
    project             TEXT,
    workspace           TEXT,
    profile             TEXT,
    title               TEXT,
    summary             TEXT NOT NULL,
    open_questions      TEXT DEFAULT '',
    outcomes            TEXT DEFAULT '',
    observation_count   INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT,
    ended_at            TEXT,
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_summaries_project
    ON session_summaries(project, created_at DESC);

CREATE TABLE IF NOT EXISTS recall_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    turn_number         INTEGER,
    query               TEXT NOT NULL,
    recall_mode         TEXT NOT NULL,
    selected_ids_json   TEXT NOT NULL DEFAULT '[]',
    injected_text       TEXT NOT NULL DEFAULT '',
    token_estimate      INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (recall_mode IN ('prefetch', 'search', 'timeline', 'get'))
);

CREATE INDEX IF NOT EXISTS idx_recall_log_session_turn
    ON recall_log(session_id, turn_number DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version             INTEGER PRIMARY KEY,
    name                TEXT NOT NULL,
    applied_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (1, 'initial hermes_mem schema');
