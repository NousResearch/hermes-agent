CREATE TABLE IF NOT EXISTS models (
  repo_id TEXT PRIMARY KEY,
  author TEXT NOT NULL,
  sha TEXT,
  last_modified TEXT,
  gated TEXT,
  private INTEGER DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active',
  source TEXT,
  tags_json TEXT,
  raw_json TEXT,
  discovered_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_author
ON models(author);

CREATE INDEX IF NOT EXISTS idx_models_status
ON models(status);

CREATE TABLE IF NOT EXISTS worker_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  repo_id TEXT,
  source TEXT,
  message TEXT,
  created_at TEXT NOT NULL
);
