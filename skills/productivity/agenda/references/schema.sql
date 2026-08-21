-- SQLite schema for Hermes Agenda system
CREATE TABLE IF NOT EXISTS agenda (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT,                  -- e.g., 'research', 'skill', 'voice', 'health', 'code'
    kind TEXT,                    -- e.g., 'paper', 'experiment', 'bugfix', 'feature'
    title TEXT NOT NULL,          -- short summary / task title
    detail TEXT,                  -- steps, context, or reference links
    priority INTEGER DEFAULT 3,   -- 1 = highest, 5 = lowest
    status TEXT DEFAULT 'pending',-- pending | active | done | recurring
    cooldown_days INTEGER DEFAULT 0, -- >0 => recurring, re-arms after cooldown days
    last_done TEXT,               -- ISO-8601 timestamp when last completed
    times_done INTEGER DEFAULT 0, -- execution count
    created TEXT NOT NULL,        -- ISO-8601 creation timestamp
    note TEXT,                    -- free-form execution notes
    surfaced INTEGER DEFAULT 0    -- 1 if already surfaced in a check-in
);

CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,             -- ISO-8601 timestamp
    agenda_id INTEGER NOT NULL,
    title TEXT,
    outcome TEXT,
    surfaced INTEGER DEFAULT 0,
    FOREIGN KEY (agenda_id) REFERENCES agenda(id)
);

CREATE TABLE IF NOT EXISTS sparks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation TEXT,             -- raw observation that sparked the idea
    idea TEXT NOT NULL,           -- the idea itself
    domain TEXT,                  -- topical domain
    score REAL,                   -- heuristic score (0.0 - 1.0)
    confidence REAL,              -- confidence estimate (0.0 - 1.0)
    decision TEXT,                -- 'pursue', 'defer', 'reject'
    status TEXT DEFAULT 'open',   -- open | pursued | declined
    kill_criteria TEXT,           -- falsification criteria
    created TEXT NOT NULL,        -- ISO-8601 creation timestamp
    note TEXT
);

-- Indexes for fast queue queries and status rollups
CREATE INDEX IF NOT EXISTS idx_agenda_priority_status ON agenda(priority, status);
CREATE INDEX IF NOT EXISTS idx_agenda_domain_kind ON agenda(domain, kind);
CREATE INDEX IF NOT EXISTS idx_log_agenda_id ON log(agenda_id);
CREATE INDEX IF NOT EXISTS idx_sparks_domain ON sparks(domain);
