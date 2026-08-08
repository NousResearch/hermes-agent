-- Long-Term Fact Store — Database Schema
-- Create this database with: sqlite3 ~/.hermes/facts.db < schema.sql
-- Or let the script auto-create it on first use.

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    meta_tags TEXT NOT NULL DEFAULT '[]',
    category TEXT DEFAULT NULL,
    date_created TEXT NOT NULL,
    last_used TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_facts_meta_tags ON facts(meta_tags);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_last_used ON facts(last_used);
CREATE INDEX IF NOT EXISTS idx_facts_date_created ON facts(date_created);