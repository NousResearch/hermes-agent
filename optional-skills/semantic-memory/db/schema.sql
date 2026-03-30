-- Memory Engine SQLite Schema (Fase 1-2)
-- Created: 2026-03-23
-- Purpose: Store facts, embeddings, and temporal metadata

-- Enable vector support
-- Requires: sqlite-vec extension loaded

-- Main facts table with vector embeddings
CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    
    -- Embeddings (vector[384] for BAAI/bge-small-en-v1.5)
    embedding VECTOR(384),
    
    -- Source tracking
    source TEXT NOT NULL,  -- 'daily', 'memory.md', 'skill', 'agents', 'obsidian'
    source_path TEXT,      -- full path or reference
    source_line INT,       -- line number if applicable
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Temporal weighting
    last_referenced TIMESTAMP,
    referenced_count INT DEFAULT 0,
    decay_weight REAL DEFAULT 1.0,  -- decays over time
    freshness_tier TEXT DEFAULT 'medium',  -- 'recent', 'medium', 'old', 'archive'
    
    -- Quality metrics
    confidence REAL DEFAULT 0.9,  -- 0-1, how confident we are in this fact
    is_active BOOLEAN DEFAULT 1,
    is_archived BOOLEAN DEFAULT 0,
    
    -- Classification
    fact_type TEXT,  -- USER_FACT, PREFERENCE, DECISION, etc
    tags TEXT,       -- JSON array of tags
    category TEXT,   -- High-level category for grouping
    
    -- Metadata
    metadata TEXT,   -- JSON: author, agent_identity, version, etc
    
    -- Deduplication
    canonical_id TEXT,  -- if this is duplicate, points to main fact
    similar_facts TEXT, -- JSON array of similar fact IDs
    
    -- Indexing helper
    content_hash TEXT UNIQUE,  -- for dedup detection
    
    FOREIGN KEY(canonical_id) REFERENCES memory_facts(id)
);

-- Temporal decay tracking (for analytics)
CREATE TABLE IF NOT EXISTS temporal_decay_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    old_weight REAL,
    new_weight REAL,
    days_since_created INT,
    decay_reason TEXT,  -- 'time', 'manual_archive', etc
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
);

-- Reference tracking (when facts are used in reasoning)
CREATE TABLE IF NOT EXISTS fact_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    referenced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    relevance_feedback TEXT,  -- 'helpful', 'partial', 'not_helpful'
    context TEXT,  -- brief context of how it was used
    FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
);

-- Fact relationships (A mentions B, A contradicts C, etc)
CREATE TABLE IF NOT EXISTS fact_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_fact_id TEXT NOT NULL,
    target_fact_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- 'mentions', 'contradicts', 'extends', 'obsoletes', 'related_to'
    confidence REAL DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_fact_id) REFERENCES memory_facts(id),
    FOREIGN KEY(target_fact_id) REFERENCES memory_facts(id)
);

-- Session context cache (for contextual windowing)
CREATE TABLE IF NOT EXISTS session_context (
    session_id TEXT PRIMARY KEY,
    session_date DATE,
    summary TEXT,
    facts TEXT,  -- JSON array of fact IDs in this session
    previous_session_id TEXT,
    next_session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quality metrics and audit trail
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,  -- 'embed', 'update', 'archive', 'deduplicate', 'correction'
    fact_id TEXT,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    agent TEXT,  -- 'hermes', 'katsumi', 'system'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
);

-- Contradiction detection log
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a TEXT NOT NULL,
    fact_id_b TEXT NOT NULL,
    contradiction_type TEXT,  -- 'direct', 'implicit', 'temporal'
    confidence REAL,
    resolution_status TEXT DEFAULT 'unresolved',  -- 'unresolved', 'merged', 'archived', 'confirmed_diff'
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_note TEXT,
    FOREIGN KEY(fact_id_a) REFERENCES memory_facts(id),
    FOREIGN KEY(fact_id_b) REFERENCES memory_facts(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_source ON memory_facts(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON memory_facts(created_at);
CREATE INDEX IF NOT EXISTS idx_last_referenced ON memory_facts(last_referenced);
CREATE INDEX IF NOT EXISTS idx_decay_weight ON memory_facts(decay_weight);
CREATE INDEX IF NOT EXISTS idx_is_active ON memory_facts(is_active);
CREATE INDEX IF NOT EXISTS idx_canonical_id ON memory_facts(canonical_id);
CREATE INDEX IF NOT EXISTS idx_content_hash ON memory_facts(content_hash);
CREATE INDEX IF NOT EXISTS idx_fact_type ON memory_facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_category ON memory_facts(category);

-- Vector index (for semantic search)
-- Note: sqlite-vec syntax, requires extension
-- CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_vec USING vec0(
--     embedding(memory_facts)
-- );

-- Metadata table for engine state
CREATE TABLE IF NOT EXISTS engine_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initialize metadata
INSERT OR IGNORE INTO engine_metadata (key, value) VALUES
    ('version', '1.0'),
    ('created_at', CURRENT_TIMESTAMP),
    ('last_processed', NULL),
    ('total_facts', '0'),
    ('total_embeddings', '0'),
    ('status', 'initialized');
