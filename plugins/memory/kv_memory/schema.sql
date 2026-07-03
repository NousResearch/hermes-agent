-- kv-memory: Model-native semantic memory schema
-- Mode A: Q4-quantized KV-cache states per turn
-- Mode B: Hidden-state embeddings per turn
-- Both: text summaries for fallback and inspection

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    model_id TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'  -- JSON: platform, agent_context, etc.
);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,                -- UUID
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    model_id TEXT NOT NULL DEFAULT '',
    num_layers INTEGER,                 -- layers used for embedding
    num_kv_heads INTEGER,               -- KV heads per layer
    head_dim INTEGER,                   -- dimension per head
    embedding_dim INTEGER NOT NULL,     -- total embedding dimension
    q4_embedding BLOB,                  -- Q4-quantized embedding (packed uint8)
    q4_scales BLOB,                     -- per-channel float32 scales
    fp16_embedding BLOB,                -- raw FP16 embedding (optional, for fidelity checks)
    summary_text TEXT,                  -- lightweight text summary
    tool_calls TEXT DEFAULT '[]',       -- JSON array of tool calls this turn
    user_feedback INTEGER DEFAULT 0,    -- -1 (bad), 0 (neutral), 1 (good)
    metadata TEXT DEFAULT '{}'          -- JSON: topic tags, importance, source
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_turns_model ON turns(model_id);

-- Cross-session links for causal/semantic relationship tracking
CREATE TABLE IF NOT EXISTS session_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    similarity REAL NOT NULL DEFAULT 0.0,
    link_type TEXT DEFAULT 'semantic',  -- 'semantic', 'causal', 'continuation'
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_links_source ON session_links(source_session_id);
CREATE INDEX IF NOT EXISTS idx_session_links_target ON session_links(target_session_id);

-- Embedding index for ANN search (requires sqlite-vec extension)
-- This is OPTIONAL. Created separately by storage.py with error handling.
-- If sqlite-vec is not installed, the retriever falls back to brute-force
-- cosine similarity (linear scan), which is fast enough for <100K embeddings.
