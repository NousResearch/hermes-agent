-- Trace System V2 - Optimized SQLite Schema
-- Three-level indexing: session_id -> trace_id -> tool_call_id

-- Sessions table (Level 1)
CREATE TABLE IF NOT EXISTS trace_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    event_count INTEGER NOT NULL DEFAULT 0,
    trace_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for session queries
CREATE INDEX IF NOT EXISTS idx_sessions_status ON trace_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON trace_sessions(started_at);

-- Traces table (Level 2)
CREATE TABLE IF NOT EXISTS trace_traces (
    trace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    event_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, trace_id),
    FOREIGN KEY (session_id) REFERENCES trace_sessions(session_id) ON DELETE CASCADE
);

-- Indexes for trace queries
CREATE INDEX IF NOT EXISTS idx_traces_session_id ON trace_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_status ON trace_traces(status);
CREATE INDEX IF NOT EXISTS idx_traces_started_at ON trace_traces(started_at);

-- Tool calls table (Level 3)
CREATE TABLE IF NOT EXISTS trace_tool_calls (
    tool_call_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT,  -- Compressed/serialized
    tool_result TEXT,  -- Compressed
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, trace_id, tool_call_id),
    FOREIGN KEY (session_id, trace_id) REFERENCES trace_traces(session_id, trace_id) ON DELETE CASCADE
);

-- Indexes for tool call queries
CREATE INDEX IF NOT EXISTS idx_tool_calls_trace_id ON trace_tool_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id ON trace_tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON trace_tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_status ON trace_tool_calls(status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_started_at ON trace_tool_calls(started_at);

-- Events table (detailed event log)
CREATE TABLE IF NOT EXISTS trace_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    tool_call_id TEXT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    duration_ms REAL,
    tool_name TEXT,
    tool_args TEXT,  -- Compressed
    tool_result TEXT,  -- Compressed
    error TEXT,
    model TEXT,
    message_count INTEGER,
    response_preview TEXT,
    extra TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES trace_sessions(session_id) ON DELETE CASCADE
);

-- Indexes for event queries (most common access patterns)
CREATE INDEX IF NOT EXISTS idx_events_session_trace ON trace_events(session_id, trace_id);
CREATE INDEX IF NOT EXISTS idx_events_session_tool_call ON trace_events(session_id, tool_call_id) WHERE tool_call_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON trace_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON trace_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_tool_name ON trace_events(tool_name) WHERE tool_name IS NOT NULL;

-- Composite index for three-level lookup
CREATE INDEX IF NOT EXISTS idx_events_three_level ON trace_events(session_id, trace_id, tool_call_id);

-- Error patterns table (for analysis)
CREATE TABLE IF NOT EXISTS trace_error_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_hash TEXT NOT NULL UNIQUE,
    error_message TEXT NOT NULL,
    tool_name TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    resolved INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for error pattern queries
CREATE INDEX IF NOT EXISTS idx_error_patterns_tool_name ON trace_error_patterns(tool_name);
CREATE INDEX IF NOT EXISTS idx_error_patterns_last_seen ON trace_error_patterns(last_seen);
CREATE INDEX IF NOT EXISTS idx_error_patterns_occurrence ON trace_error_patterns(occurrence_count DESC);

-- Feedback table (for user feedback on traces)
CREATE TABLE IF NOT EXISTS trace_feedbacks (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    tool_call_id TEXT,
    rating INTEGER,  -- 1-5 scale
    feedback_text TEXT,
    feedback_type TEXT,  -- 'positive', 'negative', 'neutral', 'suggestion'
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES trace_sessions(session_id) ON DELETE CASCADE
);

-- Index for feedback queries
CREATE INDEX IF NOT EXISTS idx_feedbacks_session_id ON trace_feedbacks(session_id);
CREATE INDEX IF NOT EXISTS idx_feedbacks_trace_id ON trace_feedbacks(trace_id);
CREATE INDEX IF NOT EXISTS idx_feedbacks_rating ON trace_feedbacks(rating);

-- Tool statistics table (aggregated stats for performance)
CREATE TABLE IF NOT EXISTS trace_tool_stats (
    tool_name TEXT NOT NULL,
    date TEXT NOT NULL,  -- YYYY-MM-DD
    call_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    total_duration_ms REAL NOT NULL DEFAULT 0,
    avg_duration_ms REAL,
    max_duration_ms REAL,
    min_duration_ms REAL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (tool_name, date)
);

-- Session boundaries table (for compression boundary detection)
CREATE TABLE IF NOT EXISTS trace_session_boundaries (
    boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    boundary_type TEXT NOT NULL,  -- 'compression', 'time_gap', 'message_count'
    boundary_at TEXT NOT NULL,
    message_count INTEGER,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES trace_sessions(session_id) ON DELETE CASCADE
);

-- Index for boundary queries
CREATE INDEX IF NOT EXISTS idx_boundaries_session_id ON trace_session_boundaries(session_id);
CREATE INDEX IF NOT EXISTS idx_boundaries_type ON trace_session_boundaries(boundary_type);
CREATE INDEX IF NOT EXISTS idx_boundaries_at ON trace_session_boundaries(boundary_at);

-- Views for common queries

-- View: Active sessions with summary
CREATE VIEW IF NOT EXISTS trace_active_sessions AS
SELECT 
    s.session_id,
    s.started_at,
    s.status,
    s.event_count,
    s.trace_count,
    COUNT(DISTINCT t.trace_id) as active_traces,
    COUNT(DISTINCT tc.tool_call_id) as active_tool_calls,
    MAX(e.timestamp) as last_activity
FROM trace_sessions s
LEFT JOIN trace_traces t ON s.session_id = t.session_id AND t.status = 'active'
LEFT JOIN trace_tool_calls tc ON s.session_id = tc.session_id AND tc.status = 'running'
LEFT JOIN trace_events e ON s.session_id = e.session_id
WHERE s.status = 'active'
GROUP BY s.session_id;

-- View: Tool performance summary
CREATE VIEW IF NOT EXISTS trace_tool_performance AS
SELECT 
    tool_name,
    COUNT(*) as total_calls,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
    AVG(duration_ms) as avg_duration_ms,
    MAX(duration_ms) as max_duration_ms,
    MIN(duration_ms) as min_duration_ms,
    SUM(duration_ms) as total_duration_ms
FROM trace_tool_calls
WHERE duration_ms IS NOT NULL
GROUP BY tool_name;

-- View: Recent errors
CREATE VIEW IF NOT EXISTS trace_recent_errors AS
SELECT 
    e.session_id,
    e.trace_id,
    e.tool_call_id,
    e.tool_name,
    e.error,
    e.timestamp,
    s.started_at as session_started
FROM trace_events e
LEFT JOIN trace_sessions s ON e.session_id = s.session_id
WHERE e.error IS NOT NULL
ORDER BY e.timestamp DESC
LIMIT 100;

-- Triggers for maintaining counts

-- Trigger: Update session event_count on insert
CREATE TRIGGER IF NOT EXISTS update_session_event_count
AFTER INSERT ON trace_events
BEGIN
    UPDATE trace_sessions 
    SET event_count = event_count + 1,
        updated_at = datetime('now')
    WHERE session_id = NEW.session_id;
END;

-- Trigger: Update trace event_count on insert
CREATE TRIGGER IF NOT EXISTS update_trace_event_count
AFTER INSERT ON trace_events
BEGIN
    UPDATE trace_traces 
    SET event_count = event_count + 1,
        updated_at = datetime('now')
    WHERE session_id = NEW.session_id AND trace_id = NEW.trace_id;
END;

-- Trigger: Update tool call status on complete
CREATE TRIGGER IF NOT EXISTS update_tool_call_on_complete
AFTER UPDATE ON trace_tool_calls
WHEN NEW.status = 'completed' AND OLD.status = 'running'
BEGIN
    UPDATE trace_traces
    SET tool_call_count = tool_call_count + 1,
        updated_at = datetime('now')
    WHERE session_id = NEW.session_id AND trace_id = NEW.trace_id;
END;

-- Trigger: Update error counts
CREATE TRIGGER IF NOT EXISTS update_error_counts
AFTER INSERT ON trace_events
WHEN NEW.error IS NOT NULL
BEGIN
    UPDATE trace_traces
    SET error_count = error_count + 1,
        updated_at = datetime('now')
    WHERE session_id = NEW.session_id AND trace_id = NEW.trace_id;
END;

-- Trigger: Update tool stats on tool call completion
CREATE TRIGGER IF NOT EXISTS update_tool_stats
AFTER UPDATE ON trace_tool_calls
WHEN NEW.status IN ('completed', 'error') AND OLD.status = 'running'
BEGIN
    INSERT OR REPLACE INTO trace_tool_stats (
        tool_name, date, call_count, success_count, error_count,
        total_duration_ms, avg_duration_ms, max_duration_ms, min_duration_ms, updated_at
    )
    VALUES (
        NEW.tool_name,
        date(NEW.completed_at),
        COALESCE((SELECT call_count FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + 1,
        COALESCE((SELECT success_count FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
        COALESCE((SELECT error_count FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + CASE WHEN NEW.status = 'error' THEN 1 ELSE 0 END,
        COALESCE((SELECT total_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + COALESCE(NEW.duration_ms, 0),
        (COALESCE((SELECT total_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + COALESCE(NEW.duration_ms, 0)) / 
            (COALESCE((SELECT call_count FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0) + 1),
        MAX(COALESCE(NEW.duration_ms, 0), COALESCE((SELECT max_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)), 0)),
        CASE 
            WHEN (SELECT min_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)) IS NULL THEN NEW.duration_ms
            WHEN NEW.duration_ms IS NULL THEN (SELECT min_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at))
            ELSE MIN(NEW.duration_ms, (SELECT min_duration_ms FROM trace_tool_stats WHERE tool_name = NEW.tool_name AND date = date(NEW.completed_at)))
        END,
        datetime('now')
    );
END;