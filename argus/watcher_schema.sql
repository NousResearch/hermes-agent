-- Agent Watcher Database Schema
-- Tracks all agent sessions, metrics, entropy, and actions

-- Sessions being monitored
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    session_type TEXT NOT NULL, -- 'cron', 'delegate_task', 'manual', 'orchestrator'
    job_id TEXT, -- cron job ID if applicable
    task_description TEXT,
    model TEXT,
    provider TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active', -- 'active', 'restarted', 'killed', 'completed'
    restart_count INTEGER DEFAULT 0,
    kill_count INTEGER DEFAULT 0,
    quality_gate_score REAL,
    entropy_score REAL,
    token_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    metadata TEXT -- JSON blob for additional info
);

-- Tool calls made by each session
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT, -- JSON blob of arguments
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    result_size INTEGER,
    file_changed BOOLEAN DEFAULT FALSE,
    file_path TEXT,
    file_hash_before TEXT,
    file_hash_after TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- File changes detected
CREATE TABLE IF NOT EXISTS file_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    change_type TEXT NOT NULL, -- 'created', 'modified', 'deleted'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tool_call_id INTEGER,
    hash_before TEXT,
    hash_after TEXT,
    size_before INTEGER,
    size_after INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- Terminal commands executed
CREATE TABLE IF NOT EXISTS terminal_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    command TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tool_call_id INTEGER,
    exit_code INTEGER,
    stdout_size INTEGER,
    stderr_size INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (tool_call_id) REFERENCES tool_calls(id)
);

-- Quality metrics tracked
CREATE TABLE IF NOT EXISTS quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    metric_type TEXT NOT NULL, -- 'trajectory_quality', 'fact_quality', 'pipeline_compliance'
    metric_value REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_file TEXT,
    details TEXT, -- JSON blob
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Entropy detections
CREATE TABLE IF NOT EXISTS entropy_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    entropy_type TEXT NOT NULL, -- 'repeat_tool_calls', 'repeat_commands', 'stuck_loop', 'no_file_changes', 'error_cascade', 'budget_pressure'
    severity TEXT NOT NULL, -- 'warning', 'critical'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_count INTEGER DEFAULT 1,
    details TEXT, -- JSON blob with specifics
    tool_call_ids TEXT, -- JSON array of related tool call IDs
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Actions taken by watcher
CREATE TABLE IF NOT EXISTS watcher_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    action_type TEXT NOT NULL, -- 'inject_prompt', 'restart', 'kill', 'alert'
    action_reason TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN,
    details TEXT, -- JSON blob
    entropy_detection_id INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (entropy_detection_id) REFERENCES entropy_detections(id)
);

-- Notifications sent
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    notification_type TEXT NOT NULL, -- 'restart', 'quality_drop', 'kill', 'daily_digest'
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivered BOOLEAN DEFAULT FALSE,
    delivery_error TEXT,
    watcher_action_id INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (watcher_action_id) REFERENCES watcher_actions(id)
);

-- Prime directive alignment checks
CREATE TABLE IF NOT EXISTS directive_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    check_type TEXT NOT NULL, -- 'pipeline_compliance', 'quality_gate', 'trajectory_generation', 'fact_extraction'
    passed BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT, -- JSON blob with specifics
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp ON tool_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_file_changes_session ON file_changes(session_id);
CREATE INDEX IF NOT EXISTS idx_terminal_commands_session ON terminal_commands(session_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_session ON quality_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_entropy_detections_session ON entropy_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_watcher_actions_session ON watcher_actions(session_id);
CREATE INDEX IF NOT EXISTS idx_notifications_session ON notifications(session_id);
CREATE INDEX IF NOT EXISTS idx_directive_checks_session ON directive_checks(session_id);

-- Views for common queries
CREATE VIEW IF NOT EXISTS active_sessions AS
SELECT * FROM sessions WHERE status = 'active';

CREATE VIEW IF NOT EXISTS high_entropy_sessions AS
SELECT s.*, ed.entropy_type, ed.severity, ed.detection_count
FROM sessions s
JOIN entropy_detections ed ON s.session_id = ed.session_id
WHERE ed.severity = 'critical' AND ed.detection_count >= 3;

CREATE VIEW IF NOT EXISTS low_quality_sessions AS
SELECT s.*, qm.metric_type, qm.metric_value
FROM sessions s
JOIN quality_metrics qm ON s.session_id = qm.session_id
WHERE qm.metric_value < 0.92 AND qm.timestamp > datetime('now', '-1 hour');

CREATE VIEW IF NOT EXISTS recent_actions AS
SELECT wa.*, s.session_type, s.job_id
FROM watcher_actions wa
JOIN sessions s ON wa.session_id = s.session_id
WHERE wa.timestamp > datetime('now', '-24 hours')
ORDER BY wa.timestamp DESC;