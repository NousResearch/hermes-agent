"""PostgreSQL schema for the existing SessionDB tables and SQL operations."""

from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS

SEARCH_CONTENT_CHARS = 8_192
SEARCH_TOOL_CALLS_CHARS = 4_096
SEARCH_TOOL_NAME_CHARS = 256
POSTGRES_SCHEMA_VERSION = 3

DELIVERY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS delivery_obligations (
    obligation_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    thread_id TEXT,
    content TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts BIGINT NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    owner_pid BIGINT,
    owner_started_at BIGINT,
    last_error TEXT,
    adapter_profile TEXT,
    owner_token TEXT,
    lease_expires_at DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_delivery_obligations_recovery
    ON delivery_obligations(state, lease_expires_at, updated_at);
CREATE INDEX IF NOT EXISTS idx_delivery_obligations_runtime
    ON delivery_obligations(platform, adapter_profile, owner_token)
    WHERE state = 'failed';
"""

CRON_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    source TEXT NOT NULL,
    process_id TEXT NOT NULL,
    pid BIGINT NOT NULL,
    process_started_at BIGINT,
    status TEXT NOT NULL CHECK(status IN
        ('claimed', 'running', 'completed', 'failed', 'unknown')),
    handoff_pending BIGINT NOT NULL DEFAULT 0,
    handoff_started_at DOUBLE PRECISION,
    claimed_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    delivery_outcome TEXT,
    scheduled_instant TEXT,
    owner_lease_expires_at DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_executions_job_claimed
    ON executions(job_id, claimed_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_executions_status_claimed
    ON executions(status, claimed_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_executions_occurrence
    ON executions(job_id, scheduled_instant) WHERE status = 'completed';
CREATE INDEX IF NOT EXISTS idx_executions_recovery
    ON executions(status, owner_lease_expires_at);

CREATE TABLE IF NOT EXISTS cron_incidents (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    error_sig TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('detected', 'alerted', 'resolved', 'closed')),
    failure_type TEXT NOT NULL DEFAULT 'unknown',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    acked_at TEXT,
    alerted_at TEXT,
    closed_at TEXT,
    error TEXT NOT NULL,
    output_file TEXT
);

CREATE INDEX IF NOT EXISTS idx_cron_incidents_job ON cron_incidents(job_id);
CREATE INDEX IF NOT EXISTS idx_cron_incidents_state ON cron_incidents(state);

CREATE TABLE IF NOT EXISTS deliveries (
    execution_id TEXT PRIMARY KEY,
    job_json TEXT NOT NULL,
    content TEXT NOT NULL,
    for_failure BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN
        ('pending', 'delivering', 'delivered', 'failed', 'unknown')),
    owner_process_id TEXT,
    owner_pid BIGINT,
    owner_started_at BIGINT,
    owner_lease_expires_at DOUBLE PRECISION,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_deliveries_pending
    ON deliveries(status, created_at, execution_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_recovery
    ON deliveries(status, owner_lease_expires_at);

CREATE TABLE IF NOT EXISTS delivery_tombstones (
    execution_id TEXT PRIMARY KEY,
    terminal_status TEXT NOT NULL CHECK(terminal_status IN
        ('delivered', 'failed', 'unknown')),
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS cron_notepad (
    job_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, key)
);
"""

POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized BIGINT DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at DOUBLE PRECISION NOT NULL,
    ended_at DOUBLE PRECISION,
    end_reason TEXT,
    message_count BIGINT DEFAULT 0,
    tool_call_count BIGINT DEFAULT 0,
    input_tokens BIGINT DEFAULT 0,
    output_tokens BIGINT DEFAULT 0,
    cache_read_tokens BIGINT DEFAULT 0,
    cache_write_tokens BIGINT DEFAULT 0,
    reasoning_tokens BIGINT DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation BIGINT NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd DOUBLE PRECISION,
    actual_cost_usd DOUBLE PRECISION,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at DOUBLE PRECISION,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count BIGINT DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until DOUBLE PRECISION,
    compression_failure_error TEXT,
    compression_fallback_streak BIGINT NOT NULL DEFAULT 0,
    compression_ineffective_count BIGINT NOT NULL DEFAULT 0,
    compression_recovery_deadline DOUBLE PRECISION,
    profile_name TEXT,
    transport_profile TEXT,
    rewind_count BIGINT NOT NULL DEFAULT 0,
    archived BIGINT NOT NULL DEFAULT 0,
    pinned BIGINT NOT NULL DEFAULT 0,
    hidden BIGINT NOT NULL DEFAULT 0,
    last_read_at DOUBLE PRECISION,
    tool_names TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp DOUBLE PRECISION NOT NULL,
    token_count BIGINT,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed BIGINT DEFAULT 0,
    _compressed_summary BIGINT NOT NULL DEFAULT 0,
    active BIGINT NOT NULL DEFAULT 1,
    compacted BIGINT NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    display_identity BYTEA,
    display_order BIGINT
);

CREATE TABLE IF NOT EXISTS session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count BIGINT NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cache_read_tokens BIGINT NOT NULL DEFAULT 0,
    cache_write_tokens BIGINT NOT NULL DEFAULT 0,
    reasoning_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    actual_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen DOUBLE PRECISION,
    last_seen DOUBLE PRECISION,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE IF NOT EXISTS gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversation_generations (
    source TEXT NOT NULL,
    session_key TEXT NOT NULL,
    generation BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (source, session_key)
);

CREATE TABLE IF NOT EXISTS gateway_heartbeats (
    backend_id TEXT PRIMARY KEY,
    pid BIGINT NOT NULL,
    started_at DOUBLE PRECISION NOT NULL,
    last_heartbeat DOUBLE PRECISION NOT NULL,
    profile TEXT NOT NULL DEFAULT '',
    host TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at DOUBLE PRECISION NOT NULL,
    completed_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts BIGINT NOT NULL DEFAULT 0,
    delivered_at DOUBLE PRECISION,
    owner_pid BIGINT,
    owner_started_at BIGINT,
    task_json TEXT,
    delivery_claim TEXT,
    delivery_claimed_at DOUBLE PRECISION,
    origin_session_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session
    ON messages(session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires ON session_turn_leases(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model);
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_display_page
    ON messages(session_id, display_order, active DESC, id DESC)
    WHERE active = 1 OR compacted = 1;
CREATE INDEX IF NOT EXISTS idx_messages_display_backfill
    ON messages(session_id) WHERE (display_order IS NULL OR display_identity IS NULL)
    AND (active = 1 OR compacted = 1);
CREATE INDEX IF NOT EXISTS idx_messages_display_identity
    ON messages(session_id, display_identity, display_order)
    WHERE display_identity IS NOT NULL AND (active = 1 OR compacted = 1);
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_effective_activity
    ON sessions(COALESCE(last_activity_at, started_at) DESC, started_at DESC);
"""

POSTGRES_SCHEMA_SQL += DELIVERY_SCHEMA_SQL
POSTGRES_SCHEMA_SQL += CRON_SCHEMA_SQL

FUNCTION_SQL = """
CREATE FUNCTION json_valid(raw TEXT) RETURNS INTEGER
LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE WHEN raw IS JSON THEN 1 ELSE 0 END
$$;
CREATE FUNCTION json_extract(raw TEXT, path TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT raw::jsonb #>> string_to_array(substr(path, 3), '.')
$$;
CREATE FUNCTION json_type(raw TEXT, path TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT CASE jsonb_typeof(raw::jsonb #> string_to_array(substr(path, 3), '.'))
        WHEN 'string' THEN 'text'
        ELSE jsonb_typeof(raw::jsonb #> string_to_array(substr(path, 3), '.')) END
$$;
CREATE FUNCTION json_remove(raw TEXT, path TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT (raw::jsonb #- string_to_array(substr(path, 3), '.'))::text
$$;
CREATE FUNCTION json_set(raw TEXT, path TEXT, value TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
    SELECT jsonb_set(raw::jsonb, string_to_array(substr(path, 3), '.'),
                    COALESCE(to_jsonb(value), 'null'::jsonb), true)::text
$$;
CREATE FUNCTION instr(haystack TEXT, needle TEXT) RETURNS INTEGER
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$ SELECT strpos(haystack, needle) $$;
CREATE FUNCTION substr(value TEXT, start_pos INTEGER) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT pg_catalog.substr(value, CASE WHEN start_pos < 0
        THEN greatest(length(value) + start_pos + 1, 1) ELSE start_pos END)
$$;
CREATE FUNCTION substr(value TEXT, start_pos INTEGER, count INTEGER) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT pg_catalog.substr(value, CASE WHEN start_pos < 0
        THEN greatest(length(value) + start_pos + 1, 1) ELSE start_pos END, count)
$$;
"""

DISPLAY_SQL = """
CREATE FUNCTION messages_display_order_insert() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.display_order IS NULL THEN
        NEW.display_order := COALESCE((
            SELECT display_order FROM messages
            WHERE session_id = NEW.session_id AND id <> NEW.id
              AND (active = 1 OR compacted = 1)
              AND display_identity = NEW.display_identity AND display_order IS NOT NULL
            ORDER BY display_order LIMIT 1
        ), NEW.id);
    END IF;
    RETURN NEW;
END
$$;
CREATE TRIGGER messages_display_order_insert BEFORE INSERT ON messages
FOR EACH ROW EXECUTE FUNCTION messages_display_order_insert();

CREATE FUNCTION messages_display_visibility_update() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.active = 1 OR NEW.compacted = 1) = (OLD.active = 1 OR OLD.compacted = 1) THEN
        RETURN NULL;
    END IF;
    IF NEW.active = 1 OR NEW.compacted = 1 THEN
        UPDATE messages SET display_order = LEAST(NEW.id, COALESCE((
            SELECT display_order FROM messages
            WHERE session_id = NEW.session_id AND id <> NEW.id
              AND (active = 1 OR compacted = 1)
              AND display_identity = NEW.display_identity AND display_order IS NOT NULL
            ORDER BY display_order LIMIT 1
        ), NEW.id)) WHERE id = NEW.id;
        UPDATE messages SET display_order = (SELECT display_order FROM messages WHERE id = NEW.id)
        WHERE session_id = NEW.session_id AND id <> NEW.id AND (active = 1 OR compacted = 1)
          AND display_identity = NEW.display_identity;
    ELSE
        UPDATE messages SET display_order = (
            SELECT MIN(peer.id) FROM messages AS peer
            WHERE peer.session_id = OLD.session_id AND (peer.active = 1 OR peer.compacted = 1)
              AND peer.display_identity = OLD.display_identity
        ) WHERE session_id = OLD.session_id AND (active = 1 OR compacted = 1)
          AND display_identity = OLD.display_identity;
    END IF;
    RETURN NULL;
END
$$;
CREATE TRIGGER messages_display_visibility_update AFTER UPDATE OF active, compacted ON messages
FOR EACH ROW EXECUTE FUNCTION messages_display_visibility_update();

CREATE FUNCTION messages_display_identity_update() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE messages SET display_identity = NULL, display_order = NULL
    WHERE id = NEW.id OR (
        session_id = OLD.session_id AND display_identity = OLD.display_identity
        AND (active = 1 OR compacted = 1)
    );
    RETURN NULL;
END
$$;
CREATE TRIGGER messages_display_identity_update
AFTER UPDATE OF role, content, timestamp, tool_call_id, tool_calls, tool_name,
                display_kind, display_metadata ON messages
FOR EACH ROW EXECUTE FUNCTION messages_display_identity_update();

CREATE FUNCTION messages_display_identity_delete() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE messages SET display_order = (
        SELECT MIN(peer.id) FROM messages AS peer
        WHERE peer.session_id = OLD.session_id AND (peer.active = 1 OR peer.compacted = 1)
          AND peer.display_identity = OLD.display_identity
    ) WHERE session_id = OLD.session_id AND (active = 1 OR compacted = 1)
      AND display_identity = OLD.display_identity;
    RETURN NULL;
END
$$;
CREATE TRIGGER messages_display_identity_delete AFTER DELETE ON messages
FOR EACH ROW WHEN (OLD.active = 1 OR OLD.compacted = 1)
EXECUTE FUNCTION messages_display_identity_delete();

CREATE TABLE state_change_revision (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    revision BIGINT NOT NULL
);
INSERT INTO state_change_revision VALUES (1, 0);
CREATE FUNCTION bump_state_change_revision() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE state_change_revision SET revision = revision + 1 WHERE id = 1;
    RETURN NULL;
END
$$;
CREATE TRIGGER sessions_changed AFTER INSERT OR UPDATE OR DELETE ON sessions
FOR EACH STATEMENT EXECUTE FUNCTION bump_state_change_revision();
CREATE TRIGGER messages_changed AFTER INSERT OR UPDATE OR DELETE ON messages
FOR EACH STATEMENT EXECUTE FUNCTION bump_state_change_revision();
"""

SEARCH_SQL = f"""
ALTER TABLE messages ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('simple', left(coalesce(content, ''),
        CASE WHEN role = 'tool' THEN {FTS_TOOL_CONTENT_PREFIX_CHARS} ELSE {SEARCH_CONTENT_CHARS} END)
        || ' ' || left(coalesce(tool_name, ''), {SEARCH_TOOL_NAME_CHARS})
        || ' ' || left(coalesce(tool_calls, ''), {SEARCH_TOOL_CALLS_CHARS}))
) STORED;
CREATE INDEX idx_messages_search_vector ON messages USING GIN(search_vector);
CREATE INDEX idx_messages_search_overflow ON messages(id)
    WHERE (role <> 'tool' AND length(content) > {SEARCH_CONTENT_CHARS})
        OR length(tool_calls) > {SEARCH_TOOL_CALLS_CHARS}
        OR length(tool_name) > {SEARCH_TOOL_NAME_CHARS};
"""

TELEGRAM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    enabled BIGINT NOT NULL DEFAULT 1,
                    activated_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    has_topics_enabled BIGINT,
                    allows_users_to_create_topics BIGINT,
                    capability_checked_at DOUBLE PRECISION,
                    intro_message_id TEXT,
                    pinned_message_id TEXT,
                    PRIMARY KEY (profile_name, chat_id)
                );
CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (profile_name, chat_id, thread_id)
                );
CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session ON telegram_dm_topic_bindings(session_id);
CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user ON telegram_dm_topic_bindings(profile_name,user_id,chat_id);
"""
