"""Versioned, reversible SQLite schema for Communication Core."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

LATEST_SCHEMA_VERSION = 2


_V1_UP = r"""
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE persons (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    pii_policy TEXT NOT NULL DEFAULT 'minimal'
        CHECK (pii_policy IN ('minimal', 'standard', 'restricted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE connected_accounts (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    account_namespace TEXT NOT NULL,
    label TEXT NOT NULL,
    owner_profile TEXT NOT NULL,
    credential_ref TEXT,
    browser_profile_ref TEXT,
    auth_status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (auth_status IN ('unknown', 'healthy', 'reauth_required', 'failed')),
    health_status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (health_status IN ('unknown', 'healthy', 'degraded', 'failed')),
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    write_policy TEXT NOT NULL DEFAULT 'disabled'
        CHECK (write_policy IN ('disabled', 'draft_only', 'approval_required')),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    rate_limit_state_json TEXT NOT NULL DEFAULT '{}',
    last_seen_at TEXT,
    last_successful_sync_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (provider, account_namespace),
    UNIQUE (owner_profile, provider, label)
);

CREATE TABLE platform_identities (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    observed_via_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    external_id TEXT NOT NULL,
    display_name TEXT,
    profile_ref TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    sync_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (provider, observed_via_account_id, external_id)
);

CREATE TABLE contact_endpoints (
    id TEXT PRIMARY KEY,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    platform_identity_id TEXT NOT NULL REFERENCES platform_identities(id),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'disabled', 'blocked')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (connected_account_id, platform_identity_id)
);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'direct'
        CHECK (kind IN ('direct', 'group', 'channel')),
    title TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    sync_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (provider, connected_account_id, external_id)
);

CREATE TABLE participants (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    platform_identity_id TEXT NOT NULL REFERENCES platform_identities(id),
    role TEXT NOT NULL DEFAULT 'member',
    PRIMARY KEY (conversation_id, platform_identity_id)
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    provider TEXT NOT NULL,
    external_id TEXT,
    stable_fingerprint TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('incoming', 'outgoing', 'system')),
    sender_identity_id TEXT REFERENCES platform_identities(id),
    body TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    sync_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE (provider, connected_account_id, external_id),
    UNIQUE (connected_account_id, stable_fingerprint)
);

CREATE TABLE sync_runs (
    id TEXT PRIMARY KEY,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT REFERENCES contact_endpoints(id),
    mode TEXT NOT NULL CHECK (mode IN ('full', 'incremental', 'retry')),
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'partial', 'failed')),
    stats_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    retry_of_id TEXT REFERENCES sync_runs(id)
);

CREATE TABLE sync_cursors (
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT REFERENCES contact_endpoints(id),
    resource TEXT NOT NULL,
    cursor_value TEXT NOT NULL,
    sync_version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (connected_account_id, endpoint_id, resource)
);

CREATE TABLE sync_issues (
    id TEXT PRIMARY KEY,
    sync_run_id TEXT NOT NULL REFERENCES sync_runs(id) ON DELETE CASCADE,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT REFERENCES contact_endpoints(id),
    code TEXT NOT NULL,
    detail_redacted TEXT NOT NULL,
    retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'retrying', 'resolved', 'quarantined')),
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE sync_locks (
    connected_account_id TEXT PRIMARY KEY REFERENCES connected_accounts(id),
    owner_token TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    acquired_at TEXT NOT NULL
);

CREATE INDEX idx_identity_person ON platform_identities(person_id);
CREATE INDEX idx_endpoint_account ON contact_endpoints(connected_account_id);
CREATE INDEX idx_conversation_endpoint ON conversations(endpoint_id);
CREATE INDEX idx_message_timeline ON messages(conversation_id, sent_at, id);
CREATE INDEX idx_sync_issue_scope ON sync_issues(connected_account_id, status);
CREATE UNIQUE INDEX idx_sync_cursor_account_resource
ON sync_cursors(connected_account_id, resource) WHERE endpoint_id IS NULL;
"""


_V2_UP = r"""
CREATE TABLE communication_journeys (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (person_id)
);

CREATE TABLE channel_episodes (
    id TEXT PRIMARY KEY,
    journey_id TEXT NOT NULL REFERENCES communication_journeys(id) ON DELETE CASCADE,
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    start_reason TEXT NOT NULL,
    end_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE channel_transitions (
    id TEXT PRIMARY KEY,
    journey_id TEXT NOT NULL REFERENCES communication_journeys(id) ON DELETE CASCADE,
    from_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    to_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    initiator TEXT NOT NULL CHECK (initiator IN ('person', 'user', 'inbound_resume')),
    evidence_type TEXT NOT NULL,
    evidence_ref TEXT NOT NULL,
    happened_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (from_endpoint_id <> to_endpoint_id)
);

CREATE TABLE channel_preferences (
    person_id TEXT NOT NULL REFERENCES persons(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    state TEXT NOT NULL CHECK (state IN ('active', 'paused', 'ended', 'return_by_request', 'blocked')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (person_id, endpoint_id)
);

CREATE TABLE account_link_policies (
    source_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    target_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    allowed INTEGER NOT NULL CHECK (allowed IN (0, 1)),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_account_id, target_account_id),
    CHECK (source_account_id <> target_account_id)
);

CREATE TABLE person_channel_routes (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    source_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    target_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (person_id, source_endpoint_id)
);

CREATE TABLE route_audit (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    source_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    target_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    source_endpoint_id TEXT,
    target_endpoint_id TEXT,
    action TEXT NOT NULL,
    allowed INTEGER NOT NULL CHECK (allowed IN (0, 1)),
    explanation TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE contact_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    exclusion INTEGER NOT NULL DEFAULT 0 CHECK (exclusion IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE contact_group_members (
    group_id TEXT NOT NULL REFERENCES contact_groups(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    PRIMARY KEY (group_id, person_id)
);

CREATE TABLE smart_segments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    query_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE contact_events (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    connected_account_id TEXT REFERENCES connected_accounts(id),
    endpoint_id TEXT REFERENCES contact_endpoints(id),
    event_type TEXT NOT NULL,
    external_id TEXT,
    happened_at TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    data_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    sync_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE (connected_account_id, external_id, event_type)
);

CREATE TABLE commitments (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    message_id TEXT REFERENCES messages(id),
    kind TEXT NOT NULL CHECK (kind IN ('promise', 'agreement', 'unanswered_question', 'follow_up')),
    summary TEXT NOT NULL,
    due_at TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'dismissed')),
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE relationship_states (
    person_id TEXT PRIMARY KEY REFERENCES persons(id),
    priority INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    last_touch_at TEXT,
    next_action_at TEXT,
    next_action_reason TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE identity_merge_audit (
    id TEXT PRIMARY KEY,
    winner_person_id TEXT NOT NULL REFERENCES persons(id),
    merged_person_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('merge', 'unmerge')),
    created_at TEXT NOT NULL
);

CREATE TABLE drafts (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    source_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    source_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    route_version TEXT NOT NULL,
    recipient_preview_json TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'cancelled', 'approved', 'queued', 'invalidated')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE approvals (
    id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL UNIQUE REFERENCES drafts(id),
    actor TEXT NOT NULL,
    person_id TEXT NOT NULL REFERENCES persons(id),
    source_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    source_endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    target_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    recipient_preview_hash TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    route_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'consumed', 'rejected', 'expired', 'invalidated')),
    approved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT
);

CREATE TABLE outbox_items (
    id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL UNIQUE REFERENCES drafts(id),
    approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(id),
    person_id TEXT NOT NULL REFERENCES persons(id),
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    endpoint_id TEXT NOT NULL REFERENCES contact_endpoints(id),
    payload_hash TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'sent', 'failed', 'uncertain', 'cancelled')),
    claim_token TEXT,
    claim_expires_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    postcondition_json TEXT,
    error_redacted TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE outbox_events (
    id TEXT PRIMARY KEY,
    outbox_id TEXT NOT NULL REFERENCES outbox_items(id) ON DELETE CASCADE,
    from_status TEXT,
    to_status TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE greeting_deliveries (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    event_id TEXT NOT NULL REFERENCES contact_events(id),
    local_date TEXT NOT NULL,
    draft_id TEXT REFERENCES drafts(id),
    status TEXT NOT NULL CHECK (status IN ('planned', 'drafted', 'excluded', 'sent')),
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (person_id, event_id, local_date)
);

CREATE TABLE legacy_id_mappings (
    source_system TEXT NOT NULL,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    entity_type TEXT NOT NULL,
    legacy_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    migrated_at TEXT NOT NULL,
    PRIMARY KEY (source_system, connected_account_id, entity_type, legacy_id)
);

CREATE TABLE migration_runs (
    id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'rolled_back')),
    counts_json TEXT NOT NULL DEFAULT '{}',
    reconciliation_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (source_system, source_hash, connected_account_id)
);

CREATE TABLE legacy_records (
    id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    connected_account_id TEXT NOT NULL REFERENCES connected_accounts(id),
    entity_type TEXT NOT NULL,
    legacy_id TEXT NOT NULL,
    person_id TEXT REFERENCES persons(id),
    payload_json TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    migrated_at TEXT NOT NULL,
    UNIQUE (source_system, connected_account_id, entity_type, legacy_id)
);

CREATE TABLE xdom_suggestions (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id),
    public_story_id TEXT NOT NULL,
    public_article_id TEXT,
    matched_topics_json TEXT NOT NULL,
    matched_entities_json TEXT NOT NULL,
    source_urls_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'suggested'
        CHECK (status IN ('suggested', 'drafted', 'dismissed')),
    draft_id TEXT REFERENCES drafts(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (person_id, public_story_id)
);

CREATE INDEX idx_episode_journey_time ON channel_episodes(journey_id, started_at);
CREATE INDEX idx_transition_journey_time ON channel_transitions(journey_id, happened_at);
CREATE INDEX idx_event_person_time ON contact_events(person_id, happened_at);
CREATE UNIQUE INDEX idx_commitment_message_kind
ON commitments(message_id, kind) WHERE message_id IS NOT NULL;
CREATE INDEX idx_draft_scope ON drafts(connected_account_id, endpoint_id, status);
CREATE INDEX idx_outbox_claim ON outbox_items(status, claim_expires_at);
CREATE INDEX idx_xdom_person_status ON xdom_suggestions(person_id, status);
CREATE INDEX idx_legacy_record_scope
ON legacy_records(source_system, connected_account_id, entity_type);

-- SQLite cannot express all account/provider ownership rules as ordinary
-- foreign keys.  These triggers make accidental cross-account joins fail at
-- the storage boundary, including code paths outside CommunicationRepository.
CREATE TRIGGER identity_account_provider_insert
BEFORE INSERT ON platform_identities
WHEN NEW.provider <> (
    SELECT provider FROM connected_accounts WHERE id = NEW.observed_via_account_id
)
BEGIN
    SELECT RAISE(ABORT, 'identity provider/account mismatch');
END;

CREATE TRIGGER identity_account_provider_update
BEFORE UPDATE OF provider, observed_via_account_id ON platform_identities
WHEN NEW.provider <> (
    SELECT provider FROM connected_accounts WHERE id = NEW.observed_via_account_id
)
BEGIN
    SELECT RAISE(ABORT, 'identity provider/account mismatch');
END;

CREATE TRIGGER endpoint_account_identity_insert
BEFORE INSERT ON contact_endpoints
WHEN NEW.connected_account_id <> (
    SELECT observed_via_account_id FROM platform_identities
    WHERE id = NEW.platform_identity_id
)
BEGIN
    SELECT RAISE(ABORT, 'endpoint account/identity mismatch');
END;

CREATE TRIGGER endpoint_account_identity_update
BEFORE UPDATE OF connected_account_id, platform_identity_id ON contact_endpoints
WHEN NEW.connected_account_id <> (
    SELECT observed_via_account_id FROM platform_identities
    WHERE id = NEW.platform_identity_id
)
BEGIN
    SELECT RAISE(ABORT, 'endpoint account/identity mismatch');
END;

CREATE TRIGGER conversation_scope_insert
BEFORE INSERT ON conversations
WHEN NEW.connected_account_id <> (
        SELECT connected_account_id FROM contact_endpoints WHERE id = NEW.endpoint_id
    ) OR NEW.provider <> (
        SELECT provider FROM connected_accounts WHERE id = NEW.connected_account_id
    )
BEGIN
    SELECT RAISE(ABORT, 'conversation scope mismatch');
END;

CREATE TRIGGER message_scope_insert
BEFORE INSERT ON messages
WHEN NEW.connected_account_id <> (
        SELECT connected_account_id FROM contact_endpoints WHERE id = NEW.endpoint_id
    ) OR NEW.connected_account_id <> (
        SELECT connected_account_id FROM conversations WHERE id = NEW.conversation_id
    ) OR NEW.endpoint_id <> (
        SELECT endpoint_id FROM conversations WHERE id = NEW.conversation_id
    ) OR NEW.provider <> (
        SELECT provider FROM connected_accounts WHERE id = NEW.connected_account_id
    ) OR (NEW.sender_identity_id IS NOT NULL AND NEW.connected_account_id <> (
        SELECT observed_via_account_id FROM platform_identities
        WHERE id = NEW.sender_identity_id
    ))
BEGIN
    SELECT RAISE(ABORT, 'message scope mismatch');
END;

CREATE TRIGGER draft_scope_insert
BEFORE INSERT ON drafts
WHEN NEW.connected_account_id <> (
        SELECT connected_account_id FROM contact_endpoints WHERE id = NEW.endpoint_id
    ) OR NEW.source_account_id <> (
        SELECT connected_account_id FROM contact_endpoints
        WHERE id = NEW.source_endpoint_id
    )
BEGIN
    SELECT RAISE(ABORT, 'draft endpoint/account mismatch');
END;

CREATE TRIGGER draft_exact_target_update
AFTER UPDATE OF person_id, connected_account_id, endpoint_id,
                source_account_id, source_endpoint_id, route_version,
                recipient_preview_json, payload, payload_hash
ON drafts
WHEN OLD.status = 'approved' AND (
    OLD.person_id <> NEW.person_id OR
    OLD.connected_account_id <> NEW.connected_account_id OR
    OLD.endpoint_id <> NEW.endpoint_id OR
    OLD.source_account_id <> NEW.source_account_id OR
    OLD.source_endpoint_id <> NEW.source_endpoint_id OR
    OLD.route_version <> NEW.route_version OR
    OLD.recipient_preview_json <> NEW.recipient_preview_json OR
    OLD.payload <> NEW.payload OR
    OLD.payload_hash <> NEW.payload_hash
)
BEGIN
    UPDATE approvals SET status = 'invalidated'
    WHERE draft_id = NEW.id AND status = 'active';
    UPDATE drafts SET status = 'invalidated' WHERE id = NEW.id;
END;

CREATE TRIGGER route_change_invalidates_approval
AFTER UPDATE OF target_endpoint_id, enabled, updated_at ON person_channel_routes
BEGIN
    UPDATE approvals SET status = 'invalidated'
    WHERE status = 'active' AND draft_id IN (
        SELECT id FROM drafts
        WHERE person_id = NEW.person_id
          AND source_endpoint_id = NEW.source_endpoint_id
          AND status = 'approved'
    );
    UPDATE drafts SET status = 'invalidated'
    WHERE person_id = NEW.person_id
      AND source_endpoint_id = NEW.source_endpoint_id
      AND status = 'approved';
END;

CREATE TRIGGER approval_scope_insert
BEFORE INSERT ON approvals
WHEN NEW.target_account_id <> (
        SELECT connected_account_id FROM contact_endpoints WHERE id = NEW.endpoint_id
    ) OR NEW.source_account_id <> (
        SELECT connected_account_id FROM contact_endpoints
        WHERE id = NEW.source_endpoint_id
    )
BEGIN
    SELECT RAISE(ABORT, 'approval endpoint/account mismatch');
END;

CREATE TRIGGER outbox_scope_insert
BEFORE INSERT ON outbox_items
WHEN NEW.connected_account_id <> (
    SELECT connected_account_id FROM contact_endpoints WHERE id = NEW.endpoint_id
)
BEGIN
    SELECT RAISE(ABORT, 'outbox endpoint/account mismatch');
END;
"""


_V2_DOWN = r"""
DROP TRIGGER IF EXISTS outbox_scope_insert;
DROP TRIGGER IF EXISTS approval_scope_insert;
DROP TRIGGER IF EXISTS route_change_invalidates_approval;
DROP TRIGGER IF EXISTS draft_exact_target_update;
DROP TRIGGER IF EXISTS draft_scope_insert;
DROP TRIGGER IF EXISTS message_scope_insert;
DROP TRIGGER IF EXISTS conversation_scope_insert;
DROP TRIGGER IF EXISTS endpoint_account_identity_update;
DROP TRIGGER IF EXISTS endpoint_account_identity_insert;
DROP TRIGGER IF EXISTS identity_account_provider_update;
DROP TRIGGER IF EXISTS identity_account_provider_insert;
DROP TABLE IF EXISTS xdom_suggestions;
DROP TABLE IF EXISTS legacy_records;
DROP TABLE IF EXISTS migration_runs;
DROP TABLE IF EXISTS legacy_id_mappings;
DROP TABLE IF EXISTS greeting_deliveries;
DROP TABLE IF EXISTS outbox_events;
DROP TABLE IF EXISTS outbox_items;
DROP TABLE IF EXISTS approvals;
DROP TABLE IF EXISTS drafts;
DROP TABLE IF EXISTS identity_merge_audit;
DROP TABLE IF EXISTS relationship_states;
DROP TABLE IF EXISTS commitments;
DROP TABLE IF EXISTS contact_events;
DROP TABLE IF EXISTS smart_segments;
DROP TABLE IF EXISTS contact_group_members;
DROP TABLE IF EXISTS contact_groups;
DROP TABLE IF EXISTS route_audit;
DROP TABLE IF EXISTS person_channel_routes;
DROP TABLE IF EXISTS account_link_policies;
DROP TABLE IF EXISTS channel_preferences;
DROP TABLE IF EXISTS channel_transitions;
DROP TABLE IF EXISTS channel_episodes;
DROP TABLE IF EXISTS communication_journeys;
"""


_V1_DOWN = r"""
DROP TABLE IF EXISTS sync_issues;
DROP TABLE IF EXISTS sync_locks;
DROP TABLE IF EXISTS sync_cursors;
DROP TABLE IF EXISTS sync_runs;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS participants;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS contact_endpoints;
DROP TABLE IF EXISTS platform_identities;
DROP TABLE IF EXISTS connected_accounts;
DROP TABLE IF EXISTS persons;
DROP TABLE IF EXISTS schema_migrations;
"""


def configure(connection: sqlite3.Connection) -> sqlite3.Connection:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open SQLite without ever creating a database from a read path."""
    path = Path(path)
    if readonly:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        return configure(sqlite3.connect(uri, uri=True, timeout=5))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = configure(sqlite3.connect(path, timeout=5, isolation_level=None))
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def current_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def _apply(connection: sqlite3.Connection, sql: str, version: int) -> None:
    metadata_sql = ""
    if version:
        metadata_sql = f"""
INSERT OR REPLACE INTO schema_migrations(version, applied_at)
VALUES ({int(version)}, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
"""
    script = f"""BEGIN IMMEDIATE;
{sql}
PRAGMA user_version = {int(version)};
{metadata_sql}
COMMIT;
"""
    try:
        connection.executescript(script)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def migrate(path: Path, target_version: int = LATEST_SCHEMA_VERSION) -> int:
    """Apply ordered idempotent migrations up to ``target_version``."""
    if target_version < 0 or target_version > LATEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {target_version}")
    connection = connect(path)
    try:
        version = current_version(connection)
        while version < target_version:
            next_version = version + 1
            _apply(connection, _V1_UP if next_version == 1 else _V2_UP, next_version)
            version = next_version
        return version
    finally:
        connection.close()


def rollback(path: Path, target_version: int) -> int:
    """Reverse migrations explicitly; intended for tested rollback/cutover."""
    if target_version < 0 or target_version > LATEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {target_version}")
    connection = connect(path)
    try:
        version = current_version(connection)
        while version > target_version:
            if version == 2:
                _apply(connection, _V2_DOWN, 1)
            elif version == 1:
                _apply(connection, _V1_DOWN, 0)
            version -= 1
        return version
    finally:
        connection.close()


MigrationHook = Callable[[sqlite3.Connection], None]
