"""Native PostgreSQL schemas and transaction boundary for API persistence."""
from functools import wraps

RESPONSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at DOUBLE PRECISION NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL);
"""
RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_idempotency (
    scope TEXT NOT NULL, idempotency_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE, status_json TEXT NOT NULL,
    owner_pid BIGINT NOT NULL DEFAULT 0, owner_started BIGINT NOT NULL DEFAULT 0,
    retention_until DOUBLE PRECISION NOT NULL DEFAULT 0, acknowledged_at DOUBLE PRECISION,
    created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
    owner_token TEXT, owner_expires_at DOUBLE PRECISION,
    PRIMARY KEY(scope, idempotency_key));
"""


def response_transaction(method):
    """Serialize response history and LRU changes across PostgreSQL API replicas."""
    @wraps(method)
    def run(self, *args, **kwargs):
        if getattr(self._conn, 'backend', 'sqlite') != 'postgres':
            return method(self, *args, **kwargs)
        with self._postgres_lock, self._conn.write():
            return method(self, *args, **kwargs)
    return run
