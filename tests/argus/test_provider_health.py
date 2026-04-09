"""Tests for ARGUS provider health monitoring module."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import provider_health


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def argus_db():
    """In-memory argus.db with provider_health + sessions + tool_calls tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Core tables from watcher_schema.sql
    cur.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            session_type TEXT NOT NULL,
            job_id TEXT,
            task_description TEXT,
            model TEXT,
            provider TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            restart_count INTEGER DEFAULT 0,
            kill_count INTEGER DEFAULT 0,
            quality_gate_score REAL,
            entropy_score REAL,
            token_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            metadata TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            tool_args TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_ms INTEGER,
            success BOOLEAN,
            error_message TEXT,
            result_size INTEGER,
            file_changed BOOLEAN DEFAULT FALSE,
            file_path TEXT,
            file_hash_before TEXT,
            file_hash_after TEXT
        )
    """)

    provider_health.ensure_table(cur, conn)
    conn.commit()
    yield cur, conn
    conn.close()


@pytest.fixture
def state_db(tmp_path):
    """Temporary state.db with sessions + messages tables."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            billing_provider TEXT,
            started_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp TEXT,
            token_count INTEGER,
            finish_reason TEXT
        )
    """)
    conn.commit()
    yield db_path, conn
    conn.close()


# ─── classify_error tests ───────────────────────────────────────────────

class TestClassifyError:
    def test_rate_limit_429(self):
        cats = provider_health.classify_error("HTTP 429 Too Many Requests")
        assert "rate_limit" in cats

    def test_rate_limit_text(self):
        cats = provider_health.classify_error("Rate limit exceeded for model")
        assert "rate_limit" in cats

    def test_timeout(self):
        cats = provider_health.classify_error("Connection timed out after 30s")
        assert "timeout" in cats

    def test_read_timeout(self):
        cats = provider_health.classify_error("ReadTimeout: HTTPSConnectionPool")
        assert "timeout" in cats

    def test_auth_401(self):
        cats = provider_health.classify_error("HTTP 401 Unauthorized")
        assert "auth_error" in cats

    def test_invalid_api_key(self):
        cats = provider_health.classify_error("Invalid API key provided")
        assert "auth_error" in cats

    def test_billing_402(self):
        cats = provider_health.classify_error("HTTP 402 Payment Required: insufficient credits")
        assert "billing" in cats

    def test_server_500(self):
        cats = provider_health.classify_error("HTTP 500 Internal Server Error")
        assert "server_error" in cats

    def test_server_502(self):
        cats = provider_health.classify_error("502 Bad Gateway")
        assert "server_error" in cats

    def test_server_503(self):
        cats = provider_health.classify_error("503 Service Unavailable")
        assert "server_error" in cats

    def test_overloaded(self):
        cats = provider_health.classify_error("Overloaded: model capacity exceeded")
        assert "server_error" in cats

    def test_model_not_found(self):
        cats = provider_health.classify_error("Model not found: gpt-5-turbo")
        assert "model_error" in cats

    def test_context_length(self):
        cats = provider_health.classify_error("This model's maximum context length is 4097 tokens")
        assert "context_length" in cats

    def test_empty_string(self):
        assert provider_health.classify_error("") == []

    def test_none(self):
        assert provider_health.classify_error(None) == []

    def test_no_match(self):
        assert provider_health.classify_error("Everything is fine") == []

    def test_multiple_categories(self):
        # A message could match both rate_limit and server_error patterns
        cats = provider_health.classify_error("429 rate limit exceeded, server overloaded")
        assert "rate_limit" in cats
        assert "server_error" in cats


# ─── _detect_outage tests ───────────────────────────────────────────────

class TestDetectOutage:
    def test_too_few_errors(self):
        errors = [{"timestamp": "2026-04-09T10:00:00", "session_id": "s1"}]
        assert provider_health._detect_outage(errors) is False

    def test_same_session_no_outage(self):
        # 3 errors from same session = not an outage
        errors = [
            {"timestamp": "2026-04-09T10:00:01", "session_id": "s1"},
            {"timestamp": "2026-04-09T10:00:02", "session_id": "s1"},
            {"timestamp": "2026-04-09T10:00:03", "session_id": "s1"},
        ]
        assert provider_health._detect_outage(errors) is False

    def test_outage_detected(self):
        # 3 different sessions within 60s window = outage
        errors = [
            {"timestamp": "2026-04-09T10:00:01", "session_id": "s1"},
            {"timestamp": "2026-04-09T10:00:02", "session_id": "s2"},
            {"timestamp": "2026-04-09T10:00:03", "session_id": "s3"},
        ]
        assert provider_health._detect_outage(errors) is True

    def test_outage_wide_window(self):
        # 3 sessions but spread over 5 minutes = no outage
        errors = [
            {"timestamp": "2026-04-09T10:00:00", "session_id": "s1"},
            {"timestamp": "2026-04-09T10:02:00", "session_id": "s2"},
            {"timestamp": "2026-04-09T10:04:00", "session_id": "s3"},
        ]
        assert provider_health._detect_outage(errors) is False

    def test_empty(self):
        assert provider_health._detect_outage([]) is False


# ─── ensure_table tests ─────────────────────────────────────────────────

class TestEnsureTable:
    def test_creates_table(self, argus_db):
        cur, conn = argus_db
        # Table already created by fixture; verify it exists
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='provider_health'"
        )
        assert cur.fetchone() is not None

    def test_idempotent(self, argus_db):
        cur, conn = argus_db
        # Calling again should not raise
        provider_health.ensure_table(cur, conn)


# ─── _provider_from_model tests ─────────────────────────────────────────

class TestProviderFromModel:
    def test_anthropic(self):
        assert provider_health._provider_from_model("claude-sonnet-4-20250514") == "anthropic"

    def test_openai(self):
        assert provider_health._provider_from_model("gpt-4o") == "openai"

    def test_google(self):
        assert provider_health._provider_from_model("gemini-2.0-flash") == "google"

    def test_mimo_nous(self):
        assert provider_health._provider_from_model("xiaomi/mimo-v2-pro") == "nous"

    def test_none(self):
        assert provider_health._provider_from_model(None) is None

    def test_unknown(self):
        assert provider_health._provider_from_model("fancy-custom-model") is None


# ─── _percentile tests ──────────────────────────────────────────────────

class TestPercentile:
    def test_empty(self):
        assert provider_health._percentile([], 95) is None

    def test_single(self):
        assert provider_health._percentile([100.0], 95) == 100.0

    def test_p50(self):
        vals = [10, 20, 30, 40, 50]
        assert provider_health._percentile(vals, 50) == 30

    def test_p95(self):
        vals = list(range(100))
        p95 = provider_health._percentile(vals, 95)
        assert p95 is not None
        assert p95 >= 94  # near top


# ─── format_alert tests ─────────────────────────────────────────────────

class TestFormatAlert:
    def test_no_alert_on_info(self):
        report = {"overall_severity": "info", "providers": {}}
        assert provider_health.format_alert(report) is None

    def test_warning_alert(self):
        report = {
            "overall_severity": "warning",
            "outage_detected": False,
            "providers": {
                "nous": {
                    "severity": "warning",
                    "total_errors": 8,
                    "rate_limit_count": 0,
                    "timeout_count": 3,
                    "auth_error_count": 0,
                    "billing_error_count": 0,
                    "server_error_count": 5,
                    "model_error_count": 0,
                    "context_length_count": 0,
                    "p95_latency_ms": 1200.0,
                    "outage_detected": False,
                    "recent_errors": [],
                }
            },
        }
        alert = provider_health.format_alert(report)
        assert alert is not None
        assert "nous" in alert
        assert "WARNING" in alert
        assert "Timeouts: 3" in alert

    def test_critical_with_outage(self):
        report = {
            "overall_severity": "critical",
            "outage_detected": True,
            "providers": {
                "openrouter": {
                    "severity": "critical",
                    "total_errors": 25,
                    "rate_limit_count": 10,
                    "timeout_count": 0,
                    "auth_error_count": 0,
                    "billing_error_count": 0,
                    "server_error_count": 15,
                    "model_error_count": 0,
                    "context_length_count": 0,
                    "p95_latency_ms": None,
                    "outage_detected": True,
                    "recent_errors": [
                        {"error_type": "rate_limit", "detail": "429 Too Many Requests"},
                    ],
                }
            },
        }
        alert = provider_health.format_alert(report)
        assert "OUTAGE DETECTED" in alert
        assert "CRITICAL" in alert
        assert "Rate limits (429): 10" in alert


# ─── cleanup_old_snapshots tests ────────────────────────────────────────

class TestCleanupOldSnapshots:
    def test_removes_old(self, argus_db):
        cur, conn = argus_db
        old = (datetime.now(UTC) - timedelta(hours=100)).isoformat()
        cur.execute(
            "INSERT INTO provider_health (provider, timestamp) VALUES (?, ?)",
            ("test-provider", old),
        )
        conn.commit()

        deleted = provider_health.cleanup_old_snapshots(cur, conn, keep_hours=72)
        assert deleted == 1

    def test_keeps_recent(self, argus_db):
        cur, conn = argus_db
        recent = datetime.now(UTC).isoformat()
        cur.execute(
            "INSERT INTO provider_health (provider, timestamp) VALUES (?, ?)",
            ("test-provider", recent),
        )
        conn.commit()

        deleted = provider_health.cleanup_old_snapshots(cur, conn, keep_hours=72)
        assert deleted == 0


# ─── Integration: run_provider_check ─────────────────────────────────────

class TestRunProviderCheck:
    def test_clean_state(self, argus_db, state_db, monkeypatch):
        """No errors → report is info severity."""
        cur, conn = argus_db
        db_path, state_conn = state_db

        monkeypatch.setattr(provider_health, "_state_db_path", lambda: db_path)

        report = provider_health.run_provider_check(cur, conn)
        assert report["overall_severity"] == "info"
        assert not report["outage_detected"]

    def test_rate_limit_detected(self, argus_db, state_db, monkeypatch):
        """429 errors in messages → critical with rate_limit count."""
        cur, conn = argus_db
        db_path, state_conn = state_db
        s_cur = state_conn.cursor()

        since = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        s_cur.execute(
            "INSERT INTO sessions (id, model, billing_provider, started_at) VALUES (?, ?, ?, ?)",
            ("sess-1", "xiaomi/mimo-v2-pro", "nous", since),
        )
        for i in range(6):
            s_cur.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                ("sess-1", "system", "HTTP 429 Too Many Requests", since),
            )
        state_conn.commit()

        monkeypatch.setattr(provider_health, "_state_db_path", lambda: db_path)

        report = provider_health.run_provider_check(cur, conn)
        assert report["overall_severity"] == "critical"
        nous_data = report["providers"].get("nous", {})
        assert nous_data.get("rate_limit_count", 0) >= 1

    def test_tool_call_errors(self, argus_db, state_db, monkeypatch):
        """Failed tool calls in argus.db → detected in report."""
        cur, conn = argus_db
        db_path, _ = state_db

        since = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        cur.execute(
            "INSERT INTO sessions (session_id, session_type, provider) VALUES (?, ?, ?)",
            ("manual_s1", "manual", "anthropic"),
        )
        for i in range(6):
            cur.execute(
                "INSERT INTO tool_calls (session_id, tool_name, success, error_message, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("manual_s1", "terminal", 0, "Connection timed out", since),
            )
        conn.commit()

        monkeypatch.setattr(provider_health, "_state_db_path", lambda: db_path)

        report = provider_health.run_provider_check(cur, conn)
        anth = report["providers"].get("anthropic", {})
        assert anth.get("timeout_count", 0) >= 1

    def test_snapshots_persisted(self, argus_db, state_db, monkeypatch):
        """After check, snapshots are written to provider_health table."""
        cur, conn = argus_db
        db_path, state_conn = state_db
        s_cur = state_conn.cursor()

        since = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        s_cur.execute(
            "INSERT INTO sessions (id, model, billing_provider, started_at) VALUES (?, ?, ?, ?)",
            ("sess-2", "gpt-4o", "openai", since),
        )
        for i in range(6):
            s_cur.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                ("sess-2", "system", "502 Bad Gateway", since),
            )
        state_conn.commit()

        monkeypatch.setattr(provider_health, "_state_db_path", lambda: db_path)

        provider_health.run_provider_check(cur, conn)

        cur.execute("SELECT COUNT(*) FROM provider_health")
        count = cur.fetchone()[0]
        assert count >= 1

        cur.execute("SELECT provider, severity FROM provider_health ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        assert row["provider"] == "openai"
        assert row["severity"] == "critical"
