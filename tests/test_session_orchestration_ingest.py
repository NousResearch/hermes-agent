"""
Tests for session_orchestration/ingest.py.

Covers:
- Valid signed payload → correct enqueue intent (correlate or adopt)
- Unknown run → adopt intent enqueued
- Duplicate event_id → no second enqueue (idempotent)
- Bad HMAC signature → rejected
- Rate limit exceeded → rejected
- Per-source rate limit is independent (different sources don't share quota)
- Disabled session_orchestration → skipped
- Missing event_id → rejected

All tests use an isolated SQLite DB (tmp_path) and monkeypatch config.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from session_orchestration.registry import (
    SessionOrchestrationRegistry,
    canonical_repo_id,
)
import session_orchestration.ingest as ingest_mod
from session_orchestration.ingest import (
    _check_rate_limit,
    _is_duplicate_event,
    _validate_hmac,
    _ensure_dedup_schema,
    process_z_harness_alert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign_payload(body: bytes, secret: str) -> str:
    """Produce sha256=<hex> signature matching notify-watchdog.sh."""
    hex_digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={hex_digest}"


def _make_payload(
    *,
    event_id: Optional[str] = None,
    run_id: str = "run-test-001",
    repo: str = "/Users/test/project",
    event: str = "watchdog_stall",
    severity: str = "warning",
    reason: str = "Process hung",
    source: str = "z-harness",
    slug: str = "my-plan",
) -> Dict[str, Any]:
    return {
        "schema_version": "1",
        "harness_version": "1.0.100",
        "source": source,
        "event": event,
        "event_id": event_id or str(uuid.uuid4()),
        "run_id": run_id,
        "slug": slug,
        "repo": repo,
        "severity": severity,
        "reason": reason,
        "ts": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


@pytest.fixture(autouse=True)
def _reset_rate_windows():
    """Clear in-memory rate windows between tests."""
    ingest_mod._rate_windows.clear()
    yield
    ingest_mod._rate_windows.clear()


@pytest.fixture()
def patch_enabled(monkeypatch, db_path):
    """Patch ingest to use tmp db_path and enable session_orchestration."""

    monkeypatch.setattr(ingest_mod, "_get_db_path", lambda: db_path)
    monkeypatch.setattr(
        ingest_mod,
        "_get_session_orchestration_config",
        lambda: {
            "enabled": True,
            "feed_channel_id": "111222333",
            "external_runs_thread_id": "999888777",
            "rate_limit_per_source": 10,
        },
    )
    monkeypatch.setattr(ingest_mod, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        ingest_mod,
        "_push_to_feed",
        lambda *a, **kw: asyncio.sleep(0),  # no-op async
    )


# ---------------------------------------------------------------------------
# HMAC validation
# ---------------------------------------------------------------------------


class TestHmacValidation:
    def test_valid_signature(self):
        secret = "my-secret"
        body = b'{"event":"test"}'
        sig = _sign_payload(body, secret)
        assert _validate_hmac(body, secret, sig) is True

    def test_wrong_secret(self):
        body = b'{"event":"test"}'
        sig = _sign_payload(body, "correct-secret")
        assert _validate_hmac(body, "wrong-secret", sig) is False

    def test_tampered_body(self):
        secret = "my-secret"
        body = b'{"event":"test"}'
        sig = _sign_payload(body, secret)
        assert _validate_hmac(b'{"event":"tampered"}', secret, sig) is False

    def test_missing_sha256_prefix(self):
        secret = "my-secret"
        body = b'{"event":"test"}'
        bare_hex = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        # No "sha256=" prefix → rejected
        assert _validate_hmac(body, secret, bare_hex) is False

    def test_empty_signature(self):
        assert _validate_hmac(b"body", "secret", "") is False


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_first_seen_not_duplicate(self, db_path):
        _ensure_dedup_schema(db_path)
        eid = str(uuid.uuid4())
        assert _is_duplicate_event(db_path, eid, "z-harness") is False

    def test_second_seen_is_duplicate(self, db_path):
        _ensure_dedup_schema(db_path)
        eid = str(uuid.uuid4())
        _is_duplicate_event(db_path, eid, "z-harness")  # first
        assert _is_duplicate_event(db_path, eid, "z-harness") is True

    def test_different_event_ids_not_duplicate(self, db_path):
        _ensure_dedup_schema(db_path)
        eid1 = str(uuid.uuid4())
        eid2 = str(uuid.uuid4())
        _is_duplicate_event(db_path, eid1, "z-harness")
        assert _is_duplicate_event(db_path, eid2, "z-harness") is False

    def test_dedup_persists_across_instances(self, db_path):
        """A new process (new connection) still sees the recorded event."""
        _ensure_dedup_schema(db_path)
        eid = str(uuid.uuid4())
        _is_duplicate_event(db_path, eid, "z-harness")
        # Simulate a new process instance using a fresh connection
        assert _is_duplicate_event(db_path, eid, "z-harness") is True


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_within_limit(self):
        source = f"test-source-{uuid.uuid4().hex}"
        for _ in range(5):
            assert _check_rate_limit(source, 10) is True

    def test_exceeded_limit(self):
        source = f"test-source-{uuid.uuid4().hex}"
        limit = 3
        for _ in range(limit):
            _check_rate_limit(source, limit)
        # Next one should be rejected
        assert _check_rate_limit(source, limit) is False

    def test_different_sources_independent(self):
        """Rate limit buckets are per-source — source A exhausted does not block source B."""
        source_a = f"source-a-{uuid.uuid4().hex}"
        source_b = f"source-b-{uuid.uuid4().hex}"
        limit = 2
        for _ in range(limit):
            _check_rate_limit(source_a, limit)
        # source_a exhausted
        assert _check_rate_limit(source_a, limit) is False
        # source_b unaffected
        assert _check_rate_limit(source_b, limit) is True


# ---------------------------------------------------------------------------
# process_z_harness_alert — integration tests
# ---------------------------------------------------------------------------


class TestProcessAlert:
    def test_disabled_returns_disabled(self, db_path, monkeypatch):
        monkeypatch.setattr(ingest_mod, "_is_enabled", lambda: False)
        monkeypatch.setattr(ingest_mod, "_get_db_path", lambda: db_path)
        payload = _make_payload()
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert result["status"] == "disabled"

    def test_missing_event_id_rejected(self, patch_enabled):
        payload = _make_payload()
        payload.pop("event_id")
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert result["status"] == "missing_fields"

    def test_bad_signature_rejected(self, patch_enabled):
        payload = _make_payload()
        body = json.dumps(payload).encode()
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(
                payload,
                raw_body=body,
                signature_header="sha256=deadbeef0000",
                hmac_secret="correct-secret",
            )
        )
        assert result["status"] == "invalid_signature"

    def test_valid_signature_accepted(self, patch_enabled):
        secret = "test-secret-123"
        payload = _make_payload()
        body = json.dumps(payload).encode()
        sig = _sign_payload(body, secret)
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(
                payload,
                raw_body=body,
                signature_header=sig,
                hmac_secret=secret,
            )
        )
        assert result["status"] == "accepted"

    def test_unknown_run_adopts(self, patch_enabled, db_path, registry):
        """Unknown (run_id, repo) pair results in an 'adopt' intent enqueued."""
        payload = _make_payload(run_id="new-run-99", repo="/some/new/repo")
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert result["status"] == "accepted"
        assert result["action"] == "adopted"

        # The adopt intent must be in the queue.
        intents = registry.drain_intents()
        adopt_intents = [i for i in intents if i["intent"] == "adopt"]
        assert len(adopt_intents) == 1
        adopt_payload = json.loads(adopt_intents[0]["payload"])
        assert adopt_payload["run_id"] == "new-run-99"
        assert adopt_payload["agent"] == "z-harness"  # agent field populated from payload source

    def test_known_run_correlates(self, patch_enabled, db_path, registry):
        """Known (run_id, repo) pair results in an 'update' intent enqueued."""
        run_id = "known-run-42"
        repo_path = "/Users/zeke/projects/myproject"
        repo_key = canonical_repo_id(workdir=repo_path)

        # Pre-seed a registry row simulating a managed session.
        existing_tid = f"task-{uuid.uuid4().hex[:8]}"
        registry.upsert(
            existing_tid,
            agent="claude-code",
            run_id=run_id,
            repo=repo_key,
            state="RUNNING",
        )

        payload = _make_payload(run_id=run_id, repo=repo_path)
        result = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert result["status"] == "accepted"
        assert result["action"] == "correlated"
        assert result["task_id"] == existing_tid

        # The update intent must be in the queue.
        intents = registry.drain_intents()
        update_intents = [i for i in intents if i["intent"] == "update"]
        assert len(update_intents) == 1
        update_payload = json.loads(update_intents[0]["payload"])
        assert update_payload["task_id"] == existing_tid

    def test_duplicate_event_id_no_second_enqueue(self, patch_enabled, db_path, registry):
        """Sending the same event_id twice must not produce a second intent."""
        eid = str(uuid.uuid4())
        payload = _make_payload(event_id=eid, run_id="dup-run", repo="/dup/repo")

        # First POST
        r1 = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert r1["status"] == "accepted"
        intents_after_first = registry.drain_intents()
        assert len(intents_after_first) == 1

        # Second POST with identical event_id → must be deduped
        r2 = asyncio.get_event_loop().run_until_complete(
            process_z_harness_alert(payload)
        )
        assert r2["status"] == "duplicate"

        # No new intent enqueued
        intents_after_second = registry.drain_intents()
        assert intents_after_second == []

    def test_rate_limit_exceeded_rejected(self, patch_enabled):
        """After exhausting the per-source rate limit, the next event is rejected."""
        source = f"z-harness-rl-{uuid.uuid4().hex}"
        limit = 10  # matches patch_enabled fixture

        # Exhaust the limit with distinct event_ids
        loop = asyncio.get_event_loop()
        for _ in range(limit):
            p = _make_payload(source=source)
            loop.run_until_complete(process_z_harness_alert(p))

        # Next event from same source must be rate-limited
        p_extra = _make_payload(source=source)
        result = loop.run_until_complete(process_z_harness_alert(p_extra))
        assert result["status"] == "rate_limited"
