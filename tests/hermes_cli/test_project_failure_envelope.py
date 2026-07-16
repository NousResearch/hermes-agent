"""Focused HOF-012B tests for validated failure-envelope persistence."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_cli.project_failure_envelope import (
    FAILURE_CLASSES,
    ProjectFailureEnvelope,
    compute_error_fingerprint,
    record_failure_envelope,
    redact_error,
)
from hermes_cli.project_finalization_contract import ensure_project_finalization_schema


def _db() -> tuple[sqlite3.Connection, Path]:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = Path(handle.name)
    handle.close()
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_project_finalization_schema(conn)
    return conn, path


def _close(conn: sqlite3.Connection, path: Path) -> None:
    conn.close()
    path.unlink(missing_ok=True)


def _record(conn: sqlite3.Connection, **overrides):
    values = {
        "board_id": "board",
        "root_task_id": "root",
        "generation": 1,
        "task_id": "task",
        "run_id": 7,
        "provider": "openai",
        "model": "model-a",
        "failure_class": "provider_timeout",
        "status_code": 504,
        "retry_after": 30,
        "redacted_error": "provider timeout",
    }
    values.update(overrides)
    return record_failure_envelope(conn, **values)


def test_all_required_failure_classes_are_accepted_and_persisted():
    conn, path = _db()
    try:
        for run_id, failure_class in enumerate(FAILURE_CLASSES, start=1):
            envelope = _record(conn, run_id=run_id, failure_class=failure_class)
            assert envelope.failure_class == failure_class
        assert conn.execute("SELECT COUNT(*) FROM project_failure_envelopes").fetchone()[0] == len(FAILURE_CLASSES)
    finally:
        _close(conn, path)


def test_redaction_removes_secret_material_and_structured_payloads():
    secrets = (
        "sk-live-1234567890",
        "Bearer abcdefghijklmnop",
        "https://example.test/callback?access_token=oauth-secret",
        "api_key=api-secret",
        "-----BEGIN PRIVATE KEY-----\nprivate-secret\n-----END PRIVATE KEY-----",
    )
    text = " ".join(secrets)
    redacted = redact_error(text)
    assert redacted is not None
    for secret in secrets:
        assert secret not in redacted
    assert "[url redacted]" in redacted
    assert redact_error('{"response": "raw provider body", "api_key": "secret"}') == "[structured error redacted]"
    assert redact_error("prompt: the user asked for private chat text") == "[structured error redacted]"


def test_fingerprint_is_stable_after_redaction_and_uses_sha256():
    conn, path = _db()
    try:
        first = _record(
            conn,
            redacted_error="failed with api_key=first-secret",
        )
        second = _record(
            conn,
            run_id=8,
            redacted_error="failed with api_key=second-secret",
        )
        assert first.error_fingerprint == second.error_fingerprint
        assert first.error_fingerprint == compute_error_fingerprint(
            {
                "board_id": "board",
                "root_task_id": "root",
                "generation": 1,
                "task_id": "task",
                "run_id": 7,
                "provider": "openai",
                "model": "model-a",
                "failure_class": "provider_timeout",
                "status_code": 504,
                "retry_after": 30,
                "redacted_error": "failed with [secret field redacted]",
                "error_fingerprint": None,
            }
        )
        assert len(first.error_fingerprint) == 64
    finally:
        _close(conn, path)


def test_exact_repeat_is_idempotent_inside_the_frozen_schema():
    conn, path = _db()
    try:
        first = _record(conn)
        repeat = _record(conn)
        assert repeat == first
        assert conn.execute("SELECT COUNT(*) FROM project_failure_envelopes").fetchone()[0] == 1
    finally:
        _close(conn, path)


def test_conflicting_identity_rejects_before_mutation():
    conn, path = _db()
    try:
        first = _record(conn)
        with pytest.raises(ValueError, match="conflicting failure envelope identity"):
            _record(conn, status_code=401, failure_class="provider_auth")
        assert conn.execute("SELECT COUNT(*) FROM project_failure_envelopes").fetchone()[0] == 1
        persisted = conn.execute(
            "SELECT status_code, failure_class, redacted_error FROM project_failure_envelopes WHERE id=?",
            (first.id,),
        ).fetchone()
        assert tuple(persisted) == (504, "provider_timeout", "provider timeout")
    finally:
        _close(conn, path)


def test_distinct_occurrences_require_distinct_run_ids():
    conn, path = _db()
    try:
        first = _record(conn, run_id=1)
        second = _record(conn, run_id=2)
        assert first.id != second.id
        assert conn.execute("SELECT COUNT(*) FROM project_failure_envelopes").fetchone()[0] == 2
    finally:
        _close(conn, path)


def test_typed_validation_and_fingerprint_conflicts_happen_before_insert():
    conn, path = _db()
    try:
        with pytest.raises(ValueError, match="failure_class"):
            _record(conn, failure_class="not-a-class")
        with pytest.raises(TypeError, match="status_code"):
            _record(conn, status_code="504")
        with pytest.raises(TypeError, match="unrecognized"):
            _record(conn, unexpected="value")
        with pytest.raises(ValueError, match="error_fingerprint"):
            _record(conn, error_fingerprint="0" * 64)
        assert conn.execute("SELECT COUNT(*) FROM project_failure_envelopes").fetchone()[0] == 0
    finally:
        _close(conn, path)


def test_hof002_compatibility_keeps_exact_frozen_fields_and_no_secret_columns():
    conn, path = _db()
    try:
        envelope = _record(conn,
            redacted_error="authorization: ***",
        )
        assert isinstance(envelope, ProjectFailureEnvelope)
        assert tuple(envelope.__dataclass_fields__) == (
            "id", "board_id", "root_task_id", "generation", "task_id", "run_id",
            "provider", "model", "failure_class", "status_code", "retry_after",
            "redacted_error", "error_fingerprint", "created_at",
        )
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(project_failure_envelopes)")]
        assert columns == [
            "id", "board_id", "root_task_id", "generation", "task_id", "run_id",
            "provider", "model", "failure_class", "status_code", "retry_after",
            "redacted_error", "error_fingerprint", "created_at",
        ]
        row_values = conn.execute("SELECT * FROM project_failure_envelopes").fetchone()
        assert all(secret not in str(value) for value in row_values for secret in ("Bearer ", "api-secret", "oauth-secret", "authorization"))
    finally:
        _close(conn, path)
