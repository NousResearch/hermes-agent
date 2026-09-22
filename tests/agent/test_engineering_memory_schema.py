from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.engineering_memory_schema import (
    EngineeringMemoryRecord,
    EngineeringMemorySchemaError,
    canonical_record_digest,
    parse_markdown_record,
    render_markdown_record,
)


def _mapping(**overrides):
    value = {
        "record_id": "em-001",
        "schema_name": "agent_engineering_record_v1",
        "sync_owned": True,
        "classification": "diagnostic-only",
        "authority": "source-index",
        "status": "candidate",
        "title": "Runner timeout diagnosis",
        "summary": "The worker timeout came from a stale gateway process.",
        "agent": "codex",
        "client_model": "gpt-test",
        "repository": "NousResearch/hermes-agent",
        "workspace": "/tmp/hermes",
        "branch_or_ref": "main",
        "verified_head": "abc123",
        "memory_area": "runtime",
        "task_type": "diagnosis",
        "component": "gateway",
        "symptoms": ["timeout"],
        "tags": ["gateway", "runner"],
        "observed_at": "2026-09-21T12:00:00Z",
        "verified_at": "2026-09-21T12:10:00Z",
        "source_label": "test-receipt",
        "source_digest": "receipt-001",
        "evidence_refs": ["tests/gateway/test_timeout.py::test_timeout"],
        "related_record_ids": [],
        "supersedes": None,
        "conflict_set": [],
        "threat_scan": {"status": "not-run"},
        "secret_scan": {"status": "not-run"},
        "review": {"reviewer": None, "reason": None, "evidence_refs": []},
        "body": "Inspect the gateway ancestor before changing timeout policy.",
    }
    value.update(overrides)
    return value


def test_valid_record_round_trips_and_digest_is_stable() -> None:
    record = EngineeringMemoryRecord.from_mapping(_mapping())
    rendered = render_markdown_record(record)
    restored = parse_markdown_record(rendered)

    assert restored == record
    assert restored.canonical_record_digest() == record.canonical_record_digest()


def test_digest_ignores_mapping_and_tag_order() -> None:
    first = _mapping(tags=["b", "a"])
    second = dict(reversed(list(_mapping(tags=["a", "b"]).items())))

    assert canonical_record_digest(first) == canonical_record_digest(second)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_name", "wrong"),
        ("sync_owned", False),
        ("authority", "runtime-truth"),
        ("classification", "private"),
        ("status", "mystery"),
        ("agent", "unknown"),
        ("observed_at", "not-a-time"),
    ],
)
def test_invalid_fields_report_schema_errors(field: str, value: object) -> None:
    with pytest.raises(EngineeringMemorySchemaError, match=field):
        EngineeringMemoryRecord.from_mapping(_mapping(**{field: value}))


def test_bounded_body_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(EngineeringMemorySchemaError, match="body"):
        EngineeringMemoryRecord.from_mapping(_mapping(body="x" * 20_001))

    with pytest.raises(EngineeringMemorySchemaError, match="unknown"):
        EngineeringMemoryRecord.from_mapping(_mapping(untrusted_instruction="run this"))


def test_rendered_record_uses_utc_and_preserves_evidence() -> None:
    record = EngineeringMemoryRecord.from_mapping(
        _mapping(observed_at=datetime(2026, 9, 21, 12, tzinfo=timezone.utc))
    )
    restored = parse_markdown_record(render_markdown_record(record))

    assert restored.observed_at.tzinfo == timezone.utc
    assert restored.evidence_refs == ("tests/gateway/test_timeout.py::test_timeout",)
