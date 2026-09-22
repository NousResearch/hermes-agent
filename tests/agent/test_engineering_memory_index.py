from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent.engineering_memory_index import EngineeringMemoryIndex
from agent.engineering_memory_schema import EngineeringMemoryRecord


def _record(record_id: str, *, status: str = "approved", component: str = "gateway", verified_head: str | None = "abc") -> EngineeringMemoryRecord:
    return EngineeringMemoryRecord.from_mapping(
        {
            "record_id": record_id,
            "schema_name": "agent_engineering_record_v1",
            "sync_owned": True,
            "classification": "diagnostic-only",
            "authority": "source-index",
            "status": status,
            "title": "Gateway worker timeout",
            "summary": "A stale gateway ancestor caused the worker timeout.",
            "agent": "codex",
            "repository": "repo",
            "memory_area": "runtime",
            "task_type": "diagnosis",
            "component": component,
            "tags": ["timeout", "gateway"],
            "verified_head": verified_head,
            "observed_at": "2026-09-21T00:00:00Z",
            "verified_at": "2026-09-21T01:00:00Z",
            "source_label": "receipt",
            "source_digest": record_id,
            "evidence_refs": ["tests/gateway/test_timeout.py"],
            "body": "Inspect the ancestor process before tuning the timeout.",
        }
    )


def test_rebuild_search_preserves_provenance_and_filters(tmp_path: Path) -> None:
    index = EngineeringMemoryIndex(tmp_path / "index.sqlite3")
    index.rebuild([_record("good"), _record("candidate", status="candidate")])

    results = index.search("stale gateway timeout", repository="repo", component="gateway")

    assert [result.record_id for result in results] == ["good"]
    assert results[0].verified_head == "abc"
    assert results[0].source_label == "receipt"
    assert results[0].reason_code in {"fts_match", "token_overlap"}
    assert index.verify().record_count == 1


def test_fallback_search_has_same_contract_and_respects_budget(tmp_path: Path) -> None:
    index = EngineeringMemoryIndex(tmp_path / "fallback.sqlite3", fts_supported=False)
    index.rebuild([_record("one"), _record("two", component="worker")])

    results = index.search("gateway timeout", limit=1, char_budget=600)

    assert len(results) == 1
    assert results[0].reason_code == "token_overlap"
    assert len(results[0].summary) <= 600


def test_conflicts_are_returned_together_when_explicitly_indexed(tmp_path: Path) -> None:
    first = _record("one")
    second = replace(_record("two"), conflict_set=("one", "two"))
    first = replace(first, conflict_set=("one", "two"))
    index = EngineeringMemoryIndex(tmp_path / "index.sqlite3")
    index.rebuild([first, second])

    results = index.search("gateway timeout", limit=8)

    assert {record.record_id for record in results} == {"one", "two"}
    assert all(record.conflict_set == ("one", "two") for record in results)
