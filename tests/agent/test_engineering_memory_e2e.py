from __future__ import annotations

from pathlib import Path

from agent.engineering_memory_curator import EngineeringMemoryCurator
from agent.engineering_memory_index import EngineeringMemoryIndex
from agent.engineering_memory_ledger import EngineeringMemoryLedger
from agent.engineering_memory_laya import LayaDecisionEngine, LayaRuntimeConfig
from agent.engineering_memory_schema import EngineeringMemoryRecord


def _record(record_id: str, repository: str) -> EngineeringMemoryRecord:
    return EngineeringMemoryRecord.from_mapping(
        {
            "record_id": record_id,
            "schema_name": "agent_engineering_record_v1",
            "sync_owned": True,
            "classification": "diagnostic-only",
            "authority": "source-index",
            "status": "candidate",
            "title": "Verified runner repair",
            "summary": "The runner was repaired after checking the current head.",
            "agent": "hermes",
            "repository": repository,
            "memory_area": "runtime",
            "task_type": "repair",
            "component": "runner",
            "verified_head": "abc123",
            "observed_at": "2026-09-21T00:00:00Z",
            "verified_at": "2026-09-21T01:00:00Z",
            "source_label": "receipt",
            "source_digest": record_id,
            "evidence_refs": ["tests/test_runner.py::test_repair"],
            "body": "The repair was validated against the current repository head.",
        }
    )


def test_end_to_end_review_rebuild_search_is_profile_isolated(tmp_path: Path) -> None:
    vault_a = tmp_path / "shared-a"
    vault_b = tmp_path / "shared-b"
    index_a = EngineeringMemoryIndex(tmp_path / "a.sqlite3")
    index_b = EngineeringMemoryIndex(tmp_path / "b.sqlite3")
    curator_a = EngineeringMemoryCurator(EngineeringMemoryLedger(vault_a))
    curator_b = EngineeringMemoryCurator(EngineeringMemoryLedger(vault_b))

    curator_a.stage(_record("a", "repo-a"))
    curator_b.stage(_record("b", "repo-b"))
    curator_a.review("a", "approve", reviewer="human", reason="A receipt checked")
    curator_b.review("b", "approve", reviewer="human", reason="B receipt checked")
    index_a.rebuild(curator_a.ledger.iter_records(statuses={"approved"}))
    index_b.rebuild(curator_b.ledger.iter_records(statuses={"approved"}))

    assert [item.record_id for item in index_a.search("runner repair")] == ["a"]
    assert [item.record_id for item in index_b.search("runner repair")] == ["b"]


def test_laya_unavailable_does_not_change_deterministic_boundary() -> None:
    evaluation = LayaDecisionEngine(None, LayaRuntimeConfig()).evaluate({}, [])
    assert evaluation.diagnostic == "laya_unavailable"
    assert evaluation.decisions == ()
