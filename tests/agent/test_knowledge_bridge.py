import json
from pathlib import Path

from agent.knowledge_bridge import (
    export_lane_item_to_governance_package,
    load_exported_packet,
)
from agent.knowledge_lanes import KnowledgeLaneStore


def test_export_lane_item_to_governance_package_writes_candidate_and_package(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    draft = store.add_draft(
        title="Routing fallback rule",
        body="Prefer recently healthy fallback lanes when the primary fails.",
        source="memory_tool:memory:add",
        provenance={"session_id": "sess-1", "tool_trace_refs": ["tool:memory:add"]},
        tags=["routing", "fallback"],
        confidence="high",
    )

    result = export_lane_item_to_governance_package(
        lane_item_id=draft["id"],
        lane="draft",
        repo_root=Path("/home/user/.hermes/hermes-agent"),
        target_surface="memory",
        target_path="memory/governed-knowledge.md",
    )

    candidate = load_exported_packet(Path(result["promotion_candidate_path"]))
    package = load_exported_packet(Path(result["ingestion_package_path"]))
    evidence = load_exported_packet(Path(result["evidence_path"]))

    assert candidate["promotion_id"].startswith("prom_")
    assert candidate["source_lane"]["lane_id"] == "hermes-knowledge-lane:draft"
    assert candidate["source_lane"]["work_item_id"] == draft["id"]
    assert candidate["insight"]["title"] == draft["title"]
    assert candidate["target"]["surface"] == "memory"
    assert candidate["source_refs"][0]["path"].endswith(f"{draft['id']}.json")

    assert package["schema_version"] == "clawd.knowledge_ingestion.package.v1"
    assert package["promotion_candidate_ref"]["promotion_id"] == candidate["promotion_id"]
    assert package["preserved_evidence"]["item_count"] == 1
    assert package["handoff"]["queue_runtime"].endswith("knowledge_promotion_queue.py")

    assert evidence["title"] == draft["title"]
    assert evidence["body"] == draft["body"]


def test_export_lane_item_to_governance_package_backlinks_into_lane_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    promoted_seed = store.add_draft(
        title="Operator surface idea",
        body="Build Hermes-native triage from existing runtime truth.",
        source="repo:inspection",
        provenance={"file": "gateway/status.py"},
        tags=["operator"],
        confidence="medium",
    )
    promoted = store.promote_draft(
        promoted_seed["id"],
        promotion_reason="validated",
        evidence=["tests"],
    )

    result = export_lane_item_to_governance_package(
        lane_item_id=promoted["id"],
        lane="promoted",
        repo_root=Path("/home/user/.hermes/hermes-agent"),
        target_surface="playbook",
        target_path="docs/ops/operator-playbook.md",
    )

    state = store.read_state()
    stored = state["promoted_items"][0]
    bridge = stored["provenance"]["governance_bridge"]

    assert bridge["promotion_candidate_path"] == result["promotion_candidate_path"]
    assert bridge["ingestion_package_path"] == result["ingestion_package_path"]
    assert bridge["target_surface"] == "playbook"
    assert bridge["target_path"] == "docs/ops/operator-playbook.md"
