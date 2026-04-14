import json

from agent.knowledge_lanes import (
    KnowledgeLaneStore,
    validate_knowledge_payload,
)


def test_add_draft_creates_typed_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()

    record = store.add_draft(
        title="Tiger Smart Invest note",
        body="Likely not true instant redemption.",
        source="chat:user",
        provenance={"message_id": "123"},
        tags=["tiger", "finance"],
        confidence="medium",
    )

    assert record["lane"] == "draft"
    assert record["status"] == "draft"
    assert record["source"] == "chat:user"
    assert record["provenance"]["message_id"] == "123"
    assert record["confidence"] == "medium"


def test_promote_draft_moves_record_to_promoted_lane_with_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    draft = store.add_draft(
        title="Context7 is useful",
        body="Use it as a read-only docs sidecar.",
        source="repo:deep-dive",
        provenance={"report": "context7.md"},
    )

    promoted = store.promote_draft(
        draft["id"],
        promotion_reason="validated against pack synthesis",
        evidence=["report:context7", "report:hermes-self-evolution"],
    )

    assert promoted["lane"] == "promoted"
    assert promoted["status"] == "promoted"
    assert promoted["promotion"]["reason"] == "validated against pack synthesis"
    assert promoted["promotion"]["evidence"] == ["report:context7", "report:hermes-self-evolution"]

    state = store.read_state()
    assert len(state["draft_items"]) == 0
    assert len(state["promoted_items"]) == 1


def test_validate_knowledge_payload_reports_invalid_shapes():
    payload = {
        "schema_version": 1,
        "draft_items": [
            {
                "id": "x",
                "lane": "draft",
                "title": "Bad draft",
                "body": "Missing provenance and bad confidence",
                "source": "chat:user",
                "provenance": "oops",
                "confidence": "certain",
                "status": "draft",
                "created_at": "",
                "tags": "bad",
            }
        ],
        "promoted_items": [],
    }

    errors = validate_knowledge_payload(payload)

    assert any("provenance" in err for err in errors)
    assert any("confidence" in err for err in errors)
    assert any("tags" in err for err in errors)
    assert any("created_at" in err for err in errors)


def test_validator_summary_marks_promoted_store_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    draft = store.add_draft(
        title="Provider routing should be validated",
        body="Invalid config should fail closed.",
        source="repo:inspection",
        provenance={"file": "gateway/run.py"},
    )
    store.promote_draft(draft["id"], promotion_reason="implemented", evidence=["tests"])

    report = store.validation_report()

    assert report["valid"] is True
    assert report["counts"]["draft"] == 0
    assert report["counts"]["promoted"] == 1
    assert report["errors"] == []


def test_state_persists_as_json_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    store.add_draft(
        title="Draft persists",
        body="Persist me",
        source="chat:user",
        provenance={"message_id": "1"},
    )

    path = tmp_path / "knowledge" / "knowledge_lanes.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["draft_items"]) == 1


def test_find_relevant_items_defaults_to_promoted_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    store.add_draft(
        title="Draft routing note",
        body="Drafts should not be retrieved by default.",
        source="chat:user",
        provenance={"message_id": "1"},
        tags=["routing"],
    )
    promoted_seed = store.add_draft(
        title="Promoted routing rule",
        body="Only promoted routing knowledge should be reused by default.",
        source="repo:inspection",
        provenance={"file": "gateway/run.py"},
        tags=["routing", "verified"],
    )
    store.promote_draft(
        promoted_seed["id"],
        promotion_reason="validated in tests",
        evidence=["tests/agent/test_knowledge_lanes.py"],
    )

    results = store.find_relevant_items("routing")

    assert [item["title"] for item in results] == ["Promoted routing rule"]
    assert all(item["lane"] == "promoted" for item in results)


def test_find_relevant_items_supports_lane_override_and_tag_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    draft = store.add_draft(
        title="Draft archery note",
        body="A draft note about deliberate practice.",
        source="chat:user",
        provenance={"message_id": "2"},
        tags=["practice", "draft"],
    )
    promoted_seed = store.add_draft(
        title="Promoted archery note",
        body="A promoted note about deliberate practice and calm repetition.",
        source="repo:deep-dive",
        provenance={"report": "archery.md"},
        tags=["practice", "verified"],
    )
    store.promote_draft(
        promoted_seed["id"],
        promotion_reason="cross-checked",
        evidence=["report:archery"],
    )

    all_results = store.find_relevant_items("practice", lane="all", tags=["practice"])
    draft_results = store.find_relevant_items("practice", lane="draft", tags=["draft"])

    assert [item["title"] for item in all_results] == ["Draft archery note", "Promoted archery note"]
    assert [item["id"] for item in draft_results] == [draft["id"]]


def test_find_relevant_items_returns_empty_for_non_positive_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    promoted_seed = store.add_draft(
        title="Promoted note",
        body="Useful verified knowledge.",
        source="repo:deep-dive",
        provenance={"report": "note.md"},
        tags=["verified"],
    )
    store.promote_draft(
        promoted_seed["id"],
        promotion_reason="cross-checked",
        evidence=["report:note"],
    )

    assert store.find_relevant_items("knowledge", limit=0) == []
    assert store.find_relevant_items("knowledge", limit=-1) == []
