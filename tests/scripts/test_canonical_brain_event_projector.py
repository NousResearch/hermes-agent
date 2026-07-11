from scripts.canonical_brain_event_projector import projection_documents, write_documents


def test_projector_is_event_type_driven_and_writes_atomic_documents(tmp_path):
    rows = [{
        "event_id": "e1",
        "event_type": "case.note",
        "case_id": "case:1",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": {"source_refs": {"platform": "discord", "thread_id": "t1", "message_id": "m1"}},
        "status": {"state": "case.note", "summary": "contains blocker risk priority words but is not classified"},
        "next_action": {},
        "payload": {},
    }]
    documents = projection_documents(rows)
    assert documents["index.json"]["semantic_classifier"] is False
    assert documents["index.json"]["case_count"] == 1
    assert documents["route_backs.json"]["items"] == []
    write_documents(tmp_path, documents)
    assert (tmp_path / "cases.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))
