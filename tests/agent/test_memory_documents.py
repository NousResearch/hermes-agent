from agent.memory_documents import MemoryDocument, MemorySourceRef


def _make_document(text: str) -> MemoryDocument:
    return MemoryDocument(
        memory_type="project_context",
        scope="repo:hermes-agent",
        source=MemorySourceRef(source_kind="file", source_id="docs/test.md", source_path="docs/test.md"),
        text=text,
        created_at="2026-05-07T00:00:00+00:00",
        updated_at="2026-05-07T01:00:00+00:00",
        freshness_hint="weekly",
        confidence=0.8,
        tags=("pinecone", "memory"),
        canonical=True,
        title="Test doc",
    )


def test_chunk_size_bounds_and_header_awareness():
    text = """# Overview
This is the overview paragraph. It explains the system.

## Details
""" + ("Sentence. " * 140) + """

## Notes
A short closing note.
"""
    doc = _make_document(text)
    chunks = doc.chunk(target_chars=220, max_chars=320)

    assert len(chunks) >= 3
    assert all(1 <= len(chunk.text) <= 320 for chunk in chunks)
    assert chunks[0].header_path == ("Overview",)
    assert any(chunk.header_path == ("Overview", "Details") for chunk in chunks)
    assert chunks[-1].header_path == ("Overview", "Notes")


def test_metadata_preservation_and_required_fields():
    doc = _make_document("# Title\nUseful memory content for retrieval.")
    chunk = doc.chunk(target_chars=200, max_chars=300)[0]

    assert chunk.memory_type == "project_context"
    assert chunk.scope == "repo:hermes-agent"
    assert chunk.source_kind == "file"
    assert chunk.source_id == "docs/test.md"
    assert chunk.source_path == "docs/test.md"
    assert chunk.created_at == "2026-05-07T00:00:00+00:00"
    assert chunk.updated_at == "2026-05-07T01:00:00+00:00"
    assert chunk.freshness_hint == "weekly"
    assert chunk.confidence == 0.8
    assert chunk.tags == ("pinecone", "memory")
    assert chunk.canonical is True
    assert chunk.metadata["source_kind"] == "file"
    assert chunk.metadata["tags"] == ["pinecone", "memory"]


def test_chunk_ids_are_stable_for_same_input():
    text = "# Same\n" + ("Repeatable text. " * 40)
    first = _make_document(text).chunk(target_chars=180, max_chars=240)
    second = _make_document(text).chunk(target_chars=180, max_chars=240)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.document_id for chunk in first] == [chunk.document_id for chunk in second]
