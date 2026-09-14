"""NOVA Knowledge: declaration, ingestion, indexing and scoped retrieval.

The tests that matter most here are the negative ones. A retrieval system that returns
good results for a good question is easy; one that cannot be talked into returning a
document it was never granted, and that does not quietly index a symlink to a private key,
is the thing an enterprise is actually buying.
"""

from __future__ import annotations

import sqlite3

import pytest

from nova.errors import NovaError, SpecError
from nova.knowledge import (
    KnowledgeIndex,
    chunk_document,
    ingest,
    iter_documents,
    load_catalog,
)
from nova.knowledge.index import fts5_available
from nova.knowledge.query import EmptyQuery, build_match_expression

HANDBOOK = """# Refund Policy

## Eligibility

A customer may request a refund within 30 days of purchase.

## Approval thresholds

Refunds above $5,000 require sign-off from the finance lead and the account
director, and must reference the original contract.
"""

ONCALL = """# On-Call Rotation

The primary on-call engineer carries the pager for seven days, starting
Monday at 09:00 UTC. Escalation goes to the secondary after 15 minutes.
"""


@pytest.fixture
def corpus(tmp_path):
    """A two-document corpus, declared and on disk."""
    docs = tmp_path / "docs"
    (docs / "hr").mkdir(parents=True)
    (docs / "refunds.md").write_text(HANDBOOK, encoding="utf-8")
    (docs / "hr" / "oncall.md").write_text(ONCALL, encoding="utf-8")
    (tmp_path / "knowledge.yaml").write_text(
        "sources:\n"
        "  - id: handbook\n"
        "    title: Company Handbook\n"
        f"    root: {docs}\n"
        '    include: ["**/*.md"]\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def index(corpus):
    ingest(load_catalog(corpus), corpus / "k.db")
    with KnowledgeIndex.open(corpus / "k.db") as handle:
        yield handle


# -- declaration ------------------------------------------------------------


def test_fts5_is_available():
    """Every other test in this file is meaningless without it."""
    assert fts5_available()


def test_an_absent_knowledge_file_is_a_valid_state(tmp_path):
    """A tenant with no corpora is not a broken tenant."""
    assert load_catalog(tmp_path).sources == ()


def test_a_relative_root_resolves_against_the_bundle(corpus):
    (corpus / "knowledge.yaml").write_text(
        "sources:\n  - id: handbook\n    root: docs\n", encoding="utf-8"
    )
    assert load_catalog(corpus).get("handbook").root == (corpus / "docs").resolve()


def test_duplicate_source_ids_are_refused(corpus):
    (corpus / "knowledge.yaml").write_text(
        "sources:\n  - id: a\n    root: docs\n  - id: a\n    root: docs\n", encoding="utf-8"
    )
    with pytest.raises(SpecError, match="duplicate knowledge source id"):
        load_catalog(corpus)


def test_an_unknown_key_is_refused(corpus):
    """A typo must not silently produce a corpus with different behaviour."""
    (corpus / "knowledge.yaml").write_text(
        "sources:\n  - id: a\n    root: docs\n    includes: ['*.md']\n", encoding="utf-8"
    )
    with pytest.raises(SpecError, match="includes"):
        load_catalog(corpus)


def test_an_agent_cannot_be_granted_an_undeclared_corpus(tmp_path):
    """Caught at bundle load, where a typo is still cheap."""
    from nova.spec import load_bundle

    (tmp_path / "organization.yaml").write_text(
        "tenant_id: t\nlegal_name: T\n", encoding="utf-8"
    )
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a.yaml").write_text(
        "id: a\nname: A\nrole: worker\nknowledge:\n  sources: [nope]\n", encoding="utf-8"
    )
    with pytest.raises(SpecError, match="has not declared: nope"):
        load_bundle(tmp_path)


# -- walking the source -----------------------------------------------------


def test_a_double_star_pattern_matches_at_the_corpus_root(corpus):
    """``**/*.md`` means every .md file, including the ones not in a subdirectory.

    fnmatch has no ``**``, so the unexpanded pattern silently skips top-level files — and
    the only symptom is an ingest report with a smaller number in it than expected.
    """
    found = {path.name for path in iter_documents(load_catalog(corpus).get("handbook")).files}
    assert found == {"refunds.md", "oncall.md"}


def test_symlinks_are_never_followed(corpus):
    """The cheapest possible way to read a private key into a searchable corpus."""
    secret = corpus / "secret.pem"
    secret.write_text("PRIVATE KEY", encoding="utf-8")
    (corpus / "docs" / "innocent.md").symlink_to(secret)

    walk = iter_documents(load_catalog(corpus).get("handbook"))
    assert "innocent.md" not in {path.name for path in walk.files}
    assert any("symlink" in entry.reason for entry in walk.skipped)


def test_dot_directories_are_never_descended(corpus):
    (corpus / "docs" / ".git").mkdir()
    (corpus / "docs" / ".git" / "config.md").write_text("[remote]", encoding="utf-8")
    walk = iter_documents(load_catalog(corpus).get("handbook"))
    assert all(".git" not in path.parts for path in walk.files)


def test_a_file_over_the_cap_is_skipped_with_a_reason(corpus):
    (corpus / "knowledge.yaml").write_text(
        f"sources:\n  - id: handbook\n    root: {corpus / 'docs'}\n"
        '    include: ["**/*.md"]\n    max_file_bytes: 50\n',
        encoding="utf-8",
    )
    walk = iter_documents(load_catalog(corpus).get("handbook"))
    assert not walk.files
    assert all("max_file_bytes" in entry.reason for entry in walk.skipped)


# -- chunking ---------------------------------------------------------------


def test_a_chunk_knows_where_it_came_from():
    chunks = chunk_document(HANDBOOK, source_id="s", doc_path="refunds.md")
    assert chunks
    for chunk in chunks:
        assert chunk.doc_path == "refunds.md"
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.citation.startswith("refunds.md:")


def test_heading_trail_is_carried_into_the_citation():
    chunks = chunk_document(HANDBOOK, source_id="s", doc_path="r.md", max_chars=200)
    trails = [chunk.heading_trail for chunk in chunks]
    assert ("Refund Policy",) in trails
    assert any(len(trail) > 1 and trail[0] == "Refund Policy" for trail in trails)


def test_chunk_ids_are_stable_across_identical_ingests():
    first = chunk_document(HANDBOOK, source_id="s", doc_path="r.md")
    second = chunk_document(HANDBOOK, source_id="s", doc_path="r.md")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_differ_across_corpora():
    """The same document in two corpora is two documents, not one shared row."""
    a = chunk_document(HANDBOOK, source_id="a", doc_path="r.md")
    b = chunk_document(HANDBOOK, source_id="b", doc_path="r.md")
    assert a[0].chunk_id != b[0].chunk_id


def test_a_single_oversized_block_is_split_not_dropped():
    """One long line must not cost the document its place in the index."""
    chunks = chunk_document("x" * 9000, source_id="s", doc_path="blob.md", max_chars=1000)
    assert len(chunks) > 1
    assert sum(len(c.text) for c in chunks) >= 9000


def test_a_citation_never_claims_lines_the_chunk_does_not_own():
    """Overlap borrows text from the previous chunk; it must not borrow its line numbers."""
    text = "\n\n".join(f"Paragraph {n} " + "word " * 60 for n in range(12))
    chunks = chunk_document(text, source_id="s", doc_path="d.md", max_chars=600, overlap_chars=150)
    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_line >= previous.start_line


def test_an_empty_document_produces_no_chunks():
    assert chunk_document("   \n\n  ", source_id="s", doc_path="d.md") == []


# -- the query builder ------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        'refund" OR 1=1',
        "NEAR(a b)",
        "source_id:*",
        "^refund",
        "it's a customer's refund",
        '"unterminated',
        "a" * 5000,
    ],
)
def test_hostile_questions_never_reach_the_fts5_parser_intact(index, question):
    """Whatever the model types, the expression is rebuilt from parts this code chose."""
    index.search(question, source_ids=["handbook"], limit=3)  # must not raise


def test_stopwords_are_dropped_but_never_all_of_them():
    assert "refund" in build_match_expression("what is the refund policy")
    # A question made entirely of stopwords is still a question.
    assert build_match_expression("what is the")


def test_a_question_with_no_words_is_rejected():
    with pytest.raises(EmptyQuery):
        build_match_expression("!!! ??? ...")


def test_quoted_phrases_survive_as_phrases():
    assert '"finance lead"' in build_match_expression('"finance lead"')


# -- indexing and retrieval -------------------------------------------------


def test_ingest_indexes_every_declared_document(corpus):
    report = ingest(load_catalog(corpus), corpus / "k.db")
    assert report.indexed == 2
    assert report.chunks >= 2


def test_search_returns_a_citable_hit(index):
    hits = index.search("refund approval finance lead", source_ids=["handbook"], limit=3)
    assert hits
    assert hits[0].source_id == "handbook"
    assert hits[0].citation.startswith("refunds.md:")
    assert "finance" in hits[0].text.lower()


def test_scores_are_not_all_zero_on_a_small_corpus(index):
    """bm25 on a tiny corpus lands near 1e-06. Rounding that to four decimal places reports
    every hit as 0.0 and throws the ranking away at the display layer."""
    hits = index.search("refund", source_ids=["handbook"], limit=3)
    assert hits and hits[0].score != 0.0


def test_snippets_mark_the_matched_terms(index):
    hits = index.search("pager", source_ids=["handbook"], limit=1)
    assert "[[pager]]" in hits[0].snippet


def test_a_corpus_that_was_not_granted_returns_nothing(index):
    assert index.search("refund", source_ids=["some-other-corpus"]) == []


def test_no_granted_corpus_means_no_results_not_all_results(index):
    """The failure that turns a scoping bug into a disclosure incident."""
    assert index.search("refund", source_ids=[]) == []


def test_reingest_of_unchanged_documents_rechunks_nothing(corpus):
    ingest(load_catalog(corpus), corpus / "k.db")
    second = ingest(load_catalog(corpus), corpus / "k.db")
    assert second.indexed == 0
    assert second.sources[0].unchanged == 2


def test_an_edited_document_is_reindexed(corpus):
    ingest(load_catalog(corpus), corpus / "k.db")
    (corpus / "docs" / "refunds.md").write_text(
        HANDBOOK + "\n## Chargebacks\n\nA chargeback is not a refund.\n", encoding="utf-8"
    )
    assert ingest(load_catalog(corpus), corpus / "k.db").indexed == 1

    with KnowledgeIndex.open(corpus / "k.db") as index:
        assert index.search("chargeback", source_ids=["handbook"])


def test_a_deleted_document_stops_being_searchable(corpus):
    """A corpus that keeps serving a withdrawn document is a live disclosure."""
    ingest(load_catalog(corpus), corpus / "k.db")
    (corpus / "docs" / "hr" / "oncall.md").unlink()

    report = ingest(load_catalog(corpus), corpus / "k.db")
    assert report.sources[0].removed == ["hr/oncall.md"]
    with KnowledgeIndex.open(corpus / "k.db") as index:
        assert index.search("pager", source_ids=["handbook"]) == []


def test_a_dry_run_writes_nothing_but_reports_what_it_would_do(corpus):
    report = ingest(load_catalog(corpus), corpus / "k.db", dry_run=True)
    assert report.indexed == 2
    assert not (corpus / "k.db").exists()


def test_a_document_that_cannot_be_extracted_is_skipped_with_a_reason(corpus):
    (corpus / "docs" / "scan.md").write_bytes(b"\xff\xfe\x00binary")
    report = ingest(load_catalog(corpus), corpus / "k.db")
    assert any(path == "scan.md" for path, _ in report.sources[0].skipped)


def test_dropping_a_source_removes_its_documents(corpus):
    ingest(load_catalog(corpus), corpus / "k.db")
    with KnowledgeIndex.open(corpus / "k.db") as index:
        assert index.drop_source("handbook") == 2
        assert index.search("refund", source_ids=["handbook"]) == []


def test_an_index_from_a_future_schema_is_refused_not_migrated(corpus):
    ingest(load_catalog(corpus), corpus / "k.db")
    connection = sqlite3.connect(str(corpus / "k.db"))
    connection.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(NovaError, match="schema version"):
        KnowledgeIndex.open(corpus / "k.db")


# -- the audit invariant ----------------------------------------------------


def test_indexing_is_recorded_as_a_model_visible_change(corpus, tmp_path):
    """Putting documents where a model can quote them is model-visible by definition."""
    from nova.audit import AuditLog

    audit = AuditLog(tmp_path / "audit.jsonl", tenant_id="t")
    ingest(load_catalog(corpus), corpus / "k.db", audit=audit)

    events = [event for event in audit.read() if event.kind == "knowledge.indexed"]
    assert [event.phase for event in events] == ["intent", "committed"]
    assert all(event.model_visible for event in events)
    assert events[-1].detail["indexed"] == 2


def test_a_failed_ingest_records_the_failure_against_its_intent(corpus, tmp_path, monkeypatch):
    from nova.audit import AuditLog

    audit = AuditLog(tmp_path / "audit.jsonl", tenant_id="t")

    def explode(*args, **kwargs):
        raise RuntimeError("disk went away")

    # Reached through importlib rather than a dotted path: the package re-exports the
    # function ``ingest``, which shadows the module ``nova.knowledge.ingest`` for both
    # ``getattr`` and monkeypatch's own path resolution.
    import importlib

    monkeypatch.setattr(
        importlib.import_module("nova.knowledge.ingest"), "_ingest_source", explode
    )
    with pytest.raises(RuntimeError):
        ingest(load_catalog(corpus), corpus / "k.db", audit=audit)

    phases = [event.phase for event in audit.read() if event.kind == "knowledge.indexed"]
    assert phases == ["intent", "failed"]
