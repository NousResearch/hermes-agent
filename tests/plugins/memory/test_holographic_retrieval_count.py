"""retrieval_count must track facts surfaced by EVERY recall path.

The store had a working counter but nothing ever called it: ``retrieval_count``
was incremented only inside ``MemoryStore.search_facts``, which no code path
reached, while the live recall surface (``FactRetriever.search`` and the
vector-space ``probe``/``related``/``reason``) never touched it. Every fact in a
long-lived store therefore sat at ``retrieval_count = 0`` forever, so "which
facts does this agent actually use" — the signal consolidation and trust decay
are supposed to consume — did not exist.

These are behavior contracts (relationships between before/after state), not
snapshots of any particular count.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Real on-disk store. NOT ':memory:' — MemoryStore resolves the path and
    shares one process-wide connection per file, so ':memory:' becomes a
    literal ./:memory: file that leaks state across tests."""
    s = MemoryStore(str(tmp_path / "retrieval_count.db"))
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """Facts whose entities and text both support keyword and vector recall."""
    for i in range(6):
        store.add_fact(
            content=f"Deployment target {i} uses the Ganymede pipeline with option {i % 3}.",
            category="project" if i % 2 else "tool",
            tags=f"deploy entity_{i % 3}",
        )
    return store


def _count(store: MemoryStore, fact_id: int) -> int:
    return store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()["retrieval_count"]


def _counts(store: MemoryStore) -> dict[int, int]:
    return {
        r["fact_id"]: r["retrieval_count"]
        for r in store._conn.execute("SELECT fact_id, retrieval_count FROM facts").fetchall()
    }


# ---------------------------------------------------------------------------
# The bug: recall paths did not count
# ---------------------------------------------------------------------------

def test_search_increments_only_returned_facts(seeded):
    """search() counts what it returns — and nothing it merely considered."""
    retriever = FactRetriever(store=seeded)
    before = _counts(seeded)

    results = retriever.search("Ganymede pipeline deployment", limit=2)

    assert results, "fixture must actually match, or this asserts nothing"
    assert len(results) <= 2
    returned = {f["fact_id"] for f in results}
    after = _counts(seeded)

    for fact_id, prior in before.items():
        expected = prior + 1 if fact_id in returned else prior
        assert after[fact_id] == expected, (
            f"fact {fact_id}: expected {expected}, got {after[fact_id]} "
            f"(returned={fact_id in returned})"
        )


def test_search_accumulates_across_calls(seeded):
    """The counter is cumulative — two recalls of the same fact count twice."""
    retriever = FactRetriever(store=seeded)

    first = retriever.search("Ganymede pipeline", limit=1)
    assert first
    fact_id = first[0]["fact_id"]
    after_one = _count(seeded, fact_id)

    second = retriever.search("Ganymede pipeline", limit=1)
    assert second and second[0]["fact_id"] == fact_id
    assert _count(seeded, fact_id) == after_one + 1


@pytest.mark.parametrize("call", [
    pytest.param(lambda r: r.probe("Ganymede"), id="probe"),
    pytest.param(lambda r: r.related("Ganymede"), id="related"),
    pytest.param(lambda r: r.reason(["Ganymede"]), id="reason"),
])
def test_vector_recall_paths_increment(seeded, call):
    """probe/related/reason rank vectors directly and bypass search() — each
    must still record retrieval, or the counter only works for keyword hits."""
    retriever = FactRetriever(store=seeded)
    before = _counts(seeded)

    results = call(retriever)

    assert results, "vector path returned nothing; the assertion below is vacuous"
    returned = {f["fact_id"] for f in results}
    after = _counts(seeded)
    assert any(after[i] > before[i] for i in returned)
    for fact_id in returned:
        assert after[fact_id] == before[fact_id] + 1


def test_returned_dict_matches_persisted_count(seeded):
    """The dict handed to the caller must not report a stale pre-write count."""
    retriever = FactRetriever(store=seeded)
    results = retriever.search("Ganymede pipeline", limit=3)

    assert results
    for fact in results:
        assert fact["retrieval_count"] == _count(seeded, fact["fact_id"])


# ---------------------------------------------------------------------------
# What must NOT count
# ---------------------------------------------------------------------------

def test_list_facts_does_not_increment(seeded):
    """Browsing the store is inspection, not use. Counting it would let the
    weekly consolidation pass inflate every fact it reviews."""
    before = _counts(seeded)
    seeded.list_facts(limit=100)
    assert _counts(seeded) == before


def test_contradict_does_not_increment(seeded):
    """contradict() is memory hygiene scanning for conflicts, not recall."""
    retriever = FactRetriever(store=seeded)
    before = _counts(seeded)
    retriever.contradict()
    assert _counts(seeded) == before


def test_empty_result_set_is_a_noop(seeded):
    """No hits, no writes — an unmatched query must not touch the table."""
    retriever = FactRetriever(store=seeded)
    before = _counts(seeded)
    assert retriever.search("zzzznonexistentqqqq") == []
    assert _counts(seeded) == before


def test_record_retrieval_ignores_empty_input(store):
    assert store.record_retrieval([]) == 0
    assert store.record_retrieval([None]) == 0  # type: ignore[list-item]


def test_record_retrieval_survives_unknown_ids(seeded):
    """Unknown ids update nothing rather than raising."""
    before = _counts(seeded)
    assert seeded.record_retrieval([999999]) == 0
    assert _counts(seeded) == before


def test_retrieval_is_best_effort_and_never_breaks_a_read(seeded, monkeypatch):
    """A failing counter must degrade to 'no stats', never to a failed recall.

    Memory is a read-mostly dependency of every turn; a bookkeeping write that
    can raise would turn a locked database into a broken conversation.
    """
    retriever = FactRetriever(store=seeded)

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(seeded, "record_retrieval", boom)
    results = retriever.search("Ganymede pipeline", limit=2)
    assert results, "read must still succeed when retrieval accounting fails"


def test_store_record_retrieval_swallows_sqlite_errors(seeded, monkeypatch):
    """The store-level guard holds even if the connection itself misbehaves."""
    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(seeded, "_write", boom)
    assert seeded.record_retrieval([1]) == 0


# ---------------------------------------------------------------------------
# Retrieval must not corrupt neighbouring state
# ---------------------------------------------------------------------------

def test_retrieval_does_not_touch_updated_at(seeded):
    """Retrieval is a read. Bumping updated_at would reset the temporal-decay
    clock, so merely finding a fact would make a stale fact look fresh.

    Backdates the rows first: SQLite's CURRENT_TIMESTAMP has one-second
    granularity, so comparing 'now' against 'now' passes even when the code
    DOES rewrite the column. A distinctly old sentinel makes the write visible.
    """
    retriever = FactRetriever(store=seeded)
    sentinel = "2001-02-03 04:05:06"
    seeded._conn.execute("UPDATE facts SET updated_at = ?", (sentinel,))
    seeded._conn.commit()

    assert retriever.search("Ganymede pipeline", limit=3)

    stamps = {
        r["updated_at"]
        for r in seeded._conn.execute("SELECT updated_at FROM facts").fetchall()
    }
    assert stamps == {sentinel}, f"retrieval rewrote updated_at: {stamps}"


def test_retrieval_does_not_disturb_trust_or_helpful_count(seeded):
    retriever = FactRetriever(store=seeded)
    before = [
        (r["fact_id"], r["trust_score"], r["helpful_count"])
        for r in seeded._conn.execute(
            "SELECT fact_id, trust_score, helpful_count FROM facts ORDER BY fact_id"
        ).fetchall()
    ]

    assert retriever.search("Ganymede pipeline", limit=3)

    after = [
        (r["fact_id"], r["trust_score"], r["helpful_count"])
        for r in seeded._conn.execute(
            "SELECT fact_id, trust_score, helpful_count FROM facts ORDER BY fact_id"
        ).fetchall()
    ]
    assert after == before


def test_search_still_matches_after_many_retrievals(seeded):
    """Regression guard for the FTS5 side of this change.

    The facts_au trigger rebuilds the full-text row on UPDATE. Counting
    retrievals fires that trigger on the hot read path, so the trigger is now
    WHEN-guarded to content/tags changes. If that guard were wrong, repeated
    recall would corrupt or empty the FTS index and search would go silent.
    """
    retriever = FactRetriever(store=seeded)
    for _ in range(10):
        assert retriever.search("Ganymede pipeline", limit=3)

    assert retriever.search("Ganymede pipeline", limit=3)
    integrity = seeded._conn.execute(
        "INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')"
    )
    assert integrity is not None  # raises above if the FTS index is corrupt


def test_content_edit_still_reindexes_after_trigger_guard(seeded):
    """The WHEN guard must not break the case the trigger exists for:
    editing content has to remain searchable under its new text."""
    retriever = FactRetriever(store=seeded)
    first = retriever.search("Ganymede pipeline", limit=1)
    assert first
    fact_id = first[0]["fact_id"]

    assert seeded.update_fact(fact_id, content="Callisto relay handles the overflow queue.")

    assert retriever.search("Callisto relay overflow"), "edited content is not searchable"
    stale = [f["fact_id"] for f in retriever.search("Ganymede pipeline", limit=10)]
    assert fact_id not in stale, "old text still indexed after an edit"


def test_tag_edit_still_reindexes_after_trigger_guard(seeded):
    """Tags are in the FTS index too — the guard covers them explicitly."""
    retriever = FactRetriever(store=seeded)
    first = retriever.search("Ganymede pipeline", limit=1)
    assert first
    fact_id = first[0]["fact_id"]

    assert seeded.update_fact(fact_id, tags="tycho brahe marker")

    assert any(
        f["fact_id"] == fact_id for f in retriever.search("tycho marker", limit=10)
    ), "edited tags are not searchable"


def test_legacy_unguarded_trigger_is_migrated(tmp_path):
    """Existing databases carry the old unguarded facts_au trigger, and
    CREATE TRIGGER IF NOT EXISTS will not replace it. _init_db must drop and
    recreate it, or upgraded users keep paying an FTS rebuild per retrieval."""
    db_path = tmp_path / "legacy.db"

    store = MemoryStore(str(db_path))
    store.add_fact(content="Legacy Ganymede fact for migration.", category="tool")
    store.close()

    # Reinstate the pre-fix trigger definition.
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        DROP TRIGGER IF EXISTS facts_au;
        CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
            INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
        END;
        """
    )
    raw.commit()
    raw.close()

    reopened = MemoryStore(str(db_path))
    try:
        sql = reopened._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'facts_au'"
        ).fetchone()["sql"]
        assert "WHEN" in sql.upper(), "legacy trigger was not migrated"

        # And the migrated database still works end to end.
        retriever = FactRetriever(store=reopened)
        results = retriever.search("Legacy Ganymede", limit=1)
        assert results
        assert _count(reopened, results[0]["fact_id"]) == 1
    finally:
        reopened.close()
