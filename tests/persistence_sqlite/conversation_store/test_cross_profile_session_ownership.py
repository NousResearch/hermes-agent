"""Cross-profile ownership of rows published through the persistence interface."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from persistence import PersistenceError
from tests.fixtures.storage_seed import seed_session


def test_existing_session_owner_refuses_a_second_store(backend, scope_a, scope_b):
    owner, other = backend.open(scope_a), backend.open(scope_b)
    seed_session(owner, "imported-id", "cli", title="Owner's title")
    with pytest.raises(PersistenceError, match="owned by profile store"):
        seed_session(other, "imported-id", "tui")
    assert owner.sessions.get("imported-id").title == "Owner's title"
    assert other.sessions.get("imported-id") is None
    seed_session(other, "distinct-id", "tui")
    assert other.sessions.get("distinct-id") is not None


def test_two_profiles_publishing_one_id_have_one_winner(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    a.ready()
    b.ready()
    barrier = Barrier(2)

    def create(store):
        barrier.wait(timeout=5)
        try:
            seed_session(store, "two-writers", "cli")
        except PersistenceError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as workers:
        published = list(workers.map(create, (a, b)))
    assert sum(published) == 1
    assert [s.sessions.get("two-writers") is not None for s in (a, b)] == published


def test_compression_refuses_foreign_child_before_ending_parent(backend, scope_a, scope_b):
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage

    owner, other = backend.open(scope_a), backend.open(scope_b)
    seed_session(owner, "reserved-child", "tui", title="Preserve foreign row")
    seed_session(other, "parent", "cli")
    handoff = [NewMessage.from_live({"role": "user", "content": "Preserve the handoff"})]

    with pytest.raises(PersistenceError, match="owned by profile store"):
        other.lineage.publish_compression_child(
            "parent", CompressionChild("reserved-child", "cli"), handoff, require_compression_lease=False,
        )
    assert other.sessions.get("parent").ended_at is None
    assert other.sessions.get("reserved-child") is None
    assert owner.sessions.get("reserved-child").title == "Preserve foreign row"

    other.lineage.publish_compression_child(
        "parent", CompressionChild("fresh-child", "cli"), handoff, require_compression_lease=False,
    )
    assert other.sessions.get("parent").end_reason == "compression"
    assert other.sessions.get("fresh-child").parent_session_id == "parent"


def test_transcript_append_refuses_ambiguous_legacy_rows(backend, scope_a, scope_b):
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_holder import raw_holder
    from tests.fixtures.storage_read import transcript_rows

    a, b = backend.open(scope_a), backend.open(scope_b)
    for store, scope in ((a, scope_a), (b, scope_b)):
        raw_holder(store)._execute_write(lambda conn: conn.execute(
            "INSERT INTO sessions(id, source, started_at, profile_name) VALUES (?, 'cli', 1, ?)",
            ("legacy-duplicate", scope.name),
        ))
    turn = NewMessage.from_live({"role": "user", "content": "Must never reach either copy"})
    with pytest.raises(PersistenceError):
        a.messages.append("legacy-duplicate", turn)
    with pytest.raises(PersistenceError):
        b.messages.append_batch("legacy-duplicate", [turn])
    assert not transcript_rows(a, "legacy-duplicate")
    assert not transcript_rows(b, "legacy-duplicate")
