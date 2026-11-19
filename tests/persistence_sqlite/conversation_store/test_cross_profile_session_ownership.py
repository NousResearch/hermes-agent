"""Cross-profile ownership of rows published through the persistence interface."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from persistence import PersistenceError
from tests.fixtures.storage_seed import seed_session


def test_independent_profiles_can_create_the_same_session_id(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    seed_session(a, "same-id", "cli", title="A")
    seed_session(b, "same-id", "tui", title="B")
    assert a.sessions.get("same-id").title == "A"
    assert b.sessions.get("same-id").title == "B"


def test_concurrent_creates_with_same_id_are_isolated(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    a.ready()
    b.ready()
    barrier = Barrier(2)

    def create(spec):
        store, title = spec
        barrier.wait(timeout=5)
        seed_session(store, "two-writers", "cli", title=title)

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(create, ((a, "A"), (b, "B"))))
    assert a.sessions.get("two-writers").title == "A"
    assert b.sessions.get("two-writers").title == "B"


def test_compression_child_id_is_scoped_to_its_profile(backend, scope_a, scope_b):
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage

    owner, other = backend.open(scope_a), backend.open(scope_b)
    seed_session(owner, "shared-child", "tui", title="Preserve independent row")
    seed_session(other, "parent", "cli")
    handoff = [NewMessage.from_live({"role": "user", "content": "Preserve the handoff"})]

    other.lineage.publish_compression_child(
        "parent", CompressionChild("shared-child", "cli"), handoff, require_compression_lease=False,
    )
    assert other.sessions.get("parent").end_reason == "compression"
    assert other.sessions.get("shared-child").parent_session_id == "parent"
    assert owner.sessions.get("shared-child").title == "Preserve independent row"
    assert owner.sessions.get("shared-child").parent_session_id is None


def test_transcript_writes_do_not_cross_profile_scopes(backend, scope_a, scope_b):
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_read import transcript_rows

    a, b = backend.open(scope_a), backend.open(scope_b)
    seed_session(a, "shared", "cli")
    seed_session(b, "shared", "cli")
    a.messages.append("shared", NewMessage.from_live({"role": "user", "content": "A only"}))
    b.messages.append_batch("shared", [NewMessage.from_live({"role": "user", "content": "B only"})])
    assert [row["content"] for row in transcript_rows(a, "shared")] == ["A only"]
    assert [row["content"] for row in transcript_rows(b, "shared")] == ["B only"]


def test_retained_legacy_accounting_marker_refuses_direct_and_queued_usage(backend, scope_a):
    import hashlib

    from hermes_constants import get_default_hermes_root
    from tests.fixtures.storage_holder import raw_holder

    store = backend.open(scope_a)
    seed_session(store, "receipt-owner", "cli")
    holder = raw_holder(store)
    directory = get_default_hermes_root() / ".session-lineage-locks" / "accounting"
    directory.mkdir(mode=0o700, parents=True)
    location = hashlib.sha256(str(holder.db_path.resolve()).encode()).hexdigest()
    marker = directory / f"usage-{location}-old-inode-retained.lock"
    marker.touch(mode=0o600)

    from persistence.sqlite.conversation._store.usage import update_token_counts

    with pytest.raises(PersistenceError, match="Unresolved legacy accounting marker"):
        update_token_counts(holder, "receipt-owner", input_tokens=7)
    with pytest.raises(PersistenceError, match="Unresolved legacy accounting marker"):
        holder.queue_token_counts("receipt-owner", input_tokens=7)
    assert marker.is_file()
    assert holder._read_one("SELECT input_tokens FROM sessions WHERE id='receipt-owner'")[0] == 0
