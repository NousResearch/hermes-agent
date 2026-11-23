"""Compression race at the flush chokepoint: a turn writing against a session
already closed by compression must adopt the LIVE continuation tip instead of
dying with ``session_persistence_failed`` and a misleading "full disk" dialog.

The store resolves the continuation chain transitively via the canonical API
``lineage.tip`` (bounded walk, excludes branch/delegate/tool
children, prefers live children over stale closed siblings). This suite proves
the agent flush path:

* adopts a unique live child (depth-1 case),
* follows a chain of >=2 compressions to the live head — THE regression the
  depth-1 ``find_live_compression_child`` API missed (#82001),
* fails closed when no continuation exists (no retry loop),
* fails closed when the resolved tip is itself closed (``ws_orphan_reap``),
* performs the tip lookup exactly once per flush (adoption budget), and
* never renders the failure with the historical full-disk misdiagnosis.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session


def _flush_agent(store, session_id):
    """Bind the real flush methods onto a stand-in whose store is the writer handle."""
    agent = SimpleNamespace(
        _session_db=store,
        persistence=store,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent

def _build_compression_chain(store, chain: list[str]) -> tuple[str, str]:
    """Create ``chain[0] -> ... -> chain[-1]`` where every session except the
    last is compression-ended and the last is live. Returns (root, live_head).
    """
    for i, sid in enumerate(chain):
        parent = chain[i - 1] if i > 0 else None
        seed_session(store, sid, "tui", parent_session_id=parent)
        if i < len(chain) - 1:
            store.sessions.end(sid, "compression")
    return chain[0], chain[-1]

def test_flush_adopts_unique_live_continuation(tmp_path: Path) -> None:
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "parent", "tui")
        seed_messages(store, "parent", {"role": "user", "content": "before split"})
        store.sessions.end("parent", "compression")
        seed_session(store, "child", "tui", parent_session_id="parent")

        agent = _flush_agent(store, "parent")
        messages = [{"role": "user", "content": "steered after compression"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is True, "flush must succeed after adopting the continuation"
        assert agent.session_id == "child"
        durable = conversation(store, "child")
        assert any(
            m.get("content") == "steered after compression" for m in durable
        ), "the user message must land in the child session, not be lost"
        # The compression-closed parent stays immutable.
        parent_rows = conversation(store, "parent")
        assert not any(
            m.get("content") == "steered after compression" for m in parent_rows
        )
        assert agent._compression_adoption_failed is False
    finally:
        store.close()

def test_flush_adopts_live_head_across_compression_chain(tmp_path: Path) -> None:
    """A stale writer behind a chain of >=2 compressions adopts the live head.

    This is the exact lineage from #82001 (`root(compressed) -> mid(compressed)
    -> tip(live)`) that a depth-1 live-child lookup cannot resolve, because the
    direct child is itself already compression-ended.
    """
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        root, head = _build_compression_chain(store, ["root", "mid", "tip"])

        agent = _flush_agent(store, root)
        messages = [{"role": "user", "content": "steered after double rotation"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is True, "flush must succeed by adopting the chain head"
        assert agent.session_id == head, "agent must move to the live chain head"
        durable = conversation(store, head)
        assert any(
            m.get("content") == "steered after double rotation" for m in durable
        ), "the user message must land in the chain head, not be lost"
    finally:
        store.close()

def test_flush_fails_closed_when_no_continuation(tmp_path: Path) -> None:
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "parent", "tui")
        seed_messages(store, "parent", {"role": "user", "content": "before split"})
        store.sessions.end("parent", "compression")

        agent = _flush_agent(store, "parent")
        messages = [{"role": "user", "content": "steered after compression"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is False, "no continuation -> fail closed (never guess)"
        assert agent.session_id == "parent", "session id must not change"
        assert agent._compression_adoption_failed is True
        assert agent._last_persistence_error_cause == "compression_closed"
    finally:
        store.close()

def test_flush_fails_closed_when_tip_is_stale_closed(tmp_path: Path) -> None:
    """The canonical tip walk may land on a stale closed sibling (e.g.
    ``ws_orphan_reap``) — a non-live tip must NOT be adopted; fail closed."""
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "parent", "tui")
        seed_messages(store, "parent", {"role": "user", "content": "before split"})
        store.sessions.end("parent", "compression")
        seed_session(store, "stale", "tui", parent_session_id="parent")
        store.sessions.end("stale", "ws_orphan_reap")

        agent = _flush_agent(store, "parent")
        messages = [{"role": "user", "content": "steered after compression"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is False, "non-live tip must fail closed (never adopt stale)"
        assert agent.session_id == "parent"
        assert agent._compression_adoption_failed is True
    finally:
        store.close()

def test_flush_adopts_exactly_once_no_retry_loop(tmp_path: Path, monkeypatch) -> None:
    """Adoption budget: the tip lookup runs at most once per flush, and a
    second closed-parent write after adoption fails closed instead of looping.
    """
    from persistence import CompressionClosed

    store = open_persistence(profile_scope(home=tmp_path))
    try:
        _build_compression_chain(store, ["root", "tip"])

        agent = _flush_agent(store, "root")

        tip_calls = {"count": 0}

        def _counting_tip(session_id):
            tip_calls["count"] += 1
            return store.lineage.tip(session_id)

        # Every batch write raises closed — including the post-adoption retry
        # against the live tip (simulating the tip rotating again mid-flush).
        def _always_closed(session_id, messages, **kwargs):
            raise CompressionClosed(session_id, session_id)

        agent.persistence = SimpleNamespace(
            read_only=False, scope=store.scope, sessions=store.sessions,
            lineage=SimpleNamespace(tip=_counting_tip), messages=SimpleNamespace(append_batch=_always_closed))

        messages = [{"role": "user", "content": "steered after compression"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is False, "second closed-parent write must fail closed"
        assert tip_calls["count"] == 1, "tip lookup must happen exactly once"
        assert agent._compression_adoption_failed is True
    finally:
        store.close()

# ---------------------------------------------------------------------------
# Diagnostics: the failure must never read like a disk problem.
# ---------------------------------------------------------------------------

def test_compression_closed_error_classifies_as_compression_closed() -> None:
    from persistence.sqlite._state.errors import (
        PERSISTENCE_ERROR_CAUSES, CompressionSessionClosedError, classify_persistence_error,
    )

    cause = classify_persistence_error(CompressionSessionClosedError("session-abc"))
    assert cause == "compression_closed"
    assert cause in PERSISTENCE_ERROR_CAUSES
    # String form (post-RPC wrapping) classifies identically.
    assert (
        classify_persistence_error(str(CompressionSessionClosedError("session-abc")))
        == "compression_closed"
    )
