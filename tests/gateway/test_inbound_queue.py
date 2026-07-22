"""Focused tests for the durable Gateway inbound queue.

The queue is intentionally backed by a small standalone SQLite database.  It
must remain usable while ``state.db`` is busy, survive process replacement,
and preserve one-at-a-time fairness between independent runtime sessions.
"""

from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from gateway.config import Platform
from gateway.inbound_queue import (
    GatewayInboxStore,
    _owner_alive,
    lookup_session_trigger_durability,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _source(
    *,
    chat_id: str = "chat-1",
    thread_id: str | None = "thread-1",
    profile: str | None = "coding",
) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_name="ops",
        chat_type="thread",
        user_id="user-1",
        user_name="Operator",
        thread_id=thread_id,
        chat_topic="Gateway incidents",
        user_id_alt="stable-user-1",
        scope_id="guild-1",
        parent_chat_id="parent-1",
        message_id="source-message-1",
        profile=profile,
        is_bot=False,
        role_authorized=True,
        delivered_via_upstream_relay=True,
    )


def _event(
    message_id: str | None = "message-1",
    *,
    text: str = "continue the interrupted task",
    source: SessionSource | None = None,
    explicit_queue: bool = False,
) -> MessageEvent:
    metadata = {
        "thread_ts": "123.4",
        "session_webhook": "https://example.invalid/send?access_token=secret",
        "_hermes_gateway_inbox": {"claim_token": "must-not-be-persisted"},
        "untrusted_secret": "must-not-be-persisted",
        "not_json": object(),
    }
    if explicit_queue:
        metadata["_hermes_explicit_queue"] = {
            "id": "q-explicit-1",
            "owner_user_id": "user-1",
            "origin": "explicit",
            "created_at": "2026-07-21T12:00:00+00:00",
            "bearer": "nested-secret",
        }
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source or _source(),
        raw_message=object(),
        message_id=message_id,
        platform_update_id=42,
        media_urls=["/tmp/image.png"],
        media_types=["image/png"],
        reply_to_message_id="reply-1",
        reply_to_text="prior context",
        reply_to_author_id="author-1",
        reply_to_author_name="Alice",
        reply_to_is_own_message=True,
        auto_skill=["hermes-development"],
        channel_prompt="Keep replies concise.",
        channel_context="Earlier channel context",
        metadata=metadata,
        timestamp=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )


def _store(tmp_path, **kwargs) -> GatewayInboxStore:
    return GatewayInboxStore(hermes_home=tmp_path, **kwargs)


def test_owner_alive_uses_cross_platform_pid_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "gateway.status.get_process_start_time", lambda _pid: None
    )
    monkeypatch.setattr(
        "gateway.status._pid_exists",
        lambda pid: calls.append(pid) or True,
    )

    assert _owner_alive(1234, None) is True
    assert calls == [1234]


def test_trigger_durability_lookup_preserves_indeterminate_db_errors():
    class RawDB:
        error = None

        def has_platform_message_id(self, session_id, trigger_identity):
            assert session_id == "session-1"
            assert trigger_identity == "message-1"
            if self.error is not None:
                raise self.error
            return True

    raw_db = RawDB()
    session_store = type("SessionStoreStub", (), {"_db": raw_db})()

    assert (
        lookup_session_trigger_durability(
            session_store, "session-1", "message-1"
        )
        is True
    )
    raw_db.error = sqlite3.OperationalError("database is locked")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        lookup_session_trigger_durability(
            session_store, "session-1", "message-1"
        )


def test_database_uses_wal_full_sync_and_private_permissions(tmp_path):
    store = _store(tmp_path)

    with sqlite3.connect(store.path()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2

    assert os.stat(store.path()).st_mode & 0o777 == 0o600


def test_message_event_round_trip_uses_an_explicit_json_whitelist(tmp_path):
    store = _store(tmp_path)
    result = store.enqueue(
        _event(explicit_queue=True),
        session_key="agent:main:discord:thread:chat-1:thread-1",
        origin="explicit",
    )

    assert result.accepted is True
    assert result.inserted is True
    assert result.row is not None
    assert result.row.queue_id == "q-explicit-1"

    restored = result.row.to_event()
    assert restored.text == "continue the interrupted task"
    assert restored.message_type is MessageType.TEXT
    assert restored.raw_message is None
    assert restored.timestamp == datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert restored.source.platform is Platform.DISCORD
    assert restored.source.scope_id == "guild-1"
    assert restored.source.profile == "coding"
    assert restored.source.delivered_via_upstream_relay is False
    assert restored.metadata["thread_ts"] == "123.4"
    assert restored.metadata["_hermes_explicit_queue"]["id"] == "q-explicit-1"
    assert restored.metadata["_hermes_gateway_inbox"]["queue_id"] == "q-explicit-1"
    assert restored.metadata["_hermes_gateway_inbox"]["claim_token"] is None
    assert restored.metadata["_hermes_gateway_inbox"]["trigger_identity"] == "message-1"
    assert "bearer" not in restored.metadata["_hermes_explicit_queue"]
    assert "session_webhook" not in restored.metadata
    assert "untrusted_secret" not in restored.metadata
    assert "not_json" not in restored.metadata

    raw_payload = (
        sqlite3
        .connect(store.path())
        .execute(
            "SELECT payload_json FROM inbound_events WHERE queue_id=?",
            (result.row.queue_id,),
        )
        .fetchone()[0]
    )
    assert "raw_message" not in raw_payload
    assert "must-not-be-persisted" not in raw_payload
    assert "delivered_via_upstream_relay" not in raw_payload
    assert "session_webhook" not in raw_payload
    assert "access_token" not in raw_payload
    assert "nested-secret" not in raw_payload
    assert "_hermes_gateway_inbox" not in raw_payload


def test_enqueue_deduplicates_platform_message_within_routed_session(tmp_path):
    store = _store(tmp_path)
    key = "agent:main:discord:thread:chat-1:thread-1"

    first = store.enqueue(_event("same-id"), session_key=key)
    duplicate = store.enqueue(_event("same-id"), session_key=key)
    other_lane = store.enqueue(
        _event("same-id", source=_source(thread_id="thread-2")),
        session_key="agent:main:discord:thread:chat-1:thread-2",
    )

    assert first.accepted and first.inserted
    assert duplicate.accepted and not duplicate.inserted
    assert duplicate.row.queue_id == first.row.queue_id
    assert other_lane.accepted and other_lane.inserted
    assert store.pending_count() == 2


def test_missing_platform_id_gets_a_stable_synthetic_trigger_identity(tmp_path):
    store = _store(tmp_path)

    result = store.enqueue(_event(None), session_key="session-a")

    assert result.accepted and result.row is not None
    assert result.row.trigger_identity == f"gateway-inbox:{result.row.queue_id}"
    assert result.row.dedupe_key.startswith("arrival:")


def test_enqueue_enforces_per_session_and_global_pending_caps(tmp_path):
    store = _store(tmp_path, max_pending_per_session=2, max_pending_total=3)

    assert store.enqueue(_event("a1"), session_key="a").accepted
    assert store.enqueue(_event("a2"), session_key="a").accepted
    per_session_full = store.enqueue(_event("a3"), session_key="a")
    assert per_session_full.accepted is False
    assert per_session_full.status == "session_full"

    assert store.enqueue(_event("b1"), session_key="b").accepted
    global_full = store.enqueue(_event("c1"), session_key="c")
    assert global_full.accepted is False
    assert global_full.status == "global_full"

    # Dedupe is checked before capacity, so a platform retry still resolves to
    # the existing durable row instead of being misreported as queue overflow.
    duplicate = store.enqueue(_event("a1"), session_key="a")
    assert duplicate.accepted and not duplicate.inserted


def test_bind_claim_complete_and_cancel_are_state_guarded(tmp_path):
    store = _store(tmp_path)
    first = store.enqueue(_event("first"), session_key="a").row
    second = store.enqueue(_event("second"), session_key="b").row

    assert store.bind(first.queue_id, "session-db-a", "platform-first") is True
    claimed = store.claim(first.queue_id)
    assert claimed is not None
    assert claimed.state == "claimed"
    assert claimed.session_id == "session-db-a"
    assert claimed.trigger_identity == "platform-first"
    assert (
        store.bind(
            first.queue_id,
            "wrong-session",
            "wrong-trigger",
            claim_token="wrong-token",
        )
        is False
    )
    assert store.bind(
        first.queue_id,
        "session-db-final",
        "platform-final",
        claim_token=claimed.claim_token,
    )
    rebound = store.claim(first.queue_id, claim_token=claimed.claim_token)
    assert rebound.session_id == "session-db-final"
    assert rebound.trigger_identity == "platform-final"

    assert store.cancel(first.queue_id) is False
    assert store.complete(first.queue_id, claimed.claim_token) is True
    assert store.complete(first.queue_id, claimed.claim_token) is False
    assert store.get(first.queue_id).state == "completed"

    assert store.cancel(second.queue_id, session_key="b") is True
    assert store.get(second.queue_id).state == "cancelled"


def test_claimed_row_cannot_be_cancelled_without_its_claim_token(tmp_path):
    store = _store(tmp_path)
    row = store.enqueue(_event("claimed"), session_key="claimed-session").row
    claimed = store.claim(row.queue_id)

    assert claimed is not None
    queue_id = claimed.queue_id
    assert store.cancel(queue_id, session_key="claimed-session") is False
    restored = store.get(queue_id)
    assert restored is not None
    assert restored.state == "claimed"


def test_claim_next_is_fifo_within_session_and_round_robin_across_sessions(tmp_path):
    store = _store(tmp_path)
    a1 = store.enqueue(_event("a1"), session_key="a").row
    a2 = store.enqueue(_event("a2"), session_key="a").row
    b1 = store.enqueue(_event("b1"), session_key="b").row
    c1 = store.enqueue(_event("c1"), session_key="c").row

    first = store.claim_next()
    assert first.queue_id == a1.queue_id
    assert store.complete(first.queue_id, first.claim_token)

    second = store.claim_next()
    assert second.queue_id == b1.queue_id
    assert store.complete(second.queue_id, second.claim_token)

    third = store.claim_next()
    assert third.queue_id == c1.queue_id
    assert store.complete(third.queue_id, third.claim_token)

    fourth = store.claim_next()
    assert fourth.queue_id == a2.queue_id


def test_claim_next_skips_inflight_excluded_and_undeliverable_sessions(tmp_path):
    store = _store(tmp_path)
    a1 = store.enqueue(_event("a1"), session_key="a").row
    store.enqueue(_event("a2"), session_key="a")
    store.enqueue(_event("b1", source=_source(profile="other")), session_key="b")
    store.enqueue(
        MessageEvent(
            text="c", source=SessionSource(Platform.SLACK, "c"), message_id="c1"
        ),
        session_key="c",
    )

    assert store.claim(a1.queue_id) is not None
    # a has an in-flight row, b is explicitly excluded, and c is not on a
    # currently deliverable platform.
    assert (
        store.claim_next(exclude_session_keys={"b"}, deliverable_platforms={"discord"})
        is None
    )


def test_rows_survive_store_reopen(tmp_path):
    first_store = _store(tmp_path)
    queued = first_store.enqueue(_event("persist-me"), session_key="session-a").row
    first_store.bind(queued.queue_id, "db-session-a")

    reopened = _store(tmp_path)
    restored = reopened.get(queued.queue_id)
    assert restored is not None
    assert restored.state == "queued"
    assert restored.session_id == "db-session-a"
    assert restored.to_event().text == "continue the interrupted task"


def test_dead_owner_without_durable_trigger_is_requeued(tmp_path, monkeypatch):
    store = _store(tmp_path)
    row = store.enqueue(_event("replay"), session_key="a").row
    store.bind(row.queue_id, "db-a", "trigger-a")
    assert store.claim(row.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    reclaimed = store.reclaim_dead_claims(lambda session_id, trigger_identity: False)

    assert [(item.queue_id, item.state) for item in reclaimed] == [
        (row.queue_id, "queued")
    ]
    assert store.get(row.queue_id).owner_pid is None


def test_dead_owner_with_durable_trigger_becomes_resume_ready(tmp_path, monkeypatch):
    store = _store(tmp_path)
    row = store.enqueue(_event("resume"), session_key="a").row
    store.bind(row.queue_id, "db-a", "trigger-a")
    assert store.claim(row.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    reclaimed = store.reclaim_dead_claims(
        lambda session_id, trigger_identity: (
            session_id == "db-a" and trigger_identity == "trigger-a"
        )
    )

    assert [(item.queue_id, item.state) for item in reclaimed] == [
        (row.queue_id, "resume_ready")
    ]
    ready = store.get(row.queue_id)
    assert ready.owner_pid is None
    assert ready.claim_token is None
    assert ready.resume_only is True

    claimed = store.claim_next()
    assert claimed.queue_id == row.queue_id
    assert claimed.resume_only is True
    assert claimed.claim_token
    assert claimed.attempts == 2
    assert claimed.owner_pid == os.getpid()


def test_indeterminate_trigger_check_leaves_dead_claim_for_retry(tmp_path, monkeypatch):
    store = _store(tmp_path)
    row = store.enqueue(_event("unknown"), session_key="a").row
    store.bind(row.queue_id, "db-a", "trigger-a")
    assert store.claim(row.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    assert store.reclaim_dead_claims(lambda *_: None) == []
    assert store.get(row.queue_id).state == "claimed"


def test_dead_claim_over_attempt_cap_moves_to_dead_letter(tmp_path, monkeypatch):
    store = _store(tmp_path, max_attempts=1)
    row = store.enqueue(_event("poison"), session_key="a").row
    assert store.claim(row.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    reclaimed = store.reclaim_dead_claims(lambda *_: False)

    assert [(item.queue_id, item.state) for item in reclaimed] == [
        (row.queue_id, "dead_letter")
    ]


def test_claim_token_guards_completion_and_owner_scoped_retry(tmp_path):
    store = _store(tmp_path)
    row = store.enqueue(_event("token-guard"), session_key="a").row

    first = store.claim(row.queue_id)
    assert first is not None and first.claim_token
    assert store.claim(row.queue_id) is None
    assert (
        store.claim(row.queue_id, claim_token=first.claim_token).claim_token
        == first.claim_token
    )
    assert store.complete(row.queue_id, "wrong-token") is False

    assert store.retry(
        row.queue_id,
        first.claim_token,
        error="transient worker failure",
        not_before=0,
    )
    ready = store.get(row.queue_id)
    assert ready.state == "queued"
    assert ready.claim_token is None
    assert ready.last_error == "transient worker failure"

    second = store.claim(row.queue_id)
    assert second.claim_token != first.claim_token
    assert store.complete(row.queue_id, first.claim_token) is False
    assert store.complete(row.queue_id, second.claim_token) is True


def test_release_preserves_resume_only_dispatch_mode(tmp_path, monkeypatch):
    store = _store(tmp_path, max_attempts=4)
    row = store.enqueue(_event("resume-release"), session_key="a").row
    store.bind(row.queue_id, "db-a", "trigger-a")
    assert store.claim(row.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)
    store.reclaim_dead_claims(lambda *_: True)

    resumed = store.claim_next()
    assert resumed.resume_only is True
    assert store.release(resumed.queue_id, resumed.claim_token, error="cancelled")
    ready = store.get(row.queue_id)
    assert ready.state == "resume_ready"
    assert ready.resume_only is True

    resumed_again = store.claim_next()
    assert resumed_again.resume_only is True
    assert resumed_again.claim_token != resumed.claim_token


def test_retry_can_promote_a_live_claim_to_resume_only(tmp_path):
    store = _store(tmp_path)
    row = store.enqueue(_event("durable-before-failure"), session_key="a").row
    claimed = store.claim(row.queue_id)

    assert store.retry(
        claimed.queue_id,
        claimed.claim_token,
        error="failed after trigger persistence",
        resume_only=True,
    )
    ready = store.get(row.queue_id)
    assert ready.state == "resume_ready"
    assert ready.resume_only is True

    resumed = store.claim_next()
    assert resumed.resume_only is True
    assert resumed.attempts == 2


def test_repeated_resume_crashes_consume_attempt_budget(tmp_path, monkeypatch):
    store = _store(tmp_path, max_attempts=2)
    row = store.enqueue(_event("resume-poison"), session_key="a").row
    store.bind(row.queue_id, "db-a", "trigger-a")
    first = store.claim(row.queue_id)
    assert first.attempts == 1
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    store.reclaim_dead_claims(lambda *_: True)
    second = store.claim_next()
    assert second.resume_only is True
    assert second.attempts == 2

    recovered = store.reclaim_dead_claims(lambda *_: True)
    assert [(item.queue_id, item.state) for item in recovered] == [
        (row.queue_id, "dead_letter")
    ]


def test_reclaim_only_touches_currently_deliverable_platforms(tmp_path, monkeypatch):
    store = _store(tmp_path)
    discord = store.enqueue(_event("discord"), session_key="discord").row
    slack = store.enqueue(
        MessageEvent(
            text="slack",
            source=SessionSource(Platform.SLACK, "slack-chat"),
            message_id="slack",
        ),
        session_key="slack",
    ).row
    store.bind(discord.queue_id, "db-discord", "discord")
    store.bind(slack.queue_id, "db-slack", "slack")
    assert store.claim(discord.queue_id) is not None
    assert store.claim(slack.queue_id) is not None
    monkeypatch.setattr("gateway.inbound_queue._owner_alive", lambda *_: False)

    reclaimed = store.reclaim_dead_claims(
        lambda *_: False,
        deliverable_platforms={Platform.DISCORD},
    )

    assert [item.queue_id for item in reclaimed] == [discord.queue_id]
    assert store.get(discord.queue_id).state == "queued"
    assert store.get(slack.queue_id).state == "claimed"


def test_queue_id_conflict_is_not_reported_as_message_dedupe(tmp_path):
    store = _store(tmp_path)
    first = store.enqueue(_event("first-id"), session_key="a", queue_id="fixed")
    duplicate = store.enqueue(_event("first-id"), session_key="a", queue_id="fixed")
    conflict = store.enqueue(_event("different-id"), session_key="b", queue_id="fixed")

    assert first.accepted and first.inserted
    assert duplicate.accepted and not duplicate.inserted
    assert duplicate.status == "duplicate"
    assert conflict.accepted is False
    assert conflict.status == "queue_id_conflict"
    assert conflict.row is None
    assert store.pending_count() == 1


def test_prune_expires_terminal_rows_and_removes_orphaned_lanes(tmp_path):
    store = _store(
        tmp_path,
        terminal_retention_seconds=10,
        prune_interval_seconds=3_600,
    )
    completed = store.enqueue(_event("completed"), session_key="a").row
    cancelled = store.enqueue(_event("cancelled"), session_key="b").row
    claim = store.claim(completed.queue_id)
    assert store.complete(completed.queue_id, claim.claim_token)
    assert store.cancel(cancelled.queue_id)
    with sqlite3.connect(store.path()) as conn:
        conn.execute(
            "UPDATE inbound_events SET updated_at=100 WHERE queue_id IN (?, ?)",
            (completed.queue_id, cancelled.queue_id),
        )

    assert store.prune(now=111) == 2
    with sqlite3.connect(store.path()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inbound_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM inbox_lanes").fetchone()[0] == 0


def test_total_row_hard_cap_rejects_or_prunes_before_insert(tmp_path):
    store = _store(
        tmp_path,
        max_pending_total=10,
        max_rows=3,
        terminal_retention_seconds=86_400,
        prune_interval_seconds=3_600,
    )
    rows = [
        store.enqueue(_event(f"active-{index}"), session_key=f"s-{index}").row
        for index in range(3)
    ]

    full = store.enqueue(_event("no-room"), session_key="s-full")
    assert full.accepted is False
    assert full.status == "storage_full"

    claim = store.claim(rows[0].queue_id)
    assert store.complete(claim.queue_id, claim.claim_token)
    inserted = store.enqueue(_event("after-terminal"), session_key="s-new")
    assert inserted.accepted and inserted.inserted
    with sqlite3.connect(store.path()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inbound_events").fetchone()[0] == 3
    assert store.get(rows[0].queue_id) is None


def test_corrupt_database_is_quarantined_and_recreated(tmp_path):
    db_dir = tmp_path / "gateway"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "gateway-inbox.db"
    db_path.write_bytes(b"this is not sqlite")

    store = _store(tmp_path)

    quarantined = list(db_dir.glob("gateway-inbox.db.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"this is not sqlite"
    assert store.enqueue(_event("after-corruption"), session_key="a").accepted


def test_corrupt_payload_is_dead_lettered_without_blocking_next_row(tmp_path):
    store = _store(tmp_path)
    broken = store.enqueue(_event("broken"), session_key="a").row
    healthy = store.enqueue(_event("healthy"), session_key="b").row
    with sqlite3.connect(store.path()) as conn:
        conn.execute(
            "UPDATE inbound_events SET payload_json='not-json' WHERE queue_id=?",
            (broken.queue_id,),
        )

    claimed = store.claim_next()

    assert claimed.queue_id == healthy.queue_id
    with sqlite3.connect(store.path()) as conn:
        state, error = conn.execute(
            "SELECT state, last_error FROM inbound_events WHERE queue_id=?",
            (broken.queue_id,),
        ).fetchone()
    assert state == "dead_letter"
    assert "payload" in error.lower()


def test_sixty_four_concurrent_writers_do_not_lock_each_other(tmp_path):
    stores = [_store(tmp_path, max_pending_total=128) for _ in range(8)]

    def _write(index: int):
        return stores[index % len(stores)].enqueue(
            _event(f"message-{index}"),
            session_key=f"session-{index}",
        )

    with ThreadPoolExecutor(max_workers=64) as pool:
        results = list(pool.map(_write, range(64)))

    assert all(result.accepted and result.inserted for result in results)
    assert stores[0].pending_count() == 64
