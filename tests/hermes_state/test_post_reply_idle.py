"""Durable post-reply idle deadline and fenced in-place compaction."""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("s", source="gateway", session_key="chat")
    for i in range(4):
        store.append_message("s", role="user" if i % 2 == 0 else "assistant", content=str(i))
    yield store
    store.close()


def _contents(db):
    return [m["content"] for m in db.get_messages("s")]


def test_arm_claim_and_commit_survive_reopen(db):
    generation = db.invalidate_post_reply_idle("s")
    watermark = db.get_active_message_watermark("s")
    now = time.time()
    assert db.arm_post_reply_idle("s", "chat", now, expected_generation=generation)
    other = SessionDB(db.db_path)
    assert other.claim_due_post_reply_idle("worker", now=now - 1, lease_ttl=30) is None
    claim = other.claim_due_post_reply_idle("worker", now=now, lease_ttl=30)
    assert claim == ("s", generation, watermark)
    assert other.archive_and_compact("s", [{"role": "assistant", "content": "summary"}],
                                     idle_claim=(generation, watermark, "worker")) == 1
    assert _contents(db) == ["summary"]
    assert other.claim_due_post_reply_idle("worker", now=now + 1, lease_ttl=30) is None
    other.close()


def test_inbound_fences_claimed_worker_without_touching_transcript(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("old", now=100, lease_ttl=30)
    db.invalidate_post_reply_idle("s")
    with pytest.raises(ValueError, match="idle compaction fence"):
        db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                               idle_claim=(claim[1], claim[2], "old"))
    assert _contents(db) == ["0", "1", "2", "3"]


def test_unfenced_append_cannot_be_folded_into_stale_summary(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("old", now=100, lease_ttl=30)
    db.append_message("s", role="user", content="new inbound")
    with pytest.raises(ValueError, match="idle compaction fence"):
        db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                               idle_claim=(claim[1], claim[2], "old"))
    assert _contents(db)[-1] == "new inbound"


def test_stale_scheduled_generation_cannot_claim_new_reply(db):
    old = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=old)
    new = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=new)
    assert db.claim_due_post_reply_idle("w", session_id="s", expected_generation=old,
                                        now=100, lease_ttl=30) is None
    assert db.claim_due_post_reply_idle("w", session_id="s", expected_generation=new,
                                        now=100, lease_ttl=30) == ("s", new, 4)


def test_stale_reply_cannot_rearm_after_inbound(db):
    gen = db.invalidate_post_reply_idle("s")
    db.invalidate_post_reply_idle("s")
    assert not db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    assert db.claim_due_post_reply_idle("w", now=100, lease_ttl=30) is None


def test_expired_lease_can_be_reclaimed_but_old_holder_cannot_commit(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    old = db.claim_due_post_reply_idle("old", now=100, lease_ttl=2)
    assert db.claim_due_post_reply_idle("new", now=101, lease_ttl=10) is None
    assert db.claim_due_post_reply_idle("new", now=103, lease_ttl=10) == old
    with pytest.raises(ValueError, match="idle compaction fence"):
        db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                               idle_claim=(gen, old[2], "old"))


def test_idle_summary_prompt_and_messages_commit_together(db):
    db.update_system_prompt("s", "before")
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", time.time() - 1, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("worker", lease_ttl=30)
    db.archive_and_compact("s", [{"role": "assistant", "content": "summary"}],
                           idle_claim=(gen, claim[2], "worker"), idle_system_prompt="after")
    assert db.get_session("s")["system_prompt"] == "after"
    assert _contents(db) == ["summary"]


def test_buffered_inbound_fences_summary_before_durable_append(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", time.time() - 1, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("idle", lease_ttl=30)
    pending = [False]
    idle_claim = (gen, claim[2], "idle", lambda: pending[0])
    pending[0] = True  # adapter holds the event; no new DB row exists yet
    with pytest.raises(ValueError, match="idle compaction fence"):
        db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                               idle_claim=idle_claim)
    assert _contents(db) == ["0", "1", "2", "3"]


def test_active_turn_lease_prevents_idle_commit(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", time.time() - 1, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("idle", lease_ttl=30)
    assert db.try_acquire_session_turn_lease("s", "human-turn")
    try:
        with pytest.raises(ValueError, match="idle compaction fence"):
            db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                                   idle_claim=(gen, claim[2], "idle"))
        assert _contents(db) == ["0", "1", "2", "3"]
    finally:
        db.release_session_turn_lease("s", "human-turn")


def test_worker_renews_claim_during_slow_preparation(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    assert db.claim_due_post_reply_idle("worker", now=100, lease_ttl=1)
    assert db.renew_post_reply_idle("s", gen, "worker", now=100.5, lease_ttl=10)
    assert db.claim_due_post_reply_idle("other", now=102, lease_ttl=1) is None
    assert db.claim_due_post_reply_idle("other", now=111, lease_ttl=1)


def test_release_backoff_and_session_rotation(db):
    gen = db.invalidate_post_reply_idle("s")
    db.arm_post_reply_idle("s", "chat", 100, expected_generation=gen)
    claim = db.claim_due_post_reply_idle("w", now=100, lease_ttl=30)
    assert db.release_post_reply_idle("s", gen, "w", retry_at=130)
    assert db.claim_due_post_reply_idle("w", now=129, lease_ttl=30) is None
    assert db.claim_due_post_reply_idle("w", now=130, lease_ttl=30) == claim
    db.end_session("s", end_reason="reset")
    with pytest.raises(ValueError, match="idle compaction fence"):
        db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                               idle_claim=(gen, claim[2], "w"))
