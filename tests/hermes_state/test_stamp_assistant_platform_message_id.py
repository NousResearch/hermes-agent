"""Stamp outbound delivery id onto the latest assistant transcript row.

Used so phone proactive delivery and session history share one public id.
"""

from hermes_state import SessionDB


def _db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", "phone")
    return db


def test_stamps_latest_assistant_when_unset(tmp_path):
    db = _db(tmp_path)
    aid = db.append_message(session_id="s1", role="assistant", content="hello there")
    assert db.stamp_latest_assistant_platform_message_id("s1", "deadbeefcafe", content="hello there")
    rows = db.get_messages("s1")
    assistant = next(r for r in rows if r.get("role") == "assistant")
    assert assistant.get("platform_message_id") == "deadbeefcafe"
    assert assistant.get("id") == aid
    assert db.has_platform_message_id("s1", "deadbeefcafe")


def test_content_match_skips_older_assistant(tmp_path):
    db = _db(tmp_path)
    db.append_message(session_id="s1", role="assistant", content="old")
    new_id = db.append_message(session_id="s1", role="assistant", content="new final")
    assert db.stamp_latest_assistant_platform_message_id("s1", "aabbccddeeff", content="new final")
    stamped = [r for r in db.get_messages("s1") if r.get("platform_message_id") == "aabbccddeeff"]
    assert len(stamped) == 1
    assert stamped[0]["content"] == "new final"
    assert new_id == stamped[0]["id"]
    old = next(r for r in db.get_messages("s1") if r.get("content") == "old")
    assert not old.get("platform_message_id")


def test_does_not_overwrite_different_platform_id(tmp_path):
    db = _db(tmp_path)
    db.append_message(
        session_id="s1",
        role="assistant",
        content="locked",
        platform_message_id="already-set",
    )
    assert not db.stamp_latest_assistant_platform_message_id("s1", "other-id", content="locked")
    assert db.has_platform_message_id("s1", "already-set")
    assert not db.has_platform_message_id("s1", "other-id")


def test_idempotent_same_id(tmp_path):
    db = _db(tmp_path)
    db.append_message(session_id="s1", role="assistant", content="x")
    assert db.stamp_latest_assistant_platform_message_id("s1", "same-id", content="x")
    assert db.stamp_latest_assistant_platform_message_id("s1", "same-id", content="x")
    assert db.has_platform_message_id("s1", "same-id")
