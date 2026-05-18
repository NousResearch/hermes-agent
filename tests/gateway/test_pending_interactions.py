from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.pending_interactions import (
    STATUS_AMBIGUOUS,
    STATUS_EXPIRED,
    STATUS_OPEN,
    STATUS_RESOLVED,
    load_pending_interactions,
    pending_store_path,
    record_pending_interaction,
    resolve_pending_reply,
)
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


REQUIRED_SCHEMA_FIELDS = {
    "id",
    "origin_profile",
    "platform",
    "channel_id",
    "thread_id",
    "user_id",
    "source_session_id",
    "job_id",
    "question_summary",
    "expected_reply_shape",
    "artifact_paths",
    "created_at",
    "expires_at",
    "status",
}


def test_pending_interaction_schema_is_profile_local(tmp_path, monkeypatch):
    profile_home = tmp_path / ".hermes" / "profiles" / "forge"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    record = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        thread_id="thread-1",
        user_id="user-1",
        source_session_id="session-1",
        question_summary="Which artifact should I continue?",
        expected_reply_shape="artifact id or instruction",
        artifact_paths=["/tmp/result.md"],
    )

    assert pending_store_path() == profile_home / "pending_interactions" / "records.json"
    assert REQUIRED_SCHEMA_FIELDS.issubset(record)
    assert record["origin_profile"] == "forge"
    assert record["user_id"] == "user-1"
    assert record["status"] == STATUS_OPEN
    assert load_pending_interactions()[0]["id"] == record["id"]


@pytest.mark.asyncio
async def test_gateway_prepares_korean_goal_continuation_reply_as_visible_handoff():
    record_pending_interaction(
        platform="discord",
        channel_id="parent-channel",
        thread_id="thread-42",
        user_id="user-1",
        source_session_id="goal-session",
        question_summary="Goal is blocked. Should I continue with the saved plan?",
        expected_reply_shape="confirmation or instruction",
        artifact_paths=["/tmp/goal-plan.md"],
    )
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    runner.session_store = None
    runner._pending_native_image_paths_by_session = {}
    runner.adapters = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_type="thread",
        user_id="user-1",
        thread_id="thread-42",
        parent_chat_id="parent-channel",
    )
    event = MessageEvent(text="그거 계속해줘", source=source)

    prepared = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert "[Pending interaction handoff]" in prepared
    assert "Goal is blocked" in prepared
    assert "그거 계속해줘" in prepared
    assert "Do not infer a different task from runtime recall" in prepared
    assert load_pending_interactions()[0]["status"] == STATUS_RESOLVED


def test_ambiguous_korean_reply_with_multiple_pending_interactions_stays_unresolved():
    first = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        question_summary="Continue cron report A?",
        expected_reply_shape="confirmation",
    )
    second = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        question_summary="Continue goal task B?",
        expected_reply_shape="confirmation",
    )

    result = resolve_pending_reply(
        platform="discord",
        channel_id="channel-1",
        reply_text="그거 계속해줘",
    )

    assert result.status == STATUS_AMBIGUOUS
    assert first["id"] in result.message
    assert second["id"] in result.message
    statuses = {record["id"]: record["status"] for record in load_pending_interactions()}
    assert statuses[first["id"]] == STATUS_OPEN
    assert statuses[second["id"]] == STATUS_OPEN


def test_expired_pending_interaction_is_not_resolved():
    old = datetime.now(timezone.utc) - timedelta(days=2)
    record = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        source_session_id="session-old",
        question_summary="Old question?",
        expected_reply_shape="free-form reply",
        created_at=old,
        ttl_seconds=1,
    )

    result = resolve_pending_reply(
        platform="discord",
        channel_id="channel-1",
        reply_text="continue",
        now=datetime.now(timezone.utc),
    )

    assert result.status == STATUS_EXPIRED
    stored = {item["id"]: item for item in load_pending_interactions()}
    assert stored[record["id"]]["status"] == STATUS_EXPIRED


def test_visible_pending_interaction_overrides_stale_runtime_recall_hint():
    record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        source_session_id="session-current",
        question_summary="Should I continue the visible Discord handoff?",
        expected_reply_shape="confirmation",
    )

    result = resolve_pending_reply(
        platform="discord",
        channel_id="channel-1",
        reply_text="그거 계속해줘",
    )

    assert result.status == STATUS_RESOLVED
    assert "visible pending interaction as the primary context" in result.message
    assert "runtime recall" in result.message


def test_pending_interaction_does_not_resolve_for_different_user():
    record = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        user_id="user-1",
        source_session_id="session-current",
        question_summary="Should I continue the visible Discord handoff?",
        expected_reply_shape="confirmation",
    )

    result = resolve_pending_reply(
        platform="discord",
        channel_id="channel-1",
        user_id="user-2",
        reply_text="그거 계속해줘",
    )

    assert result.status == "none"
    assert load_pending_interactions()[0]["status"] == STATUS_OPEN
    assert load_pending_interactions()[0]["id"] == record["id"]


def test_pending_interaction_does_not_consume_unrelated_channel_chatter():
    record = record_pending_interaction(
        platform="discord",
        channel_id="channel-1",
        user_id="user-1",
        source_session_id="session-current",
        question_summary="Should I continue the visible Discord handoff?",
        expected_reply_shape="confirmation",
    )

    result = resolve_pending_reply(
        platform="discord",
        channel_id="channel-1",
        user_id="user-1",
        reply_text="오늘 날씨는 어때",
    )

    assert result.status == "none"
    assert load_pending_interactions()[0]["status"] == STATUS_OPEN
    assert load_pending_interactions()[0]["id"] == record["id"]
