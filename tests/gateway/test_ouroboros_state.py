"""Tests for the Ouroboros gateway-scoped recent-ID state store."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time

import pytest

from gateway.ouroboros_state import (
    OooRecentState,
    OooStateContext,
    OooStateStore,
    extract_ids,
)


def _ctx(
    *,
    guild_id: str | None = "guild-1",
    channel_id: str | None = "channel-1",
    thread_id: str | None = "thread-1",
    user_id: str | None = "user-1",
    profile: str = "default",
) -> OooStateContext:
    return OooStateContext(
        platform="discord",
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        user_id=user_id,
        profile=profile,
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    return parsed


def test_context_key_differs_between_two_discord_threads_same_user(tmp_path):
    store = OooStateStore(tmp_path / "state.json")

    first = store.context_key(_ctx(thread_id="thread-a"))
    second = store.context_key(_ctx(thread_id="thread-b"))

    assert first != second


def test_context_key_differs_between_two_users_same_thread(tmp_path):
    store = OooStateStore(tmp_path / "state.json")

    first = store.context_key(_ctx(user_id="user-a"))
    second = store.context_key(_ctx(user_id="user-b"))

    assert first != second


def test_context_key_differs_between_guild_and_dm_context(tmp_path):
    store = OooStateStore(tmp_path / "state.json")

    guild_key = store.context_key(_ctx(guild_id="guild-1", channel_id="channel-1", thread_id=None))
    dm_key = store.context_key(_ctx(guild_id=None, channel_id="dm-channel", thread_id=None))

    assert guild_key != dm_key


def test_context_key_differs_between_profiles(tmp_path):
    store = OooStateStore(tmp_path / "state.json")

    default_key = store.context_key(_ctx(profile="default"))
    other_key = store.context_key(_ctx(profile="coder"))

    assert default_key != other_key


def test_update_then_load_returns_stored_fields_and_iso_utc_updated_at(tmp_path):
    store = OooStateStore(tmp_path / "state.json")
    ctx = _ctx()

    updated = store.update(ctx, interview_session_id="interview-1", last_job_id="job-1")
    loaded = store.load(ctx)

    assert updated.interview_session_id == "interview-1"
    assert loaded.interview_session_id == "interview-1"
    assert loaded.last_job_id == "job-1"
    assert loaded.updated_at is not None
    _parse_utc(loaded.updated_at)


def test_update_merges_fields_without_deleting_previous_ids_and_changes_updated_at(tmp_path):
    store = OooStateStore(tmp_path / "state.json")
    ctx = _ctx()

    first = store.update(ctx, interview_session_id="interview-1", last_job_id="job-1")
    time.sleep(0.001)
    second = store.update(ctx, auto_session_id="auto-1")

    assert second.interview_session_id == "interview-1"
    assert second.last_job_id == "job-1"
    assert second.auto_session_id == "auto-1"
    assert first.updated_at is not None
    assert second.updated_at is not None
    assert second.updated_at != first.updated_at
    assert _parse_utc(second.updated_at) >= _parse_utc(first.updated_at)


def test_concurrent_updates_via_two_instances_preserve_both_fields(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    ctx = _ctx()
    first_store = OooStateStore(path)
    second_store = OooStateStore(path)
    read_barrier = threading.Barrier(2)
    original_read_store = OooStateStore._read_store

    def delayed_read_store(self):
        data = original_read_store(self)
        try:
            read_barrier.wait(timeout=0.25)
        except threading.BrokenBarrierError:
            pass
        return data

    monkeypatch.setattr(OooStateStore, "_read_store", delayed_read_store)

    errors: list[BaseException] = []

    def run_update(store: OooStateStore, field_name: str, value: str) -> None:
        try:
            store.update(ctx, **{field_name: value})
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below.
            errors.append(exc)

    threads = [
        threading.Thread(
            target=run_update,
            args=(first_store, "interview_session_id", "interview-1"),
        ),
        threading.Thread(
            target=run_update,
            args=(second_store, "last_job_id", "job-1"),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert [thread.is_alive() for thread in threads] == [False, False]
    assert errors == []

    monkeypatch.undo()
    loaded = OooStateStore(path).load(ctx)
    assert loaded.interview_session_id == "interview-1"
    assert loaded.last_job_id == "job-1"


def test_update_rejects_unknown_field(tmp_path):
    store = OooStateStore(tmp_path / "state.json")

    with pytest.raises(ValueError, match="not_a_state_field"):
        store.update(_ctx(), not_a_state_field="value")


def test_new_store_instance_sees_persisted_state(tmp_path):
    path = tmp_path / "state.json"
    ctx = _ctx()

    OooStateStore(path).update(ctx, last_lineage_id="lineage-1", last_seed_id="seed-1")
    loaded = OooStateStore(path).load(ctx)

    assert loaded.last_lineage_id == "lineage-1"
    assert loaded.last_seed_id == "seed-1"


def test_corrupt_json_recovers_without_crash_and_returns_empty_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")

    loaded = OooStateStore(path).load(_ctx())

    assert loaded == OooRecentState()
    quarantined = list(tmp_path.glob("state.json.corrupt.*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not valid json"


def test_extract_ids_handles_common_flat_payload():
    payload = {
        "session_id": "session-1",
        "job_id": "job-1",
        "execution_id": "execution-1",
        "auto_session_id": "auto-1",
        "lineage_id": "lineage-1",
        "seed_id": "seed-1",
    }

    assert extract_ids(payload) == {
        "last_session_id": "session-1",
        "last_job_id": "job-1",
        "last_execution_id": "execution-1",
        "auto_session_id": "auto-1",
        "last_lineage_id": "lineage-1",
        "last_seed_id": "seed-1",
    }


def test_extract_ids_handles_nested_result_dict():
    assert extract_ids({"result": {"job_id": "job-1"}}) == {"last_job_id": "job-1"}


def test_extract_ids_handles_nested_structured_content_dict():
    assert extract_ids({"structuredContent": {"session_id": "session-1"}}) == {
        "last_session_id": "session-1"
    }


def test_extract_ids_handles_result_json_string_dict():
    assert extract_ids({"result": '{"execution_id": "execution-1"}'}) == {
        "last_execution_id": "execution-1"
    }
