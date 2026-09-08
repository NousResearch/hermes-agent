import time

import gateway.cron_reply_context as crc


def test_profile_store_is_resolved_per_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(crc, "_STORE_PATH", None)
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(crc, "get_hermes_home", lambda: first)
    crc.record_reply_context("teams", "chat", "first profile", thread_id="one")
    monkeypatch.setattr(crc, "get_hermes_home", lambda: second)
    assert crc.find_reply_context("teams", "chat", thread_id="one") is None
    crc.record_reply_context("teams", "chat", "second profile", thread_id="one")
    monkeypatch.setattr(crc, "get_hermes_home", lambda: first)
    assert crc.find_reply_context("teams", "chat", thread_id="one")["content"] == "first profile"


def test_platform_adapter_api_records_and_finds_exact_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_reply_context(
        "teams",
        "19:channel@thread.tacv2",
        "TaskOps Decisions Needed",
        thread_id="root-message-1",
        message_id="root-message-1",
    )

    record = crc.find_reply_context(
        "teams",
        "19:channel@thread.tacv2;messageid=root-message-1",
        thread_id="root-message-1",
    )

    assert record is not None
    assert record["content"] == "TaskOps Decisions Needed"
    assert record["message_id"] == "root-message-1"
    assert record["job_id"] is None


def test_records_and_finds_exact_thread_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context(
        "teams",
        "conv-1",
        "Cron message body",
        thread_id="root-message-1",
        message_id="root-message-1",
        job_id="job-1",
    )

    record = crc.find_cron_reply_context(
        "teams",
        "conv-1",
        thread_id="root-message-1",
    )

    assert record is not None
    assert record["content"] == "Cron message body"
    assert record["job_id"] == "job-1"


def test_message_without_reply_id_does_not_borrow_latest_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context("teams", "conv-1", "Old", thread_id="old")
    assert crc.find_cron_reply_context("teams", "conv-1") is None


def test_explicit_unknown_thread_does_not_borrow_latest_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context(
        "teams",
        "conv-1",
        "Different thread",
        thread_id="other-root",
    )

    assert crc.find_cron_reply_context(
        "teams",
        "conv-1",
        thread_id="uncached-root",
    ) is None


def test_normalizes_teams_thread_conversation_ids(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context(
        "teams",
        "19:channel@thread.tacv2",
        "Cron context",
        thread_id="1780267076971",
    )

    record = crc.find_cron_reply_context(
        "teams",
        "19:channel@thread.tacv2;messageid=1780267076971",
        thread_id="1780267076971",
    )

    assert record is not None
    assert record["content"] == "Cron context"


def test_ignores_stale_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context("teams", "conv-1", "Old", thread_id="old")
    records = crc._load_records()
    records["teams::conv-1::old"]["updated_at"] = time.time() - 10_000
    crc._write_records(records)

    assert crc.find_cron_reply_context(
        "teams",
        "conv-1",
        thread_id="old",
        max_age_seconds=60,
    ) is None


def test_record_prunes_expired_contexts(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context("teams", "conv-1", "Old", thread_id="old")
    records = crc._load_records()
    records["teams::conv-1::old"]["updated_at"] = (
        time.time() - crc._DEFAULT_MAX_AGE_SECONDS - 1
    )
    crc._write_records(records)

    crc.record_cron_reply_context("teams", "conv-1", "New", thread_id="new")

    assert set(crc._load_records()) == {"teams::conv-1::new"}


def test_record_caps_context_store_to_newest_records(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)
    monkeypatch.setattr(crc, "_MAX_RECORDS", 3)

    for index in range(5):
        monkeypatch.setattr(crc.time, "time", lambda index=index: 1_000 + index)
        crc.record_cron_reply_context(
            "teams",
            "conv-1",
            f"Message {index}",
            thread_id=f"thread-{index}",
        )

    assert set(crc._load_records()) == {
        "teams::conv-1::thread-2",
        "teams::conv-1::thread-3",
        "teams::conv-1::thread-4",
    }
