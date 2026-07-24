import time

import gateway.cron_reply_context as crc


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


def test_falls_back_to_latest_chat_context(tmp_path, monkeypatch):
    store = tmp_path / "cron_reply_contexts.json"
    monkeypatch.setattr(crc, "_STORE_PATH", store)

    crc.record_cron_reply_context("teams", "conv-1", "Old", thread_id="old")
    records = crc._load_records()
    records["teams::conv-1::old"]["updated_at"] = time.time() - 60
    records["teams::conv-1::new"] = {
        "platform": "teams",
        "chat_id": "conv-1",
        "thread_id": "new",
        "message_id": "new",
        "job_id": "job-new",
        "content": "New",
        "updated_at": time.time(),
    }
    crc._write_records(records)

    record = crc.find_cron_reply_context("teams", "conv-1")

    assert record is not None
    assert record["content"] == "New"


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

    assert crc.find_cron_reply_context("teams", "conv-1", max_age_seconds=60) is None
