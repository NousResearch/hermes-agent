from datetime import datetime, timezone

from agent.slack_work_items import (
    SlackSourceRef,
    append_work_item,
    find_work_item,
    latest_work_items,
    load_work_items,
    make_work_id,
    parse_slack_source_ref,
    update_work_item,
)


def test_parse_thread_permalink_with_thread_ts():
    ref = parse_slack_source_ref(
        "<https://ffxivkr.slack.com/archives/C0B7QVCLQF9/p1783498191737439?thread_ts=1783489727.804269&cid=C0B7QVCLQF9|link>"
    )
    assert ref == SlackSourceRef(
        source_type="thread",
        permalink="https://ffxivkr.slack.com/archives/C0B7QVCLQF9/p1783498191737439?thread_ts=1783489727.804269&cid=C0B7QVCLQF9",
        channel_id="C0B7QVCLQF9",
        message_ts="1783498191.737439",
        thread_ts="1783489727.804269",
        file_id=None,
    )


def test_parse_thread_permalink_without_thread_ts_uses_message_ts():
    ref = parse_slack_source_ref(
        "https://ffxivkr.slack.com/archives/C0B7QVCLQF9/p1783498191737439"
    )
    assert ref is not None
    assert ref.channel_id == "C0B7QVCLQF9"
    assert ref.message_ts == "1783498191.737439"
    assert ref.thread_ts == "1783498191.737439"


def test_parse_bare_file_id_as_file_ref():
    ref = parse_slack_source_ref("F0AC9G3PYE9")
    assert ref == SlackSourceRef(
        source_type="file",
        permalink="",
        channel_id=None,
        message_ts=None,
        thread_ts=None,
        file_id="F0AC9G3PYE9",
    )


def test_work_item_append_lookup_and_update(tmp_path):
    path = tmp_path / "slack_work_items.jsonl"
    ref = parse_slack_source_ref(
        "https://ffxivkr.slack.com/archives/C0B7QVCLQF9/p1783498191737439?thread_ts=1783489727.804269&cid=C0B7QVCLQF9"
    )
    assert ref is not None
    work_id = make_work_id(
        datetime(2026, 7, 8, tzinfo=timezone.utc),
        seed="C0B7QVCLQF9:1783489727.804269",
    )
    item = append_work_item(
        path,
        {
            "work_id": work_id,
            "source_type": ref.source_type,
            "source_permalink": ref.permalink,
            "channel_id": ref.channel_id,
            "thread_ts": ref.thread_ts,
            "message_ts": ref.message_ts,
            "file_id": ref.file_id,
            "requested_by_user_id": "U05NZ8WD30C",
            "request_text": "@캐트시 인제스트해",
            "intent": "ingest",
            "status": "received",
            "result_summary": "",
            "artifact_paths": [],
            "jarvis_handoff_id": None,
            "jarvis_notification_status": "read_from_jarvis_queue",
            "created_at": "2026-07-08T00:00:00+00:00",
            "updated_at": "2026-07-08T00:00:00+00:00",
        },
    )
    assert item["work_id"] == work_id
    items = load_work_items(path)
    assert find_work_item(items, ref)["work_id"] == work_id
    updated = update_work_item(
        path, work_id, {"status": "completed", "result_summary": "처리 완료"}
    )
    assert updated["status"] == "completed"
    records = load_work_items(path)
    assert len(records) == 2
    assert latest_work_items(records)[0]["result_summary"] == "처리 완료"


def test_find_work_item_prefers_canonical_thread_key_over_permalink(tmp_path):
    path = tmp_path / "slack_work_items.jsonl"
    append_work_item(
        path,
        {
            "work_id": "sw_test",
            "source_type": "thread",
            "source_permalink": "https://old.example/archives/C0B7QVCLQF9/p1783498191737439",
            "channel_id": "C0B7QVCLQF9",
            "thread_ts": "1783489727.804269",
            "message_ts": "1783498191.737439",
            "file_id": None,
            "status": "received",
        },
    )
    ref = parse_slack_source_ref(
        "https://ffxivkr.slack.com/archives/C0B7QVCLQF9/p1783498191737439?thread_ts=1783489727.804269&cid=C0B7QVCLQF9"
    )
    assert ref is not None
    assert find_work_item(load_work_items(path), ref)["work_id"] == "sw_test"
