import json

from cron import webui_notifications
from cron.webui_notifications import (
    append_notification,
    ensure_store,
    list_notifications,
    mark_read,
    notifications_file,
)


def test_append_creates_owner_only_store_and_normalizes_record(tmp_path):
    home = tmp_path / "profiles" / "newsletteros"
    record = append_notification(
        {
            "job_id": "abc123",
            "job_name": "Daily Brief",
            "title": "Cronjob Response: Daily Brief",
            "body": "hello",
            "media": [(tmp_path / "voice.ogg", True)],
        },
        home=home,
    )

    path = notifications_file(home)
    assert path.exists()
    assert record["id"].startswith("notif_")
    assert record["schema_version"] == 1
    assert record["source"] == "cron"
    assert record["profile"] == "newsletteros"
    assert record["read_at"] is None
    assert record["media"] == [{"path": str(tmp_path / "voice.ogg"), "is_voice": True}]
    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["id"] == record["id"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.with_suffix(path.suffix + ".lock").stat().st_mode & 0o777 == 0o600


def test_list_returns_newest_first_and_skips_malformed_lines(tmp_path):
    home = tmp_path / ".hermes"
    older = append_notification({"id": "older", "body": "old", "created_at": "2026-01-01T00:00:00Z"}, home=home)
    newer = append_notification({"id": "newer", "body": "new", "created_at": "2026-01-02T00:00:00Z"}, home=home)
    path = notifications_file(home)
    path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")

    records = list_notifications(home=home, limit=10)

    assert [r["id"] for r in records] == [newer["id"], older["id"]]


def test_mark_read_updates_one_record_only_and_preserves_invalid_lines(tmp_path):
    home = tmp_path / ".hermes"
    first = append_notification({"id": "first", "body": "one", "created_at": "2026-01-01T00:00:00Z"}, home=home)
    second = append_notification({"id": "second", "body": "two", "created_at": "2026-01-02T00:00:00Z"}, home=home)
    path = notifications_file(home)
    path.write_text("bad-json\n" + path.read_text(encoding="utf-8"), encoding="utf-8")

    updated = mark_read(first["id"], home=home)

    assert updated is not None
    assert updated["id"] == first["id"]
    assert updated["read_at"]
    raw = path.read_text(encoding="utf-8")
    assert "bad-json" in raw
    listed = {r["id"]: r for r in list_notifications(home=home, limit=10)}
    assert listed[first["id"]]["read_at"]
    assert listed[second["id"]]["read_at"] is None


def test_mark_read_missing_id_returns_none(tmp_path):
    home = tmp_path / ".hermes"
    ensure_store(home)
    assert mark_read("missing", home=home) is None


def test_append_bounds_large_body_and_media(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_notifications, "_MAX_BODY_CHARS", 8)
    monkeypatch.setattr(webui_notifications, "_MAX_MEDIA_ITEMS", 2)

    record = append_notification(
        {
            "body": "0123456789",
            "media": ["one", "two", "three"],
        },
        home=tmp_path,
    )

    assert record["body"].startswith("01234567")
    assert record["body"].endswith("[notification body truncated]")
    assert record["body_truncated"] is True
    assert [item["path"] for item in record["media"]] == ["one", "two"]


def test_append_retains_only_newest_bounded_records(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_notifications, "_MAX_RECORDS", 3)

    for idx in range(5):
        append_notification({"id": f"n-{idx}", "body": str(idx)}, home=tmp_path)

    records = list_notifications(home=tmp_path, limit=10)
    assert [row["id"] for row in records] == ["n-4", "n-3", "n-2"]
