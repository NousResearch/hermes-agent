"""Tests for cron/pending_notices.py — the push-side buffer for cron
session-awareness (record on delivery, drain on next interactive turn)."""

from cron.pending_notices import record, drain, _MAX_PER_KEY


class TestRecordDrain:
    def test_record_then_drain_roundtrip(self, tmp_path):
        assert record("telegram", "123", "PR Watch", "found 3 issues", base_dir=tmp_path)
        got = drain("telegram", "123", base_dir=tmp_path)
        assert len(got) == 1
        assert got[0]["job_name"] == "PR Watch"
        assert got[0]["text"] == "found 3 issues"

    def test_drain_clears_buffer(self, tmp_path):
        record("telegram", "123", "j", "x", base_dir=tmp_path)
        drain("telegram", "123", base_dir=tmp_path)
        assert drain("telegram", "123", base_dir=tmp_path) == []

    def test_multiple_entries_preserve_order(self, tmp_path):
        record("telegram", "123", "j1", "first", base_dir=tmp_path)
        record("telegram", "123", "j2", "second", base_dir=tmp_path)
        got = drain("telegram", "123", base_dir=tmp_path)
        assert [e["text"] for e in got] == ["first", "second"]

    def test_keys_are_isolated_by_platform_and_chat(self, tmp_path):
        record("telegram", "123", "j", "tg", base_dir=tmp_path)
        record("sendblue", "123", "j", "sb", base_dir=tmp_path)
        record("telegram", "999", "j", "other", base_dir=tmp_path)
        assert [e["text"] for e in drain("telegram", "123", base_dir=tmp_path)] == ["tg"]
        assert [e["text"] for e in drain("sendblue", "123", base_dir=tmp_path)] == ["sb"]
        assert [e["text"] for e in drain("telegram", "999", base_dir=tmp_path)] == ["other"]

    def test_platform_key_is_case_insensitive(self, tmp_path):
        record("Telegram", "123", "j", "x", base_dir=tmp_path)
        assert len(drain("telegram", "123", base_dir=tmp_path)) == 1

    def test_empty_text_not_recorded(self, tmp_path):
        assert record("telegram", "123", "j", "   ", base_dir=tmp_path) is False
        assert drain("telegram", "123", base_dir=tmp_path) == []

    def test_missing_chat_id_not_recorded(self, tmp_path):
        assert record("telegram", None, "j", "x", base_dir=tmp_path) is False
        assert record("telegram", "", "j", "x", base_dir=tmp_path) is False

    def test_entries_capped_at_max(self, tmp_path):
        for i in range(_MAX_PER_KEY + 5):
            record("telegram", "123", "j", f"n{i}", base_dir=tmp_path)
        got = drain("telegram", "123", base_dir=tmp_path)
        assert len(got) == _MAX_PER_KEY
        # oldest dropped, newest kept
        assert got[-1]["text"] == f"n{_MAX_PER_KEY + 4}"
        assert got[0]["text"] == "n5"

    def test_drain_unknown_key_is_empty(self, tmp_path):
        assert drain("telegram", "nope", base_dir=tmp_path) == []

    def test_thread_id_preserved(self, tmp_path):
        record("telegram", "123", "j", "x", thread_id="42", base_dir=tmp_path)
        got = drain("telegram", "123", base_dir=tmp_path)
        assert got[0]["thread_id"] == "42"
