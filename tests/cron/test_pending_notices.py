"""Tests for cron/pending_notices.py — the push-side buffer for cron
session-awareness (record on delivery, drain on next interactive turn)."""

from cron.pending_notices import (
    record,
    drain,
    mark_accepted,
    dismiss,
    new_notice_id,
    normalize_notify_mode,
    _MAX_PER_KEY,
)


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


class TestInjectGating:
    """Button mode buffers entries with inject=False until the user accepts.

    drain() returns only injectable entries and leaves the rest, so the
    gateway's system-prompt fold (run.py) stays mode-agnostic.
    """

    def test_record_assigns_id_and_defaults_inject_true(self, tmp_path):
        nid = record("telegram", "123", "j", "x", base_dir=tmp_path)
        assert isinstance(nid, str) and nid
        got = drain("telegram", "123", base_dir=tmp_path)
        assert got[0]["id"] == nid
        assert got[0]["inject"] is True

    def test_record_with_explicit_id(self, tmp_path):
        nid = record("telegram", "123", "j", "x", notice_id="abc123", base_dir=tmp_path)
        assert nid == "abc123"
        assert drain("telegram", "123", base_dir=tmp_path)[0]["id"] == "abc123"

    def test_inject_false_held_until_accepted(self, tmp_path):
        record("telegram", "123", "j", "held", notice_id="n1", inject=False, base_dir=tmp_path)
        # not injected while pending, but not lost
        assert drain("telegram", "123", base_dir=tmp_path) == []
        assert mark_accepted("telegram", "123", "n1", base_dir=tmp_path) is True
        got = drain("telegram", "123", base_dir=tmp_path)
        assert [e["text"] for e in got] == ["held"]

    def test_mark_accepted_unknown_id(self, tmp_path):
        record("telegram", "123", "j", "x", notice_id="n1", inject=False, base_dir=tmp_path)
        assert mark_accepted("telegram", "123", "nope", base_dir=tmp_path) is False

    def test_dismiss_removes_entry(self, tmp_path):
        record("telegram", "123", "j", "x", notice_id="n1", inject=False, base_dir=tmp_path)
        assert dismiss("telegram", "123", "n1", base_dir=tmp_path) is True
        assert drain("telegram", "123", base_dir=tmp_path) == []
        # gone for good
        assert mark_accepted("telegram", "123", "n1", base_dir=tmp_path) is False

    def test_dismiss_unknown_id(self, tmp_path):
        assert dismiss("telegram", "123", "nope", base_dir=tmp_path) is False

    def test_drain_returns_injectable_leaves_pending(self, tmp_path):
        record("telegram", "123", "j", "auto", notice_id="a", base_dir=tmp_path)
        record("telegram", "123", "j", "held", notice_id="b", inject=False, base_dir=tmp_path)
        assert [e["text"] for e in drain("telegram", "123", base_dir=tmp_path)] == ["auto"]
        # the pending one survived the drain and can still be accepted later
        assert mark_accepted("telegram", "123", "b", base_dir=tmp_path) is True
        assert [e["text"] for e in drain("telegram", "123", base_dir=tmp_path)] == ["held"]

    def test_legacy_entry_without_inject_is_drained(self, tmp_path):
        # entries written by the pre-button record() have no inject/id field
        import json
        from cron.pending_notices import _store_path
        p = _store_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"telegram:123": [{"ts": "t", "job_name": "j", "thread_id": None, "text": "old"}]}
        ))
        assert [e["text"] for e in drain("telegram", "123", base_dir=tmp_path)] == ["old"]

    def test_new_notice_id_unique_and_short(self):
        ids = {new_notice_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(isinstance(i, str) and 0 < len(i) <= 16 for i in ids)


class TestNotifyMode:
    """cron.notify_session normalizes to off / auto / button, preserving the
    legacy bool semantics (True == on == auto, False/None == off)."""

    def test_true_is_auto(self):
        assert normalize_notify_mode(True) == "auto"

    def test_false_is_off(self):
        assert normalize_notify_mode(False) == "off"

    def test_none_is_off(self):
        assert normalize_notify_mode(None) == "off"

    def test_button_aliases(self):
        assert normalize_notify_mode("button") == "button"
        assert normalize_notify_mode("buttons") == "button"

    def test_off_aliases(self):
        for v in ("off", "no", "false", "disabled", ""):
            assert normalize_notify_mode(v) == "off"

    def test_auto_aliases(self):
        for v in ("auto", "on", "yes", "true"):
            assert normalize_notify_mode(v) == "auto"

    def test_case_insensitive(self):
        assert normalize_notify_mode("Button") == "button"
        assert normalize_notify_mode("OFF") == "off"

    def test_unknown_present_value_defaults_to_auto(self):
        # a non-empty but unrecognized value stays on (matches the old
        # "any truthy config value enabled it" behavior)
        assert normalize_notify_mode("wat") == "auto"
