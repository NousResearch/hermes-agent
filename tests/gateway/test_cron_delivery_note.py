"""Tests for GatewayRunner._build_cron_delivery_note — the push-side injection
that folds buffered cron deliveries into the next turn's system prompt."""

from types import SimpleNamespace
from unittest.mock import patch

from gateway.run import GatewayRunner


def _src(platform="telegram", chat_id="123"):
    return SimpleNamespace(platform=SimpleNamespace(value=platform), chat_id=chat_id)


def _note(source):
    # method doesn't use self, so a dummy self is fine
    return GatewayRunner._build_cron_delivery_note(None, source)


class TestBuildCronDeliveryNote:
    def test_none_when_no_entries(self):
        with patch("cron.pending_notices.drain", return_value=[]):
            assert _note(_src()) is None

    def test_drains_with_platform_value_and_chat_id(self):
        with patch("cron.pending_notices.drain", return_value=[]) as d:
            _note(_src(platform="telegram", chat_id="999"))
        d.assert_called_once_with("telegram", "999")

    def test_formats_entries_into_system_note(self):
        entries = [
            {"job_name": "PR Watch", "ts": "2026-06-01T16:00:00", "text": "found 3 issues"},
            {"job_name": "Nutrition", "ts": "2026-06-01T21:30:00", "text": "score 82"},
        ]
        with patch("cron.pending_notices.drain", return_value=entries):
            note = _note(_src())
        assert note is not None
        assert note.startswith("[System note:")
        assert note.rstrip().endswith("]")
        assert "PR Watch" in note and "found 3 issues" in note
        assert "Nutrition" in note and "score 82" in note
        assert "NOT in your message history" in note

    def test_long_text_truncated(self):
        entries = [{"job_name": "j", "ts": "", "text": "x" * 5000}]
        with patch("cron.pending_notices.drain", return_value=entries):
            note = _note(_src())
        assert "[…truncated]" in note
        assert len(note) < 5000

    def test_none_when_chat_id_missing(self):
        with patch("cron.pending_notices.drain", return_value=[]) as d:
            assert _note(_src(chat_id=None)) is None
        d.assert_not_called()

    def test_plain_string_platform_supported(self):
        src = SimpleNamespace(platform="sendblue", chat_id="123")
        with patch("cron.pending_notices.drain", return_value=[]) as d:
            _note(src)
        d.assert_called_once_with("sendblue", "123")
