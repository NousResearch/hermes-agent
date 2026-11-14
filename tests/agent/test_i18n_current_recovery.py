"""Localized recovery retains the structured delivery markers introduced upstream."""

from agent import i18n
from agent.context_engine import CompactionProgressText, automatic_compaction_status_message
from agent.conversation_compression import COMPACTION_STATUS
from agent.stream_max_lifetime import StreamLifetimeExceeded
from agent.turn_failure_copy import exhausted_copy


def test_localized_compaction_keeps_its_progress_marker(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")
    i18n.reset_language_cache()
    message = automatic_compaction_status_message(object(), phase="start", default_message=COMPACTION_STATUS)
    assert isinstance(message, CompactionProgressText)
    assert message == COMPACTION_STATUS
    assert i18n.render_localized(message) != str(message)


def test_stream_lifetime_failure_is_localized_only_for_display(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")
    i18n.reset_language_cache()
    fields = dict(label="OpenAI", attempts=3, summary="timeout", error=StreamLifetimeExceeded(300, 300))
    stored = exhausted_copy("timeout", **fields)
    shown = exhausted_copy("timeout", lang=None, **fields)
    assert stored.startswith("The response was still streaming when agent.stream_max_lifetime (300s)")
    assert "На всех 3 попытках" in shown and "agent.stream_max_lifetime" in shown
