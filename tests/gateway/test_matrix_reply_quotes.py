"""Matrix reply anchor behavior."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import _reply_anchor_for_event


def _event(chat_type: str = "group") -> SimpleNamespace:
    return SimpleNamespace(
        message_id="mx-123",
        source=SimpleNamespace(platform=Platform.MATRIX, chat_type=chat_type),
    )


def test_matrix_group_replies_quote_by_default(monkeypatch):
    monkeypatch.delenv("MATRIX_QUOTE_REPLIES", raising=False)

    assert _reply_anchor_for_event(_event()) == "mx-123"


def test_matrix_group_reply_quotes_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MATRIX_QUOTE_REPLIES", "false")

    assert _reply_anchor_for_event(_event()) is None


def test_matrix_dm_replies_also_suppressed_when_quotes_disabled(monkeypatch):
    monkeypatch.setenv("MATRIX_QUOTE_REPLIES", "false")

    assert _reply_anchor_for_event(_event("dm")) is None
