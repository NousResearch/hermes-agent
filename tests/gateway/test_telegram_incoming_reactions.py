"""Tests for incoming Telegram emoji reactions as user feedback signals.

These tests exercise TelegramAdapter._handle_reaction(), which processes
MessageReaction updates from Telegram and writes structured feedback to
~/.hermes/feedback.jsonl and (for certain reaction types) to
~/.hermes/memory/feedback-log.md.
"""

import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    return adapter


def _make_reaction_update(emoji: str | None, *, chat_id: int = 999, message_id: int = 42):
    """Build a minimal fake telegram Update with a message_reaction payload."""
    chat = SimpleNamespace(id=chat_id)
    if emoji is None:
        reaction_obj = None
    else:
        reaction_obj = SimpleNamespace(emoji=emoji)

    new_reactions = [] if emoji is None else [reaction_obj]

    message_reaction = SimpleNamespace(
        chat=chat,
        message_id=message_id,
        new_reaction=new_reactions,
    )
    update = MagicMock()
    update.message_reaction = message_reaction
    return update


def _make_empty_reaction_update(*, chat_id: int = 999, message_id: int = 42):
    """Build a reaction update with an empty new_reaction list (reaction removed)."""
    chat = SimpleNamespace(id=chat_id)
    message_reaction = SimpleNamespace(
        chat=chat,
        message_id=message_id,
        new_reaction=[],
    )
    update = MagicMock()
    update.message_reaction = message_reaction
    return update


def _make_none_reaction_update():
    """Build an update where message_reaction is None."""
    update = MagicMock()
    update.message_reaction = None
    return update


# ── Tests: feedback.jsonl logging ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thumbs_up_logs_positive_feedback(tmp_path, monkeypatch):
    """👍 reaction should write a 'positive' entry to feedback.jsonl."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f44d")

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    assert feedback_file.exists()
    entry = json.loads(feedback_file.read_text().strip())
    assert entry["emoji"] == "\U0001f44d"
    assert entry["feedback_type"] == "positive"
    assert entry["chat_id"] == "999"
    assert entry["message_id"] == 42
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_thumbs_down_logs_negative_feedback(tmp_path, monkeypatch):
    """👎 reaction should write a 'negative' entry to feedback.jsonl."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f44e")

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    entry = json.loads(feedback_file.read_text().strip())
    assert entry["feedback_type"] == "negative"


@pytest.mark.asyncio
async def test_heart_logs_save_feedback(tmp_path, monkeypatch):
    """❤️ reaction should write a 'save' entry to feedback.jsonl."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U00002764")

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    entry = json.loads(feedback_file.read_text().strip())
    assert entry["feedback_type"] == "save"
    assert entry["emoji"] == "\U00002764"


@pytest.mark.asyncio
async def test_fire_logs_strong_positive_feedback(tmp_path, monkeypatch):
    """🔥 reaction should write a 'strong_positive' entry to feedback.jsonl."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f525")

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    entry = json.loads(feedback_file.read_text().strip())
    assert entry["feedback_type"] == "strong_positive"
    assert entry["emoji"] == "\U0001f525"


# ── Tests: memory/feedback-log.md ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thumbs_down_writes_to_memory_log(tmp_path, monkeypatch):
    """👎 reaction should write a note to memory/feedback-log.md."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f44e", message_id=77, chat_id=555)

    await adapter._handle_reaction(update, MagicMock())

    log_file = tmp_path / ".hermes" / "memory" / "feedback-log.md"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Negative feedback" in content
    assert "77" in content   # message_id
    assert "555" in content  # chat_id


@pytest.mark.asyncio
async def test_heart_writes_to_memory_log(tmp_path, monkeypatch):
    """❤️ reaction should write a saved-response note to memory/feedback-log.md."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U00002764", message_id=88, chat_id=666)

    await adapter._handle_reaction(update, MagicMock())

    log_file = tmp_path / ".hermes" / "memory" / "feedback-log.md"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Saved response" in content
    assert "88" in content
    assert "666" in content


@pytest.mark.asyncio
async def test_fire_writes_to_memory_log(tmp_path, monkeypatch):
    """🔥 reaction should write a saved-response note to memory/feedback-log.md."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f525", message_id=99, chat_id=777)

    await adapter._handle_reaction(update, MagicMock())

    log_file = tmp_path / ".hermes" / "memory" / "feedback-log.md"
    assert log_file.exists()
    content = log_file.read_text()
    assert "Saved response" in content


@pytest.mark.asyncio
async def test_thumbs_up_does_not_write_to_memory_log(tmp_path, monkeypatch):
    """👍 (positive) reaction should NOT write to memory/feedback-log.md."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f44d")

    await adapter._handle_reaction(update, MagicMock())

    log_file = tmp_path / ".hermes" / "memory" / "feedback-log.md"
    assert not log_file.exists()


# ── Tests: edge cases ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_emoji_is_ignored(tmp_path, monkeypatch):
    """An unsupported emoji reaction should produce no output files."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_reaction_update("\U0001f600")  # 😀 — not in FEEDBACK_MAP

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    assert not feedback_file.exists()


@pytest.mark.asyncio
async def test_empty_new_reaction_list_is_ignored(tmp_path, monkeypatch):
    """A reaction-removal event (empty new_reaction list) should do nothing."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_empty_reaction_update()

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    assert not feedback_file.exists()


@pytest.mark.asyncio
async def test_none_message_reaction_is_ignored(tmp_path, monkeypatch):
    """An update with message_reaction=None should return immediately."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()
    update = _make_none_reaction_update()

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    assert not feedback_file.exists()


@pytest.mark.asyncio
async def test_multiple_reactions_all_logged(tmp_path, monkeypatch):
    """When multiple reactions are added at once, each should be logged."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    adapter = _make_adapter()

    # Build an update with two reactions at once
    chat = SimpleNamespace(id=123)
    new_reactions = [
        SimpleNamespace(emoji="\U0001f44d"),
        SimpleNamespace(emoji="\U0001f525"),
    ]
    message_reaction = SimpleNamespace(chat=chat, message_id=10, new_reaction=new_reactions)
    update = MagicMock()
    update.message_reaction = message_reaction

    await adapter._handle_reaction(update, MagicMock())

    feedback_file = tmp_path / ".hermes" / "feedback.jsonl"
    lines = feedback_file.read_text().strip().splitlines()
    assert len(lines) == 2
    types = {json.loads(l)["feedback_type"] for l in lines}
    assert types == {"positive", "strong_positive"}
