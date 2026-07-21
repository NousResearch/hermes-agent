"""Regression tests for #68716 — Telegram ``allow_from`` short-circuit.

Previously, the adapter-level ``allow_from`` was treated as the *sole*
authority by ``_is_user_authorized_from_message`` for group messages.
A sender excluded from the global allowlist was rejected even when they
were explicitly included in ``group_allow_from`` or the chat was in
``group_allowed_chats``.

This contradicted the documented orthogonal authorization semantics
(restricted DMs + open groups).

Fix scope: the new deferral only activates when BOTH a group-scope config
(``group_allow_from`` or ``group_allowed_chats``) is present AND the
message is in a group/forum/supergroup. Plain DMs and configs without
group scope keep the original short-circuit behaviour, so the existing
``allow_from=["222"]`` style configs are not affected.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(allow_from=None, group_allow_from=None, group_allowed_chats=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


def _make_message(*, from_user_id=222, chat_id=-100, chat_type="group"):
    return SimpleNamespace(
        message_id=42,
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title="Test",
            is_forum=False,
        ),
        from_user=SimpleNamespace(id=from_user_id, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


# ── DM path — unchanged behaviour ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_dm_with_global_allowlist_still_short_circuits():
    """DM sender outside global allowlist is rejected — unchanged behaviour."""
    adapter = _make_adapter(allow_from=["111"])
    msg = _make_message(from_user_id=222, chat_type="private", chat_id=12345)
    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.asyncio
async def test_dm_with_global_allowlist_member_passes():
    """DM sender inside global allowlist is accepted — unchanged behaviour."""
    adapter = _make_adapter(allow_from=["222"])
    msg = _make_message(from_user_id=222, chat_type="private", chat_id=12345)
    assert adapter._is_user_authorized_from_message(msg) is True


# ── Group path without group-scope config — unchanged behaviour ────────────


def test_group_sender_not_in_allow_from_still_blocked_without_group_scope():
    """Without group_allow_from / group_allowed_chats, the original short-circuit holds."""
    adapter = _make_adapter(allow_from=["111"])
    msg = _make_message(from_user_id=222, chat_id=-100, chat_type="group")
    assert adapter._is_user_authorized_from_message(msg) is False


def test_group_sender_in_allow_from_passes_without_group_scope():
    """Without group scope config, in-allowlist group sender still passes."""
    adapter = _make_adapter(allow_from=["222"])
    msg = _make_message(from_user_id=222, chat_id=-100, chat_type="group")
    assert adapter._is_user_authorized_from_message(msg) is True


# ── Group path WITH group-scope config — new deferral (#68716) ─────────────


def test_group_sender_in_group_allow_from_defers_to_runner(monkeypatch):
    """Group sender in group_allow_from but NOT in allow_from defers to the runner path."""
    adapter = _make_adapter(
        allow_from=["111"],  # sender 222 NOT here
        group_allow_from=["222"],  # sender 222 IS here
    )

    runner_called = {"flag": False}

    class _StubRunner:
        def _is_user_authorized(self, source):
            runner_called["flag"] = True
            return True

    adapter._message_handler = SimpleNamespace(__self__=_StubRunner())
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    msg = _make_message(from_user_id=222, chat_id=-100, chat_type="group")
    result = adapter._is_user_authorized_from_message(msg)

    assert result is True, "runner should have authorized the sender"
    assert runner_called["flag"] is True, "runner auth_fn must be reached"


def test_group_chat_in_group_allowed_chats_defers_to_runner(monkeypatch):
    """Chat in group_allowed_chats defers to runner even if sender not in allow_from."""
    adapter = _make_adapter(
        allow_from=["111"],
        group_allowed_chats=["-100"],
    )

    runner_called = {"flag": False}

    class _StubRunner:
        def _is_user_authorized(self, source):
            runner_called["flag"] = True
            return True

    adapter._message_handler = SimpleNamespace(__self__=_StubRunner())
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    msg = _make_message(from_user_id=222, chat_id=-100, chat_type="group")
    result = adapter._is_user_authorized_from_message(msg)

    assert result is True
    assert runner_called["flag"] is True, "group_allowed_chats must reach runner"


def test_group_sender_in_global_allowlist_short_circuits_with_group_scope(monkeypatch):
    """Group sender in global allowlist short-circuits to True even with group-scope config."""
    adapter = _make_adapter(
        allow_from=["222"],
        group_allow_from=["999"],
        group_allowed_chats=["-100"],
    )

    runner_called = {"flag": False}

    class _StubRunner:
        def _is_user_authorized(self, source):
            runner_called["flag"] = True
            return True

    adapter._message_handler = SimpleNamespace(__self__=_StubRunner())
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    msg = _make_message(from_user_id=222, chat_id=-100, chat_type="group")
    result = adapter._is_user_authorized_from_message(msg)

    assert result is True
    assert runner_called["flag"] is False, "global allowlist member should not invoke runner"