"""Regression tests for #81370: Telegram ingress fragments buffered in the
adapter's text/photo/media-group debounce slots must be invalidated at
every conversation boundary (/stop, /new, /reset, auto-reset) so they
can never merge into the next unrelated user turn.

Before the fix, only ``disconnect()`` cleared these slots. A forward or
media download that finished after a conversation reset kept its
fragments in the debounce slots, and the next unrelated user message
merged them (production log: "Merged 1 Telegram startup attachment(s)"
two minutes after a /stop).
"""

import asyncio

import pytest

from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    """Build a TelegramAdapter without running its full constructor side
    effects (config, network stack).  The buffers under test are plain
    dicts initialized in __init__, so a minimal object via __new__ plus
    manual init of the relevant attrs is enough for the unit-level
    regression tests below."""
    adapter = object.__new__(TelegramAdapter)
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._pending_messages = {}
    adapter._text_debounce = {}
    adapter._active_sessions = {}
    adapter._session_tasks = {}
    # ``build_session_key`` reads ``self.config.extra`` to derive the
    # ``group_sessions_per_user`` / ``thread_sessions_per_user`` flags.
    # Default both to True (matches the gateway default and is enough for
    # DM key derivation).
    class _Cfg:
        extra = {
            "group_sessions_per_user": True,
            "thread_sessions_per_user": True,
        }
    adapter.config = _Cfg()
    adapter._gateway_profile_name = None
    return adapter


class TestDiscardSessionIngress:
    def test_text_batch_for_session_is_dropped(self):
        adapter = _make_adapter()
        key = "telegram:chat:123"
        adapter._pending_text_batches[key] = "pending text"
        # A completed task must not be cancelled (no error); a pending one
        # would be cancelled by the flush. Use a sentinel completed task.
        adapter._pending_text_batch_tasks[key] = asyncio.Task(
            asyncio.sleep(0)
        )
        adapter._pending_text_batch_tasks[key].cancel()
        try:
            asyncio.get_running_loop().run_until_complete(
                asyncio.sleep(0)
            )
        except RuntimeError:
            pass  # no running loop in this context

        adapter._discard_session_ingress(key)

        assert key not in adapter._pending_text_batches
        assert key not in adapter._pending_text_batch_tasks

    def test_other_sessions_batches_survive(self):
        adapter = _make_adapter()
        mine = "telegram:chat:123"
        other = "telegram:chat:456"
        adapter._pending_text_batches[mine] = "mine"
        adapter._pending_text_batches[other] = "other"
        adapter._pending_text_batch_tasks[mine] = None
        adapter._pending_text_batch_tasks[other] = None

        adapter._discard_session_ingress(mine)

        # My batch is gone; the unrelated session's is untouched.
        assert mine not in adapter._pending_text_batches
        assert other in adapter._pending_text_batches
        assert adapter._pending_text_batches[other] == "other"

    def test_photo_burst_and_album_slots_for_session_are_dropped(self):
        adapter = _make_adapter()
        session_key = "telegram:chat:123"
        burst_key = f"{session_key}:photo-burst"
        album_key = f"{session_key}:album:media_grp_9"
        other_key = "telegram:chat:456:photo-burst"
        adapter._pending_photo_batches[burst_key] = "burst"
        adapter._pending_photo_batches[album_key] = "album"
        adapter._pending_photo_batches[other_key] = "other"
        adapter._pending_photo_batch_tasks[burst_key] = None
        adapter._pending_photo_batch_tasks[album_key] = None
        adapter._pending_photo_batch_tasks[other_key] = None

        adapter._discard_session_ingress(session_key)

        assert burst_key not in adapter._pending_photo_batches
        assert album_key not in adapter._pending_photo_batches
        # Other session's burst survives.
        assert other_key in adapter._pending_photo_batches

    def test_media_group_slots_for_session_are_dropped(self):
        adapter = _make_adapter()
        # Build real SessionSource objects so the adapter's media-group
        # branch has to call ``build_session_key``.  The earlier fixture
        # used a fabricated ``source.session_key`` attribute that masked
        # the dead-code defect — SessionSource has no such attribute, so
        # the cleanup branch never matched in production (#81370).
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        source_a = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        source_b = SessionSource(platform=Platform.TELEGRAM, chat_id="456")

        class _Event:
            def __init__(self, source):
                self.source = source

        adapter._media_group_events["grp_a"] = _Event(source_a)
        adapter._media_group_events["grp_b"] = _Event(source_b)
        adapter._media_group_tasks["grp_a"] = None
        adapter._media_group_tasks["grp_b"] = None

        # Compute the key the same way the gateway does so the test
        # mirrors the production flow end-to-end.
        session_key = build_session_key(
            source_a,
            group_sessions_per_user=True,
            thread_sessions_per_user=True,
        )
        adapter._discard_session_ingress(session_key)

        assert "grp_a" not in adapter._media_group_events
        # Unrelated session's media group survives.
        assert "grp_b" in adapter._media_group_events


class TestBaseHookFunnel:
    def test_base_discard_calls_adapter_override(self):
        """_discard_text_debounce must funnel into the adapter override so
        every boundary that clears the base debounce also clears Telegram's
        own slots."""
        adapter = _make_adapter()
        session_key = "telegram:chat:123"
        adapter._pending_text_batches[session_key] = "fragment"
        adapter._pending_text_batch_tasks[session_key] = None

        # _discard_text_debounce -> _discard_session_ingress (override)
        BasePlatformAdapter._discard_text_debounce(adapter, session_key)

        assert session_key not in adapter._pending_text_batches
