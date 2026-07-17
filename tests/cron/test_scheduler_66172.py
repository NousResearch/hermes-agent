"""Regression tests for #66172 — Discord continuable-cron thread seed uses
``chat_id = thread_id`` to match the inbound handler's session key.

Background:
    Discord threads ARE channels in their own right. The Discord inbound
    message handler (plugins/platforms/discord/adapter.py) builds
    ``SessionSource`` with ``chat_id = str(effective_channel.id)`` where
    ``effective_channel`` is the *thread* object — i.e. ``chat_id`` equals
    ``thread_id`` for thread messages.

    The cron thread-seed path (``_seed_cron_thread_session``) was building
    ``SessionSource`` with ``chat_id = parent_channel_id``, producing a
    session key of ``agent:main:discord:thread:<CHANNEL_ID>:<THREAD_ID>``.
    The user's first reply in-thread resolved to
    ``agent:main:discord:thread:<THREAD_ID>:<THREAD_ID>`` — an orphan
    session with no seeded brief (#66172).

    The fix seeds Discord thread sessions with ``chat_id = thread_id``,
    matching the inbound handler, so seed and reply resolve to the same row.
"""

import pytest
from unittest.mock import MagicMock, patch


def _seed(adapter, platform_name, parent_chat_id, thread_id, mirror_text="Daily brief"):
    """Invoke the private seeder and return (store_mock, mirror_mock) for asserts."""
    from cron.scheduler import _seed_cron_thread_session

    store = MagicMock()
    adapter._session_store = store
    with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
        _seed_cron_thread_session(
            {"id": "j1", "name": "daily"}, adapter, platform_name,
            parent_chat_id, thread_id, mirror_text,
            chat_name="Ops",
        )
    return store, mirror_mock


class TestSeedCronThreadSessionDiscordChatIdOverride:
    """#66172 — Discord thread seed must use thread_id as chat_id."""

    def test_discord_uses_thread_id_as_chat_id(self):
        """For Discord, the seeded SessionSource.chat_id equals thread_id,
        not the parent channel id — matching the inbound handler's key.
        """
        adapter = MagicMock()
        store, mirror = _seed(adapter, "discord", parent_chat_id="100", thread_id="9001")

        store.get_or_create_session.assert_called_once()
        seeded_source = store.get_or_create_session.call_args[0][0]
        assert seeded_source.platform.value == "discord"
        assert seeded_source.chat_type == "thread"
        assert seeded_source.thread_id == "9001"
        # The bug fix: chat_id must equal thread_id for Discord.
        assert seeded_source.chat_id == "9001", (
            "Discord thread seed must key chat_id by thread_id "
            f"(got {seeded_source.chat_id!r}, expected '9001') — #66172"
        )

    def test_discord_mirror_to_session_uses_thread_id_as_chat_id(self):
        """mirror_to_session must be called with the SAME chat_id the
        SessionSource was created with — otherwise _find_session_id by
        origin.chat_id would fail to resolve the row we just created.
        """
        adapter = MagicMock()
        store, mirror = _seed(adapter, "discord", parent_chat_id="100", thread_id="9001")

        mirror.assert_called_once()
        mirrored_chat_id = mirror.call_args.args[1]
        assert mirrored_chat_id == "9001", (
            "mirror_to_session for Discord must receive thread_id as chat_id "
            f"(got {mirrored_chat_id!r}) — #66172"
        )
        assert mirror.call_args.kwargs.get("thread_id") == "9001"

    def test_telegram_unchanged_uses_parent_channel_id(self):
        """Sanity: Telegram is NOT Discord — Telegram forum topics keep
        chat_id = parent group id even for thread messages, so the seed
        must keep using the parent channel id (no override).
        """
        adapter = MagicMock()
        store, mirror = _seed(adapter, "telegram", parent_chat_id="123", thread_id="9001")

        store.get_or_create_session.assert_called_once()
        seeded_source = store.get_or_create_session.call_args[0][0]
        assert seeded_source.platform.value == "telegram"
        assert seeded_source.chat_type == "thread"
        assert seeded_source.thread_id == "9001"
        assert seeded_source.chat_id == "123", (
            "Telegram thread seed must keep parent chat_id "
            f"(got {seeded_source.chat_id!r}, expected '123')"
        )
        mirrored_chat_id = mirror.call_args.args[1]
        assert mirrored_chat_id == "123"

    def test_slack_unchanged_uses_parent_channel_id(self):
        """Sanity: Slack is NOT Discord — Slack threaded replies keep
        chat_id = channel id, so the seed must keep the parent channel id.
        """
        adapter = MagicMock()
        store, mirror = _seed(adapter, "slack", parent_chat_id="C123", thread_id="1700000000.0")

        store.get_or_create_session.assert_called_once()
        seeded_source = store.get_or_create_session.call_args[0][0]
        assert seeded_source.platform.value == "slack"
        assert seeded_source.chat_type == "thread"
        assert seeded_source.thread_id == "1700000000.0"
        assert seeded_source.chat_id == "C123", (
            "Slack thread seed must keep parent chat_id "
            f"(got {seeded_source.chat_id!r}, expected 'C123')"
        )

    def test_discord_session_key_matches_inbound_handler_key(self):
        """End-to-end: build_session_key(seeded_source) == build_session_key(inbound_source).

        This is the property that #66172 actually requires — the seeded
        row must be reachable by the lookup that happens on the user's
        first reply. We use the real ``build_session_key`` to prove the
        two paths now agree for Discord.
        """
        adapter = MagicMock()
        store, _ = _seed(adapter, "discord", parent_chat_id="100", thread_id="9001")
        seeded_source = store.get_or_create_session.call_args[0][0]

        # Inbound handler builds source with chat_id = effective_channel.id
        # where effective_channel is the thread, so chat_id == thread_id.
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        inbound_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="9001",          # thread.id
            chat_type="thread",
            thread_id="9001",         # thread.id
            user_id="123456789",     # message.author.id
            user_name="replying-user",
        )

        seed_key = build_session_key(seeded_source)
        inbound_key = build_session_key(inbound_source)
        assert seed_key == inbound_key, (
            f"Seed key {seed_key!r} != inbound key {inbound_key!r} — "
            "user's first reply will not resolve to the seeded session (#66172)"
        )

    def test_telegram_session_key_still_matches_inbound(self):
        """Sanity: the fix must not change Telegram behavior. The seeded
        key for a Telegram forum topic must still match the key the inbound
        handler produces (chat_id = group id, thread_id = topic id).
        """
        adapter = MagicMock()
        store, _ = _seed(adapter, "telegram", parent_chat_id="-1001234567890", thread_id="42")
        seeded_source = store.get_or_create_session.call_args[0][0]

        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        inbound_source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234567890",
            chat_type="thread",
            thread_id="42",
            user_id="111",
            user_name="alice",
        )

        seed_key = build_session_key(seeded_source)
        inbound_key = build_session_key(inbound_source)
        assert seed_key == inbound_key, (
            f"Telegram seed key {seed_key!r} != inbound key {inbound_key!r} — "
            "the Discord fix accidentally broke Telegram forum topics"
        )

    def test_seed_idempotent_when_called_twice(self):
        """Two seeds for the same Discord thread produce two identical
        SessionSource rows (modulo the chat_id override). The bug fix
        must be deterministic.
        """
        adapter = MagicMock()
        store, _ = _seed(adapter, "discord", "100", "9001")
        first = store.get_or_create_session.call_args[0][0]

        store2, _ = _seed(adapter, "discord", "100", "9001")
        second = store2.get_or_create_session.call_args[0][0]

        assert first.chat_id == second.chat_id == "9001"
        assert first.thread_id == second.thread_id == "9001"

    def test_discord_no_session_store_still_calls_mirror(self):
        """Defensive: when adapter has no _session_store, we still call
        mirror_to_session with the correct chat_id (thread_id for Discord).
        This guards against accidental flow breakage.
        """
        adapter = MagicMock()
        adapter._session_store = None
        with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            from cron.scheduler import _seed_cron_thread_session
            _seed_cron_thread_session(
                {"id": "j1"}, adapter, "discord", "100", "9001", "Brief",
            )
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.args[1] == "9001"
