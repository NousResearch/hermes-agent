"""An observe-mode addressed turn keeps its trusted sender for actor-bound tools (#140059).

Telegram observe mode shares one sender-less session source so observed chatter and addressed
turns form one conversation. That must not lose the CURRENT message's author: the live event
carries the platform update's sender, and the gateway binds HERMES_SESSION_USER_ID / the turn
actor from it when (and only when) the shared source has no user identity of its own.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context,
    turn_actor_from_event,
)
from gateway.session_context import clear_session_vars, get_session_env, _VAR_MAP, _UNSET

from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={
        "require_mention": True,
        "allowed_chats": ["-100"],
        "group_allowed_chats": ["-100"],
        "allowed_topics": [],
        "observe_unmentioned_group_messages": True,
    })
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = None
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


def _group_message(text, *, from_user_id, from_user_name, entities=None):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=entities or [],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=-100, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name=from_user_name, is_bot=False),
        reply_to_message=None,
        date=None,
    )


def _addressed_event(adapter, from_user_id=222, from_user_name="Bob Example"):
    text = "@hermes_bot summarize the thread"
    offset = text.index("@hermes_bot")
    msg = _group_message(
        text, from_user_id=from_user_id, from_user_name=from_user_name,
        entities=[SimpleNamespace(type="mention", offset=offset, length=len("@hermes_bot"))],
    )
    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=1001)
    event.text = adapter._clean_bot_trigger_text(event.text)
    return event


def _reset_vars():
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_addressed_observe_turn_keeps_trusted_sender_on_event():
    """Shared source stays sender-less; the event itself still names the trusted sender."""
    adapter = _make_adapter()
    event = _addressed_event(adapter)
    assert event.source.user_id == "222"  # pre-attribution: the update's sender

    attributed = adapter._apply_telegram_group_observe_attribution(event)

    # The session source remains the shared, sender-less one (routing unchanged).
    assert attributed.source.chat_id == "-100"
    assert attributed.source.user_id is None
    assert attributed.source.user_name is None
    # The live event keeps the trusted author for the current turn.
    assert attributed.user_id == "222"
    assert attributed.user_name == "Bob Example"
    assert attributed.text.startswith("[Bob Example|222]\n")


def test_observe_turn_command_keeps_source_identity():
    """Commands were already exempt: source identity and event fields stay untouched."""
    adapter = _make_adapter()
    msg = _group_message("/status", from_user_id=222, from_user_name="Bob Example")
    event = adapter._build_message_event(msg, MessageType.COMMAND, update_id=1002)

    attributed = adapter._apply_telegram_group_observe_attribution(event)

    assert attributed.source.user_id == "222"
    assert attributed.user_id is None  # command path never stamped the event author


def test_turn_actor_from_event_reads_only_the_event():
    assert turn_actor_from_event(SimpleNamespace(user_id="222", user_name="Bob")) == {
        "id": "222", "name": "Bob"}
    # Blank / absent authors carry no actor; nothing is inferred from elsewhere.
    assert turn_actor_from_event(SimpleNamespace(user_id="", user_name=None)) is None
    assert turn_actor_from_event(SimpleNamespace(user_id=None, user_name="X")) is None


def test_build_session_context_carries_turn_actor():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group")
    config = SimpleNamespace(
        get_connected_platforms=list, get_home_channel=lambda p: None,
        group_sessions_per_user=False, thread_sessions_per_user=False,
    )
    context = build_session_context(
        source, config, turn_actor=turn_actor_from_event(SimpleNamespace(user_id="222", user_name="Bob")))
    assert context.turn_actor == {"id": "222", "name": "Bob"}
    assert build_session_context(source, config).turn_actor is None


def test_session_env_binds_turn_actor_when_source_is_senderless(monkeypatch):
    """HERMES_SESSION_USER_ID must name the triggering sender even on a shared session."""
    try:
        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group")
        context = SessionContext(
            source=source, connected_platforms=[], home_channels={},
            turn_actor={"id": "222", "name": "Bob Example"})

        tokens = runner._set_session_env(context)

        assert get_session_env("HERMES_SESSION_USER_ID") == "222"
        assert get_session_env("HERMES_SESSION_USER_NAME") == "Bob Example"
        # The shared-session routing identity is NOT rewritten: chat stays the group lane.
        assert get_session_env("HERMES_SESSION_CHAT_ID") == "-100"
    finally:
        clear_session_vars(tokens)
        _reset_vars()


def test_session_env_prefers_sourced_user_id_over_turn_actor():
    """A source that already identifies its user keeps today's values (no behavior change)."""
    try:
        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="111", chat_type="dm",
            user_id="111", user_name="Alice")
        context = SessionContext(
            source=source, connected_platforms=[], home_channels={},
            turn_actor={"id": "222", "name": "Bob Example"})

        tokens = runner._set_session_env(context)

        assert get_session_env("HERMES_SESSION_USER_ID") == "111"
        assert get_session_env("HERMES_SESSION_USER_NAME") == "Alice"
    finally:
        clear_session_vars(tokens)
        _reset_vars()


def test_session_env_without_actor_keeps_senderless_blank():
    """No event author (unscoped reader, non-observe platform) must not invent one."""
    try:
        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group")
        context = SessionContext(source=source, connected_platforms=[], home_channels={})

        tokens = runner._set_session_env(context)

        assert get_session_env("HERMES_SESSION_USER_ID") == ""
        assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    finally:
        clear_session_vars(tokens)
        _reset_vars()
