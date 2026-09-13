"""Tests for Telegram DM Private Chat Topics (Bot API 9.4).

Covers:
- _setup_dm_topics: loading persisted thread_ids from config
- _setup_dm_topics: creating new topics via API when no thread_id
- _persist_dm_topic_thread_id: saving thread_id back to config.yaml
- _get_dm_topic_info: looking up topic config by thread_id
- _cache_dm_topic_from_message: caching thread_ids from incoming messages
- _build_message_event: DM topic resolution in message events
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


# Use the shared, comprehensive telegram mock from conftest instead of a
# file-local one. The previous local installer differed from every other
# telegram test's stub in two ways — it registered a SEPARATE string-valued
# ``telegram.constants`` module (others register the root mock, so ParseMode
# members stay auto-generated MagicMock attributes) and its ``telegram.error``
# was a bare MagicMock (conftest defines real exception subclasses with PTB's
# hierarchy) — and it installed UNCONDITIONALLY (no real-library guard).
# Because it also force-reimported the adapter, the divergent stub leaked into
# sys.modules for the rest of the session: every later telegram test that
# asserts ParseMode repr or isinstance against telegram.error classes failed
# order-dependently in full runs while passing in isolation.
from tests.gateway.conftest import _ensure_telegram_mock  # noqa: E402

_ensure_telegram_mock()
# Force reimport so the adapter binds to whatever sys.modules now holds
# (the shared mock, or the real library when it is installed) rather than a
# stub an earlier test file may have bound it to.
sys.modules.pop("plugins.platforms.telegram.adapter", None)

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter(dm_topics_config=None, group_topics_config=None):
    """Create a TelegramAdapter with optional DM/group topics config."""
    extra = {}
    if dm_topics_config is not None:
        extra["dm_topics"] = dm_topics_config
    if group_topics_config is not None:
        extra["group_topics"] = group_topics_config
    config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter = TelegramAdapter(config)
    return adapter


# ── _setup_dm_topics: load persisted thread_ids ──


@pytest.mark.asyncio
async def test_setup_dm_topics_creates_when_no_thread_id():
    """Topics without thread_id should be created via API."""
    adapter = _make_adapter([
        {
            "chat_id": 222,
            "topics": [
                {"name": "NewTopic", "icon_color": 7322096},
            ],
        }
    ])
    adapter._bot = AsyncMock()
    mock_topic = SimpleNamespace(message_thread_id=999)
    adapter._bot.create_forum_topic.return_value = mock_topic

    # Mock the persist method so it doesn't touch the filesystem
    adapter._persist_dm_topic_thread_id = MagicMock()

    await adapter._setup_dm_topics()

    # Should have been created
    adapter._bot.create_forum_topic.assert_called_once_with(
        chat_id=222, name="NewTopic", icon_color=7322096,
    )
    # Should be in cache
    assert adapter._dm_topics["222:NewTopic"] == 999
    # Should persist
    adapter._persist_dm_topic_thread_id.assert_called_once_with(222, "NewTopic", 999)


@pytest.mark.asyncio
async def test_setup_dm_topics_mixed_persisted_and_new():
    """Mix of persisted and new topics should work correctly."""
    adapter = _make_adapter([
        {
            "chat_id": 333,
            "topics": [
                {"name": "Existing", "thread_id": 50},
                {"name": "New", "icon_color": 123},
            ],
        }
    ])
    adapter._bot = AsyncMock()
    mock_topic = SimpleNamespace(message_thread_id=777)
    adapter._bot.create_forum_topic.return_value = mock_topic
    adapter._persist_dm_topic_thread_id = MagicMock()

    await adapter._setup_dm_topics()

    # Existing loaded from config
    assert adapter._dm_topics["333:Existing"] == 50
    # New created via API
    assert adapter._dm_topics["333:New"] == 777
    # Only one API call (for "New")
    adapter._bot.create_forum_topic.assert_called_once()


# ── _create_dm_topic: error handling ──


@pytest.mark.asyncio
async def test_create_dm_topic_handles_duplicate_error():
    """Duplicate topic error should return None gracefully."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.side_effect = Exception("topic_name_duplicate")

    result = await adapter._create_dm_topic(chat_id=111, name="General")

    assert result is None


@pytest.mark.asyncio
async def test_create_dm_topic_handles_generic_error():
    """Generic error should return None with warning."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.side_effect = Exception("some random error")

    result = await adapter._create_dm_topic(chat_id=111, name="General")

    assert result is None


@pytest.mark.asyncio
async def test_ensure_dm_topic_creates_on_demand_and_persists():
    """Named delivery targets should create missing private DM topics on demand."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=444)
    adapter._persist_dm_topic_thread_id = MagicMock()

    result = await adapter.ensure_dm_topic("111", "On Demand")

    assert result == "444"
    adapter._bot.create_forum_topic.assert_called_once_with(
        chat_id=111,
        name="On Demand",
    )
    assert adapter._dm_topics["111:On Demand"] == 444
    assert adapter._dm_topics_config == [
        {"chat_id": 111, "topics": [{"name": "On Demand", "thread_id": 444}]}
    ]
    adapter._persist_dm_topic_thread_id.assert_called_once_with(
        111, "On Demand", 444, replace_existing=False
    )


# ── _persist_dm_topic_thread_id ──


def test_persist_dm_topic_thread_id_writes_config(tmp_path):
    """Should write thread_id into the correct topic in config.yaml."""
    import yaml

    config_data = {
        "platforms": {
            "telegram": {
                "extra": {
                    "dm_topics": [
                        {
                            "chat_id": 111,
                            "topics": [
                                {"name": "General", "icon_color": 123},
                                {"name": "Work", "icon_color": 456},
                            ],
                        }
                    ]
                }
            }
        }
    }

    config_file = tmp_path / ".hermes" / "config.yaml"
    config_file.parent.mkdir(parents=True)
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    adapter = _make_adapter()

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
        adapter._persist_dm_topic_thread_id(111, "General", 999)

    with open(config_file) as f:
        result = yaml.safe_load(f)

    topics = result["platforms"]["telegram"]["extra"]["dm_topics"][0]["topics"]
    assert topics[0]["thread_id"] == 999
    assert "thread_id" not in topics[1]  # "Work" should be untouched


# ── _get_dm_topic_info ──


def test_persist_dm_topic_thread_id_preserves_config_on_write_failure(tmp_path):
    """Failed writes should leave the original config.yaml intact."""
    import yaml

    config_data = {
        "platforms": {
            "telegram": {
                "extra": {
                    "dm_topics": [
                        {
                            "chat_id": 111,
                            "topics": [
                                {"name": "General", "icon_color": 123},
                            ],
                        }
                    ]
                }
            }
        }
    }

    config_file = tmp_path / ".hermes" / "config.yaml"
    config_file.parent.mkdir(parents=True)
    original_text = yaml.dump(config_data)
    config_file.write_text(original_text, encoding="utf-8")

    adapter = _make_adapter()

    def fail_dump(*args, **kwargs):
        raise RuntimeError("boom")

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}), \
         patch("yaml.dump", side_effect=fail_dump):
        adapter._persist_dm_topic_thread_id(111, "General", 999)

    assert config_file.read_text(encoding="utf-8") == original_text
    result = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    topics = result["platforms"]["telegram"]["extra"]["dm_topics"][0]["topics"]
    assert "thread_id" not in topics[0]


def test_get_dm_topic_info_finds_cached_topic():
    """Should return topic config when thread_id is in cache."""
    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [
                {"name": "General", "skill": "my-skill"},
            ],
        }
    ])
    adapter._dm_topics["111:General"] = 100

    result = adapter._get_dm_topic_info("111", "100")

    assert result is not None
    assert result["name"] == "General"
    assert result["skill"] == "my-skill"


def test_get_dm_topic_info_hot_reloads_from_config(tmp_path):
    """Should find a topic added to config after startup (hot-reload)."""
    import yaml

    # Start with empty topics
    adapter = _make_adapter([
        {"chat_id": 111, "topics": []}
    ])

    # Write config with a new topic + thread_id
    config_data = {
        "platforms": {
            "telegram": {
                "extra": {
                    "dm_topics": [
                        {
                            "chat_id": 111,
                            "topics": [
                                {"name": "NewProject", "thread_id": 555},
                            ],
                        }
                    ]
                }
            }
        }
    }
    config_file = tmp_path / ".hermes" / "config.yaml"
    config_file.parent.mkdir(parents=True)
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
        result = adapter._get_dm_topic_info("111", "555")

    assert result is not None
    assert result["name"] == "NewProject"
    # Should now be cached
    assert adapter._dm_topics["111:NewProject"] == 555


# ── _cache_dm_topic_from_message ──


def test_cache_dm_topic_from_message_no_overwrite():
    """Should not overwrite an existing cached topic."""
    adapter = _make_adapter()
    adapter._dm_topics["111:General"] = 100

    adapter._cache_dm_topic_from_message("111", "999", "General")

    assert adapter._dm_topics["111:General"] == 100  # unchanged


# ── _build_message_event: auto_skill binding ──


def _make_mock_message(chat_id=111, chat_type="private", text="hello", thread_id=None,
                       user_id=42, user_name="Test User", forum_topic_created=None,
                       is_topic_message=None, is_forum=None):
    """Create a mock Telegram Message for _build_message_event tests."""
    chat = SimpleNamespace(
        id=chat_id,
        type=chat_type,
        title=None,
    )
    if is_forum is not None:
        chat.is_forum = is_forum
    # Add full_name attribute for DM chats
    if not hasattr(chat, "full_name"):
        chat.full_name = user_name

    user = SimpleNamespace(
        id=user_id,
        full_name=user_name,
    )

    if is_topic_message is None:
        is_topic_message = bool(thread_id) if chat_type == "private" else None

    msg = SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=thread_id,
        is_topic_message=is_topic_message,
        message_id=1001,
        reply_to_message=None,
        date=None,
        forum_topic_created=forum_topic_created,
    )
    return msg


def test_build_message_event_sets_auto_skill():
    """When topic has a skill binding, auto_skill should be set on the event."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [
                {"name": "My Project", "skill": "accessibility-auditor", "thread_id": 100},
            ],
        }
    ])
    adapter._dm_topics["111:My Project"] = 100

    msg = _make_mock_message(chat_id=111, thread_id=100, text="check this page")
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "accessibility-auditor"
    # chat_topic should be the clean topic name, no [skill: ...] suffix
    assert event.source.chat_topic == "My Project"


def test_build_message_event_no_auto_skill_without_binding():
    """Topics without skill binding should have auto_skill=None."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [
                {"name": "General", "thread_id": 200},
            ],
        }
    ])
    adapter._dm_topics["111:General"] = 200

    msg = _make_mock_message(chat_id=111, thread_id=200)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None
    assert event.source.chat_topic == "General"


# ── _build_message_event: group_topics skill binding ──

# The telegram mock sets sys.modules["telegram.constants"] = telegram_mod (root mock),
# so `from telegram.constants import ChatType` in telegram.py resolves to
# telegram_mod.ChatType — not telegram_mod.constants.ChatType.  We must use
# the same ChatType object the production code sees so equality checks work.
from telegram.constants import ChatType as _ChatType  # noqa: E402


def test_group_topic_skill_binding():
    """Group topic with skill config should set auto_skill on the event."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Engineering", "thread_id": 5, "skill": "software-development"},
                {"name": "Sales", "thread_id": 12, "skill": "sales-framework"},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1001234567890,
        chat_type=_ChatType.SUPERGROUP,
        thread_id=5,
        text="hello",
        is_topic_message=True,
        is_forum=True,
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "software-development"
    assert event.source.chat_topic == "Engineering"


def test_group_topic_skill_binding_second_topic():
    """A different thread_id in the same group should resolve its own skill."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Engineering", "thread_id": 5, "skill": "software-development"},
                {"name": "Sales", "thread_id": 12, "skill": "sales-framework"},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1001234567890,
        chat_type=_ChatType.SUPERGROUP,
        thread_id=12,
        text="deal update",
        is_topic_message=True,
        is_forum=True,
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "sales-framework"
    assert event.source.chat_topic == "Sales"


# ── _build_message_event: from_user=None fallback in DMs ──


# ── _build_message_event: DM-topic lane inheritance for stamp-less media (#109527) ──


def test_stampless_photo_inherits_last_topic_lane():
    """A photo with no topic stamp at all inherits the sender's last stamped topic lane instead
    of landing in the chat's default lane."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    text_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(text_msg, MessageType.TEXT)

    photo_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="")
    event = adapter._build_message_event(photo_msg, MessageType.PHOTO)

    assert event.source.thread_id == "100"


def test_stampless_photo_stays_in_default_lane_without_prior_topic():
    """No prior stamped message for this (chat, user) means there is no lane to inherit."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    photo_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="")
    event = adapter._build_message_event(photo_msg, MessageType.PHOTO)

    assert event.source.thread_id is None


def test_stampless_photo_lane_expires_after_ttl():
    """A lane older than the TTL window is not inherited — falls back to the default lane."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    text_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(text_msg, MessageType.TEXT)

    future = time.monotonic() + TelegramAdapter._DM_TOPIC_LANE_TTL_SECONDS + 1
    with patch("plugins.platforms.telegram.adapter.time.monotonic", return_value=future):
        photo_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="")
        event = adapter._build_message_event(photo_msg, MessageType.PHOTO)

    assert event.source.thread_id is None


def test_stampless_photo_does_not_inherit_another_users_lane():
    """A lane recorded for one user must not leak to a different user in the same chat."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    text_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(text_msg, MessageType.TEXT)

    photo_msg = _make_mock_message(chat_id=111, user_id=99, thread_id=None, text="")
    event = adapter._build_message_event(photo_msg, MessageType.PHOTO)

    assert event.source.thread_id is None


# ── _build_message_event: lane inheritance is restricted to stamp-losing media types ──
# Text, commands and location messages always carry a real stamp (or none at all when there is
# genuinely no topic), so recalling a lane for them would misroute an ordinary plain-DM message
# into a stale topic instead of leaving it in the default lane.


def test_stampless_text_does_not_inherit_topic_lane():
    """A stamp-less text message must NOT inherit a previously observed topic lane."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    stamped = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(stamped, MessageType.TEXT)

    plain_text = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="hello again")
    event = adapter._build_message_event(plain_text, MessageType.TEXT)

    assert event.source.thread_id is None


def test_stampless_command_does_not_inherit_topic_lane():
    """A stamp-less command must NOT inherit a previously observed topic lane."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    stamped = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(stamped, MessageType.TEXT)

    command_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="/start")
    event = adapter._build_message_event(command_msg, MessageType.COMMAND)

    assert event.source.thread_id is None


def test_stampless_location_does_not_inherit_topic_lane():
    """A stamp-less location/venue share must NOT inherit a previously observed topic lane."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()

    stamped = _make_mock_message(chat_id=111, user_id=42, thread_id=100, text="hi")
    adapter._build_message_event(stamped, MessageType.TEXT)

    location_msg = _make_mock_message(chat_id=111, user_id=42, thread_id=None, text="")
    event = adapter._build_message_event(location_msg, MessageType.LOCATION)

    assert event.source.thread_id is None


# ── DM-topic lane inheritance through the real dispatch handlers (#109527 follow-up) ──
# The tests above call _build_message_event() directly. These go through the actual
# _handle_text_message / _handle_command / _handle_location_message entry points so the
# restriction is proven at the point where routing/session-key decisions are made too.


def _make_dispatch_message(chat_id=111, user_id=42, thread_id=None, text="hi", is_topic_message=None,
                            venue=None, location=None):
    msg = _make_mock_message(
        chat_id=chat_id, chat_type="private", text=text, thread_id=thread_id, user_id=user_id,
        is_topic_message=is_topic_message)
    msg.caption = None
    msg.entities = []
    msg.caption_entities = []
    msg.media_group_id = None
    msg.venue = venue
    msg.location = location
    msg.photo = None
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.document = None
    msg.sticker = None
    return msg


def _make_dispatch_update(msg, update_id=1):
    return SimpleNamespace(update_id=update_id, message=msg, effective_message=msg)


@pytest.mark.asyncio
async def test_dispatch_stampless_text_does_not_inherit_topic_lane():
    """Through _handle_text_message, a stamp-less DM text stays in the default lane."""
    adapter = _make_adapter()
    adapter._bot = None
    adapter.handle_message = AsyncMock()
    adapter._text_batch_delay_seconds = 0.01

    stamped = _make_dispatch_message(thread_id=100, text="hi", is_topic_message=True)
    await adapter._handle_text_message(_make_dispatch_update(stamped, update_id=1), SimpleNamespace())
    await asyncio.sleep(0.05)

    plain = _make_dispatch_message(thread_id=None, text="hello again")
    await adapter._handle_text_message(_make_dispatch_update(plain, update_id=2), SimpleNamespace())
    await asyncio.sleep(0.05)

    assert adapter.handle_message.await_count == 2
    last_event = adapter.handle_message.await_args_list[-1].args[0]
    assert last_event.source.thread_id is None


@pytest.mark.asyncio
async def test_dispatch_stampless_command_does_not_inherit_topic_lane():
    """Through _handle_command, a stamp-less DM command stays in the default lane."""
    adapter = _make_adapter()
    adapter._bot = None
    adapter.handle_message = AsyncMock()

    stamped = _make_dispatch_message(thread_id=100, text="hi", is_topic_message=True)
    await adapter._handle_text_message(_make_dispatch_update(stamped, update_id=1), SimpleNamespace())
    await asyncio.sleep(0.05)

    command_msg = _make_dispatch_message(thread_id=None, text="/start")
    await adapter._handle_command(_make_dispatch_update(command_msg, update_id=2), SimpleNamespace())

    event = adapter.handle_message.await_args_list[-1].args[0]
    assert event.source.thread_id is None


@pytest.mark.asyncio
async def test_dispatch_stampless_location_does_not_inherit_topic_lane():
    """Through _handle_location_message, a stamp-less DM location stays in the default lane."""
    adapter = _make_adapter()
    adapter._bot = None
    adapter.handle_message = AsyncMock()

    stamped = _make_dispatch_message(thread_id=100, text="hi", is_topic_message=True)
    await adapter._handle_text_message(_make_dispatch_update(stamped, update_id=1), SimpleNamespace())
    await asyncio.sleep(0.05)

    location_msg = _make_dispatch_message(thread_id=None, text="")
    location_msg.location = SimpleNamespace(latitude=1.23, longitude=4.56)
    await adapter._handle_location_message(_make_dispatch_update(location_msg, update_id=2), SimpleNamespace())

    event = adapter.handle_message.await_args_list[-1].args[0]
    assert event.source.thread_id is None
