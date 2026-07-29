"""Tests for Telegram DM Private Chat Topics (Bot API 9.4).

Covers:
- _setup_dm_topics: loading persisted thread_ids from config
- _setup_dm_topics: creating new topics via API when no thread_id
- _persist_dm_topic_thread_id: saving thread_id back to config.yaml
- _get_dm_topic_info: looking up topic config by thread_id
- _cache_dm_topic_from_message: caching thread_ids from incoming messages
- _build_message_event: DM topic resolution in message events
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)

    # Register telegram.constants as a separate module mock so that
    # ``from telegram.constants import ChatType`` resolves to our mock
    # with string-valued members (not auto-generated MagicMocks).
    constants_mod = MagicMock()
    constants_mod.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    constants_mod.ChatType.GROUP = "group"
    constants_mod.ChatType.SUPERGROUP = "supergroup"
    constants_mod.ChatType.CHANNEL = "channel"
    constants_mod.ChatType.PRIVATE = "private"

    sys.modules["telegram"] = telegram_mod
    sys.modules["telegram.ext"] = telegram_mod.ext
    sys.modules["telegram.constants"] = constants_mod
    sys.modules["telegram.request"] = telegram_mod.request

    # Force reimport so the adapter picks up the mock ChatType.
    sys.modules.pop("plugins.platforms.telegram.adapter", None)


_ensure_telegram_mock()

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
    from gateway.platforms.base import MessageType

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
    from gateway.platforms.base import MessageType

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
    from gateway.platforms.base import MessageType

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
    from gateway.platforms.base import MessageType

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


