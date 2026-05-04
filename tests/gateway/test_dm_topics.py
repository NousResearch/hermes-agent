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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


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
async def test_setup_dm_topics_loads_persisted_thread_ids():
    """Topics with thread_id in config should be loaded into cache, not created."""
    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [
                {"name": "General", "thread_id": 100},
                {"name": "Work", "thread_id": 200},
            ],
        }
    ])
    adapter._bot = AsyncMock()

    await adapter._setup_dm_topics()

    # Both should be in cache
    assert adapter._dm_topics["111:General"] == 100
    assert adapter._dm_topics["111:Work"] == 200
    # create_forum_topic should NOT have been called
    adapter._bot.create_forum_topic.assert_not_called()


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


@pytest.mark.asyncio
async def test_setup_dm_topics_skips_empty_config():
    """Empty dm_topics config should be a no-op."""
    adapter = _make_adapter([])
    adapter._bot = AsyncMock()

    await adapter._setup_dm_topics()

    adapter._bot.create_forum_topic.assert_not_called()
    assert adapter._dm_topics == {}


@pytest.mark.asyncio
async def test_setup_dm_topics_no_config():
    """No dm_topics in config at all should be a no-op."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()

    await adapter._setup_dm_topics()

    adapter._bot.create_forum_topic.assert_not_called()


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
async def test_create_dm_topic_returns_none_without_bot():
    """No bot instance should return None."""
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter._create_dm_topic(chat_id=111, name="General")

    assert result is None


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


def test_persist_dm_topic_thread_id_skips_if_already_set(tmp_path):
    """Should not overwrite an existing thread_id."""
    import yaml

    config_data = {
        "platforms": {
            "telegram": {
                "extra": {
                    "dm_topics": [
                        {
                            "chat_id": 111,
                            "topics": [
                                {"name": "General", "icon_color": 123, "thread_id": 500},
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

    with patch.object(Path, "home", return_value=tmp_path):
        adapter._persist_dm_topic_thread_id(111, "General", 999)

    with open(config_file) as f:
        result = yaml.safe_load(f)

    topics = result["platforms"]["telegram"]["extra"]["dm_topics"][0]["topics"]
    assert topics[0]["thread_id"] == 500  # unchanged


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


def test_get_dm_topic_info_returns_none_for_unknown():
    """Should return None for unknown thread_id."""
    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [{"name": "General"}],
        }
    ])
    # Mock reload to avoid filesystem access
    adapter._reload_dm_topics_from_config = lambda: None

    result = adapter._get_dm_topic_info("111", "999")

    assert result is None


def test_get_dm_topic_info_returns_none_without_config():
    """Should return None if no dm_topics config."""
    adapter = _make_adapter()
    adapter._reload_dm_topics_from_config = lambda: None

    result = adapter._get_dm_topic_info("111", "100")

    assert result is None


def test_get_dm_topic_info_returns_none_for_none_thread():
    """Should return None if thread_id is None."""
    adapter = _make_adapter([
        {"chat_id": 111, "topics": [{"name": "General"}]}
    ])

    result = adapter._get_dm_topic_info("111", None)

    assert result is None


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


def test_cache_dm_topic_from_message():
    """Should cache a new topic mapping."""
    adapter = _make_adapter()

    adapter._cache_dm_topic_from_message("111", "100", "General")

    assert adapter._dm_topics["111:General"] == 100


def test_cache_dm_topic_from_message_no_overwrite():
    """Should not overwrite an existing cached topic."""
    adapter = _make_adapter()
    adapter._dm_topics["111:General"] = 100

    adapter._cache_dm_topic_from_message("111", "999", "General")

    assert adapter._dm_topics["111:General"] == 100  # unchanged


# ── _build_message_event: auto_skill binding ──


def _make_mock_message(chat_id=111, chat_type="private", text="hello", thread_id=None,
                       user_id=42, user_name="Test User", forum_topic_created=None):
    """Create a mock Telegram Message for _build_message_event tests."""
    chat = SimpleNamespace(
        id=chat_id,
        type=chat_type,
        title=None,
    )
    # Add full_name attribute for DM chats
    if not hasattr(chat, "full_name"):
        chat.full_name = user_name

    user = SimpleNamespace(
        id=user_id,
        full_name=user_name,
    )

    msg = SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=thread_id,
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


def test_build_message_event_no_auto_skill_without_thread():
    """Regular DM messages (no thread_id) should have auto_skill=None."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_mock_message(chat_id=111, thread_id=None)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None


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
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=5, text="hello"
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
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=12, text="deal update"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "sales-framework"
    assert event.source.chat_topic == "Sales"


def test_group_topic_no_skill_binding():
    """Group topic without a skill key should have auto_skill=None but set chat_topic."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "General", "thread_id": 1},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=1, text="hey"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None
    assert event.source.chat_topic == "General"


def test_group_topic_unmapped_thread_id():
    """Thread ID not in config should fall through — no skill, no topic name."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Engineering", "thread_id": 5, "skill": "software-development"},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=999, text="random"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None
    assert event.source.chat_topic is None


def test_group_topic_unmapped_chat_id():
    """Chat ID not in group_topics config should fall through silently."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Engineering", "thread_id": 5, "skill": "software-development"},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1009999999999, chat_type=_ChatType.SUPERGROUP, thread_id=5, text="wrong group"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None
    assert event.source.chat_topic is None


def test_group_topic_no_config():
    """No group_topics config at all should be fine — no skill, no topic."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()  # no group_topics_config

    msg = _make_mock_message(
        chat_id=-1001234567890, chat_type=_ChatType.GROUP, thread_id=5, text="hi"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None
    assert event.source.chat_topic is None


def test_group_topic_chat_id_int_string_coercion():
    """chat_id as string in config should match integer chat.id via str() coercion."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": "-1001234567890",  # string, not int
            "topics": [
                {"name": "Dev", "thread_id": "7", "skill": "hermes-agent-dev"},
            ],
        }
    ])

    msg = _make_mock_message(
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=7, text="test"
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "hermes-agent-dev"
    assert event.source.chat_topic == "Dev"


# ── _build_message_event: from_user=None fallback in DMs ──


def test_build_message_event_dm_from_user_none_falls_back_to_chat_id():
    """When from_user is None in a DM, user_id should fall back to chat.id."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_mock_message(chat_id=12345, user_id=42, user_name="Alice")
    # Simulate from_user being None (edge case on fresh restart / forwarded msg)
    msg.from_user = None

    event = adapter._build_message_event(msg, MessageType.TEXT)

    # Should fall back to chat.id since chat_type is "dm"
    assert event.source.user_id == "12345"
    assert event.source.user_name == "Alice"  # falls back to chat.full_name


def test_build_message_event_group_from_user_none_stays_none():
    """When from_user is None in a group, user_id should remain None."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_mock_message(
        chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP,
        user_id=42, user_name="Alice"
    )
    msg.from_user = None

    event = adapter._build_message_event(msg, MessageType.TEXT)

    # Groups should NOT fall back — anonymous senders stay None
    assert event.source.user_id is None
    assert event.source.user_name is None


def test_build_message_event_dm_from_user_present_uses_user():
    """When from_user is present in a DM, it should be used (no fallback)."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_mock_message(chat_id=12345, user_id=99999, user_name="Bob")

    event = adapter._build_message_event(msg, MessageType.TEXT)

    # Normal case — from_user is used directly
    assert event.source.user_id == "99999"
    assert event.source.user_name == "Bob"


# ── _clean_dm_topic_title: LLM output normalization ──


def test_clean_dm_topic_title_strips_closed_think_block():
    """<think>...</think> blocks (reasoning models) must be stripped."""
    raw = "<think>analysing the user request</think>\nMPF Customer Research"
    assert TelegramAdapter._clean_dm_topic_title(raw) == "MPF Customer Research"


def test_clean_dm_topic_title_strips_thinking_variant():
    """<thinking>...</thinking> alternate tag must be stripped."""
    raw = "<thinking>let me think</thinking>\nABC Company Meeting"
    assert TelegramAdapter._clean_dm_topic_title(raw) == "ABC Company Meeting"


def test_clean_dm_topic_title_returns_empty_for_unterminated_think():
    """An unterminated <think> block (truncated by max_tokens) must consume to EOF and yield empty."""
    raw = "<think>\nThe user is writing in Cantonese (using traditional…"
    assert TelegramAdapter._clean_dm_topic_title(raw) == ""


def test_clean_dm_topic_title_strips_arbitrary_tags():
    """Any remaining XML-like tag must be stripped (defence in depth)."""
    raw = "<title>SomeName</title>"
    assert TelegramAdapter._clean_dm_topic_title(raw) == "SomeName"


def test_clean_dm_topic_title_takes_first_nonempty_line():
    """Multi-line scratchpad output should reduce to the first non-empty line."""
    raw = "\n\nClean Title\nthis is the reasoning"
    assert TelegramAdapter._clean_dm_topic_title(raw) == "Clean Title"


def test_clean_dm_topic_title_strips_quotes_and_brackets():
    """Surrounding quotes/brackets common in LLM output must be removed."""
    assert TelegramAdapter._clean_dm_topic_title('"Birthday list"') == "Birthday list"
    assert TelegramAdapter._clean_dm_topic_title("「Customer notes」") == "Customer notes"


def test_clean_dm_topic_title_strips_title_prefix():
    """Title:/Topic:/Name: prefixes should be stripped."""
    assert TelegramAdapter._clean_dm_topic_title("Title: AIA Email") == "AIA Email"
    assert TelegramAdapter._clean_dm_topic_title("Topic：客戶 MPF") == "客戶 MPF"


def test_clean_dm_topic_title_strips_trailing_punctuation():
    """Trailing punctuation common in user questions should be stripped."""
    assert TelegramAdapter._clean_dm_topic_title("AIA Email？") == "AIA Email"
    assert TelegramAdapter._clean_dm_topic_title("Notes!") == "Notes"


def test_clean_dm_topic_title_truncates_to_60():
    """Output longer than 60 chars must be ellipsis-truncated."""
    out = TelegramAdapter._clean_dm_topic_title("a" * 100)
    assert len(out) == 60
    assert out.endswith("…")


def test_clean_dm_topic_title_rejects_residual_markup():
    """If cleaning leaves a stray < or > (unbalanced bracket) it must be rejected as empty."""
    # An unmatched '<' has no closing '>', so the tag-stripping regex can't
    # remove it. The residual-markup guard must reject it.
    raw = "unmatched < bracket"
    assert TelegramAdapter._clean_dm_topic_title(raw) == ""


def test_clean_dm_topic_title_handles_empty_input():
    """Empty / whitespace input returns empty string."""
    assert TelegramAdapter._clean_dm_topic_title("") == ""
    assert TelegramAdapter._clean_dm_topic_title("   \n  \n") == ""


# ── _send_dm_topic_seed_message ──


@pytest.mark.asyncio
async def test_send_dm_topic_seed_message_uses_provided_name():
    """Seed message text should be '📌 <name>' when name is short."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    await adapter._send_dm_topic_seed_message(111, 222, "Birthdays")
    adapter._bot.send_message.assert_called_once()
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 111
    assert kwargs["message_thread_id"] == 222
    assert kwargs["text"] == "📌 Birthdays"


@pytest.mark.asyncio
async def test_send_dm_topic_seed_message_truncates_long_names():
    """Long topic names should be ellipsis-truncated in the seed text."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    long_name = "x" * 200
    await adapter._send_dm_topic_seed_message(111, 222, long_name)
    text = adapter._bot.send_message.call_args.kwargs["text"]
    assert text.startswith("📌 ")
    assert text.endswith("…")
    # 64-char visible cap on the name portion.
    assert len(text) <= 2 + 64  # "📌 " + 64 chars


@pytest.mark.asyncio
async def test_send_dm_topic_seed_message_falls_back_for_empty_name():
    """Missing/whitespace-only name should fall back to a generic seed text."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    await adapter._send_dm_topic_seed_message(111, 222, "  ")
    assert adapter._bot.send_message.call_args.kwargs["text"] == "📌 Topic ready."


@pytest.mark.asyncio
async def test_send_dm_topic_seed_message_no_op_without_bot():
    """No bot instance should be a silent no-op."""
    adapter = _make_adapter()
    adapter._bot = None
    # Must not raise.
    await adapter._send_dm_topic_seed_message(111, 222, "Birthdays")


@pytest.mark.asyncio
async def test_send_dm_topic_seed_message_swallows_send_errors():
    """A send_message failure must be logged but never raised."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.send_message.side_effect = Exception("boom")
    # Must not raise.
    await adapter._send_dm_topic_seed_message(111, 222, "Birthdays")


# ── _create_dm_topic: seed message integration ──


@pytest.mark.asyncio
async def test_create_dm_topic_sends_seed_after_success():
    """Successful topic creation must trigger a seed message in the new thread."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=888)

    thread_id = await adapter._create_dm_topic(chat_id=111, name="Birthdays")

    assert thread_id == 888
    # Seed message should have gone into the new thread.
    adapter._bot.send_message.assert_called_once()
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 111
    assert kwargs["message_thread_id"] == 888
    assert kwargs["text"] == "📌 Birthdays"


@pytest.mark.asyncio
async def test_create_dm_topic_no_seed_on_failure():
    """A failed topic creation must NOT send a seed message."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.side_effect = Exception("topic_name_duplicate")

    thread_id = await adapter._create_dm_topic(chat_id=111, name="Birthdays")

    assert thread_id is None
    adapter._bot.send_message.assert_not_called()


# ── _cache_dm_topic_from_message: seed + rename scheduling ──


@pytest.mark.asyncio
async def test_cache_dm_topic_schedules_seed_for_new_entry():
    """Caching a new topic must schedule a seed message on the running loop."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()

    adapter._cache_dm_topic_from_message("111", "555", "Customer notes")

    # Let the scheduled task run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert adapter._dm_topics["111:Customer notes"] == 555
    adapter._bot.send_message.assert_called_once()
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 111
    assert kwargs["message_thread_id"] == 555
    assert kwargs["text"] == "📌 Customer notes"


@pytest.mark.asyncio
async def test_cache_dm_topic_idempotent_no_duplicate_seed():
    """Re-caching the same key must not schedule a second seed."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()

    adapter._cache_dm_topic_from_message("111", "555", "Customer notes")
    await asyncio.sleep(0)
    adapter._bot.send_message.reset_mock()
    # Second call with same key.
    adapter._cache_dm_topic_from_message("111", "555", "Customer notes")
    await asyncio.sleep(0)

    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_cache_dm_topic_skips_rename_when_flag_disabled():
    """Without auto_rename_dm_topics, no rename must be scheduled."""
    adapter = _make_adapter()
    assert adapter._auto_rename_dm_topics is False
    adapter._bot = AsyncMock()
    adapter._maybe_rename_dm_topic = AsyncMock()  # type: ignore[assignment]

    adapter._cache_dm_topic_from_message("111", "555", "Long sentence-like name?")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    adapter._maybe_rename_dm_topic.assert_not_called()


@pytest.mark.asyncio
async def test_cache_dm_topic_schedules_rename_when_flag_enabled():
    """With auto_rename_dm_topics enabled, rename must be scheduled for new entries."""
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={"auto_rename_dm_topics": True},
    )
    adapter = TelegramAdapter(config)
    assert adapter._auto_rename_dm_topics is True
    adapter._bot = AsyncMock()
    adapter._maybe_rename_dm_topic = AsyncMock()  # type: ignore[assignment]

    adapter._cache_dm_topic_from_message("111", "555", "Long sentence-like name?")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    adapter._maybe_rename_dm_topic.assert_called_once_with(
        111, 555, "Long sentence-like name?"
    )


# ── _handle_forum_topic_created ──


@pytest.mark.asyncio
async def test_handle_forum_topic_created_caches_topic():
    """forum_topic_created service messages must reach _cache_dm_topic_from_message."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()

    msg = MagicMock()
    msg.forum_topic_created = SimpleNamespace(name="Newly created")
    msg.message_thread_id = 999
    msg.chat = SimpleNamespace(id=111)
    update = SimpleNamespace(effective_message=msg, message=msg)

    await adapter._handle_forum_topic_created(update, None)
    await asyncio.sleep(0)

    assert adapter._dm_topics["111:Newly created"] == 999


@pytest.mark.asyncio
async def test_handle_forum_topic_created_ignores_non_topic_messages():
    """Messages without forum_topic_created attribute must be ignored."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()

    msg = MagicMock()
    msg.forum_topic_created = None
    update = SimpleNamespace(effective_message=msg, message=msg)

    await adapter._handle_forum_topic_created(update, None)
    assert adapter._dm_topics == {}


# ── _maybe_rename_dm_topic ──


@pytest.mark.asyncio
async def test_maybe_rename_dm_topic_calls_edit_when_title_generated():
    """A successful LLM title must be applied via editForumTopic."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._generate_dm_topic_title = AsyncMock(return_value="客戶 MPF 研究")

    await adapter._maybe_rename_dm_topic(111, 555, "我想研究吓一啲客戶 MPF 嘅情況")

    adapter._bot.edit_forum_topic.assert_called_once_with(
        chat_id=111, message_thread_id=555, name="客戶 MPF 研究",
    )


@pytest.mark.asyncio
async def test_maybe_rename_dm_topic_skips_if_llm_returns_none():
    """When the LLM cleaner produces empty output, no rename is sent."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._generate_dm_topic_title = AsyncMock(return_value=None)

    await adapter._maybe_rename_dm_topic(111, 555, "原本嘅名")

    adapter._bot.edit_forum_topic.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_rename_dm_topic_skips_if_unchanged():
    """If the LLM returns the same name, no rename is sent."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._generate_dm_topic_title = AsyncMock(return_value="客戶 MPF 研究")

    await adapter._maybe_rename_dm_topic(111, 555, "客戶 MPF 研究")

    adapter._bot.edit_forum_topic.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_rename_dm_topic_swallows_edit_errors():
    """An editForumTopic failure must be logged but never raised."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.edit_forum_topic.side_effect = Exception("boom")
    adapter._generate_dm_topic_title = AsyncMock(return_value="New Name")

    # Must not raise.
    await adapter._maybe_rename_dm_topic(111, 555, "Old Name")
