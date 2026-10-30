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
from datetime import datetime, timedelta
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
from tests.gateway.conftest import _ensure_telegram_mock

_ensure_telegram_mock()
# Force reimport so the adapter binds to whatever sys.modules now holds
# (the shared mock, or the real library when it is installed) rather than a
# stub an earlier test file may have bound it to.
sys.modules.pop("plugins.platforms.telegram.adapter", None)

from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(
    dm_topics_config=None,
    group_topics_config=None,
    channel_skill_bindings=None,
):
    """Create a TelegramAdapter with optional topic and chat skill config."""
    extra = {}
    if dm_topics_config is not None:
        extra["dm_topics"] = dm_topics_config
    if group_topics_config is not None:
        extra["group_topics"] = group_topics_config
    if channel_skill_bindings is not None:
        extra["channel_skill_bindings"] = channel_skill_bindings
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
    import hermes_yaml as yaml

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
        yaml.safe_dump(config_data, f)

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
    import hermes_yaml as yaml

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
    original_text = yaml.safe_dump(config_data)
    config_file.write_text(original_text, encoding="utf-8")

    adapter = _make_adapter()

    def fail_dump(*args, **kwargs):
        raise RuntimeError("boom")

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}), \
         patch("ruamel.yaml.YAML.dump", side_effect=fail_dump):
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
    import hermes_yaml as yaml

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
        yaml.safe_dump(config_data, f)

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


def test_build_message_event_inherits_chat_skill_for_unknown_topic():
    """User-created DM topics should inherit their chat's skill binding."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(
        channel_skill_bindings=[{"id": "111", "skills": ["arxiv"]}]
    )
    adapter._reload_dm_topics_from_config = lambda: None

    msg = _make_mock_message(chat_id=111, thread_id=999, is_topic_message=True)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == ["arxiv"]
    assert event.source.thread_id == "999"
    assert event.source.chat_topic is None


def test_build_message_event_applies_chat_skill_to_root_dm():
    """A DM chat binding should also apply outside a topic."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(
        channel_skill_bindings=[{"id": "111", "skill": "arxiv"}]
    )

    msg = _make_mock_message(chat_id=111, thread_id=None)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == ["arxiv"]


def test_build_message_event_stacks_topic_skill_after_chat_skills():
    """A topic-specific skill should stack after inherited chat skills."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(
        dm_topics_config=[
            {
                "chat_id": 111,
                "topics": [
                    {"name": "Project", "thread_id": 100, "skill": "project-work"},
                ],
            }
        ],
        channel_skill_bindings=[
            {"id": "111", "skills": ["arxiv", "writing"]},
        ],
    )
    adapter._dm_topics["111:Project"] = 100

    msg = _make_mock_message(chat_id=111, thread_id=100)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == ["arxiv", "writing", "project-work"]


def test_build_message_event_deduplicates_chat_and_topic_skill():
    """The same chat and topic skill should only be loaded once."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(
        dm_topics_config=[
            {
                "chat_id": 111,
                "topics": [
                    {"name": "General", "thread_id": 100, "skill": "arxiv"},
                ],
            }
        ],
        channel_skill_bindings=[{"id": "111", "skills": ["arxiv"]}],
    )
    adapter._dm_topics["111:General"] = 100

    msg = _make_mock_message(chat_id=111, thread_id=100)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == ["arxiv"]


def test_build_message_event_does_not_inherit_other_chat_skill():
    """A chat binding must not leak into another Telegram DM."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(
        channel_skill_bindings=[{"id": "111", "skills": ["arxiv"]}]
    )
    adapter._reload_dm_topics_from_config = lambda: None

    msg = _make_mock_message(chat_id=222, thread_id=999, is_topic_message=True)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill is None


@pytest.mark.parametrize("session_state", ["new", "manual-reset", "continuing"])
@pytest.mark.asyncio
async def test_chat_skill_reaches_gateway_session_start(
    monkeypatch, tmp_path, session_state
):
    """A configured DM binding reaches the real gateway injection path when a session starts,
    and a continuing session is left alone (its history already carries the skill)."""
    import gateway.run as gateway_run
    from agent import skill_commands
    from gateway.config import Platform, load_gateway_config
    from gateway.platforms.base import MessageType
    from gateway.session import SessionEntry

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  channel_skill_bindings:\n"
        '    - id: "111"\n'
        "      skills:\n"
        "        - arxiv\n"
        "platforms:\n"
        "  telegram:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()
    adapter = TelegramAdapter(config.platforms[Platform.TELEGRAM])
    adapter._reload_dm_topics_from_config = lambda: None
    event = adapter._build_message_event(
        _make_mock_message(chat_id=111, thread_id=999, is_topic_message=True),
        MessageType.TEXT,
    )
    assert event.auto_skill == ["arxiv"]

    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    now = datetime.now()
    created_at = now if session_state == "new" else now - timedelta(seconds=1)
    session_entry = SessionEntry(
        session_key="agent:main:telegram:dm:111:999",
        session_id="session-1",
        created_at=created_at,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        is_fresh_reset=session_state == "manual-reset",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    load_skill = MagicMock(
        return_value=({"name": "arxiv"}, tmp_path, "arxiv")
    )
    monkeypatch.setattr(skill_commands, "_load_skill_payload", load_skill)
    monkeypatch.setattr(
        skill_commands,
        "_build_skill_message",
        lambda *_args, **_kwargs: "<skill>arxiv</skill>",
    )
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "failed": False,
        }
    )

    await runner._handle_message_with_agent(
        event,
        event.source,
        session_entry.session_key,
        1,
    )

    message = runner._run_agent.await_args.kwargs["message"]
    if session_state == "continuing":
        assert not message.startswith("<skill>")
        load_skill.assert_not_called()
        return
    assert message.startswith("<skill>arxiv</skill>\n\nhello")
    load_skill.assert_called_once_with(
        "arxiv", task_id=session_entry.session_key
    )
    assert session_entry.is_fresh_reset is False


# ── _build_message_event: group_topics skill binding ──

# The telegram mock sets sys.modules["telegram.constants"] = telegram_mod (root mock),
# so `from telegram.constants import ChatType` in telegram.py resolves to
# telegram_mod.ChatType — not telegram_mod.constants.ChatType.  We must use
# the same ChatType object the production code sees so equality checks work.
from telegram.constants import ChatType as _ChatType


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


def test_group_chat_binding_leaves_group_topics_unchanged():
    """Chat-wide bindings apply to DMs only: a group keeps its group_topics skill."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter(
        group_topics_config=[{
            "chat_id": -1001234567890,
            "topics": [{"name": "Engineering", "thread_id": 5, "skill": "software-development"}],
        }],
        channel_skill_bindings=[{"id": "-1001234567890", "skills": ["arxiv"]}],
    )

    msg = _make_mock_message(
        chat_id=-1001234567890,
        chat_type=_ChatType.SUPERGROUP,
        thread_id=5,
        is_topic_message=True,
        is_forum=True,
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.auto_skill == "software-development"


# ── _build_message_event: from_user=None fallback in DMs ──


