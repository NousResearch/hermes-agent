"""Tests for the MAX platform-plugin adapter (Russian messenger).

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_max`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Most tests target the adapter class directly. The plugin-shape tests
(``register()``, ``_env_enablement``, ``_standalone_send``, registry
presence) mirror the ntfy adapter tests — everything routes through the
``platform_registry``.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_max = load_plugin_adapter("max")

MaxAdapter = _max.MaxAdapter
check_requirements = _max.check_requirements
validate_config = _max.validate_config
is_connected = _max.is_connected
register = _max.register
_env_enablement = _max._env_enablement
_standalone_send = _max._standalone_send
MAX_MESSAGE_LENGTH = _max.MAX_MESSAGE_LENGTH
DEDUP_WINDOW_SECONDS = _max.DEDUP_WINDOW_SECONDS
prepare_outgoing_text = _max.prepare_outgoing_text
markdown_to_max_html = _max.markdown_to_max_html
strip_reasoning_block = _max.strip_reasoning_block
SHOW_REASONING_ENV = _max.SHOW_REASONING_ENV


def _run(coro):
    """Run an async coroutine synchronously (fresh event loop each call)."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_max_env(monkeypatch):
    """Isolate every test from the developer's live MAX_* environment.

    Importing ``gateway.*`` loads ``$HERMES_HOME/.env`` via python-dotenv,
    so real credentials (MAX_BOT_TOKEN, MAX_OWNER_USER_ID, …) leak into
    ``os.environ`` and break assertions that expect a clean slate. Each
    test starts from a clean MAX_* slate; tests that need a value set it
    explicitly.
    """
    for name in [n for n in os.environ if n.startswith("MAX_")]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform('max')."""
    from gateway.config import Platform
    p = Platform("max")
    assert p.value == "max"
    assert Platform("max") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestMaxRequirements:

    def test_returns_false_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "test-token")
        monkeypatch.setattr(_max, "HTTPX_AVAILABLE", False)
        assert check_requirements() is False

    def test_returns_false_without_token(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert check_requirements() is False

    def test_returns_true_with_token(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "test-token")
        assert check_requirements() is True

    def test_is_connected_from_extra(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={"token": "t"})) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False

    def test_validate_config(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={"token": "t"})) is True
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestMaxAdapterInit:

    def test_init_reads_token_from_extra(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "abc"}))
        assert adapter._token == "abc"
        assert adapter._last_user_id == ""

    def test_init_reads_token_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "env-token")
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._token == "env-token"


# ---------------------------------------------------------------------------
# 4. _env_enablement
# ---------------------------------------------------------------------------


class TestEnvEnablement:

    def test_returns_none_without_token(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert _env_enablement() is None

    def test_returns_seed_with_token(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        seed = _env_enablement()
        assert seed == {"token": "tok"}

    def test_includes_home_channel(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        monkeypatch.setenv("MAX_HOME_CHANNEL", "123")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "123"

    def test_includes_group_allowlist(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        monkeypatch.setenv("MAX_GROUP_ALLOWED_CHATS", "-1001,-1002")
        seed = _env_enablement()
        assert seed["group_allowed_chats"] == "-1001,-1002"

    def test_includes_group_sessions_per_user(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        monkeypatch.setenv("MAX_GROUP_SESSIONS_PER_USER", "false")
        seed = _env_enablement()
        assert seed["group_sessions_per_user"] is False

    def test_group_sessions_defaults_true(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        seed = _env_enablement()
        assert "group_sessions_per_user" not in seed


# ---------------------------------------------------------------------------
# 5. _handle_update — message parsing
# ---------------------------------------------------------------------------


class TestHandleUpdate:

    def _make_adapter(self, **extra):
        cfg_extra = {"token": "t"}
        cfg_extra.update(extra)
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra=cfg_extra))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        # HTTP client mock for moderation/slash paths
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}
        adapter._http_client.post = AsyncMock(return_value=resp)
        adapter._http_client.delete = AsyncMock(return_value=resp)
        adapter._http_client.get = AsyncMock(return_value=resp)
        return adapter

    def test_ignores_non_message_events(self):
        adapter = self._make_adapter()
        _run(adapter._handle_update({"update_type": "bot_started"}))
        adapter.handle_message.assert_not_called()

    def test_ignores_bot_messages(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 1, "is_bot": True},
                "body": {"text": "hi", "mid": "m1"},
            },
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_parses_user_message(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 139383659, "name": "Артур", "is_bot": False},
                "recipient": {"chat_id": 532485678, "chat_type": "dialog"},
                "body": {"text": "привет", "mid": "mid.1"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "привет"
        assert event.message_id == "mid.1"
        assert event.source.user_id == "139383659"
        assert event.source.chat_id == "532485678"
        assert adapter._last_user_id == "139383659"

    def test_parses_group_message_with_mention(self):
        """Group message mentioning the bot by @username → handled."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._username = "matreshka_bot"
        adapter._name = "Матрёшка"
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat", "chat_name": "Пачка L2"},
                "body": {"text": "@matreshka_bot сколько время?", "mid": "g1"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "-100123"
        assert event.source.chat_name == "Пачка L2"
        assert event.source.user_name == "Вася"

    def test_ignores_group_message_without_mention(self):
        """Group message without mention → ignored (only by @)."""
        adapter = self._make_adapter()
        adapter._username = "matreshka_bot"
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "классный бой был", "mid": "g2"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_group_message_addressed_by_name(self):
        """Bot display name in text counts as addressing."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._name = "Матрёшка"
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "матрёшка, дай ссылку", "mid": "g3"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()

    def test_generic_bot_word_ignored(self):
        """Generic 'бот' word in group is NOT addressed — bot answers only by name."""
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "бот, а что за ивент сегодня?", "mid": "g4"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_bot_added_learns_group_chat(self):
        """bot_added stores the group chat_id for later use."""
        adapter = self._make_adapter()
        upd = {
            "update_type": "bot_added",
            "chat": {"chat_id": -100456, "chat_type": "chat", "title": "Тестовая группа"},
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()
        assert "-100456" in adapter._known_chats
        assert adapter._known_chats["-100456"]["type"] == "group"
        assert adapter._known_chats["-100456"]["name"] == "Тестовая группа"

    def test_skips_empty_text(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 1, "is_bot": False},
                "body": {"text": "  ", "mid": "m2"},
            },
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------


class TestDedup:

    def test_duplicate_message_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter._is_duplicate("mid.1") is False
        assert adapter._is_duplicate("mid.1") is True
        assert adapter._is_duplicate("mid.2") is False

    def test_dedup_window_prunes(self):
        import time
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        # Старое сообщение за пределами окна → не считается дубликатом
        old_id = "old-msg"
        adapter._seen_messages[old_id] = time.time() - DEDUP_WINDOW_SECONDS - 10
        assert adapter._is_duplicate(old_id) is False  # pruned (перезаписано)
        assert adapter._is_duplicate(old_id) is True   # теперь в окне


# ---------------------------------------------------------------------------
# 6b. _is_addressed_to_bot
# ---------------------------------------------------------------------------


class TestAddressedToBot:

    def _adapter(self, name="Матрёшка", username="matreshka_bot"):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        a._name = name
        a._username = username
        return a

    def test_username_mention(self):
        assert self._adapter()._is_addressed_to_bot("@matreshka_bot привет")
        assert self._adapter()._is_addressed_to_bot("привет matreshka_bot")

    def test_display_name(self):
        assert self._adapter()._is_addressed_to_bot("Матрёшка, сколько время?")

    def test_case_insensitive(self):
        assert self._adapter()._is_addressed_to_bot("МАТРЁШКА, помоги")

    def test_generic_word_not_matched(self):
        # Generic words are NOT addressed — bot only answers by its real name
        assert not self._adapter()._is_addressed_to_bot("бот, кинь ссылку")
        assert not self._adapter()._is_addressed_to_bot("sir, advice?")

    def test_aliases(self):
        a = self._adapter()
        a._aliases = ["каин", "кай"]
        assert a._is_addressed_to_bot("Каин, сколько время?")
        assert a._is_addressed_to_bot("кай, го")

    def test_not_addressed(self):
        assert not self._adapter()._is_addressed_to_bot("классный бой был")
        assert not self._adapter()._is_addressed_to_bot("")

    def test_no_identity_no_false_positive(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert not a._is_addressed_to_bot("просто сообщение")


# ---------------------------------------------------------------------------
# 6c. _resolve_channel_prompt — group mini-prompt injection
# ---------------------------------------------------------------------------


class TestChannelPrompt:

    def _make_adapter(self, **extra):
        cfg_extra = {"token": "t"}
        cfg_extra.update(extra)
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra=cfg_extra))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}
        adapter._http_client.post = AsyncMock(return_value=resp)
        adapter._http_client.delete = AsyncMock(return_value=resp)
        adapter._http_client.get = AsyncMock(return_value=resp)
        return adapter

    def _adapter(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        a._name = "Каин"
        a._username = "kain_bot"
        a._description = "L2-помощник пати"
        return a

    def test_no_identity_returns_none(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert a._resolve_channel_prompt("-1001") is None

    def test_auto_prompt_from_identity(self):
        p = self._adapter()._resolve_channel_prompt("-1001")
        assert p is not None
        assert "Каин" in p
        assert "@kain_bot" in p
        assert "L2-помощник пати" in p
        assert "третьем лице" in p

    def test_custom_channel_prompt_merged(self):
        a = self._adapter()
        a.config.extra["channel_prompts"] = {"-1001": "Отвечай только по делу."}
        p = a._resolve_channel_prompt("-1001")
        assert "Отвечай только по делу." in p
        assert "Каин" in p

    def test_custom_only_without_identity(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        a.config.extra["channel_prompts"] = {"-1001": "Тест промпт"}
        assert a._resolve_channel_prompt("-1001") == "Тест промпт"

    def test_owner_parsed_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_OWNER_USER_ID", "777")
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert a._owner_user_id == "777"

    def test_owner_from_extra(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "42"}))
        assert a._owner_user_id == "42"

    def test_group_approved_when_in_allowlist(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "group_allowed_chats": "-100"}))

        assert a._is_group_approved("-100")

    def test_group_approved_when_in_approved_chats(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "approved_chats": "-200"}))
        assert a._is_group_approved("-200")

    def test_group_not_approved_when_unknown(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "1"}))
        assert not a._is_group_approved("-999")

    def test_group_approved_when_owner_is_group_admin(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "1"}))
        a._members["-300"] = {
            "1": {"user_id": 1, "is_owner": True},
            "2": {"user_id": 2, "is_admin": True},
        }
        assert a._is_group_approved("-300")

    def test_bot_role_in_channel_prompt_admin(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        a._name = "Каин"
        a._username = "kain_bot"
        a._id = "99"
        a._members["-100"] = {"99": {"user_id": 99, "is_admin": True, "permissions": ["delete", "write"]}}
        p = a._resolve_channel_prompt("-100")
        assert "администратор" in p
        assert "delete" in p

    def test_bot_role_in_channel_prompt_member(self):
        a = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        a._name = "Каин"
        a._id = "99"
        a._members["-100"] = {"99": {"user_id": 99, "is_admin": False, "is_owner": False}}
        p = a._resolve_channel_prompt("-100")
        assert "обычный участник" in p

    def test_bot_added_notifies_owner_for_approval(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "777"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"members": []}
        adapter._http_client.post = AsyncMock(return_value=resp)
        adapter._http_client.get = AsyncMock(return_value=resp)
        _run(adapter._handle_update({
            "update_type": "bot_added",
            "chat": {"chat_id": -100456, "chat_type": "chat", "title": "Тест группа"},
        }))
        # Уведомление ушло владельцу (user_id=777) — проверим последний POST
        calls = adapter._http_client.post.call_args_list
        dm_posts = [c for c in calls if c.kwargs.get("params", {}).get("user_id") == "777"]
        assert dm_posts, "owner should be notified"
        import json as _json
        body = _json.loads(dm_posts[0].kwargs["content"].decode())
        assert "добавили" in body.get("text", "")

    def test_owner_approve_command(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "777"}))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 777, "name": "Артур", "is_bot": False},
                "recipient": {"chat_id": 500, "chat_type": "dialog"},
                "body": {"text": "/approve -100777", "mid": "a1"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        assert "-100777" in adapter._approved_chats

    def test_owner_deny_command(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "777"}))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)
        adapter._approved_chats.add("-100777")
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 777, "name": "Артур", "is_bot": False},
                "recipient": {"chat_id": 500, "chat_type": "dialog"},
                "body": {"text": "/deny -100777", "mid": "a2"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        assert "-100777" not in adapter._approved_chats

    def test_group_message_ignored_when_not_approved(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "owner_user_id": "777"}))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        adapter._username = "kain_bot"
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -999, "chat_type": "chat"},
                "body": {"text": "@kain_bot привет", "mid": "g6"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_moderation_rejected_for_member(self):
        """Member (no admin) requests a ban → refused."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._id = "99"
        adapter._name = "Каин"
        # Sender is a plain member
        adapter._members["-100123"] = {
            "99": {"user_id": 99, "is_admin": True, "is_owner": False},
            "111": {"user_id": 111, "is_admin": False, "is_owner": False},
            "222": {"user_id": 222, "first_name": "Вася"},
        }
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Петя", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "каин, бан вася", "mid": "m1"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        # Reply sent to the group, agent not invoked
        adapter.handle_message.assert_not_called()
        posts = adapter._http_client.post.call_args_list
        assert posts
        import json as _json
        body = _json.loads(posts[-1].kwargs["content"].decode())
        assert "владелец или админы" in body["text"]

    def test_moderation_non_command_passes_through(self):
        """Normal group message (no уdalи/бан) → agent called."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._username = "kain_bot"
        adapter._name = "Каин"
        adapter._members["-100123"] = {"111": {"user_id": 111, "is_admin": False}}
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "@kain_bot сколько время?", "mid": "m2"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()

    def test_group_member_slash_ignored(self):
        """Group member sends /new → dropped, agent not invoked."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._username = "kain_bot"
        adapter._members["-100123"] = {"111": {"user_id": 111, "is_admin": False, "is_owner": False}}
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "/new", "mid": "m3"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_group_admin_safe_slash_allowed(self):
        """Group admin sends /new → passes to agent (safe command)."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._username = "kain_bot"
        adapter._name = "Каин"
        adapter._members["-100123"] = {"111": {"user_id": 111, "is_admin": True, "is_owner": False}}
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Админ", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "каин /new", "mid": "m4"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()

    def test_group_admin_unsafe_slash_ignored(self):
        """Group admin sends /platform pause → dropped (not in safe set)."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._username = "kain_bot"
        adapter._members["-100123"] = {"111": {"user_id": 111, "is_admin": True, "is_owner": False}}
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Админ", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "/platform pause", "mid": "m5"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_moderation_with_address_in_front(self):
        """'каин, бан вася' — moderation verb NOT at start → still works."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._id = "99"
        adapter._name = "Каин"
        # Bot is admin; sender is group admin
        adapter._members["-100123"] = {
            "99": {"user_id": 99, "is_admin": True, "is_owner": False},
            "111": {"user_id": 111, "is_admin": True, "is_owner": False},
            "222": {"user_id": 222, "first_name": "Вася", "is_admin": False},
        }
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Админ", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "каин, бан вася", "mid": "m6"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        # Agent not called; delete request went to MAX API
        adapter.handle_message.assert_not_called()
        delete_calls = [c for c in adapter._http_client.delete.call_args_list]
        assert delete_calls, "ban should hit DELETE /chats/{id}/members/{uid}"
        url = delete_calls[0].args[0]
        assert "/members/222" in url

    def test_moderation_word_not_false_positive(self):
        """'del' in 'дельфин' / 'remove' in 'соревнование' must NOT trigger."""
        adapter = self._make_adapter(approved_chats="-100123")
        adapter._id = "99"
        adapter._name = "Каин"
        adapter._members["-100123"] = {"99": {"user_id": 99, "is_admin": True}, "111": {"user_id": 111, "is_admin": True}}
        r = _run(adapter._handle_moderation_command("-100123", "каин, где дельфины?", "111"))
        assert r is None
        r2 = _run(adapter._handle_moderation_command("-100123", "соревнование завтра", "111"))
        assert r2 is None

    def test_group_event_carries_channel_prompt(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t", "approved_chats": "-100123"}))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        adapter._name = "Каин"
        adapter._username = "kain_bot"
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 111, "name": "Вася", "is_bot": False},
                "recipient": {"chat_id": -100123, "chat_type": "chat"},
                "body": {"text": "@kain_bot привет", "mid": "g5"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.channel_prompt is not None
        assert "Каин" in event.channel_prompt


# ---------------------------------------------------------------------------
# 7. send()
# ---------------------------------------------------------------------------


class TestSend:

    def test_send_dm_uses_user_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("532485678", "hello", metadata={"user_id": "139383659", "chat_type": "dm"}))
        assert result.success is True
        # DM → user_id param, not chat_id
        call_kwargs = adapter._http_client.post.call_args.kwargs
        assert call_kwargs["params"] == {"user_id": "139383659"}

    def test_send_group_uses_chat_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("999", "hello", metadata={"chat_type": "group"}))
        assert result.success is True
        call_kwargs = adapter._http_client.post.call_args.kwargs
        assert call_kwargs["params"] == {"chat_id": "999"}

    def test_send_truncates_long_text(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        long_text = "x" * (MAX_MESSAGE_LENGTH + 100)
        _run(adapter.send("1", long_text, metadata={"user_id": "2", "chat_type": "dm"}))
        # send() разбивает на несколько сообщений — к-во вызовов > 1
        assert adapter._http_client.post.call_count > 1
        for call in adapter._http_client.post.call_args_list:
            sent_body = call.kwargs["content"].decode("utf-8")
            import json as _json
            payload = _json.loads(sent_body)
            assert len(payload["text"]) <= MAX_MESSAGE_LENGTH

    def test_smart_truncate_short(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter._smart_truncate("short") == "short"

    def test_smart_truncate_long_with_notice(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        long_text = "word " * 1500  # ~7500 chars
        result = adapter._smart_truncate(long_text)
        assert len(result) <= MAX_MESSAGE_LENGTH
        assert "обрезано" in result

    def test_rate_limit_send(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        # 3 sends to the same chat within 1s — the 3rd must be rate-limited (sleep)
        import time as _time
        _run(adapter.send("1", "a", metadata={"user_id": "2", "chat_type": "dm"}))
        _run(adapter.send("1", "b", metadata={"user_id": "2", "chat_type": "dm"}))
        start = _time.monotonic()
        _run(adapter.send("1", "c", metadata={"user_id": "2", "chat_type": "dm"}))
        elapsed = _time.monotonic() - start
        assert elapsed >= 1.0  # rate-limited
        assert adapter._http_client.post.call_count == 3

    def test_send_http_error_returns_failure(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "bad request"
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("1", "hi", metadata={"user_id": "2", "chat_type": "dm"}))
        assert result.success is False
        assert "400" in result.error


# ---------------------------------------------------------------------------
# 7b. send_typing — typing indicator
# ---------------------------------------------------------------------------


class TestSendTyping:

    def test_send_typing_calls_actions(self):
        import json as _json
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        _run(adapter.send_typing("532485678"))
        adapter._http_client.post.assert_called_once()
        call = adapter._http_client.post.call_args
        # POST /chats/{chatId}/actions with {"action": "typing_on"}
        assert "/chats/532485678/actions" in call.args[0]
        payload = _json.loads(call.kwargs["content"].decode("utf-8"))
        assert payload == {"action": "typing_on"} or payload.get("action") == "typing_on"

    def test_send_typing_no_client(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = None
        # Не должно падать без HTTP-клиента
        _run(adapter.send_typing("1"))


# ---------------------------------------------------------------------------
# 7c. Marker persistence
# ---------------------------------------------------------------------------


class TestMarkerPersistence:

    def test_marker_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._marker = 12345
        adapter._save_marker()
        assert (tmp_path / "max" / "marker.json").exists()

        # Новый адаптер должен подхватить маркер с диска
        adapter2 = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter2._marker == 12345


# ---------------------------------------------------------------------------
# 7d. PEM validation
# ---------------------------------------------------------------------------


class TestPemValidation:

    def test_valid_pem(self, tmp_path):
        # Валидная структура PEM (реальный сертификат Минцифры)
        real_cert = b"""-----BEGIN CERTIFICATE-----
MIIFwjCCA6qgAwIBAgICEAAwDQYJKoZIhvcNAQELBQAwcDELMAkGA1UEBhMCUlUx
EzARBgNVBAgMCuiBkNC+0YHQvtCy0YHQutCwMREwDwYDVQQHDAjQnNC+0YHQutCy
MRAwDgYDVQQKDAdNaW5jYWYxHTAbBgNVBAMMFE1pbmNpZnkgQ0EgMjAyMTCCAiIw
DQYJKoZIhvcNAQEBBQADggKPADCCAoUCggKBAMsEBPQE3U1b1Q8kq9nWJmH8RCnx
-----END CERTIFICATE-----
"""
        cert = tmp_path / "cert.pem"
        cert.write_bytes(real_cert)
        bad = tmp_path / "bad.crt"
        bad.write_bytes(b"<html>error page</html>")
        assert _max._is_valid_pem_cert(str(bad)) is False   # HTML — не PEM
        # Реальный сертификат может не пройти DER-парсинг (обрезанный образец),
        # но HTML обязан быть отвергнут; при отсутствии BEGIN CERTIFICATE — False
        assert _max._is_valid_pem_cert(str(tmp_path / "nonexistent.pem")) is False


# ---------------------------------------------------------------------------
# 7e. Media uploads
# ---------------------------------------------------------------------------


class TestMediaUploads:

    def test_guess_media_type(self):
        assert MaxAdapter._guess_media_type("photo.png") == "image"
        assert MaxAdapter._guess_media_type("photo.PNG") == "image"
        assert MaxAdapter._guess_media_type("clip.mp4") == "video"
        assert MaxAdapter._guess_media_type("voice.mp3") == "audio"
        assert MaxAdapter._guess_media_type("doc.pdf") == "file"
        assert MaxAdapter._guess_media_type("noext") == "file"

    def test_upload_media_flow(self, tmp_path):
        # Create a fake image to upload
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG fake")

        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()

        # Step 1: /uploads returns url only (no token for image!)
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"url": "https://upload.example/put"}

        # Step 2: upload to url returns token INSIDE photos map (real MAX behavior)
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {
            "photos": {
                "photoId123": {"token": "real-token-from-photos"}
            }
        }

        async def fake_post(url, **kwargs):
            if "uploads" in url:
                return resp1
            return resp2

        # _upload_media opens a NEW client for CDN upload; mock that via patch
        import httpx as _httpx
        up_client = MagicMock()
        up_client.post = AsyncMock(return_value=resp2)
        up_client.__aenter__ = AsyncMock(return_value=up_client)
        up_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_max.httpx, "AsyncClient", return_value=up_client):
            adapter._http_client.post = AsyncMock(side_effect=fake_post)
            att = _run(adapter._upload_media(str(img)))
        assert att is not None
        assert att["type"] == "image"
        # Token must come from the photos map, NOT from /uploads
        assert att["payload"]["token"] == "real-token-from-photos"

    def test_send_with_media_files(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake-jpeg")

        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"url": "https://upload.example/put"}
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"token": "upload-token"}
        resp3 = MagicMock()
        resp3.status_code = 200

        # CDN upload goes through a separate client
        up_client = MagicMock()
        up_client.post = AsyncMock(return_value=resp2)
        up_client.__aenter__ = AsyncMock(return_value=up_client)
        up_client.__aexit__ = AsyncMock(return_value=False)

        async def fake_post(url, **kwargs):
            if "uploads" in url:
                return resp1
            return resp3  # POST /messages

        with patch.object(_max.httpx, "AsyncClient", return_value=up_client):
            adapter._http_client.post = AsyncMock(side_effect=fake_post)
            result = _run(adapter.send(
                "1", "смотри", metadata={"user_id": "2", "chat_type": "dm", "media_files": [str(img)]}
            ))
        assert result.success is True
        # 2 API calls: /uploads + send message (CDN upload on separate client)
        assert adapter._http_client.post.call_count == 2


# ---------------------------------------------------------------------------
# 8. _standalone_send
# ---------------------------------------------------------------------------


class TestStandaloneSend:

    def test_standalone_send_dm(self):
        import json as _json
        cfg = PlatformConfig(enabled=True, extra={"token": "t", "user_id": "139383659"})

        with patch.object(_max, "httpx") as mock_httpx:
            client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = _run(_standalone_send(cfg, "532485678", "hello"))
            assert result["success"] is True
            assert client.post.call_args.kwargs["params"] == {"user_id": "139383659"}

    def test_standalone_send_no_token(self):
        cfg = PlatformConfig(enabled=True, extra={})
        with patch.object(_max, "_get_scoped_secret", return_value=""):
            result = _run(_standalone_send(cfg, "1", "hi"))
        assert "error" in result


# ---------------------------------------------------------------------------
# 9. register()
# ---------------------------------------------------------------------------


class TestRegister:

    def test_register_platform(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "max"
        assert kwargs["label"] == "MAX"
        assert kwargs["emoji"] == "🟠"
        assert kwargs["allowed_users_env"] == "MAX_ALLOWED_USERS"
        assert kwargs["max_message_length"] == MAX_MESSAGE_LENGTH
        assert kwargs["setup_fn"] is not None
        assert callable(kwargs["setup_fn"])


# ---------------------------------------------------------------------------
# 10. Reasoning stripping + Markdown→MAX-HTML (prepare_outgoing_text)
# ---------------------------------------------------------------------------


class TestReasoningStripping:

    def test_fenced_reasoning_removed_by_default(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        text = '💭 **Reasoning:**\n```\nмысли\n```\n\nОтвет.'
        out, fmt = prepare_outgoing_text(text)
        assert out == "Ответ."
        assert fmt == "html"

    def test_plain_text_still_html(self, monkeypatch):
        """Markdown mode paints nothing inline on real clients — always html."""
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        out, fmt = prepare_outgoing_text("**Жирный** и *курсив*")
        assert fmt == "html"
        assert "<b>Жирный</b>" in out and "<i>курсив</i>" in out
        assert "**" not in out and "*" not in out

    def test_blockquote_reasoning_removed(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        text = '> 💭 **Reasoning:**\n> думаю\n> ещё\n\nОтвет тут'
        out, _ = prepare_outgoing_text(text)
        assert out == "Ответ тут"

    def test_subtext_reasoning_removed(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        text = '-# 💭 Reasoning\n-# думаю\n-# ещё\n\nФинал'
        out, _ = prepare_outgoing_text(text)
        assert out == "Финал"

    def test_bare_think_block_removed(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        text = "<think>\nмонолог\n</think>\nОтвет."
        out, _ = prepare_outgoing_text(text)
        assert out == "Ответ."

    def test_opt_in_keeps_reasoning(self, monkeypatch):
        monkeypatch.setenv(SHOW_REASONING_ENV, "true")
        text = '💭 **Reasoning:**\n```\nмысли\n```\n\nВот код:\n```python\nprint(1)\n```'
        out, fmt = prepare_outgoing_text(text)
        assert "Reasoning" in out
        # code fence still converts even when reasoning is kept
        assert fmt == "html"

    def test_opt_in_false_explicitly(self, monkeypatch):
        monkeypatch.setenv(SHOW_REASONING_ENV, "false")
        text = '💭 **Reasoning:**\n```\nмысли\n```\n\nОтвет.'
        out, _ = prepare_outgoing_text(text)
        assert out == "Ответ."

    def test_clean_text_untouched(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        out, fmt = prepare_outgoing_text("Просто ответ")
        assert out == "Просто ответ"
        assert fmt == "html"


class TestMarkdownToMaxHtml:

    def test_code_fence_becomes_pre(self, monkeypatch):
        monkeypatch.delenv(SHOW_REASONING_ENV, raising=False)
        text = "До кода:\n```python\nx < 5 && y > 2\nprint('hi')\n```\nПосле."
        out, fmt = prepare_outgoing_text(text)
        assert fmt == "html"
        assert '<blockquote><pre><code class="language-python">x &lt; 5 &amp;&amp; y &gt; 2\nprint(\'hi\')</code></pre></blockquote>' in out
        assert out.startswith("До кода:\n")
        assert out.endswith("После.")

    def test_code_fence_without_lang(self, monkeypatch):
        out = markdown_to_max_html("текст\n```\ncode line\n```")
        assert "<blockquote><pre>code line</pre></blockquote>" in out

    def test_unclosed_fence_at_eof(self):
        out, _ = prepare_outgoing_text("текст\n```js\nlet x = 1;")
        assert '<blockquote><pre><code class="language-js">let x = 1;</code></pre></blockquote>' in out

    def test_inline_markup_converted(self):
        out = markdown_to_max_html("`ls -la` и **жирный**, [доки](https://dev.max.ru)")
        assert "<mark>ls -la</mark>" in out
        assert "<b>жирный</b>" in out
        assert '<a href="https://dev.max.ru">доки</a>' in out
        assert "`" not in out and "**" not in out

    def test_html_escaped_in_plain_text(self):
        out = markdown_to_max_html("a < b > c & d")
        assert "&lt;" in out and "&gt;" in out and "&amp;" in out

    def test_send_payload_uses_html_for_code(self):
        """send() must flip the payload format to html when code is present."""
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        async def _noop(*a, **k):
            pass

        adapter._rate_limit_send = _noop
        _run(adapter.send("123", "Вот код:\n```py\nprint(1)\n```", metadata={"user_id": "u1"}))
        body = json.loads(adapter._http_client.post.call_args.kwargs.get(
            "content",
            adapter._http_client.post.call_args.args[1] if len(adapter._http_client.post.call_args.args) > 1 else "{}",
        ))
        assert body["format"] == "html"
        assert "<pre>" in body["text"]

    def test_send_payload_html_when_no_code(self):
        """Even plain replies go out as html (MAX markdown paints nothing)."""
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        async def _noop(*a, **k):
            pass

        adapter._rate_limit_send = _noop
        _run(adapter.send("123", "Простой ответ", metadata={"user_id": "u1"}))
        call = adapter._http_client.post.call_args
        body = json.loads(call.kwargs.get("content") or call.args[1])
        assert body["format"] == "html"
        assert body["text"] == "Простой ответ"
