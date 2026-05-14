"""Tests for the allowed_{channels,chats,rooms} whitelist extension
added alongside PR #7401 (Slack).

Covers: Telegram, Matrix, Mattermost, DingTalk.

For each platform:
- Empty = no restriction (fully backward compatible).
- When set, messages from non-listed chats/rooms are silently ignored.
- DMs are never filtered.
- @mention does NOT bypass the whitelist.
- config.yaml → env var bridging (via load_gateway_config) where applicable.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _make_telegram_adapter(*, allowed_chats=None, require_mention=None, guest_mode=False):
    from gateway.platforms.telegram import TelegramAdapter

    extra = {"guest_mode": guest_mode}
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if require_mention is not None:
        extra["require_mention"] = require_mention

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._message_handler = AsyncMock()
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _tg_group_message(chat_id=-100, text="hello"):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=chat_id, type="group"),
        from_user=SimpleNamespace(id=111),
        reply_to_message=None,
    )


def _tg_dm_message(text="hello"):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=111, type="private"),
        from_user=SimpleNamespace(id=111),
        reply_to_message=None,
    )


class TestTelegramAllowedChats:
    def test_empty_is_no_restriction(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        adapter = _make_telegram_adapter()
        assert adapter._telegram_allowed_chats() == set()
        assert adapter._should_process_message(_tg_group_message(-100)) is True

    def test_list_form(self):
        adapter = _make_telegram_adapter(allowed_chats=[-100, -200])
        assert adapter._telegram_allowed_chats() == {"-100", "-200"}

    def test_csv_form(self):
        adapter = _make_telegram_adapter(allowed_chats="-100, -200")
        assert adapter._telegram_allowed_chats() == {"-100", "-200"}

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-100,-200")
        adapter = _make_telegram_adapter()  # no extra → falls back to env
        assert adapter._telegram_allowed_chats() == {"-100", "-200"}

    def test_blocks_non_whitelisted_group(self):
        adapter = _make_telegram_adapter(allowed_chats=["-100"])
        assert adapter._should_process_message(_tg_group_message(-999)) is False

    def test_permits_whitelisted_group(self):
        adapter = _make_telegram_adapter(
            allowed_chats=["-100"], require_mention=False,
        )
        assert adapter._should_process_message(_tg_group_message(-100)) is True

    def test_mention_cannot_bypass_whitelist(self):
        """@mention in a non-allowed chat is still ignored."""
        adapter = _make_telegram_adapter(allowed_chats=["-100"])
        msg = _tg_group_message(-999, text="@hermes_bot hello")
        msg.entities = [SimpleNamespace(
            type="mention", offset=0, length=len("@hermes_bot"),
        )]
        assert adapter._should_process_message(msg) is False

    def test_dms_unaffected(self):
        """DMs bypass the allowed_chats whitelist entirely."""
        adapter = _make_telegram_adapter(allowed_chats=["-100"])
        assert adapter._should_process_message(_tg_dm_message()) is True

    def test_config_bridge(self, monkeypatch, tmp_path):
        """slack-style config.yaml → env var bridge works."""
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "telegram:\n"
            "  allowed_chats:\n"
            "    - -100\n"
            "    - -200\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "__sentinel__")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS")

        load_gateway_config()

        import os as _os
        assert _os.environ["TELEGRAM_ALLOWED_CHATS"] == "-100,-200"

    def test_config_bridge_env_takes_precedence(self, monkeypatch, tmp_path):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "telegram:\n"
            "  allowed_chats: -100\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-999")

        load_gateway_config()

        import os as _os
        assert _os.environ["TELEGRAM_ALLOWED_CHATS"] == "-999"


# ---------------------------------------------------------------------------
# DingTalk
# ---------------------------------------------------------------------------

def _make_dingtalk_adapter(*, allowed_chats=None, require_mention=None):
    # Import lazily — DingTalk SDK may not be installed.
    pytest.importorskip("gateway.platforms.dingtalk", reason="DingTalk adapter not importable")
    from gateway.platforms.dingtalk import DingTalkAdapter

    extra = {}
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if require_mention is not None:
        extra["require_mention"] = require_mention

    adapter = object.__new__(DingTalkAdapter)
    adapter.platform = Platform.DINGTALK
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    return adapter


class TestDingTalkAllowedChats:
    def test_empty_is_no_restriction(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_ALLOWED_CHATS", raising=False)
        adapter = _make_dingtalk_adapter()
        assert adapter._dingtalk_allowed_chats() == set()

    def test_list_form(self):
        adapter = _make_dingtalk_adapter(allowed_chats=["cidABC", "cidDEF"])
        assert adapter._dingtalk_allowed_chats() == {"cidABC", "cidDEF"}

    def test_csv_form(self):
        adapter = _make_dingtalk_adapter(allowed_chats="cidABC, cidDEF")
        assert adapter._dingtalk_allowed_chats() == {"cidABC", "cidDEF"}

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_ALLOWED_CHATS", "cidABC,cidDEF")
        adapter = _make_dingtalk_adapter()
        assert adapter._dingtalk_allowed_chats() == {"cidABC", "cidDEF"}

    def test_blocks_non_whitelisted_group(self):
        adapter = _make_dingtalk_adapter(allowed_chats=["cidABC"])
        assert adapter._should_process_message(
            message=None, text="hello", is_group=True, chat_id="cidXYZ",
        ) is False

    def test_dm_unaffected(self):
        """DMs (is_group=False) bypass the whitelist."""
        adapter = _make_dingtalk_adapter(allowed_chats=["cidABC"])
        assert adapter._should_process_message(
            message=None, text="hello", is_group=False, chat_id="cidXYZ",
        ) is True

    def test_config_bridge(self, monkeypatch, tmp_path):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "dingtalk:\n"
            "  allowed_chats:\n"
            "    - cidABC\n"
            "    - cidDEF\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("DINGTALK_ALLOWED_CHATS", "__sentinel__")
        monkeypatch.delenv("DINGTALK_ALLOWED_CHATS")

        load_gateway_config()

        import os as _os
        assert _os.environ["DINGTALK_ALLOWED_CHATS"] == "cidABC,cidDEF"


# ---------------------------------------------------------------------------
# Mattermost (env-var only — no config.yaml bridge)
# ---------------------------------------------------------------------------

class TestMattermostAllowedChannels:
    """Mattermost whitelist logic — replicated since the adapter reads config
    with env-var fallback inline inside _handle_post rather than through a
    helper method."""

    @staticmethod
    def _would_process(channel_id, channel_type="O", allowed_cfg=None, allowed_env=""):
        """Replicate the whitelist gate from gateway/platforms/mattermost.py."""
        import os as _os
        if channel_type == "D":
            return True
        # config-first, env-var fallback (matching the adapter)
        allowed_raw = allowed_cfg
        if allowed_raw is None:
            allowed_raw = allowed_env
        if isinstance(allowed_raw, list):
            allowed = {str(c).strip() for c in allowed_raw if str(c).strip()}
        else:
            allowed = {c.strip() for c in str(allowed_raw).split(",") if c.strip()}
        if allowed and channel_id not in allowed:
            return False
        return True

    def test_empty_config_is_no_restriction(self):
        assert self._would_process("chan123", allowed_cfg=None, allowed_env="") is True

    def test_config_list_blocks_non_whitelisted_channel(self):
        assert self._would_process(
            "chanXYZ", allowed_cfg=["chanABC", "chanDEF"],
        ) is False

    def test_config_list_permits_whitelisted_channel(self):
        assert self._would_process(
            "chanABC", allowed_cfg=["chanABC", "chanDEF"],
        ) is True

    def test_env_var_fallback_when_no_config(self):
        assert self._would_process(
            "chanXYZ", allowed_cfg=None, allowed_env="chanABC,chanDEF",
        ) is False

    def test_dm_unaffected(self):
        assert self._would_process(
            "chanXYZ", channel_type="D", allowed_cfg=["chanABC"],
        ) is True

    def test_config_bridge(self, monkeypatch, tmp_path):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "mattermost:\n"
            "  allowed_channels:\n"
            "    - chanABC\n"
            "    - chanDEF\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Pre-register the key with monkeypatch so teardown cleans it up
        # even though load_gateway_config mutates os.environ directly
        # (monkeypatch only restores keys it's touched via setenv/delenv;
        # delenv on an absent key is a no-op for teardown purposes).
        monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", "__sentinel__")
        monkeypatch.delenv("MATTERMOST_ALLOWED_CHANNELS")

        load_gateway_config()

        import os as _os
        assert _os.environ["MATTERMOST_ALLOWED_CHANNELS"] == "chanABC,chanDEF"


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

class TestMatrixAllowedRooms:
    """Matrix whitelist behavior — tested via the env-var-initialized
    instance attribute _allowed_rooms."""

    def test_empty_env_empty_set(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ALLOWED_ROOMS", raising=False)
        # Replicate __init__ parsing without needing the real adapter.
        raw = "" or ""
        allowed = {r.strip() for r in raw.split(",") if r.strip()}
        assert allowed == set()

    def test_env_var_parsed_to_set(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!room1:srv,!room2:srv")
        import os as _os
        raw = _os.environ["MATRIX_ALLOWED_ROOMS"]
        allowed = {r.strip() for r in raw.split(",") if r.strip()}
        assert allowed == {"!room1:srv", "!room2:srv"}

    def test_block_logic(self):
        """Replicates the matrix.py gate: if allowed non-empty and room not in it, drop."""
        allowed = {"!allowed:srv"}

        # Non-allowed room in group (is_dm=False) → blocked
        def would_process(room_id, is_dm):
            if is_dm:
                return True
            if allowed and room_id not in allowed:
                return False
            return True

        assert would_process("!blocked:srv", is_dm=False) is False
        assert would_process("!allowed:srv", is_dm=False) is True
        # DM always allowed
        assert would_process("!blocked:srv", is_dm=True) is True

    def test_config_bridge(self, monkeypatch, tmp_path):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "matrix:\n"
            "  allowed_rooms:\n"
            "    - '!room1:srv'\n"
            "    - '!room2:srv'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "__sentinel__")
        monkeypatch.delenv("MATRIX_ALLOWED_ROOMS")

        load_gateway_config()

        import os as _os
        assert _os.environ["MATRIX_ALLOWED_ROOMS"] == "!room1:srv,!room2:srv"
