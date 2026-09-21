"""Tests for multi-bot voice floor (turn-taking, echo suppression).

The voice floor uses filesystem-lock files so cross-process coordination
doesn't need shared SQLite (which caused the May-26 kanban DB corruption).
"""

import os
import sys
import time
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Discord mock bootstrap
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

_VC_CHANNEL_ID = 55555
_BOT_A_ID = "111"
_BOT_B_ID = "222"


def _adapter(bot_id: str, tmp_floor_dir: Path, *, multi_agent_vc: int = _VC_CHANNEL_ID) -> DiscordAdapter:
    extra = {"multi_agent_voice_channel_id": multi_agent_vc, "voice_floor_ttl_seconds": 10}
    config = PlatformConfig(enabled=True, token="fake", extra=extra)
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=bot_id))
    # Point floor dir to tmp
    adapter._voice_floor_dir = str(tmp_floor_dir)
    return adapter


# ---------------------------------------------------------------------------
# Floor acquire / release
# ---------------------------------------------------------------------------

class TestVoiceFloor:
    def test_acquire_creates_lock_file(self, tmp_path):
        adapter = _adapter(_BOT_A_ID, tmp_path)
        result = adapter._acquire_voice_floor(_VC_CHANNEL_ID)
        assert result is True
        lock = tmp_path / f"{_VC_CHANNEL_ID}.lock"
        assert lock.exists()
        payload = json.loads(lock.read_bytes())
        assert payload["bot_id"] == _BOT_A_ID

    def test_second_bot_cannot_acquire_held_floor(self, tmp_path):
        bot_a = _adapter(_BOT_A_ID, tmp_path)
        bot_b = _adapter(_BOT_B_ID, tmp_path)
        assert bot_a._acquire_voice_floor(_VC_CHANNEL_ID) is True
        assert bot_b._acquire_voice_floor(_VC_CHANNEL_ID) is False

    def test_release_removes_lock_file(self, tmp_path):
        adapter = _adapter(_BOT_A_ID, tmp_path)
        adapter._acquire_voice_floor(_VC_CHANNEL_ID)
        adapter._release_voice_floor(_VC_CHANNEL_ID)
        lock = tmp_path / f"{_VC_CHANNEL_ID}.lock"
        assert not lock.exists()

    def test_stale_lock_is_reclaimed(self, tmp_path):
        # Write a lock file with ancient mtime
        lock = tmp_path / f"{_VC_CHANNEL_ID}.lock"
        lock.write_bytes(json.dumps({"bot_id": _BOT_B_ID}).encode())
        # Set mtime to 120 seconds ago (beyond 10s TTL)
        old_time = time.time() - 120
        os.utime(str(lock), (old_time, old_time))

        bot_a = _adapter(_BOT_A_ID, tmp_path)
        result = bot_a._acquire_voice_floor(_VC_CHANNEL_ID)
        assert result is True

    def test_held_by_other_true_when_other_holds(self, tmp_path):
        bot_a = _adapter(_BOT_A_ID, tmp_path)
        bot_b = _adapter(_BOT_B_ID, tmp_path)
        bot_a._acquire_voice_floor(_VC_CHANNEL_ID)
        assert bot_b._voice_floor_held_by_other(_VC_CHANNEL_ID) is True

    def test_held_by_other_false_when_self_holds(self, tmp_path):
        adapter = _adapter(_BOT_A_ID, tmp_path)
        adapter._acquire_voice_floor(_VC_CHANNEL_ID)
        assert adapter._voice_floor_held_by_other(_VC_CHANNEL_ID) is False

    def test_held_by_other_false_when_no_lock(self, tmp_path):
        adapter = _adapter(_BOT_A_ID, tmp_path)
        assert adapter._voice_floor_held_by_other(_VC_CHANNEL_ID) is False


# ---------------------------------------------------------------------------
# Non-multi-agent channel: floor logic is bypassed
# ---------------------------------------------------------------------------

class TestFloorGating:
    def test_non_multi_agent_channel_no_floor(self, tmp_path):
        """Floor acquire/release must not fire for 1:1 Misa-Misa channels."""
        adapter = _adapter(_BOT_A_ID, tmp_path, multi_agent_vc=99999)  # different VC
        # Even if we call acquire for a different channel, the listen-loop
        # check only fires when vc_channel_id == multi_agent_voice_channel_id.
        # Verify the guard on _voice_floor_held_by_other for an unrelated channel.
        assert adapter._voice_floor_held_by_other(11111) is False

    def test_adapter_with_no_multi_agent_config_skips_floor(self, tmp_path):
        config = PlatformConfig(enabled=True, token="fake", extra={})
        adapter = DiscordAdapter(config)
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=_BOT_A_ID))
        adapter._voice_floor_dir = str(tmp_path)
        # multi_agent_voice_channel_id is None — held_by_other should return False
        assert adapter._multi_agent_voice_channel_id is None
        assert adapter._voice_floor_held_by_other(_VC_CHANNEL_ID) is False
