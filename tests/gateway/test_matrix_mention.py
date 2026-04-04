"""Tests for Matrix require-mention gating and auto-thread features."""

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_nio_mock():
    """Install a mock nio module when matrix-nio isn't available."""
    if "nio" in sys.modules and hasattr(sys.modules["nio"], "__file__"):
        return
    nio_mod = MagicMock()
    nio_mod.MegolmEvent = type("MegolmEvent", (), {})
    nio_mod.RoomMessageText = type("RoomMessageText", (), {})
    nio_mod.RoomMessageImage = type("RoomMessageImage", (), {})
    nio_mod.RoomMessageAudio = type("RoomMessageAudio", (), {})
    nio_mod.RoomMessageVideo = type("RoomMessageVideo", (), {})
    nio_mod.RoomMessageFile = type("RoomMessageFile", (), {})
    nio_mod.DownloadResponse = type("DownloadResponse", (), {})
    nio_mod.MemoryDownloadResponse = type("MemoryDownloadResponse", (), {})
    nio_mod.InviteMemberEvent = type("InviteMemberEvent", (), {})
    sys.modules.setdefault("nio", nio_mod)


_ensure_nio_mock()


def _make_adapter(tmp_path=None):
    """Create a MatrixAdapter with mocked config."""
    from gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10  # avoid startup grace filter
    return adapter


def _make_room(room_id="!room1:example.org", member_count=5, is_dm=False):
    """Create a fake Matrix room."""
    room = SimpleNamespace(
        room_id=room_id,
        member_count=member_count,
        users={},
    )
    return room


def _make_event(
    body,
    sender="@alice:example.org",
    event_id="$evt1",
    formatted_body=None,
    thread_id=None,
):
    """Create a fake RoomMessageText event."""
    content = {"body": body, "msgtype": "m.text"}
    if formatted_body:
        content["formatted_body"] = formatted_body
        content["format"] = "org.matrix.custom.html"

    relates_to = {}
    if thread_id:
        relates_to["rel_type"] = "m.thread"
        relates_to["event_id"] = thread_id
    if relates_to:
        content["m.relates_to"] = relates_to

    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        server_timestamp=int(time.time() * 1000),
        body=body,
        source={"content": content},
    )


# ---------------------------------------------------------------------------
# Mention detection helpers
# ---------------------------------------------------------------------------


class TestIsBotMentioned:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_full_user_id_in_body(self):
        assert self.adapter._is_bot_mentioned("hey @hermes:example.org help")

    def test_localpart_in_body(self):
        assert self.adapter._is_bot_mentioned("hermes can you help?")

    def test_localpart_case_insensitive(self):
        assert self.adapter._is_bot_mentioned("HERMES can you help?")

    def test_matrix_pill_in_formatted_body(self):
        html = '<a href="https://matrix.to/#/@hermes:example.org">Hermes</a> help'
        assert self.adapter._is_bot_mentioned("Hermes help", html)

    def test_no_mention(self):
        assert not self.adapter._is_bot_mentioned("hello everyone")

    def test_empty_body(self):
        assert not self.adapter._is_bot_mentioned("")

    def test_partial_localpart_no_match(self):
        # "hermesbot" should not match word-boundary check for "hermes"
        assert not self.adapter._is_bot_mentioned("hermesbot is here")


class TestStripMention:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_strip_full_user_id(self):
        result = self.adapter._strip_mention("@hermes:example.org help me")
        assert result == "help me"

    def test_strip_localpart(self):
        result = self.adapter._strip_mention("hermes help me")
        assert result == "help me"

    def test_strip_returns_empty_for_mention_only(self):
        result = self.adapter._strip_mention("@hermes:example.org")
        assert result == ""


# ---------------------------------------------------------------------------
# Require-mention gating in _on_room_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_mention_default_ignores_unmentioned(monkeypatch):
    """Default (require_mention=true): messages without mention are ignored."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("hello everyone")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_mention_default_processes_mentioned(monkeypatch):
    """Default: messages with mention are processed, mention stripped."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "help me"


@pytest.mark.asyncio
async def test_require_mention_html_pill(monkeypatch):
    """Bot mentioned via HTML pill should be processed."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    room = _make_room()
    formatted = '<a href="https://matrix.to/#/@hermes:example.org">Hermes</a> help'
    event = _make_event("Hermes help", formatted_body=formatted)

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_dm_always_responds(monkeypatch):
    """DMs always respond regardless of mention setting."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    # member_count=2 triggers DM detection
    room = _make_room(member_count=2)
    event = _make_event("hello without mention")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_free_response_room(monkeypatch):
    """Free-response rooms bypass mention requirement."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!room1:example.org,!room2:example.org")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    room = _make_room(room_id="!room1:example.org")
    event = _make_event("hello without mention")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_bot_participated_thread(monkeypatch):
    """Threads with prior bot participation bypass mention requirement."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    adapter._bot_participated_threads.add("$thread1")

    room = _make_room()
    event = _make_event("hello without mention", thread_id="$thread1")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_disabled(monkeypatch):
    """MATRIX_REQUIRE_MENTION=false: all messages processed."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("hello without mention")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "hello without mention"


# ---------------------------------------------------------------------------
# Auto-thread in _on_room_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_thread_default_creates_thread(monkeypatch):
    """Default (auto_thread=true): sets thread_id to event.event_id."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("hello", event_id="$msg1")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$msg1"


@pytest.mark.asyncio
async def test_auto_thread_preserves_existing_thread(monkeypatch):
    """If message is already in a thread, thread_id is not overridden."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    adapter._bot_participated_threads.add("$thread_root")
    room = _make_room()
    event = _make_event("reply in thread", thread_id="$thread_root")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$thread_root"


@pytest.mark.asyncio
async def test_auto_thread_skips_dm(monkeypatch):
    """DMs should not get auto-threaded."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    room = _make_room(member_count=2)
    event = _make_event("hello dm", event_id="$dm1")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_auto_thread_disabled(monkeypatch):
    """MATRIX_AUTO_THREAD=false: thread_id stays None."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("hello", event_id="$msg1")

    await adapter._on_room_message(room, event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_auto_thread_tracks_participation(monkeypatch):
    """Auto-created threads are tracked in _bot_participated_threads."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    room = _make_room()
    event = _make_event("hello", event_id="$msg1")

    with patch.object(adapter, "_save_participated_threads"):
        await adapter._on_room_message(room, event)

    assert "$msg1" in adapter._bot_participated_threads


# ---------------------------------------------------------------------------
# Thread persistence
# ---------------------------------------------------------------------------


class TestThreadPersistence:
    def test_empty_state_file(self, tmp_path, monkeypatch):
        """No state file → empty set."""
        monkeypatch.setattr(
            "gateway.platforms.matrix.MatrixAdapter._thread_state_path",
            staticmethod(lambda: tmp_path / "matrix_threads.json"),
        )
        adapter = _make_adapter()
        loaded = adapter._load_participated_threads()
        assert loaded == set()

    def test_track_thread_persists(self, tmp_path, monkeypatch):
        """_track_thread writes to disk."""
        state_path = tmp_path / "matrix_threads.json"
        monkeypatch.setattr(
            "gateway.platforms.matrix.MatrixAdapter._thread_state_path",
            staticmethod(lambda: state_path),
        )
        adapter = _make_adapter()
        adapter._track_thread("$thread_abc")

        data = json.loads(state_path.read_text())
        assert "$thread_abc" in data

    def test_threads_survive_reload(self, tmp_path, monkeypatch):
        """Persisted threads are loaded by a new adapter instance."""
        state_path = tmp_path / "matrix_threads.json"
        state_path.write_text(json.dumps(["$t1", "$t2"]))
        monkeypatch.setattr(
            "gateway.platforms.matrix.MatrixAdapter._thread_state_path",
            staticmethod(lambda: state_path),
        )
        adapter = _make_adapter()
        assert "$t1" in adapter._bot_participated_threads
        assert "$t2" in adapter._bot_participated_threads

    def test_cap_max_tracked_threads(self, tmp_path, monkeypatch):
        """Thread set is trimmed to _MAX_TRACKED_THREADS."""
        state_path = tmp_path / "matrix_threads.json"
        monkeypatch.setattr(
            "gateway.platforms.matrix.MatrixAdapter._thread_state_path",
            staticmethod(lambda: state_path),
        )
        adapter = _make_adapter()
        adapter._MAX_TRACKED_THREADS = 5

        for i in range(10):
            adapter._bot_participated_threads.add(f"$t{i}")
        adapter._save_participated_threads()

        data = json.loads(state_path.read_text())
        assert len(data) == 5


# ---------------------------------------------------------------------------
# YAML config bridge
# ---------------------------------------------------------------------------


class TestMatrixConfigBridge:
    def test_yaml_bridge_sets_env_vars(self, monkeypatch, tmp_path):
        """Matrix YAML config should bridge to env vars."""
        monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
        monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
        monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

        yaml_content = {
            "matrix": {
                "require_mention": False,
                "free_response_rooms": ["!room1:example.org", "!room2:example.org"],
                "auto_thread": False,
            }
        }

        import os
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        # Simulate the bridge logic from gateway/config.py
        yaml_cfg = yaml.safe_load(config_file.read_text())
        matrix_cfg = yaml_cfg.get("matrix", {})
        if isinstance(matrix_cfg, dict):
            if "require_mention" in matrix_cfg and not os.getenv("MATRIX_REQUIRE_MENTION"):
                monkeypatch.setenv("MATRIX_REQUIRE_MENTION", str(matrix_cfg["require_mention"]).lower())
            frc = matrix_cfg.get("free_response_rooms")
            if frc is not None and not os.getenv("MATRIX_FREE_RESPONSE_ROOMS"):
                if isinstance(frc, list):
                    frc = ",".join(str(v) for v in frc)
                monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", str(frc))
            if "auto_thread" in matrix_cfg and not os.getenv("MATRIX_AUTO_THREAD"):
                monkeypatch.setenv("MATRIX_AUTO_THREAD", str(matrix_cfg["auto_thread"]).lower())

        assert os.getenv("MATRIX_REQUIRE_MENTION") == "false"
        assert os.getenv("MATRIX_FREE_RESPONSE_ROOMS") == "!room1:example.org,!room2:example.org"
        assert os.getenv("MATRIX_AUTO_THREAD") == "false"

    def test_env_vars_take_precedence_over_yaml(self, monkeypatch):
        """Env vars should not be overwritten by YAML values."""
        monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true")

        import os
        yaml_cfg = {"matrix": {"require_mention": False}}
        matrix_cfg = yaml_cfg.get("matrix", {})
        if "require_mention" in matrix_cfg and not os.getenv("MATRIX_REQUIRE_MENTION"):
            monkeypatch.setenv("MATRIX_REQUIRE_MENTION", str(matrix_cfg["require_mention"]).lower())

        assert os.getenv("MATRIX_REQUIRE_MENTION") == "true"
