"""Tests for Matrix require-mention gating and auto-thread features."""

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

# The matrix adapter module is importable without mautrix installed
# (module-level imports use try/except with stubs).  No need for
# module-level mock installation — tests that call adapter methods
# needing real mautrix APIs mock them individually.


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
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10  # avoid startup grace filter
    return adapter


def _set_dm(adapter, room_id="!room1:example.org", is_dm=True):
    """Mark a room as DM (or not) in the adapter's cache."""
    adapter._dm_rooms[room_id] = is_dm


def _make_event(
    body,
    sender="@alice:example.org",
    event_id="$evt1",
    room_id="!room1:example.org",
    formatted_body=None,
    thread_id=None,
    mention_user_ids=None,
):
    """Create a fake room message event.

    The mautrix adapter reads ``event.room_id``, ``event.sender``,
    ``event.event_id``, ``event.timestamp``, and ``event.content``
    (a dict with ``msgtype``, ``body``, etc.).
    """
    content = {"body": body, "msgtype": "m.text"}
    if formatted_body:
        content["formatted_body"] = formatted_body
        content["format"] = "org.matrix.custom.html"

    if mention_user_ids is not None:
        content["m.mentions"] = {"user_ids": mention_user_ids}

    relates_to = {}
    if thread_id:
        relates_to["rel_type"] = "m.thread"
        relates_to["event_id"] = thread_id
    if relates_to:
        content["m.relates_to"] = relates_to

    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id=room_id,
        timestamp=int(time.time() * 1000),
        content=content,
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

    # m.mentions.user_ids — MSC3952 / Matrix v1.7 authoritative mentions
    # Ported from openclaw/openclaw#64796

    def test_m_mentions_user_ids_authoritative(self):
        """m.mentions.user_ids alone is sufficient — no body text needed."""
        assert self.adapter._is_bot_mentioned(
            "please reply",  # no @hermes anywhere in body
            mention_user_ids=["@hermes:example.org"],
        )

    def test_m_mentions_user_ids_with_body_mention(self):
        """Both m.mentions and body mention — should still be True."""
        assert self.adapter._is_bot_mentioned(
            "hey @hermes:example.org help",
            mention_user_ids=["@hermes:example.org"],
        )

    def test_m_mentions_user_ids_other_user_only(self):
        """m.mentions with a different user — bot is NOT mentioned."""
        assert not self.adapter._is_bot_mentioned(
            "hello",
            mention_user_ids=["@alice:example.org"],
        )

    def test_m_mentions_user_ids_empty_list(self):
        """Empty user_ids list — falls through to text detection."""
        assert not self.adapter._is_bot_mentioned(
            "hello everyone",
            mention_user_ids=[],
        )

    def test_m_mentions_user_ids_none(self):
        """None mention_user_ids — falls through to text detection."""
        assert not self.adapter._is_bot_mentioned(
            "hello everyone",
            mention_user_ids=None,
        )


class TestStripMention:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_strip_full_user_id(self):
        result = self.adapter._strip_mention("@hermes:example.org help me")
        assert result == "help me"

    def test_localpart_preserved(self):
        """Localpart-only text is no longer stripped — avoids false positives in paths."""
        result = self.adapter._strip_mention("hermes help me")
        assert result == "hermes help me"

    def test_localpart_in_path_preserved(self):
        """Localpart inside a file path must not be damaged."""
        result = self.adapter._strip_mention("read /home/hermes/config.yaml")
        assert result == "read /home/hermes/config.yaml"

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
    event = _make_event("hello everyone")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_mention_default_processes_mentioned(monkeypatch):
    """Default: messages with mention are processed, mention stripped."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(event)
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
    formatted = '<a href="https://matrix.to/#/@hermes:example.org">Hermes</a> help'
    event = _make_event("Hermes help", formatted_body=formatted)

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_m_mentions_user_ids(monkeypatch):
    """m.mentions.user_ids is authoritative per MSC3952 — no body mention needed.

    Ported from openclaw/openclaw#64796.
    """
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    # Body has NO mention, but m.mentions.user_ids includes the bot.
    event = _make_event(
        "please reply",
        mention_user_ids=["@hermes:example.org"],
    )

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_m_mentions_other_user_ignored(monkeypatch):
    """m.mentions.user_ids mentioning another user should NOT activate the bot."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event(
        "hey alice check this",
        mention_user_ids=["@alice:example.org"],
    )

    await adapter._on_room_message(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_mention_dm_always_responds(monkeypatch):
    """DMs always respond regardless of mention setting."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    # Mark the room as a DM via the adapter's cache.
    _set_dm(adapter)
    event = _make_event("hello without mention")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_dm_strips_full_mxid(monkeypatch):
    """DMs strip the full MXID from body when require_mention is on (default)."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "help me"


@pytest.mark.asyncio
async def test_dm_preserves_localpart_in_body(monkeypatch):
    """DMs no longer strip bare localpart — only the full MXID is removed."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("hermes help me")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "hermes help me"


@pytest.mark.asyncio
async def test_bare_mention_passes_empty_string(monkeypatch):
    """A message that is only a mention should pass through as empty, not be dropped."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("@hermes:example.org")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == ""


@pytest.mark.asyncio
async def test_require_mention_free_response_room(monkeypatch):
    """Free-response rooms bypass mention requirement."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.setenv(
        "MATRIX_FREE_RESPONSE_ROOMS", "!room1:example.org,!room2:example.org"
    )
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("hello without mention", room_id="!room1:example.org")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_bot_participated_thread(monkeypatch):
    """Threads with prior bot participation bypass mention requirement."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    adapter._threads.mark("$thread1")

    event = _make_event("hello without mention", thread_id="$thread1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_disabled(monkeypatch):
    """MATRIX_REQUIRE_MENTION=false: all messages processed."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("hello without mention")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "hello without mention"


@pytest.mark.asyncio
async def test_require_mention_disabled_skips_stripping(monkeypatch):
    """MATRIX_REQUIRE_MENTION=false: mention text is NOT stripped from body."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_FREE_RESPONSE_ROOMS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("@hermes:example.org help me")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "@hermes:example.org help me"


# ---------------------------------------------------------------------------
# Auto-thread in _on_room_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_thread_default_creates_thread(monkeypatch):
    """Default (auto_thread=true): sets thread_id to event.event_id."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    event = _make_event("hello", event_id="$msg1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$msg1"


@pytest.mark.asyncio
async def test_auto_thread_preserves_existing_thread(monkeypatch):
    """If message is already in a thread, thread_id is not overridden."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    adapter._threads.mark("$thread_root")
    event = _make_event("reply in thread", thread_id="$thread_root")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$thread_root"


@pytest.mark.asyncio
async def test_auto_thread_skips_dm(monkeypatch):
    """DMs should not get auto-threaded."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("hello dm", event_id="$dm1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_auto_thread_disabled(monkeypatch):
    """MATRIX_AUTO_THREAD=false: thread_id stays None."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    event = _make_event("hello", event_id="$msg1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_auto_thread_tracks_participation(monkeypatch):
    """Auto-created threads are tracked in _threads."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.delenv("MATRIX_AUTO_THREAD", raising=False)

    adapter = _make_adapter()
    event = _make_event("hello", event_id="$msg1")

    with patch.object(adapter._threads, "_save"):
        await adapter._on_room_message(event)

    assert "$msg1" in adapter._threads


# ---------------------------------------------------------------------------
# Thread persistence
# ---------------------------------------------------------------------------


class TestThreadPersistence:
    def test_empty_state_file(self, tmp_path, monkeypatch):
        """No state file → empty set."""
        from gateway.platforms.helpers import ThreadParticipationTracker

        monkeypatch.setattr(
            ThreadParticipationTracker,
            "_state_path",
            lambda self: tmp_path / "matrix_threads.json",
        )
        adapter = _make_adapter()
        assert "$nonexistent" not in adapter._threads

    def test_track_thread_persists(self, tmp_path, monkeypatch):
        """mark() writes to disk."""
        from gateway.platforms.helpers import ThreadParticipationTracker

        state_path = tmp_path / "matrix_threads.json"
        monkeypatch.setattr(
            ThreadParticipationTracker,
            "_state_path",
            lambda self: state_path,
        )
        adapter = _make_adapter()
        adapter._threads.mark("$thread_abc")

        data = json.loads(state_path.read_text())
        assert "$thread_abc" in data

    def test_threads_survive_reload(self, tmp_path, monkeypatch):
        """Persisted threads are loaded by a new adapter instance."""
        from gateway.platforms.helpers import ThreadParticipationTracker

        state_path = tmp_path / "matrix_threads.json"
        state_path.write_text(json.dumps(["$t1", "$t2"]))
        monkeypatch.setattr(
            ThreadParticipationTracker,
            "_state_path",
            lambda self: state_path,
        )
        adapter = _make_adapter()
        assert "$t1" in adapter._threads
        assert "$t2" in adapter._threads

    def test_cap_max_tracked_threads(self, tmp_path, monkeypatch):
        """Thread set is trimmed to max_tracked."""
        from gateway.platforms.helpers import ThreadParticipationTracker

        state_path = tmp_path / "matrix_threads.json"
        monkeypatch.setattr(
            ThreadParticipationTracker,
            "_state_path",
            lambda self: state_path,
        )
        adapter = _make_adapter()
        adapter._threads._max_tracked = 5

        for i in range(10):
            adapter._threads.mark(f"$t{i}")

        data = json.loads(state_path.read_text())
        assert len(data) == 5


# ---------------------------------------------------------------------------
# DM mention-thread feature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_mention_thread_disabled_by_default(monkeypatch):
    """Default (dm_mention_threads=false): DM with mention should NOT create a thread."""
    monkeypatch.delenv("MATRIX_DM_MENTION_THREADS", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help me", event_id="$dm1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_dm_mention_thread_creates_thread(monkeypatch):
    """MATRIX_DM_MENTION_THREADS=true: DM with @mention creates a thread."""
    monkeypatch.setenv("MATRIX_DM_MENTION_THREADS", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help me", event_id="$dm1")

    with patch.object(adapter._threads, "_save"):
        await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$dm1"
    assert msg.text == "help me"


@pytest.mark.asyncio
async def test_dm_mention_thread_no_mention_no_thread(monkeypatch):
    """MATRIX_DM_MENTION_THREADS=true: DM without mention does NOT create a thread."""
    monkeypatch.setenv("MATRIX_DM_MENTION_THREADS", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("hello without mention", event_id="$dm1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id is None


@pytest.mark.asyncio
async def test_dm_mention_thread_preserves_existing_thread(monkeypatch):
    """MATRIX_DM_MENTION_THREADS=true: DM already in a thread keeps that thread_id."""
    monkeypatch.setenv("MATRIX_DM_MENTION_THREADS", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    adapter._threads.mark("$existing_thread")
    event = _make_event("@hermes:example.org help me", thread_id="$existing_thread")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$existing_thread"


@pytest.mark.asyncio
async def test_dm_mention_thread_tracks_participation(monkeypatch):
    """DM mention-thread tracks the thread in _threads."""
    monkeypatch.setenv("MATRIX_DM_MENTION_THREADS", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")

    adapter = _make_adapter()
    _set_dm(adapter)
    event = _make_event("@hermes:example.org help", event_id="$dm1")

    with patch.object(adapter._threads, "_save"):
        await adapter._on_room_message(event)

    assert "$dm1" in adapter._threads


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
            if "require_mention" in matrix_cfg and not os.getenv(
                "MATRIX_REQUIRE_MENTION"
            ):
                monkeypatch.setenv(
                    "MATRIX_REQUIRE_MENTION", str(matrix_cfg["require_mention"]).lower()
                )
            frc = matrix_cfg.get("free_response_rooms")
            if frc is not None and not os.getenv("MATRIX_FREE_RESPONSE_ROOMS"):
                if isinstance(frc, list):
                    frc = ",".join(str(v) for v in frc)
                monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", str(frc))
            if "auto_thread" in matrix_cfg and not os.getenv("MATRIX_AUTO_THREAD"):
                monkeypatch.setenv(
                    "MATRIX_AUTO_THREAD", str(matrix_cfg["auto_thread"]).lower()
                )

        assert os.getenv("MATRIX_REQUIRE_MENTION") == "false"
        assert (
            os.getenv("MATRIX_FREE_RESPONSE_ROOMS")
            == "!room1:example.org,!room2:example.org"
        )
        assert os.getenv("MATRIX_AUTO_THREAD") == "false"

    def test_yaml_bridge_sets_dm_mention_threads(self, monkeypatch, tmp_path):
        """Matrix YAML dm_mention_threads should bridge to env var."""
        monkeypatch.delenv("MATRIX_DM_MENTION_THREADS", raising=False)

        import os

        import yaml

        yaml_content = {"matrix": {"dm_mention_threads": True}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        yaml_cfg = yaml.safe_load(config_file.read_text())
        matrix_cfg = yaml_cfg.get("matrix", {})
        if isinstance(matrix_cfg, dict):
            if "dm_mention_threads" in matrix_cfg and not os.getenv(
                "MATRIX_DM_MENTION_THREADS"
            ):
                monkeypatch.setenv(
                    "MATRIX_DM_MENTION_THREADS",
                    str(matrix_cfg["dm_mention_threads"]).lower(),
                )

        assert os.getenv("MATRIX_DM_MENTION_THREADS") == "true"

    def test_env_vars_take_precedence_over_yaml(self, monkeypatch):
        """Env vars should not be overwritten by YAML values."""
        monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true")

        import os

        yaml_cfg = {"matrix": {"require_mention": False}}
        matrix_cfg = yaml_cfg.get("matrix", {})
        if "require_mention" in matrix_cfg and not os.getenv("MATRIX_REQUIRE_MENTION"):
            monkeypatch.setenv(
                "MATRIX_REQUIRE_MENTION", str(matrix_cfg["require_mention"]).lower()
            )

        assert os.getenv("MATRIX_REQUIRE_MENTION") == "true"
