"""Tests for Matrix require-mention gating and auto-thread features."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

# The matrix adapter module is importable without mautrix installed
# (module-level imports use try/except with stubs).  No need for
# module-level mock installation — tests that call adapter methods
# needing real mautrix APIs mock them individually.


def _make_adapter(tmp_path=None, extra=None):
    """Create a MatrixAdapter with mocked config."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
            **(extra or {}),
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


    def test_matrix_pill_in_formatted_body(self):
        html = '<a href="https://matrix.to/#/@hermes:example.org">Hermes</a> help'
        assert self.adapter._is_bot_mentioned("Hermes help", html)


    # m.mentions.user_ids — MSC3952 / Matrix v1.7 authoritative mentions
    # Ported from openclaw/openclaw#64796

    def test_m_mentions_user_ids_authoritative(self):
        """m.mentions.user_ids alone is sufficient — no body text needed."""
        assert self.adapter._is_bot_mentioned(
            "please reply",  # no @hermes anywhere in body
            mention_user_ids=["@hermes:example.org"],
        )


class TestPeerAgentBudgetConfig:
    def test_defaults_are_inert(self):
        adapter = _make_adapter()

        assert adapter._peer_agent_ids == set()
        assert adapter._peer_reply_budget_per_human_message == 0
        assert adapter._peer_reply_budget_remaining == {}

    def test_parses_peer_ids_from_list(self):
        adapter = _make_adapter(
            extra={
                "peer_agent_ids": ["@ran:example.org", " @yomi:example.org "],
                "peer_reply_budget_per_human_message": 2,
            }
        )

        assert adapter._peer_agent_ids == {
            "@ran:example.org",
            "@yomi:example.org",
        }
        assert adapter._peer_reply_budget_per_human_message == 2

    def test_parses_peer_ids_from_comma_string(self):
        adapter = _make_adapter(
            extra={
                "peer_agent_ids": "@ran:example.org, @yomi:example.org",
                "peer_reply_budget_per_human_message": "3",
            }
        )

        assert adapter._peer_agent_ids == {
            "@ran:example.org",
            "@yomi:example.org",
        }
        assert adapter._peer_reply_budget_per_human_message == 3

    @pytest.mark.parametrize("value", [-1, "invalid", None])
    def test_invalid_budget_fails_closed(self, value):
        adapter = _make_adapter(
            extra={
                "peer_agent_ids": ["@yomi:example.org"],
                "peer_reply_budget_per_human_message": value,
            }
        )

        assert adapter._peer_reply_budget_per_human_message == 0


class TestPeerAgentBudgetGating:
    @staticmethod
    def _adapter(monkeypatch, *, require_mention=False):
        monkeypatch.setenv(
            "MATRIX_ALLOWED_USERS",
            "@alice:example.org,@hermes:example.org,@yomi:example.org",
        )
        monkeypatch.setenv(
            "MATRIX_REQUIRE_MENTION", str(require_mention).lower()
        )
        monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
        return _make_adapter(
            extra={
                "peer_agent_ids": ["@yomi:example.org"],
                "peer_reply_budget_per_human_message": 2,
            }
        )

    @pytest.mark.asyncio
    async def test_authorized_human_untagged_message_resets_budget(self, monkeypatch):
        adapter = self._adapter(monkeypatch)

        await adapter._on_room_message(
            _make_event("hello everyone", sender="@alice:example.org")
        )

        adapter.handle_message.assert_awaited_once()
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 2

    @pytest.mark.asyncio
    async def test_rejected_untagged_human_does_not_reset_budget(self, monkeypatch):
        adapter = self._adapter(monkeypatch, require_mention=True)
        adapter._peer_reply_budget_remaining["!room1:example.org"] = 0

        await adapter._on_room_message(
            _make_event("hello everyone", sender="@alice:example.org")
        )

        adapter.handle_message.assert_not_awaited()
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 0

    @pytest.mark.asyncio
    async def test_unknown_sender_does_not_reset_budget(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        adapter._peer_reply_budget_remaining["!room1:example.org"] = 0

        await adapter._on_room_message(
            _make_event("hello everyone", sender="@mallory:example.org")
        )

        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 0

    @pytest.mark.asyncio
    async def test_untagged_peer_is_dropped_without_consuming_budget(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        adapter._peer_reply_budget_remaining["!room1:example.org"] = 2

        await adapter._on_room_message(
            _make_event("hello sister", sender="@yomi:example.org")
        )

        adapter.handle_message.assert_not_awaited()
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 2

    @pytest.mark.asyncio
    async def test_tagged_peer_is_dropped_before_human_opens_round(self, monkeypatch):
        adapter = self._adapter(monkeypatch)

        await adapter._on_room_message(
            _make_event(
                "@hermes:example.org hello sister",
                sender="@yomi:example.org",
            )
        )

        adapter.handle_message.assert_not_awaited()
        assert adapter._peer_reply_budget_remaining.get("!room1:example.org", 0) == 0

    @pytest.mark.asyncio
    async def test_two_tagged_peer_calls_are_allowed_then_third_is_dropped(
        self, monkeypatch
    ):
        adapter = self._adapter(monkeypatch)
        await adapter._on_room_message(
            _make_event("hello everyone", sender="@alice:example.org", event_id="$human")
        )
        adapter.handle_message.reset_mock()

        for index in range(3):
            await adapter._on_room_message(
                _make_event(
                    "@hermes:example.org continue",
                    sender="@yomi:example.org",
                    event_id=f"$peer{index}",
                )
            )

        assert adapter.handle_message.await_count == 2
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 0
        first_message = adapter.handle_message.await_args_list[0].args[0]
        assert first_message.text == "continue"

    @pytest.mark.asyncio
    async def test_next_authorized_human_message_reopens_round(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        adapter._peer_reply_budget_remaining["!room1:example.org"] = 0

        await adapter._on_room_message(
            _make_event("new subject", sender="@alice:example.org", event_id="$human2")
        )
        adapter.handle_message.reset_mock()
        await adapter._on_room_message(
            _make_event(
                "@hermes:example.org your turn",
                sender="@yomi:example.org",
                event_id="$peer-next",
            )
        )

        adapter.handle_message.assert_awaited_once()
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 1

    @pytest.mark.asyncio
    async def test_peer_command_still_requires_mention_and_budget(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        adapter._peer_reply_budget_remaining["!room1:example.org"] = 2

        await adapter._on_room_message(
            _make_event("/status", sender="@yomi:example.org")
        )

        adapter.handle_message.assert_not_awaited()
        assert adapter._peer_reply_budget_remaining["!room1:example.org"] == 2


class TestStripMention:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_strip_full_user_id(self):
        result = self.adapter._strip_mention("@hermes:example.org help me")
        assert result == "help me"

    def test_localpart_preserved(self):
        """Bare localpart (no @) is preserved — avoids false positives in paths."""
        result = self.adapter._strip_mention("hermes help me")
        assert result == "hermes help me"


# ---------------------------------------------------------------------------
# Outbound mention payloads
# ---------------------------------------------------------------------------


class TestOutboundMentions:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mock_client = MagicMock()
        self.mock_client.send_message_event = AsyncMock(return_value="$evt1")
        self.adapter._client = self.mock_client

    @staticmethod
    def _sent_content(mock_client):
        call_args = mock_client.send_message_event.call_args
        return call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_send_adds_matrix_mentions_and_formatted_body(self):
        result = await self.adapter.send(
            "!room1:example.org",
            "Hello @alice:example.org, please check this.",
        )

        assert result.success is True
        content = self._sent_content(self.mock_client)
        assert content["m.mentions"] == {"user_ids": ["@alice:example.org"]}
        assert content["formatted_body"] == (
            'Hello <a href="https://matrix.to/#/@alice:example.org">'
            "@alice:example.org</a>, please check this."
        )


# ---------------------------------------------------------------------------
# Require-mention gating in _on_room_message
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Auto-thread in _on_room_message
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DM mention-thread feature
# ---------------------------------------------------------------------------


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


