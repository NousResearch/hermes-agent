"""Tests for Matrix reply_to_mode functionality.

Mirrors test_discord_reply_mode.py / test_telegram_reply_mode.py for the Matrix
adapter's reply-reference behavior control:

- "off": never attach the m.in_reply_to quote anchor (plain messages, no quote pill)
- "first": default; only chunk 0 of a split response carries the anchor
- "all": every chunk of a split response carries the anchor

Threaded sends keep their m.thread relation in every mode: the in_reply_to inside
a thread relation is the spec's fallback for unthreaded clients (is_falling_back),
not a rich reply, so "off" must not strip it.

Also covers config plumbing: PlatformConfig.reply_to_mode (with the YAML 1.1
bare-off bool quirk), the extra fallback, the MATRIX_REPLY_TO_MODE env override,
and the matrix: YAML bridge.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig, GatewayConfig, Platform, _apply_env_overrides
from plugins.platforms.matrix.adapter import MatrixAdapter, _apply_yaml_config


def _make_adapter(reply_to_mode="first", extra=None):
    """MatrixAdapter with the token credential set and no client attached."""
    config_extra = {"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}
    if extra:
        config_extra.update(extra)
    config = PlatformConfig(enabled=True, token="syt_test", reply_to_mode=reply_to_mode, extra=config_extra)
    return MatrixAdapter(config)


def _wired(adapter):
    """Attach a mock mautrix client whose send_message_event captures content."""
    client = MagicMock()
    client.send_message_event = AsyncMock(return_value="$sent")
    adapter._client = client
    return client


def _sent_contents(client) -> list:
    return [call.args[2] for call in client.send_message_event.await_args_list]


class TestReplyToModeConfig:
    """reply_to_mode resolution at __init__ time."""

    def test_default_mode_is_first(self):
        adapter = _make_adapter()
        assert adapter._reply_to_mode == "first"

    def test_off_mode(self):
        adapter = _make_adapter("off")
        assert adapter._reply_to_mode == "off"

    def test_all_mode(self):
        adapter = _make_adapter("all")
        assert adapter._reply_to_mode == "all"

    def test_invalid_falls_back_to_first(self):
        adapter = _make_adapter("banana")
        assert adapter._reply_to_mode == "first"

    def test_bool_false_means_off(self):
        """YAML 1.1 parses a bare `off` as False; the adapter must normalize."""
        adapter = _make_adapter(False)  # type: ignore[arg-value]
        assert adapter._reply_to_mode == "off"

    def test_bool_true_means_all(self):
        adapter = _make_adapter(True)  # type: ignore[arg-value]
        assert adapter._reply_to_mode == "all"

    def test_extra_fallback_resolved_by_bridge(self, monkeypatch):
        """The `extra:` spelling is resolved at config-load time (bridge → env →
        PlatformConfig), not in the adapter — mirror of the discord contract."""
        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
        seeded = _apply_yaml_config({}, {"extra": {"reply_to_mode": "off"}})
        assert seeded is not None and seeded["reply_to_mode"] == "off"
        assert os.environ.get("MATRIX_REPLY_TO_MODE") == "off"


class TestSendWithReplyToMode:
    """send() honors reply_to_mode when building m.relates_to."""

    @pytest.mark.asyncio
    async def test_off_sends_plain_message(self):
        adapter = _make_adapter("off")
        client = _wired(adapter)

        await adapter.send("!room:example.org", "plain answer", reply_to="$trigger")

        (content,) = _sent_contents(client)
        assert "m.relates_to" not in content
        assert content["body"] == "plain answer"

    @pytest.mark.asyncio
    async def test_first_anchors_the_reply(self):
        adapter = _make_adapter("first")
        client = _wired(adapter)

        await adapter.send("!room:example.org", "anchored", reply_to="$trigger")

        (content,) = _sent_contents(client)
        assert content["m.relates_to"] == {"m.in_reply_to": {"event_id": "$trigger"}}

    @pytest.mark.asyncio
    async def test_all_anchors_the_reply(self):
        adapter = _make_adapter("all")
        client = _wired(adapter)

        await adapter.send("!room:example.org", "anchored", reply_to="$trigger")

        (content,) = _sent_contents(client)
        assert content["m.relates_to"] == {"m.in_reply_to": {"event_id": "$trigger"}}

    @pytest.mark.asyncio
    async def test_off_without_reply_to_is_unchanged(self):
        """No reply_to → no relation either way; "off" must not mutate other sends."""
        adapter = _make_adapter("off")
        client = _wired(adapter)

        await adapter.send("!room:example.org", "plain")

        (content,) = _sent_contents(client)
        assert "m.relates_to" not in content

    @pytest.mark.asyncio
    async def test_first_mode_anchors_only_first_chunk(self):
        """Split responses anchor chunk 0 only under "first" (mirrors
        test_telegram_reply_mode.py::test_first_mode_only_first_chunk_threads)."""
        adapter = _make_adapter("first")
        client = _wired(adapter)
        adapter.truncate_message = lambda content, max_length=4096, len_fn=None: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("!room:example.org", "long answer", reply_to="$trigger")

        contents = _sent_contents(client)
        assert len(contents) == 3
        assert contents[0]["m.relates_to"] == {"m.in_reply_to": {"event_id": "$trigger"}}
        assert "m.relates_to" not in contents[1]
        assert "m.relates_to" not in contents[2]

    @pytest.mark.asyncio
    async def test_all_mode_anchors_every_chunk(self):
        adapter = _make_adapter("all")
        client = _wired(adapter)
        adapter.truncate_message = lambda content, max_length=4096, len_fn=None: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("!room:example.org", "long answer", reply_to="$trigger")

        contents = _sent_contents(client)
        assert len(contents) == 3
        for content in contents:
            assert content["m.relates_to"] == {"m.in_reply_to": {"event_id": "$trigger"}}

    @pytest.mark.asyncio
    async def test_off_mode_anchors_no_chunk(self):
        adapter = _make_adapter("off")
        client = _wired(adapter)
        adapter.truncate_message = lambda content, max_length=4096, len_fn=None: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("!room:example.org", "long answer", reply_to="$trigger")

        contents = _sent_contents(client)
        assert len(contents) == 3
        for content in contents:
            assert "m.relates_to" not in content


class TestThreadedSendsKeepRelation:
    """A thread relation is not a rich reply: reply_to_mode "off" must keep it."""

    @pytest.mark.asyncio
    async def test_off_keeps_thread_relation(self):
        adapter = _make_adapter("off")
        client = _wired(adapter)

        await adapter.send(
            "!room:example.org", "in thread",
            reply_to="$root", metadata={"thread_id": "$root"},
        )

        (content,) = _sent_contents(client)
        relates = content["m.relates_to"]
        assert relates["rel_type"] == "m.thread"
        assert relates["event_id"] == "$root"
        assert relates["is_falling_back"] is True
        # Spec fallback for unthreaded clients: still present, pointing at the thread root.
        assert relates["m.in_reply_to"] == {"event_id": "$root"}

    @pytest.mark.asyncio
    async def test_first_keeps_thread_relation(self):
        adapter = _make_adapter("first")
        client = _wired(adapter)

        await adapter.send(
            "!room:example.org", "in thread",
            reply_to="$root", metadata={"thread_id": "$root"},
        )

        (content,) = _sent_contents(client)
        relates = content["m.relates_to"]
        assert relates["rel_type"] == "m.thread"
        assert relates["m.in_reply_to"] == {"event_id": "$root"}

    @pytest.mark.asyncio
    async def test_first_mode_thread_relation_kept_on_every_chunk(self):
        """Thread relations are per-send, not chunk-gated: every chunk of a split
        thread reply keeps its m.thread relation even under "first"."""
        adapter = _make_adapter("first")
        client = _wired(adapter)
        adapter.truncate_message = lambda content, max_length=4096, len_fn=None: ["chunk1", "chunk2"]

        await adapter.send(
            "!room:example.org", "long thread answer",
            reply_to="$root", metadata={"thread_id": "$root"},
        )

        contents = _sent_contents(client)
        assert len(contents) == 2
        for content in contents:
            relates = content["m.relates_to"]
            assert relates["rel_type"] == "m.thread"
            assert relates["event_id"] == "$root"

    @pytest.mark.asyncio
    async def test_off_in_thread_reply_falls_back_to_thread_root(self):
        """Under "off", a reply to a specific in-thread message loses its rich-reply
        anchor; the thread relation survives with its fallback pointing at the root."""
        adapter = _make_adapter("off")
        client = _wired(adapter)

        await adapter.send(
            "!room:example.org", "in thread",
            reply_to="$specific", metadata={"thread_id": "$root"},
        )

        (content,) = _sent_contents(client)
        relates = content["m.relates_to"]
        assert relates["rel_type"] == "m.thread"
        assert relates["event_id"] == "$root"
        assert relates["m.in_reply_to"] == {"event_id": "$root"}


class TestProgressSendsHonorMode:
    """Progress/status bubbles flow through send() with reply_to set — they get
    the same treatment, no special-casing."""

    @pytest.mark.asyncio
    async def test_off_progress_send_is_plain(self):
        adapter = _make_adapter("off")
        client = _wired(adapter)

        await adapter.send("!room:example.org", "thinking...", reply_to="$anchor")

        (content,) = _sent_contents(client)
        assert "m.relates_to" not in content


class TestEnvVarOverride:
    """MATRIX_REPLY_TO_MODE environment variable override."""

    def _make_config(self):
        config = GatewayConfig()
        config.platforms[Platform.MATRIX] = PlatformConfig(
            enabled=True, token="syt_test",
            extra={"homeserver": "https://matrix.example.org"},
        )
        return config

    def test_env_var_sets_off_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"MATRIX_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].reply_to_mode == "off"

    def test_env_var_rejects_invalid_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"MATRIX_REPLY_TO_MODE": "sometimes"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].reply_to_mode == "first"


class TestYamlConfigLoading:
    """matrix.reply_to_mode in config.yaml reaches the adapter."""

    def test_yaml_bool_off_bridges_to_env_as_off(self, tmp_path, monkeypatch):
        """A bare YAML `off` parses as bool False; the bridge must normalize it to
        the string "off" (which _ReplyMode accepts) rather than "false"."""
        from plugins.platforms.matrix.adapter import _YAML_BRIDGE

        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
        seeded = _apply_yaml_config({}, {"reply_to_mode": False})
        assert seeded is not None and seeded["reply_to_mode"] == "off"
        assert os.environ.get("MATRIX_REPLY_TO_MODE") == "off"
        assert ("reply_to_mode", "MATRIX_REPLY_TO_MODE", "lower") in _YAML_BRIDGE

    def test_yaml_string_passes_through(self, monkeypatch):
        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
        seeded = _apply_yaml_config({}, {"reply_to_mode": "first"})
        assert seeded["reply_to_mode"] == "first"
        assert os.environ.get("MATRIX_REPLY_TO_MODE") == "first"

    def test_yaml_all_bool_bridges_to_all(self, monkeypatch):
        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
        seeded = _apply_yaml_config({}, {"reply_to_mode": True})
        assert seeded is not None and seeded["reply_to_mode"] == "all"
        assert os.environ.get("MATRIX_REPLY_TO_MODE") == "all"


class TestConfigSerialization:
    """reply_to_mode round-trips through PlatformConfig (shared machinery)."""

    def test_from_dict_loads_reply_to_mode(self):
        config = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": "off"})
        assert config.reply_to_mode == "off"