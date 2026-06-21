"""Tests for gateway.channel_context_map — per-chat context injection."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import Platform, SessionSource


# ---------------------------------------------------------------------------
# _load_channel_context_map unit tests
# ---------------------------------------------------------------------------


class TestLoadChannelContextMap:
    """Unit tests for the module-level _load_channel_context_map() helper."""

    def test_no_config_returns_empty(self):
        """When gateway.channel_context_map is absent, return {}."""
        from gateway.run import _load_channel_context_map

        with patch("hermes_cli.config.load_config", return_value={}):
            assert _load_channel_context_map() == {}

    def test_empty_string_returns_empty(self):
        from gateway.run import _load_channel_context_map

        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": ""}},
        ):
            assert _load_channel_context_map() == {}

    def test_inline_dict(self):
        """An inline dict in config is returned directly."""
        from gateway.run import _load_channel_context_map

        ctx = {"telegram:123": "Session A context", "discord:456": "Session B"}
        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": ctx}},
        ):
            result = _load_channel_context_map()
        assert result == ctx

    def test_inline_dict_skips_falsy_values(self):
        from gateway.run import _load_channel_context_map

        ctx = {"a": "ok", "b": "", "c": None}
        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": ctx}},
        ):
            result = _load_channel_context_map()
        assert result == {"a": "ok"}

    def test_file_path_loads_json(self, tmp_path):
        """A string config value is treated as a file path."""
        from gateway.run import _load_channel_context_map

        map_file = tmp_path / "chat-context.json"
        map_file.write_text(json.dumps({"tg:1": "context one"}))

        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(map_file)}},
        ):
            result = _load_channel_context_map()
        assert result == {"tg:1": "context one"}

    def test_file_path_missing_file_returns_empty(self, tmp_path):
        from gateway.run import _load_channel_context_map

        missing = tmp_path / "nonexistent.json"
        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(missing)}},
        ):
            assert _load_channel_context_map() == {}

    def test_file_path_invalid_json_returns_empty(self, tmp_path):
        from gateway.run import _load_channel_context_map

        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{")
        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(bad)}},
        ):
            assert _load_channel_context_map() == {}

    def test_file_path_non_dict_json_returns_empty(self, tmp_path):
        from gateway.run import _load_channel_context_map

        arr = tmp_path / "arr.json"
        arr.write_text(json.dumps(["a", "b"]))
        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(arr)}},
        ):
            assert _load_channel_context_map() == {}

    def test_mtime_cache_reuses_unchanged_file(self, tmp_path):
        """Second call within same mtime returns cached result."""
        import os
        import gateway.run as gr

        map_file = tmp_path / "ctx.json"
        map_file.write_text(json.dumps({"k": "v1"}))

        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(map_file)}},
        ):
            r1 = gr._load_channel_context_map()
            # Rewrite content but preserve the original mtime exactly
            orig_stat = map_file.stat()
            map_file.write_text(json.dumps({"k": "v2"}))
            os.utime(map_file, (orig_stat.st_atime, orig_stat.st_mtime))
            r2 = gr._load_channel_context_map()
        # Same mtime → cached value
        assert r2 == {"k": "v1"}

    def test_mtime_cache_refreshes_on_change(self, tmp_path):
        """File change with new mtime triggers reload."""
        import gateway.run as gr

        map_file = tmp_path / "ctx.json"
        map_file.write_text(json.dumps({"k": "v1"}))

        with patch(
            "hermes_cli.config.load_config",
            return_value={"gateway": {"channel_context_map": str(map_file)}},
        ):
            r1 = gr._load_channel_context_map()
            assert r1 == {"k": "v1"}

            # Force mtime change
            time.sleep(0.05)
            map_file.write_text(json.dumps({"k": "v2"}))
            r2 = gr._load_channel_context_map()
        assert r2 == {"k": "v2"}


# ---------------------------------------------------------------------------
# Integration: _prepare_inbound_message_text with channel_context_map
# ---------------------------------------------------------------------------


class TestChannelContextMapInjection:
    """Integration tests: config map context is injected into inbound messages."""

    @pytest.fixture()
    def runner(self):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        r = GatewayRunner.__new__(GatewayRunner)
        r.config = GatewayConfig(group_sessions_per_user=False)
        r.adapters = {}
        r._model = "test-model"
        r._base_url = ""
        r._has_setup_skill = lambda: False
        return r

    @pytest.fixture()
    def source(self):
        """Group chat source — sender prefix is applied for shared sessions."""
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="tg:12345",
            chat_type="group",
            user_name="Alice",
        )

    @pytest.mark.asyncio
    async def test_config_context_injected(self, runner, source):
        """Config map context is prepended to the message."""
        event = MessageEvent(text="hello", source=source)
        with patch(
            "gateway.run._load_channel_context_map",
            return_value={"tg:12345": "This is dev-session A."},
        ):
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[],
            )
        assert "This is dev-session A." in result
        assert "[New message]" in result
        assert "[Alice] hello" in result

    @pytest.mark.asyncio
    async def test_adapter_context_takes_precedence(self, runner, source):
        """When adapter already set channel_context, config context is appended."""
        event = MessageEvent(
            text="hello",
            source=source,
            channel_context="[Adapter context]",
        )
        with patch(
            "gateway.run._load_channel_context_map",
            return_value={"tg:12345": "Config context."},
        ):
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[],
            )
        # Adapter context comes first
        assert result.startswith("[Adapter context]")
        assert "Config context." in result
        assert "[New message]" in result

    @pytest.mark.asyncio
    async def test_no_config_map_no_injection(self, runner, source):
        """When config map is empty, no extra context is injected."""
        event = MessageEvent(text="hello", source=source)
        with patch(
            "gateway.run._load_channel_context_map",
            return_value={},
        ):
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[],
            )
        assert "[New message]" not in result
        assert result == "[Alice] hello"

    @pytest.mark.asyncio
    async def test_chat_id_not_in_map_no_injection(self, runner, source):
        """When chat_id is not in the map, no extra context is injected."""
        event = MessageEvent(text="hello", source=source)
        with patch(
            "gateway.run._load_channel_context_map",
            return_value={"other:chat": "Some context."},
        ):
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[],
            )
        assert "[New message]" not in result
        assert result == "[Alice] hello"

    @pytest.mark.asyncio
    async def test_config_context_only_no_adapter(self, runner, source):
        """Config context without adapter context works correctly."""
        event = MessageEvent(text="hi", source=source)
        with patch(
            "gateway.run._load_channel_context_map",
            return_value={"tg:12345": "Bound to workspace X."},
        ):
            result = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[],
            )
        assert result.startswith("Bound to workspace X.")
        assert "[New message]" in result
        assert "[Alice] hi" in result
