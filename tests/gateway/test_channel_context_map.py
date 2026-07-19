"""Behavior tests for config-driven per-chat inbound context."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import gateway.run as gateway_run
from gateway.channel_context import (
    CONFIGURED_CHANNEL_CONTEXT_HEADER,
    MAX_CHANNEL_CONTEXT_ENTRIES,
    MAX_CHANNEL_CONTEXT_FILE_BYTES,
    MAX_CHANNEL_CONTEXT_VALUE_BYTES,
    ChannelContextResolver,
    canonical_channel_context_key,
    merge_channel_context,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return None

    async def disconnect(self):
        return None

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _adapter(platform: Platform) -> _StubAdapter:
    return _StubAdapter(PlatformConfig(enabled=True, token="test"), platform)


def _source(platform: Platform = Platform.TELEGRAM, chat_id: str = "123") -> SessionSource:
    return SessionSource(platform=platform, chat_id=chat_id, chat_type="dm")


def _runner(*, multiplex_profiles: bool = False) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        group_sessions_per_user=False,
        multiplex_profiles=multiplex_profiles,
    )
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


@pytest.fixture(autouse=True)
def _isolated_runtime_resolver(monkeypatch):
    """Prevent resolver state from leaking between tests in this file."""

    monkeypatch.setattr(
        gateway_run,
        "_channel_context_resolver",
        ChannelContextResolver(),
    )


def _write_runtime_config(home: Path, yaml_body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml_body, encoding="utf-8")


def test_adapter_source_uses_raw_chat_id_but_lookup_key_is_platform_qualified():
    source = _adapter(Platform.TELEGRAM).build_source(123456, chat_type="group")

    assert source.chat_id == "123456"
    assert canonical_channel_context_key(source) == "telegram:123456"


def test_platform_qualified_keys_isolate_identical_raw_chat_ids(tmp_path):
    resolver = ChannelContextResolver()
    config = {
        "telegram:42": "Telegram workspace",
        "discord:42": "Discord workspace",
    }

    assert resolver.context_for(
        _source(Platform.TELEGRAM, "42"), config, config_home=tmp_path
    ) == "Telegram workspace"
    assert resolver.context_for(
        _source(Platform.DISCORD, "42"), config, config_home=tmp_path
    ) == "Discord workspace"
    assert resolver.context_for(
        _source(Platform.SLACK, "42"), config, config_home=tmp_path
    ) is None


def test_canonical_key_preserves_native_colons():
    source = _source(Platform.MATRIX, "!room-id:example.org")

    assert canonical_channel_context_key(source) == "matrix:!room-id:example.org"


def test_merge_marks_configured_context_after_adapter_context():
    merged = merge_channel_context("[Recent messages]\nAlice: earlier", "Workspace A")

    assert merged == (
        "[Recent messages]\nAlice: earlier\n\n"
        f"{CONFIGURED_CHANNEL_CONTEXT_HEADER}\nWorkspace A"
    )


def test_inline_mapping_rejects_invalid_entries_without_logging_secrets(
    tmp_path,
    caplog,
):
    resolver = ChannelContextResolver()
    secret_chat_id = "telegram:998877665544332211"
    secret_context = "do-not-log-this-context"
    config = {
        secret_chat_id: secret_context + ("x" * MAX_CHANNEL_CONTEXT_VALUE_BYTES),
        123: "non-string key",
        "telegram:numeric-value": 123,
        "missing-platform-prefix": "invalid key",
        "telegram:blank": "   ",
        "telegram:valid": "safe",
    }
    caplog.set_level(logging.WARNING, logger="gateway.channel_context")

    assert resolver.context_for(
        _source(chat_id="valid"), config, config_home=tmp_path
    ) == "safe"
    assert resolver.context_for(
        _source(chat_id="998877665544332211"), config, config_home=tmp_path
    ) is None
    assert secret_chat_id not in caplog.text
    assert secret_context not in caplog.text


def test_inline_mapping_over_entry_limit_fails_closed(tmp_path):
    resolver = ChannelContextResolver()
    config = {
        f"telegram:{index}": "context"
        for index in range(MAX_CHANNEL_CONTEXT_ENTRIES + 1)
    }

    assert resolver.context_for(
        _source(chat_id="0"), config, config_home=tmp_path
    ) is None


def test_unchanged_file_is_parsed_only_once(tmp_path):
    resolver = ChannelContextResolver()
    map_file = tmp_path / "chat-context.json"
    map_file.write_text(json.dumps({"telegram:42": "first"}), encoding="utf-8")

    with patch("gateway.channel_context.json.loads", wraps=json.loads) as loads:
        assert resolver.context_for(
            _source(chat_id="42"), str(map_file), config_home=tmp_path
        ) == "first"
        assert resolver.context_for(
            _source(chat_id="42"), str(map_file), config_home=tmp_path
        ) == "first"

    assert loads.call_count == 1


def test_atomic_replace_invalidates_same_mtime_and_size(tmp_path):
    resolver = ChannelContextResolver()
    map_file = tmp_path / "chat-context.json"
    replacement = tmp_path / "chat-context.next"
    first = json.dumps({"telegram:42": "one"})
    second = json.dumps({"telegram:42": "two"})
    assert len(first.encode()) == len(second.encode())
    map_file.write_text(first, encoding="utf-8")

    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) == "one"

    original_stat = map_file.stat()
    replacement.write_text(second, encoding="utf-8")
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    os.replace(replacement, map_file)

    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) == "two"


def test_switching_paths_with_same_mtime_does_not_reuse_other_file(tmp_path):
    resolver = ChannelContextResolver()
    first_file = tmp_path / "first.json"
    second_file = tmp_path / "second.json"
    first_file.write_text(json.dumps({"telegram:42": "one"}), encoding="utf-8")
    second_file.write_text(json.dumps({"telegram:42": "two"}), encoding="utf-8")
    first_stat = first_file.stat()
    os.utime(second_file, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))

    assert resolver.context_for(
        _source(chat_id="42"), str(first_file), config_home=tmp_path
    ) == "one"
    assert resolver.context_for(
        _source(chat_id="42"), str(second_file), config_home=tmp_path
    ) == "two"


def test_file_cache_evicts_least_recently_used_path(tmp_path):
    resolver = ChannelContextResolver(max_cached_files=1)
    first_file = tmp_path / "first.json"
    second_file = tmp_path / "second.json"
    first_file.write_text(json.dumps({"telegram:42": "one"}), encoding="utf-8")
    second_file.write_text(json.dumps({"telegram:42": "two"}), encoding="utf-8")

    assert resolver.context_for(
        _source(chat_id="42"), str(first_file), config_home=tmp_path
    ) == "one"
    assert resolver.context_for(
        _source(chat_id="42"), str(second_file), config_home=tmp_path
    ) == "two"

    with patch("gateway.channel_context.json.loads", wraps=json.loads) as loads:
        assert resolver.context_for(
            _source(chat_id="42"), str(first_file), config_home=tmp_path
        ) == "one"

    assert loads.call_count == 1


def test_missing_or_invalid_file_does_not_reuse_cached_context(tmp_path):
    resolver = ChannelContextResolver()
    map_file = tmp_path / "chat-context.json"
    map_file.write_text(json.dumps({"telegram:42": "stale"}), encoding="utf-8")

    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) == "stale"

    map_file.unlink()
    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) is None

    map_file.write_text("{partial", encoding="utf-8")
    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) is None

    map_file.write_text(json.dumps({"telegram:42": "fresh"}), encoding="utf-8")
    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) == "fresh"


def test_oversized_file_fails_closed(tmp_path):
    resolver = ChannelContextResolver()
    map_file = tmp_path / "chat-context.json"
    map_file.write_bytes(b"x" * (MAX_CHANNEL_CONTEXT_FILE_BYTES + 1))

    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) is None


def test_invalid_file_path_fails_closed(tmp_path):
    resolver = ChannelContextResolver()

    assert resolver.context_for(
        _source(chat_id="42"), "chat\x00context.json", config_home=tmp_path
    ) is None


@pytest.mark.parametrize(
    "payload",
    [
        b"[" * 1200 + b"]" * 1200,
        b'{"telegram:42": ' + (b"9" * 5000) + b"}",
    ],
    ids=["excessive-nesting", "integer-digit-limit"],
)
def test_json_parser_limits_fail_closed(tmp_path, payload):
    resolver = ChannelContextResolver()
    map_file = tmp_path / "chat-context.json"
    map_file.write_bytes(payload)

    assert resolver.context_for(
        _source(chat_id="42"), str(map_file), config_home=tmp_path
    ) is None


@pytest.mark.asyncio
async def test_inline_runtime_config_injects_without_mutating_event(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "default-profile"
    _write_runtime_config(
        home,
        """gateway:
  channel_context_map:
    "telegram:123456": "Bound to workspace A."
""",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    source = _adapter(Platform.TELEGRAM).build_source(
        "123456",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(
        text="status",
        source=source,
        channel_context="[Recent messages]\nBob: earlier",
    )
    runner = _runner()

    first = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )
    second = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    expected = (
        "[Recent messages]\nBob: earlier\n\n"
        f"{CONFIGURED_CHANNEL_CONTEXT_HEADER}\nBound to workspace A.\n\n"
        "[New message]\n[Alice] status"
    )
    assert first == expected
    assert second == expected
    assert event.channel_context == "[Recent messages]\nBob: earlier"


@pytest.mark.asyncio
async def test_file_runtime_config_resolves_relative_to_routed_profile(
    tmp_path,
    monkeypatch,
):
    default_home = tmp_path / "default"
    secondary_home = tmp_path / "profiles" / "secondary"
    _write_runtime_config(
        default_home,
        """gateway:
  channel_context_map:
    "telegram:42": "Wrong profile"
""",
    )
    _write_runtime_config(
        secondary_home,
        """gateway:
  channel_context_map: maps/chat-context.json
""",
    )
    map_dir = secondary_home / "maps"
    map_dir.mkdir()
    (map_dir / "chat-context.json").write_text(
        json.dumps({"telegram:42": "Secondary profile workspace"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)

    source = _adapter(Platform.TELEGRAM).build_source("42")
    source.profile = "secondary"
    runner = _runner(multiplex_profiles=True)
    monkeypatch.setattr(
        runner,
        "_resolve_profile_home_for_source",
        lambda _source: secondary_home,
    )

    result = await runner._prepare_profile_scoped_inbound_message_text(
        event=MessageEvent(text="hello", source=source),
        source=source,
        history=[],
        session_key="agent:secondary:telegram:dm:42",
    )

    assert result == (
        f"{CONFIGURED_CHANNEL_CONTEXT_HEADER}\nSecondary profile workspace\n\n"
        "[New message]\nhello"
    )
