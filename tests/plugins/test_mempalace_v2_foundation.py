"""Foundation tests for the modular MemPalace plugin rewrite.

These tests cover the first extraction wave only:
- error model
- config parsing
- collection naming / room derivation
- metadata schema builder
- plugin package export / register hook
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest  # type: ignore[unresolved-import]
import yaml

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.mempalace import MemPalaceMemoryProvider, register
from plugins.memory.mempalace.collections import (
    resolve_collection_name,
    resolve_room,
    slugify_identifier,
)
from plugins.memory.mempalace.config import (
    DEFAULT_COLLECTION_TEMPLATE,
    DEFAULT_FIXED_ROOM,
    DEFAULT_N_RESULTS,
    DEFAULT_ROOM_STRATEGY,
    DEFAULT_TOOL_MAX_RESULTS,
    DEFAULT_WING,
    MemPalaceConfig,
    load_mempalace_config,
)
from plugins.memory.mempalace.errors import (
    MEMPALACE_CONFIG_INVALID,
    MEMPALACE_QUERY_FAILED,
    MemPalaceConfigError,
    MemPalaceError,
    MemPalaceToolError,
)
from plugins.memory.mempalace.metadata import build_metadata
from plugins.memory.mempalace.store import make_drawer_id


class TestErrorModel:
    def test_base_error_exposes_code_message_and_details(self):
        err = MemPalaceError(
            code=MEMPALACE_QUERY_FAILED,
            message="query exploded",
            details={"room": "prefs"},
        )

        assert err.code == MEMPALACE_QUERY_FAILED
        assert err.message == "query exploded"
        assert err.details == {"room": "prefs"}
        assert err.to_dict()["error"]["code"] == MEMPALACE_QUERY_FAILED

    def test_config_error_uses_default_code(self):
        err = MemPalaceConfigError("bad config")
        assert err.code == MEMPALACE_CONFIG_INVALID
        assert err.message == "bad config"

    def test_tool_error_inherits_base_shape(self):
        err = MemPalaceToolError("tool failed")
        payload = err.to_dict()
        assert payload["success"] is False
        assert payload["error"]["message"] == "tool failed"


class TestConfigParsing:
    def test_loads_nested_plugin_config(self, tmp_path):
        cfg = load_mempalace_config(
            {
                "mempalace": {
                    "wing": "projects",
                    "n_results": 9,
                    "enable_kg": False,
                    "palace_path": "~/mp-test",
                    "collection_template": "agent_{user_id}",
                    "room_strategy": "platform_session",
                }
            },
            hermes_home=str(tmp_path / ".hermes"),
        )

        assert cfg.wing == "projects"
        assert cfg.n_results == 9
        assert cfg.enable_kg is False
        assert cfg.collection_template == "agent_{user_id}"
        assert cfg.room_strategy == "platform_session"
        assert cfg.palace_path.endswith("mp-test")

    def test_supports_flat_config_fallback(self, tmp_path):
        cfg = load_mempalace_config(
            {"wing": "flat-wing", "n_results": 7},
            hermes_home=str(tmp_path / ".hermes"),
        )
        assert cfg.wing == "flat-wing"
        assert cfg.n_results == 7

    def test_invalid_values_fall_back_to_defaults(self, tmp_path):
        cfg = load_mempalace_config(
            {
                "mempalace": {
                    "n_results": 0,
                    "tool_max_results": -1,
                    "enable_kg": "not-bool",
                    "room_strategy": "nope",
                }
            },
            hermes_home=str(tmp_path / ".hermes"),
        )

        assert cfg.n_results == 5
        assert cfg.tool_max_results == 20
        assert cfg.enable_kg is True
        assert cfg.room_strategy == "platform_session"

    def test_tool_max_results_is_raised_to_n_results(self, tmp_path):
        cfg = load_mempalace_config(
            {"mempalace": {"n_results": 11, "tool_max_results": 3}},
            hermes_home=str(tmp_path / ".hermes"),
        )
        assert cfg.n_results == 11
        assert cfg.tool_max_results == 11

    def test_invalid_collection_template_reverts_to_default(self, tmp_path):
        cfg = load_mempalace_config(
            {"mempalace": {"collection_template": "broken-{unknown}"}},
            hermes_home=str(tmp_path / ".hermes"),
        )
        assert cfg.collection_template == "hermes-{platform}-{user_id}"

    def test_collection_name_and_fixed_room_are_sanitized(self, tmp_path):
        cfg = load_mempalace_config(
            {
                "mempalace": {
                    "collection_name": "  Custom Room/Name  ",
                    "fixed_room": " My Room #1 ",
                }
            },
            hermes_home=str(tmp_path / ".hermes"),
        )
        assert cfg.collection_name == "custom-room-name"
        assert cfg.fixed_room == "my-room-1"

    def test_default_path_uses_hermes_home_when_missing(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        cfg = load_mempalace_config({}, hermes_home=str(hermes_home))
        assert cfg.palace_path == str(hermes_home / "mempalace")


class TestCollections:
    def test_slugify_identifier_normalizes_text(self):
        assert (
            slugify_identifier(" Jessica / Telegram Topic #1 ")
            == "jessica-telegram-topic-1"
        )

    def test_slugify_identifier_truncates_and_falls_back_when_needed(self):
        assert slugify_identifier("!!!") == "default"
        assert len(slugify_identifier("A" * 200)) <= 63

    def test_resolve_collection_name_prefers_explicit_name(self):
        cfg = MemPalaceConfig(collection_name="Custom Collection")
        name = resolve_collection_name(cfg, {"user_id": "jessica"})
        assert name == "custom-collection"

    def test_resolve_collection_name_renders_template(self):
        cfg = MemPalaceConfig(collection_template="hermes-{platform}-{user_id}")
        name = resolve_collection_name(
            cfg, {"platform": "telegram", "user_id": "Jessica 123"}
        )
        assert name == "hermes-telegram-jessica-123"

    def test_resolve_collection_name_uses_agent_and_session_fields(self):
        cfg = MemPalaceConfig(collection_template="mem-{agent_id}-{session_id}")
        name = resolve_collection_name(
            cfg, {"agent_id": "Hermes Bot", "session_id": "Thread 42"}
        )
        assert name == "mem-hermes-bot-thread-42"

    def test_resolve_collection_name_falls_back_without_user_id(self):
        cfg = MemPalaceConfig(collection_template="hermes-{user_id}")
        name = resolve_collection_name(cfg, {"session_id": "abc123"})
        assert name == "hermes-default"

    def test_resolve_room_uses_explicit_room_first(self):
        cfg = MemPalaceConfig(room_strategy="session")
        room = resolve_room(
            cfg, {"session_id": "s1"}, explicit_room="Project X / Notes"
        )
        assert room == "project-x-notes"

    @pytest.mark.parametrize(
        ("strategy", "runtime_ctx", "expected"),
        [
            ("fixed", {}, "memory"),
            ("session", {"session_id": "sess-9"}, "sess-9"),
            (
                "platform_session",
                {"platform": "telegram", "session_id": "sess-9"},
                "telegram-sess-9",
            ),
            (
                "user_platform",
                {"user_id": "jessica", "platform": "telegram"},
                "jessica-telegram",
            ),
        ],
    )
    def test_resolve_room_strategies(self, strategy, runtime_ctx, expected):
        cfg = MemPalaceConfig(room_strategy=strategy, fixed_room="memory")
        assert resolve_room(cfg, runtime_ctx) == expected


class TestMetadata:
    def test_build_metadata_contains_required_fields(self):
        data = build_metadata(
            {
                "session_id": "sess-1",
                "platform": "telegram",
                "user_id": "u1",
                "agent_id": "hermes",
            },
            room="project-x",
            source="tool",
            message_kind="explicit_memory",
            memory_type="preference",
            importance=0.9,
        )

        assert data["room"] == "project-x"
        assert data["source"] == "tool"
        assert data["message_kind"] == "explicit_memory"
        assert data["memory_type"] == "preference"
        assert data["importance"] == 0.9
        assert data["session_id"] == "sess-1"
        assert data["platform"] == "telegram"
        assert data["user_id"] == "u1"
        assert data["agent_id"] == "hermes"
        assert datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

    def test_build_metadata_omits_optional_fields_when_not_provided(self):
        data = build_metadata(
            {"session_id": None, "platform": None, "user_id": None, "agent_id": None},
            room="memory",
            source="compression",
            message_kind="compressed_context",
        )

        assert data["room"] == "memory"
        assert data["source"] == "compression"
        assert data["message_kind"] == "compressed_context"
        assert data["session_id"] == ""
        assert data["platform"] == ""
        assert data["user_id"] == ""
        assert data["agent_id"] == ""
        assert "memory_type" not in data
        assert "importance" not in data


class TestStore:
    def test_make_drawer_id_includes_session_and_content_signal(self):
        base = dict(
            wing="conversations",
            room="telegram-s1",
            source_file="conversation",
            chunk_index=0,
        )
        typed_base: dict[str, Any] = base
        id1 = make_drawer_id(
            wing=str(typed_base["wing"]),
            room=str(typed_base["room"]),
            source_file=str(typed_base["source_file"]),
            chunk_index=int(typed_base["chunk_index"]),
            content="alpha",
            session_id="s1",
        )
        id2 = make_drawer_id(
            wing=str(typed_base["wing"]),
            room=str(typed_base["room"]),
            source_file=str(typed_base["source_file"]),
            chunk_index=int(typed_base["chunk_index"]),
            content="beta",
            session_id="s1",
        )
        id3 = make_drawer_id(
            wing=str(typed_base["wing"]),
            room=str(typed_base["room"]),
            source_file=str(typed_base["source_file"]),
            chunk_index=int(typed_base["chunk_index"]),
            content="alpha",
            session_id="s2",
        )
        assert id1 != id2
        assert id1 != id3


class TestPluginManifest:
    def test_plugin_yaml_defaults_match_runtime_config(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "plugins"
            / "memory"
            / "mempalace"
            / "plugin.yaml"
        )
        manifest = yaml.safe_load(manifest_path.read_text())
        defaults = {item["key"]: item.get("default") for item in manifest["config"]}

        assert defaults["wing"] == DEFAULT_WING
        assert defaults["n_results"] == DEFAULT_N_RESULTS
        assert defaults["tool_max_results"] == DEFAULT_TOOL_MAX_RESULTS
        assert defaults["collection_template"] == DEFAULT_COLLECTION_TEMPLATE
        assert defaults["room_strategy"] == DEFAULT_ROOM_STRATEGY
        assert defaults["fixed_room"] == DEFAULT_FIXED_ROOM


class TestPackageExports:
    def test_provider_class_still_exports_from_package(self):
        provider = MemPalaceMemoryProvider()
        assert provider.name == "mempalace"

    def test_register_registers_provider(self):
        ctx = MagicMock()
        original = MemPalaceMemoryProvider.is_available
        setattr(MemPalaceMemoryProvider, "is_available", lambda self: True)
        try:
            register(ctx)
        finally:
            setattr(MemPalaceMemoryProvider, "is_available", original)
        ctx.register_memory_provider.assert_called_once()
