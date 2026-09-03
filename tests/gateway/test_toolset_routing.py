from dataclasses import dataclass

from gateway.toolset_routing import (
    SASHA_DISCORD_CODING_TOOLSETS,
    SASHA_DISCORD_HEAVY_TOOLSETS,
    SASHA_DISCORD_LEAN_TOOLSETS,
    route_toolsets_for_source,
)
from gateway.config import Platform
from gateway.run import GatewayRunner


@dataclass
class FakeSource:
    platform: Platform = Platform.DISCORD
    chat_id: str = "999"
    chat_name: str = "general"
    parent_chat_id: str | None = None


def test_home_channel_is_always_lean_even_for_coding_text():
    source = FakeSource(chat_id="1513198617198858434", chat_name="Home")

    assert route_toolsets_for_source(source, message="edit files and run tests") == SASHA_DISCORD_LEAN_TOOLSETS


def test_home_thread_is_always_lean_even_for_coding_text():
    source = FakeSource(
        chat_id="thread-1",
        parent_chat_id="1513198617198858434",
        chat_name="Home thread",
    )

    assert route_toolsets_for_source(source, message="use browser and terminal") == SASHA_DISCORD_LEAN_TOOLSETS


def test_proj_channel_is_coding_by_default():
    source = FakeSource(chat_id="999", chat_name="proj-hermes-sashabot")

    assert route_toolsets_for_source(source, message="what next?") == SASHA_DISCORD_CODING_TOOLSETS


def test_proj_thread_uses_parent_project_channel_name_from_chat_name_path():
    source = FakeSource(chat_id="thread-1", chat_name="Coding Agents / #proj-hermes-sashabot / task")

    assert route_toolsets_for_source(source, message="inspect the repo") == SASHA_DISCORD_CODING_TOOLSETS


def test_non_project_discord_channel_defaults_lean():
    source = FakeSource(chat_id="999", chat_name="general")

    assert route_toolsets_for_source(source, message="hello") == SASHA_DISCORD_LEAN_TOOLSETS


def test_explicit_heavy_intent_routes_heavy_outside_home():
    source = FakeSource(chat_id="999", chat_name="proj-hermes-sashabot")

    assert route_toolsets_for_source(source, message="use browser to QA this") == SASHA_DISCORD_HEAVY_TOOLSETS


def test_non_discord_keeps_existing_platform_resolution():
    source = FakeSource(platform=Platform.TELEGRAM, chat_id="999", chat_name="proj-hermes")

    assert route_toolsets_for_source(source, message="use browser") is None


def test_gateway_resolver_applies_discord_route_before_agent_creation():
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: None
    source = FakeSource(chat_id="999", chat_name="proj-hermes-sashabot")

    enabled = GatewayRunner._resolve_enabled_toolsets_for_source(
        runner,
        {"platform_toolsets": {"discord": ["clarify"]}},
        source,
        "discord",
        message="inspect repo",
    )

    assert "terminal" in enabled
    assert "file" in enabled
    assert "computer_use" not in enabled
    assert "cronjob" not in enabled


def test_gateway_resolver_forces_home_lean_before_agent_creation():
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: None
    source = FakeSource(chat_id="1513198617198858434", chat_name="Home")

    enabled = GatewayRunner._resolve_enabled_toolsets_for_source(
        runner,
        {"platform_toolsets": {"discord": ["terminal", "file"]}},
        source,
        "discord",
        message="run tests",
    )

    assert "homeassistant" in enabled
    assert "terminal" not in enabled
    assert "file" not in enabled
