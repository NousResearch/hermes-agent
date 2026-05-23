"""Regression tests for gateway channel/thread profile routing.

The gateway process can run under the default profile while individual Discord
channels are routed to profile-specific agents. Slash commands such as
``/profile`` and ``/model`` must use the effective chat profile, not only the
process profile.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_runner():
    runner = cast(Any, object.__new__(gateway_run.GatewayRunner))
    runner.adapters = {}
    runner.session_store = None
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    runner._session_model_overrides = {}
    return runner


def test_effective_profile_uses_discord_channel_route(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "discord": {
                "channel_profiles": {
                    "channel-builder": "local-builder",
                    "channel-research": "local-research",
                }
            }
        },
    )
    runner = _make_runner()

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-research",
        chat_type="channel",
    )

    assert runner._effective_profile_name_for_source(source) == "local-research"


def test_effective_profile_inherits_parent_channel_route_for_thread(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "discord": {
                "channel_profiles": {
                    "parent-channel": "local-builder",
                }
            }
        },
    )
    runner = _make_runner()

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-id",
        thread_id="thread-id",
        parent_chat_id="parent-channel",
        chat_type="thread",
    )

    assert runner._effective_profile_name_for_source(source) == "local-builder"


def test_profile_command_reports_effective_profile_not_process_profile(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "discord": {
                "channel_profiles": {
                    "channel-research": "local-research",
                }
            }
        },
    )

    import hermes_cli.profiles as profiles

    monkeypatch.setattr(
        profiles,
        "resolve_profile_env",
        lambda profile: f"/tmp/hermes/profiles/{profile}",
    )

    runner = _make_runner()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-research",
        chat_type="channel",
    )
    event = cast(Any, SimpleNamespace(source=source))

    response = asyncio.run(runner._handle_profile_command(event))

    assert "local-research" in response
    assert "/tmp/hermes/profiles/local-research" in response
    assert "Gateway process profile" in response
    assert "default" in response


def test_model_command_reads_effective_profile_config_for_routed_channel(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {"default": "gpt-5.5", "provider": "openai-codex"},
            "discord": {
                "channel_profiles": {
                    "channel-research": "local-research",
                }
            },
        },
    )
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_load_profile_config",
        lambda self, profile_name: {
            "model": {"default": "aeon-ultimate", "provider": "custom:local"},
            "custom_providers": [
                {
                    "name": "local",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "models": {"aeon-ultimate": 262144},
                }
            ],
        } if profile_name == "local-research" else {},
    )

    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **kwargs: [])

    runner = _make_runner()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-research",
        chat_type="channel",
    )
    event = cast(Any, SimpleNamespace(source=source, get_command_args=lambda: ""))

    response = asyncio.run(runner._handle_model_command(event))

    assert response is not None
    assert "aeon-ultimate" in response
    assert "gpt-5.5" not in response
