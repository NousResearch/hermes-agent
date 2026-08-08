"""Regression tests for #72838: ``/model`` (no args) must display the
channel_overrides model, not the global default.

Before fix: the ``/model`` handler read ``current_model`` from
``model.default`` and only checked session-level ``_session_model_overrides``
— it never consulted ``channel_overrides``. So the text-list fallback (and
the picker "Current model" label) showed the global default even when the
channel actually ran an override model.

After fix: channel_overrides is resolved (via ``_get_channel_override``)
after source normalization and before the session-override block, mirroring
the priority used by the actual turn dispatch
(session ``/model`` > ``channel_overrides`` > global default).

Provider-only overrides (set via ``channel_overrides[chat_id].provider``
without an explicit ``model``) must additionally adopt the provider's
bundled/default model returned by
``_resolve_runtime_agent_kwargs_for_provider``. Without this, the
``channel_overrides[chat_id].provider`` field would silently apply at
turn-dispatch but the ``/model`` display would still show the global
default — a confusing UX mismatch. The wrapper previously dropped
``model`` from its return dict, so the fix includes re-attaching it.
"""

import yaml
import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "12345": ChannelOverride(model="channel-override-model"),
                },
            ),
        },
    )
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm",
        ),
    )


def _setup_isolated_home(tmp_path, monkeypatch):
    """Write a config.yaml whose ``model.default`` differs from the channel
    override so we can tell them apart."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "model": {"default": "global-default-model", "provider": "openrouter"},
            "providers": {},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


@pytest.mark.asyncio
async def test_model_no_args_shows_channel_override(tmp_path, monkeypatch):
    """``/model`` with no args must show the channel_overrides model in the
    text-list fallback, not the global default."""
    _setup_isolated_home(tmp_path, monkeypatch)
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "channel-override-model" in result
    assert "global-default-model" not in result


@pytest.mark.asyncio
async def test_model_no_args_shows_global_when_no_override(tmp_path, monkeypatch):
    """When the source's channel has no override, the global default is
    shown — same as before the fix."""
    _setup_isolated_home(tmp_path, monkeypatch)
    runner = _make_runner()
    # Override applies to chat_id "12345"; this event comes from "99999".
    event = MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="99999", chat_type="dm",
        ),
    )

    result = await runner._handle_model_command(event)

    assert result is not None
    assert "global-default-model" in result


@pytest.mark.asyncio
async def test_model_no_args_provider_only_uses_resolver_model(tmp_path, monkeypatch):
    """Provider-only channel override (no explicit model) must adopt the
    provider's bundled/default model from
    _resolve_runtime_agent_kwargs_for_provider, not the global default."""
    import gateway.run as gateway_run

    _setup_isolated_home(tmp_path, monkeypatch)

    calls = []
    def mock_resolver(provider):
        calls.append(provider)
        return {
            "provider": "custom:channel",
            "base_url": "http://channel.example.com",
            "model": "provider-default-model",
            "api_key": "",
        }
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", mock_resolver)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "12345": ChannelOverride(provider="custom:channel"),
                },
            ),
        },
    )

    result = await runner._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "provider-default-model" in result
    assert "global-default-model" not in result
    assert calls == ["custom:channel"]


@pytest.mark.asyncio
async def test_model_no_args_model_and_provider_shows_explicit_model(tmp_path, monkeypatch):
    """When channel override sets both model and provider, the explicit model
    takes priority over the resolver's default model, but the resolver is
    still called for base_url resolution."""
    import gateway.run as gateway_run

    _setup_isolated_home(tmp_path, monkeypatch)

    calls = []
    def mock_resolver(provider):
        calls.append(provider)
        return {
            "provider": "custom:channel",
            "base_url": "http://channel.example.com",
            "model": "provider-default-model",
            "api_key": "",
        }
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", mock_resolver)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "12345": ChannelOverride(model="channel-model", provider="custom:channel"),
                },
            ),
        },
    )

    result = await runner._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "channel-model" in result
    assert "global-default-model" not in result
    assert calls == ["custom:channel"]
