"""Telegram DM topic runtime overrides on the channel-override architecture."""

from types import SimpleNamespace

import gateway.run as gateway_run
from gateway.config import ChannelOverride, Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        user_id="user-1",
        thread_id="42",
    )


def _runner(override: ChannelOverride):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={"111:42": override},
            )
        }
    )
    runner._session_model_overrides = {}
    runner._last_resolved_model = {}
    runner._rehydrate_session_model_override = lambda _key: None
    runner.session_store = SimpleNamespace(
        _generate_session_key=lambda _source: "telegram-topic-session"
    )
    return runner


def test_topic_lookup_prefers_composite_chat_thread_key():
    keys = gateway_run._channel_override_lookup_keys(
        "111", thread_id="42", parent_id="parent"
    )

    assert keys == ["111:42", "42", "111", "parent"]


def test_provider_only_topic_uses_provider_runtime_model(monkeypatch):
    runner = _runner(ChannelOverride(provider="topic-provider"))
    monkeypatch.setattr(
        gateway_run, "_resolve_gateway_model", lambda _config=None: "global-model"
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "global-provider", "api_key": "global-key"},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "model": "provider-default-model",
            "api_key": "topic-key",
        },
    )

    model, runtime = runner._resolve_session_agent_runtime(
        source=_source(), user_config={"model": {"default": "global-model"}}
    )

    assert model == "provider-default-model"
    assert runtime["provider"] == "topic-provider"


def test_topic_toolsets_resolve_without_erasing_default_surface(monkeypatch):
    import hermes_cli.tools_config as tools_config

    runner = _runner(ChannelOverride(toolsets=["topic-tools"]))

    def fake_platform_tools(config, _platform):
        configured = (config.get("platform_toolsets") or {}).get("telegram")
        if configured == ["topic-tools"]:
            return {"web"}
        return {"terminal", "file"}

    monkeypatch.setattr(tools_config, "_get_platform_tools", fake_platform_tools)

    assert runner._resolve_toolsets_for_source({}, "telegram", _source()) == ["web"]

    runner.config.platforms[Platform.TELEGRAM].channel_overrides[
        "111:42"
    ] = ChannelOverride(toolsets=["unknown"])

    def empty_invalid_toolsets(config, _platform):
        configured = (config.get("platform_toolsets") or {}).get("telegram")
        return set() if configured == ["unknown"] else {"terminal", "file"}

    monkeypatch.setattr(
        tools_config, "_get_platform_tools", empty_invalid_toolsets
    )
    assert runner._resolve_toolsets_for_source({}, "telegram", _source()) == [
        "file",
        "terminal",
    ]


def test_dm_topic_info_registers_runtime_override_on_plugin_adapter():
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "dm_topics": [
                {
                    "chat_id": "111",
                    "topics": [
                        {
                            "name": "Research",
                            "thread_id": 42,
                            "provider": "topic-provider",
                            "model": "topic-model",
                            "toolsets": ["web"],
                        }
                    ],
                }
            ]
        },
    )
    adapter = TelegramAdapter(config)
    adapter._dm_topics["111:Research"] = 42

    topic = adapter._get_dm_topic_info("111", "42")

    assert topic["name"] == "Research"
    override = config.channel_overrides["111:42"]
    assert override.provider == "topic-provider"
    assert override.model == "topic-model"
    assert override.toolsets == ["web"]


def test_explicit_channel_override_wins_over_dm_topic_bridge():
    explicit = ChannelOverride(model="explicit-model")
    config = PlatformConfig(
        enabled=True,
        token="***",
        channel_overrides={"111:42": explicit},
    )
    adapter = TelegramAdapter(config)

    adapter._register_dm_topic_runtime_override(
        "111",
        "42",
        {"model": "topic-model", "provider": "topic-provider"},
    )

    assert config.channel_overrides["111:42"] is explicit
