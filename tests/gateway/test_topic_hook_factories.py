"""Lazy, config-driven topic hook activation."""

import pytest

from gateway.config import PlatformConfig
from gateway.topic_hooks import (
    TopicHookRegistry,
    TopicPluginHook,
    create_topic_hook,
    register_topic_hook_factory,
)


class _Hook(TopicPluginHook):
    def __init__(self, profile: str) -> None:
        self.profile = profile


class _LifecycleHook(_Hook):
    def __init__(self, profile: str) -> None:
        super().__init__(profile)
        self.events = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


def _config(*, profile: str, hooks=None) -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        token="synthetic-test-token",
        extra={
            "topic_routing": {
                "mode": "strict",
                "routes": [
                    {"chat_id": "12345", "thread_id": 1, "profile": profile}
                ],
                **({} if hooks is None else {"hooks": hooks}),
            }
        },
    )


def test_factory_registration_is_lazy() -> None:
    calls = []
    register_topic_hook_factory(
        "synthetic-lazy",
        lambda: calls.append("called") or _Hook("synthetic-lazy"),
        owner="test-lazy",
    )
    assert calls == []
    assert create_topic_hook("synthetic-lazy", owner="test-lazy").profile == "synthetic-lazy"
    assert calls == ["called"]


@pytest.mark.asyncio
async def test_registry_starts_and_stops_hooks_exactly_once() -> None:
    registry = TopicHookRegistry()
    hook = _LifecycleHook("synthetic-lifecycle")
    registry.register(hook)

    await registry.start()
    await registry.start()
    assert hook.events == ["start"]

    await registry.stop()
    await registry.stop()
    assert hook.events == ["start", "stop"]


def test_plugin_context_registers_factory_with_plugin_owner() -> None:
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    context = PluginContext(
        PluginManifest(name="synthetic-plugin", key="synthetic-plugin-key"),
        PluginManager(),
    )
    context.register_topic_hook_factory(
        "synthetic-context", lambda: _Hook("synthetic-context")
    )
    assert (
        create_topic_hook("synthetic-context", owner="synthetic-plugin-key").profile
        == "synthetic-context"
    )


def test_same_owner_can_refresh_factory_but_other_owner_cannot() -> None:
    register_topic_hook_factory(
        "synthetic-owned", lambda: _Hook("synthetic-owned"), owner="owner-a"
    )
    replacement = lambda: _Hook("synthetic-owned")
    register_topic_hook_factory(
        "synthetic-owned", replacement, owner="owner-a"
    )
    assert create_topic_hook("synthetic-owned", owner="owner-a").profile == "synthetic-owned"
    with pytest.raises(ValueError, match="duplicate topic hook factory profile"):
        register_topic_hook_factory(
            "synthetic-owned", replacement, owner="owner-b"
        )


def test_factory_cannot_redirect_profile() -> None:
    register_topic_hook_factory(
        "synthetic-requested", lambda: _Hook("other"), owner="test-mismatch"
    )
    with pytest.raises(ValueError, match="profile mismatch"):
        create_topic_hook("synthetic-requested", owner="test-mismatch")


def test_force_plugin_rescan_drops_stale_topic_hook_factories(monkeypatch) -> None:
    from hermes_cli.plugins import PluginManager

    register_topic_hook_factory(
        "synthetic-stale", lambda: _Hook("synthetic-stale"), owner="removed-plugin"
    )
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
    manager.discover_and_load(force=True)
    with pytest.raises(ValueError, match="no registered factory"):
        create_topic_hook("synthetic-stale", owner="removed-plugin")


def test_configured_owner_must_match_registered_plugin() -> None:
    register_topic_hook_factory(
        "synthetic-owner-bound",
        lambda: _Hook("synthetic-owner-bound"),
        owner="expected-plugin",
    )
    with pytest.raises(ValueError, match="owner mismatch"):
        create_topic_hook("synthetic-owner-bound", owner="other-plugin")


def test_adapter_does_not_instantiate_unconfigured_factory() -> None:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    calls = []
    register_topic_hook_factory(
        "synthetic-inert",
        lambda: calls.append("called") or _Hook("synthetic-inert"),
        owner="test-inert",
    )
    adapter = TelegramAdapter(_config(profile="synthetic-inert"))
    assert calls == []
    assert adapter._topic_hooks._by_profile == {}


def test_adapter_instantiates_only_explicit_registered_route_hook() -> None:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    register_topic_hook_factory(
        "synthetic-enabled",
        lambda: _Hook("synthetic-enabled"),
        owner="test-enabled",
    )
    adapter = TelegramAdapter(
        _config(
            profile="synthetic-enabled",
            hooks=[{"profile": "synthetic-enabled", "plugin": "test-enabled"}],
        )
    )
    route = adapter._topic_route_registry.resolve("12345", 1)
    assert adapter._topic_hooks.hook_for(route).profile == "synthetic-enabled"


def test_unknown_or_unrouted_hook_fails_adapter_startup() -> None:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    with pytest.raises(ValueError, match="no registered factory"):
        TelegramAdapter(
            _config(
                profile="synthetic-unknown",
                hooks=[{"profile": "synthetic-unknown", "plugin": "missing"}],
            )
        )

    register_topic_hook_factory(
        "synthetic-unrouted",
        lambda: _Hook("synthetic-unrouted"),
        owner="test-unrouted",
    )
    with pytest.raises(ValueError, match="no registered route"):
        TelegramAdapter(
            _config(
                profile="sol",
                hooks=[{"profile": "synthetic-unrouted", "plugin": "test-unrouted"}],
            )
        )


@pytest.mark.parametrize(
    "hooks",
    [
        "sol",
        ["sol"],
        [{"profile": "", "plugin": "sol-food"}],
        [{"profile": "sol", "plugin": ""}],
        [{"profile": "sol", "plugin": "sol-food", "extra": True}],
    ],
)
def test_invalid_hook_configuration_fails_closed(hooks) -> None:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    with pytest.raises(ValueError, match="topic_routing.hooks"):
        TelegramAdapter(_config(profile="sol", hooks=hooks))


def test_duplicate_hook_profiles_fail_closed() -> None:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    register_topic_hook_factory(
        "synthetic-duplicate",
        lambda: _Hook("synthetic-duplicate"),
        owner="test-duplicate",
    )
    hooks = [
        {"profile": "synthetic-duplicate", "plugin": "test-duplicate"},
        {"profile": "synthetic-duplicate", "plugin": "test-duplicate"},
    ]
    with pytest.raises(ValueError, match="duplicate profiles"):
        TelegramAdapter(_config(profile="synthetic-duplicate", hooks=hooks))
