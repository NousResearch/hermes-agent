"""PlatformEntry seams a plugin platform uses instead of a core special case.

``trusted_inbound`` (no human sender: allowlists/pairing do not apply) and ``display_tier`` (built-in
display defaults) were Home Assistant branches in core; the homeassistant catalog plugin is their
consumer. Driven through the real registry, authz and display resolver.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource


@pytest.fixture
def plugin_platform():
    names = []

    def _register(name: str, **fields):
        platform_registry.register(PlatformEntry(
            name=name, label=name.title(), adapter_factory=lambda cfg: None, check_fn=lambda: True,
            source="plugin", **fields))
        names.append(name)
        return Platform(name)

    yield _register
    for name in names:
        platform_registry.unregister(name)


def _runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = None
    return runner


def _event_source(platform):
    return SessionSource(platform=platform, user_id="event-bus", chat_id="events", user_name="bus", chat_type="dm")


def test_trusted_inbound_platform_bypasses_allowlists(plugin_platform, monkeypatch):
    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    trusted = plugin_platform("seamtrusted", trusted_inbound=True)
    plain = plugin_platform("seamplain")
    runner = _runner()
    assert runner._is_user_authorized(_event_source(trusted)) is True
    # Control: the default stays default-deny for an unknown sender.
    assert runner._is_user_authorized(_event_source(plain)) is False


def test_display_tier_supplies_builtin_defaults_below_user_overrides(plugin_platform):
    plugin_platform("seamminimal", display_tier="minimal")
    plugin_platform("seamnotier")
    assert resolve_display_setting({}, "seamminimal", "tool_progress") == "off"
    assert resolve_display_setting({}, "seamminimal", "tool_preview_length") == 0
    assert resolve_display_setting({}, "seamminimal", "streaming") is False
    # The user's per-platform override still wins over the registered tier.
    user = {"display": {"platforms": {"seamminimal": {"tool_progress": "all"}}}}
    assert resolve_display_setting(user, "seamminimal", "tool_progress") == "all"
    # No tier: the global defaults, as before.
    assert resolve_display_setting({}, "seamnotier", "tool_progress") == resolve_display_setting({}, "nosuchplatform", "tool_progress")


def test_platform_entry_seam_defaults_are_inert():
    entry = PlatformEntry(name="x", label="X", adapter_factory=lambda cfg: None, check_fn=lambda: True)
    assert (entry.trusted_inbound, entry.display_tier, entry.shared_env_prefixes) == (False, "", ())


def test_user_plugin_cannot_mark_a_core_platform_trusted_inbound():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    ctx = PluginContext(PluginManifest(name="seamevil", key="seamevil", source="user"),
                        PluginManager(scope_key=platform_registry.current_scope_key()))
    with pytest.raises(ValueError, match="trusted_inbound"):
        ctx.register_platform("telegram", "Telegram", adapter_factory=lambda cfg: None, check_fn=lambda: True,
                              trusted_inbound=True)
    # Control: a platform core does not ship (Home Assistant's case) may declare it.
    try:
        ctx.register_platform("seameventbus", "Bus", adapter_factory=lambda cfg: None, check_fn=lambda: True,
                              trusted_inbound=True)
        assert platform_registry.get("seameventbus").trusted_inbound is True
    finally:
        platform_registry.unregister("seameventbus", scope=platform_registry.current_scope_key())
