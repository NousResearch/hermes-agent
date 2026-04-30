"""Tests for the platform adapter registry and dynamic Platform enum."""

import os
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from gateway.platform_registry import PlatformRegistry, PlatformEntry, platform_registry
from gateway.config import Platform, PlatformConfig, GatewayConfig


# ── Platform enum dynamic members ─────────────────────────────────────────


class TestPlatformEnumDynamic:
    """Test that Platform enum accepts unknown values for plugin platforms."""

    def test_builtin_members_still_work(self):
        assert Platform.TELEGRAM.value == "telegram"
        assert Platform("telegram") is Platform.TELEGRAM

    def test_dynamic_member_created(self):
        p = Platform("irc")
        assert p.value == "irc"
        assert p.name == "IRC"

    def test_dynamic_member_identity_stable(self):
        """Same value returns same object (cached)."""
        a = Platform("irc")
        b = Platform("irc")
        assert a is b

    def test_dynamic_member_case_normalised(self):
        """Mixed case normalised to lowercase."""
        a = Platform("IRC")
        b = Platform("irc")
        assert a is b
        assert a.value == "irc"

    def test_dynamic_member_with_hyphens(self):
        """Registered plugin platforms with hyphens work once registered."""
        from gateway.platform_registry import platform_registry as _reg

        entry = PlatformEntry(
            name="my-platform",
            label="My Platform",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="plugin",
        )
        _reg.register(entry)
        try:
            p = Platform("my-platform")
            assert p.value == "my-platform"
            assert p.name == "MY_PLATFORM"
        finally:
            _reg.unregister("my-platform")

    def test_dynamic_member_rejects_unregistered(self):
        """Arbitrary strings are rejected to prevent enum pollution."""
        with pytest.raises(ValueError):
            Platform("totally-fake-platform")

    def test_dynamic_member_rejects_non_string(self):
        with pytest.raises(ValueError):
            Platform(123)

    def test_dynamic_member_rejects_empty(self):
        with pytest.raises(ValueError):
            Platform("")

    def test_dynamic_member_rejects_whitespace_only(self):
        with pytest.raises(ValueError):
            Platform("   ")


# ── PlatformRegistry ──────────────────────────────────────────────────────


class TestPlatformRegistry:
    """Test the PlatformRegistry itself."""

    def _make_entry(self, name="test", check_ok=True, validate_ok=True, factory_ok=True):
        adapter_mock = MagicMock()
        return PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg, _m=adapter_mock: _m if factory_ok else (_ for _ in ()).throw(RuntimeError("factory error")),
            check_fn=lambda: check_ok,
            validate_config=lambda cfg: validate_ok,
            required_env=[],
            source="plugin",
        ), adapter_mock

    def test_register_and_get(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("alpha")
        reg.register(entry)
        assert reg.get("alpha") is entry
        assert reg.is_registered("alpha")

    def test_get_unknown_returns_none(self):
        reg = PlatformRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("beta")
        reg.register(entry)
        assert reg.unregister("beta") is True
        assert reg.get("beta") is None
        assert reg.unregister("beta") is False  # already gone

    def test_create_adapter_success(self):
        reg = PlatformRegistry()
        entry, mock_adapter = self._make_entry("gamma")
        reg.register(entry)
        result = reg.create_adapter("gamma", MagicMock())
        assert result is mock_adapter

    def test_create_adapter_unknown_name(self):
        reg = PlatformRegistry()
        assert reg.create_adapter("unknown", MagicMock()) is None

    def test_create_adapter_check_fails(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("delta", check_ok=False)
        reg.register(entry)
        assert reg.create_adapter("delta", MagicMock()) is None

    def test_create_adapter_validate_fails(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("epsilon", validate_ok=False)
        reg.register(entry)
        assert reg.create_adapter("epsilon", MagicMock()) is None

    def test_create_adapter_factory_exception(self):
        reg = PlatformRegistry()
        entry = PlatformEntry(
            name="broken",
            label="Broken",
            adapter_factory=lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")),
            check_fn=lambda: True,
            validate_config=None,
            source="plugin",
        )
        reg.register(entry)
        # factory raises → create_adapter returns None instead of propagating
        assert reg.create_adapter("broken", MagicMock()) is None

    def test_create_adapter_no_validate(self):
        """When validate_config is None, skip validation."""
        reg = PlatformRegistry()
        mock_adapter = MagicMock()
        entry = PlatformEntry(
            name="novalidate",
            label="NoValidate",
            adapter_factory=lambda cfg: mock_adapter,
            check_fn=lambda: True,
            validate_config=None,
            source="plugin",
        )
        reg.register(entry)
        assert reg.create_adapter("novalidate", MagicMock()) is mock_adapter

    def test_all_entries(self):
        reg = PlatformRegistry()
        e1, _ = self._make_entry("one")
        e2, _ = self._make_entry("two")
        reg.register(e1)
        reg.register(e2)
        names = {e.name for e in reg.all_entries()}
        assert names == {"one", "two"}

    def test_plugin_entries(self):
        reg = PlatformRegistry()
        plugin_entry, _ = self._make_entry("plugged")
        builtin_entry = PlatformEntry(
            name="core",
            label="Core",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="builtin",
        )
        reg.register(plugin_entry)
        reg.register(builtin_entry)
        plugin_names = {e.name for e in reg.plugin_entries()}
        assert plugin_names == {"plugged"}

    def test_re_register_replaces(self):
        reg = PlatformRegistry()
        entry1, mock1 = self._make_entry("dup")
        entry2 = PlatformEntry(
            name="dup",
            label="Dup v2",
            adapter_factory=lambda cfg: "v2",
            check_fn=lambda: True,
            source="plugin",
        )
        reg.register(entry1)
        reg.register(entry2)
        assert reg.get("dup").label == "Dup v2"


# ── GatewayConfig integration ────────────────────────────────────────────


class TestGatewayConfigPluginPlatform:
    """Test that GatewayConfig parses and validates plugin platforms."""

    def test_from_dict_accepts_plugin_platform(self):
        data = {
            "platforms": {
                "telegram": {"enabled": True, "token": "test-token"},
                "irc": {"enabled": True, "extra": {"server": "irc.libera.chat"}},
            }
        }
        cfg = GatewayConfig.from_dict(data)
        platform_values = {p.value for p in cfg.platforms}
        assert "telegram" in platform_values
        assert "irc" in platform_values

    def test_get_connected_platforms_includes_registered_plugin(self):
        """Plugin platform with registry entry passes get_connected_platforms."""
        # Register a fake plugin platform
        from gateway.platform_registry import platform_registry as _reg

        test_entry = PlatformEntry(
            name="testplat",
            label="TestPlat",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            validate_config=lambda cfg: bool(cfg.extra.get("token")),
            source="plugin",
        )
        _reg.register(test_entry)
        try:
            data = {
                "platforms": {
                    "testplat": {"enabled": True, "extra": {"token": "abc"}},
                }
            }
            cfg = GatewayConfig.from_dict(data)
            connected = cfg.get_connected_platforms()
            connected_values = {p.value for p in connected}
            assert "testplat" in connected_values
        finally:
            _reg.unregister("testplat")

    def test_get_connected_platforms_excludes_unregistered_plugin(self):
        """Plugin platform without registry entry is excluded."""
        data = {
            "platforms": {
                "unknown_plugin": {"enabled": True, "extra": {"token": "abc"}},
            }
        }
        cfg = GatewayConfig.from_dict(data)
        connected = cfg.get_connected_platforms()
        connected_values = {p.value for p in connected}
        assert "unknown_plugin" not in connected_values

    def test_get_connected_platforms_excludes_invalid_config(self):
        """Plugin platform with failing validate_config is excluded."""
        from gateway.platform_registry import platform_registry as _reg

        test_entry = PlatformEntry(
            name="badconfig",
            label="BadConfig",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            validate_config=lambda cfg: False,  # always fails
            source="plugin",
        )
        _reg.register(test_entry)
        try:
            data = {
                "platforms": {
                    "badconfig": {"enabled": True, "extra": {}},
                }
            }
            cfg = GatewayConfig.from_dict(data)
            connected = cfg.get_connected_platforms()
            connected_values = {p.value for p in connected}
            assert "badconfig" not in connected_values
        finally:
            _reg.unregister("badconfig")


# ── Extended PlatformEntry fields ─────────────────────────────────────


class TestPlatformEntryExtendedFields:
    """Test the auth, message length, and display fields on PlatformEntry."""

    def test_default_field_values(self):
        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
        assert entry.allowed_users_env == ""
        assert entry.allow_all_env == ""
        assert entry.max_message_length == 0
        assert entry.pii_safe is False
        assert entry.emoji == "🔌"
        assert entry.allow_update_command is True

    def test_custom_auth_fields(self):
        entry = PlatformEntry(
            name="irc",
            label="IRC",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            allowed_users_env="IRC_ALLOWED_USERS",
            allow_all_env="IRC_ALLOW_ALL_USERS",
            max_message_length=450,
            pii_safe=False,
            emoji="💬",
        )
        assert entry.allowed_users_env == "IRC_ALLOWED_USERS"
        assert entry.allow_all_env == "IRC_ALLOW_ALL_USERS"
        assert entry.max_message_length == 450
        assert entry.emoji == "💬"


# ── Cron platform resolution ─────────────────────────────────────────


class TestCronPlatformResolution:
    """Test that cron delivery accepts plugin platform names."""

    def test_builtin_platform_resolves(self):
        """Built-in platform names resolve via Platform() call."""
        p = Platform("telegram")
        assert p is Platform.TELEGRAM

    def test_plugin_platform_resolves(self):
        """Plugin platform names create dynamic enum members."""
        p = Platform("irc")
        assert p.value == "irc"

    def test_invalid_platform_type_rejected(self):
        """Non-string values are still rejected."""
        with pytest.raises(ValueError):
            Platform(None)


# ── platforms.py integration ──────────────────────────────────────────


class TestPlatformsMerge:
    """Test get_all_platforms() merges with registry."""

    def test_get_all_platforms_includes_builtins(self):
        from hermes_cli.platforms import get_all_platforms, PLATFORMS
        merged = get_all_platforms()
        for key in PLATFORMS:
            assert key in merged

    def test_get_all_platforms_includes_plugin(self):
        from hermes_cli.platforms import get_all_platforms
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="testmerge",
            label="TestMerge",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            emoji="🧪",
        ))
        try:
            merged = get_all_platforms()
            assert "testmerge" in merged
            assert "TestMerge" in merged["testmerge"].label
        finally:
            _reg.unregister("testmerge")

    def test_platform_label_plugin_fallback(self):
        from hermes_cli.platforms import platform_label
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="labeltest",
            label="LabelTest",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            emoji="🏷️",
        ))
        try:
            label = platform_label("labeltest")
            assert "LabelTest" in label
        finally:
            _reg.unregister("labeltest")
