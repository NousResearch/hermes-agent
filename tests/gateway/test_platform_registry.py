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


# ── apply_yaml_config_fn (PlatformEntry field + load_gateway_config dispatch) ──


class TestApplyYamlConfigFnField:
    """The hook field itself — defaults, custom values, signature."""

    def test_default_is_none(self):
        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
        assert entry.apply_yaml_config_fn is None

    def test_accepts_callable(self):
        def _hook(yaml_cfg, platform_cfg):
            return None

        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            apply_yaml_config_fn=_hook,
        )
        assert entry.apply_yaml_config_fn is _hook
        # Sanity-check the signature contract.
        assert entry.apply_yaml_config_fn({"x": 1}, {"y": 2}) is None


class TestApplyYamlConfigFnDispatch:
    """End-to-end dispatch through load_gateway_config().

    Each test registers a temporary PlatformEntry, writes a config.yaml in
    a tmp HERMES_HOME, calls load_gateway_config(), and asserts the hook
    was invoked correctly.  Cleanup unregisters the entry.
    """

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def _register_hook(self, name, hook_fn):
        from gateway.platform_registry import platform_registry as _reg

        entry = PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=hook_fn,
        )
        _reg.register(entry)
        return _reg

    def test_hook_can_mutate_environ(self, tmp_path, monkeypatch):
        """A hook that mutates os.environ has its env vars set after load."""
        env_var = "MYHOOKPLAT_FLAG"
        monkeypatch.delenv(env_var, raising=False)

        def _hook(yaml_cfg, platform_cfg):
            if "flag" in platform_cfg and not os.getenv(env_var):
                os.environ[env_var] = str(platform_cfg["flag"]).lower()
            return None

        reg = self._register_hook("myhookplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myhookplat:\n  flag: true\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert os.environ.get(env_var) == "true"
        finally:
            reg.unregister("myhookplat")
            os.environ.pop(env_var, None)

    def test_hook_returned_dict_merges_into_extra(self, tmp_path, monkeypatch):
        """A hook that returns a dict has it merged into PlatformConfig.extra."""

        def _hook(yaml_cfg, platform_cfg):
            return {"seeded_key": "seeded_value", "flag": platform_cfg.get("flag")}

        reg = self._register_hook("myextraplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myextraplat:\n  flag: yes\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            cfg = load_gateway_config()

            plat = Platform("myextraplat")
            assert plat in cfg.platforms
            extra = cfg.platforms[plat].extra
            assert extra.get("seeded_key") == "seeded_value"
            # flag value carried through from yaml_cfg arg.
            assert extra.get("flag") is True
        finally:
            reg.unregister("myextraplat")

    def test_hook_receives_full_yaml_and_platform_subdict(
        self, tmp_path, monkeypatch
    ):
        """Hook receives both the full yaml_cfg and its own platform sub-dict."""
        captured: dict = {}

        def _hook(yaml_cfg, platform_cfg):
            captured["yaml_cfg"] = yaml_cfg
            captured["platform_cfg"] = platform_cfg
            return None

        reg = self._register_hook("mycaptureplat", _hook)
        try:
            home = self._write_config(
                tmp_path,
                "top_level_key: 1\n"
                "mycaptureplat:\n"
                "  inner_key: deep\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert captured["yaml_cfg"].get("top_level_key") == 1
            assert captured["platform_cfg"] == {"inner_key": "deep"}
        finally:
            reg.unregister("mycaptureplat")

    def test_hook_exception_swallowed(self, tmp_path, monkeypatch):
        """A misbehaving hook never aborts load_gateway_config()."""

        def _bad_hook(yaml_cfg, platform_cfg):
            raise RuntimeError("plugin author bug")

        # Also register a well-behaved hook to ensure dispatch continues
        # iterating after a bad one.
        good_called = {"count": 0}

        def _good_hook(yaml_cfg, platform_cfg):
            good_called["count"] += 1
            return None

        from gateway.platform_registry import platform_registry as _reg
        _reg.register(PlatformEntry(
            name="mybadplat",
            label="MyBad",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=_bad_hook,
        ))
        _reg.register(PlatformEntry(
            name="mygoodplat",
            label="MyGood",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=_good_hook,
        ))
        try:
            home = self._write_config(
                tmp_path,
                "mybadplat:\n  k: v\n"
                "mygoodplat:\n  k: v\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            # Must not raise.
            from gateway.config import load_gateway_config
            load_gateway_config()

            assert good_called["count"] == 1
        finally:
            _reg.unregister("mybadplat")
            _reg.unregister("mygoodplat")

    def test_hook_skipped_when_platform_section_missing(
        self, tmp_path, monkeypatch
    ):
        """Hook is NOT called when the platform's YAML section is absent."""
        called = {"count": 0}

        def _hook(yaml_cfg, platform_cfg):
            called["count"] += 1
            return None

        reg = self._register_hook("myabsentplat", _hook)
        try:
            home = self._write_config(tmp_path, "telegram:\n  k: v\n")
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert called["count"] == 0
        finally:
            reg.unregister("myabsentplat")

    def test_hook_skipped_when_platform_section_not_dict(
        self, tmp_path, monkeypatch
    ):
        """Hook is NOT called when the platform's YAML section isn't a dict."""
        called = {"count": 0}

        def _hook(yaml_cfg, platform_cfg):
            called["count"] += 1
            return None

        reg = self._register_hook("mybadshapeplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "mybadshapeplat: just-a-string\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert called["count"] == 0
        finally:
            reg.unregister("mybadshapeplat")

    def test_env_var_takes_precedence_when_hook_uses_getenv_guard(
        self, tmp_path, monkeypatch
    ):
        """The standard `not os.getenv(...)` guard preserves env > YAML."""
        env_var = "MYPRECPLAT_FLAG"
        monkeypatch.setenv(env_var, "preexisting")

        def _hook(yaml_cfg, platform_cfg):
            if "flag" in platform_cfg and not os.getenv(env_var):
                os.environ[env_var] = str(platform_cfg["flag"]).lower()
            return None

        reg = self._register_hook("myprecplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myprecplat:\n  flag: yaml-value\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            # Pre-existing env var was NOT clobbered by the hook.
            assert os.environ.get(env_var) == "preexisting"
        finally:
            reg.unregister("myprecplat")
            os.environ.pop(env_var, None)


class TestPluginPlatformSharedKeyBridge:
    """Plugin-registered platforms get the same shared-key bridging as built-ins.

    Without this, plugin authors using ``apply_yaml_config_fn`` would have to
    re-implement bridging for every common key (``unauthorized_dm_behavior``,
    ``notice_delivery``, ``reply_prefix``, ``require_mention``, ``dm_policy``,
    ``allow_from``, etc.) — defeating the hook's whole point of letting
    plugins focus on their *platform-specific* keys.
    """

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def test_shared_keys_bridged_for_plugin_platform(self, tmp_path, monkeypatch):
        """A plugin platform's ``require_mention``/``dm_policy``/etc. flow into
        ``PlatformConfig.extra`` without the plugin needing its own bridge."""
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="mysharedplat",
            label="MySharedPlat",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
        ))
        try:
            home = self._write_config(
                tmp_path,
                "mysharedplat:\n"
                "  require_mention: true\n"
                "  dm_policy: allow\n"
                "  reply_prefix: \"→ \"\n"
                "  allow_from: [\"alice\", \"bob\"]\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("mysharedplat")
            assert plat in cfg.platforms
            extra = cfg.platforms[plat].extra
            assert extra.get("require_mention") is True
            assert extra.get("dm_policy") == "allow"
            assert extra.get("reply_prefix") == "→ "
            assert extra.get("allow_from") == ["alice", "bob"]
        finally:
            _reg.unregister("mysharedplat")
