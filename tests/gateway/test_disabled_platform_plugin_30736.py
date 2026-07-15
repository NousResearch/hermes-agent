"""Regression tests for #30736 — bundled platform plugin disable now sticks.

User-visible bug: a user runs ``hermes plugins disable platforms/discord``,
``hermes plugins list`` cheerfully reports the plugin as ``disabled``, but
the gateway then:

* loads the discord adapter anyway,
* logs ``ERROR hermes_plugins.discord_platform.adapter: [Discord] No bot
  token configured`` on every start and every reconnect attempt,
* queues discord into the reconnect watcher with exponential backoff up to
  the per-platform ``_PAUSE_AFTER_FAILURES`` circuit-breaker pause.

Root cause: ``hermes plugins list`` and ``hermes plugins disable`` use the
*path-derived* key (``platforms/discord``), while ``PluginManager.discover_and_load``
was scanning ``plugins/platforms/`` with no category prefix and therefore
keyed the bundled adapters by their *manifest name* (``discord-platform``).
Because the loader's disable check looked at the manifest name only when
the path-derived key didn't match, ``platforms/discord`` in the disabled
list never matched the in-memory ``discord-platform`` key, and the plugin
loaded as if nothing happened.

These tests pin three things in place:

1. **Key alignment** — every bundled platform plugin is registered with a
   path-derived key (``platforms/<name>``), so the CLI's view of the
   plugin matches the loader's view.

2. **Disable check** — both the new path-derived key and the legacy
   manifest-name key are honored by the loader's disable check, so
   no user gets stuck because their config still uses the old form.

3. **Gateway runtime** — when the platform's plugin is disabled, the
   gateway startup loop skips the platform cleanly: no adapter
   instantiation, no ERROR log, no entry in ``_failed_platforms`` (so the
   reconnect watcher never fires for it), and a single user-readable INFO
   log naming the action that would re-enable it.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status
from hermes_cli.plugins import (
    PluginManager,
    get_bundled_plugins_dir,
    is_platform_plugin_disabled,
)


# ────────────────────────────────────────────────────────────────────────────
# Plugin loader: key alignment + disable matching
# ────────────────────────────────────────────────────────────────────────────


class TestBundledPlatformKeyAlignment:
    """Bundled platform plugins must register under ``platforms/<name>``."""

    @pytest.fixture()
    def loaded_platform_plugins(self) -> dict[str, Any]:
        """Run the real loader and return its bundled platform manifests.

        The fixture exercises the production discover_and_load() path —
        the same one the gateway invokes at startup — so any future change
        that re-introduces the key mismatch fails these tests immediately.
        """
        manager = PluginManager()
        # Avoid mutating the user's HERMES_HOME during the test.
        with patch("hermes_cli.plugins._get_disabled_plugins", return_value=set()):
            manager.discover_and_load()
        return {
            key: loaded
            for key, loaded in manager._plugins.items()
            if loaded.manifest.kind == "platform"
        }

    def test_discord_uses_path_derived_key(self, loaded_platform_plugins):
        """discord must show up as 'platforms/discord', not 'discord-platform'."""
        assert "platforms/discord" in loaded_platform_plugins, (
            f"Expected 'platforms/discord' key, got: "
            f"{sorted(loaded_platform_plugins)}"
        )
        assert "discord-platform" not in loaded_platform_plugins

    def test_all_bundled_platforms_use_path_derived_keys(self, loaded_platform_plugins):
        """Every bundled adapter under plugins/platforms/ must use the
        category-prefixed key so the CLI list / disable commands stay in
        sync with the loader.
        """
        expected = {
            p.name
            for p in sorted((get_bundled_plugins_dir() / "platforms").iterdir())
            if p.is_dir() and (p / "plugin.yaml").exists()
        }
        # All bundled platforms should be discoverable through the
        # path-derived key.  Empty set would mean the bundle moved.
        assert expected, "No bundled platform plugins discovered — test setup is off"
        observed_keys = set(loaded_platform_plugins)
        for platform_dir in expected:
            assert f"platforms/{platform_dir}" in observed_keys, (
                f"Bundled platform {platform_dir!r} did not register under "
                f"'platforms/{platform_dir}'. Got keys: {sorted(observed_keys)}"
            )

    def test_path_derived_key_preserved_on_manifest(self, loaded_platform_plugins):
        """The PluginManifest.key field itself should also carry the
        category prefix, not just the lookup-key in _plugins.
        """
        for key, loaded in loaded_platform_plugins.items():
            assert loaded.manifest.key == key, (
                f"Manifest.key {loaded.manifest.key!r} does not match "
                f"lookup key {key!r}"
            )
            assert loaded.manifest.key.startswith("platforms/"), (
                f"Bundled platform plugin {key!r} keyed without category prefix"
            )


class TestDisableHonoursBothKeyForms:
    """`hermes plugins disable platforms/discord` actually disables it."""

    def _load_with_disabled(self, disabled: set[str]) -> PluginManager:
        manager = PluginManager()
        with patch(
            "hermes_cli.plugins._get_disabled_plugins", return_value=disabled
        ):
            manager.discover_and_load()
        return manager

    def test_disable_with_path_derived_key_blocks_plugin(self):
        """User runs `hermes plugins disable platforms/discord` → discord
        plugin does not load.
        """
        manager = self._load_with_disabled({"platforms/discord"})
        discord = manager._plugins.get("platforms/discord")
        assert discord is not None, (
            "Discord manifest should still be discovered (just not loaded) "
            "so `hermes plugins list` can render its 'disabled' state"
        )
        assert discord.enabled is False
        assert discord.error == "disabled via config"
        assert discord.module is None, "Disabled plugin must not have an imported module"

    def test_disable_with_legacy_manifest_name_still_blocks_plugin(self):
        """Back-compat: users who edited config.yaml by hand before the
        key alignment landed may have ``discord-platform`` in their
        disabled list.  That form must still disable the plugin.
        """
        manager = self._load_with_disabled({"discord-platform"})
        discord = manager._plugins.get("platforms/discord")
        assert discord is not None
        assert discord.enabled is False
        assert discord.error == "disabled via config"

    def test_disable_other_platform_does_not_affect_discord(self):
        """Disabling teams must not bleed over into discord."""
        manager = self._load_with_disabled({"platforms/teams"})
        discord = manager._plugins.get("platforms/discord")
        teams = manager._plugins.get("platforms/teams")
        assert discord is not None
        assert teams is not None
        assert teams.enabled is False
        assert discord.enabled is True, (
            "Disabling teams must not disable discord — disable is per-plugin"
        )

    def test_empty_disabled_set_loads_all_platforms(self):
        manager = self._load_with_disabled(set())
        for key, loaded in manager._plugins.items():
            if loaded.manifest.kind != "platform":
                continue
            assert loaded.enabled is True, (
                f"Bundled platform {key!r} should auto-load when nothing is disabled"
            )

    def test_is_platform_plugin_disabled_path_form(self):
        with patch(
            "hermes_cli.plugins._get_disabled_plugins",
            return_value={"platforms/discord"},
        ):
            assert is_platform_plugin_disabled("discord") is True
            assert is_platform_plugin_disabled("teams") is False

    def test_is_platform_plugin_disabled_legacy_name_form(self):
        """Legacy ``<name>-platform`` form (manifest.name) still resolves."""
        with patch(
            "hermes_cli.plugins._get_disabled_plugins",
            return_value={"discord-platform"},
        ):
            assert is_platform_plugin_disabled("discord") is True
            assert is_platform_plugin_disabled("teams") is False

    def test_is_platform_plugin_disabled_empty_set(self):
        with patch("hermes_cli.plugins._get_disabled_plugins", return_value=set()):
            assert is_platform_plugin_disabled("discord") is False

    def test_is_platform_plugin_disabled_empty_name_is_safe(self):
        with patch(
            "hermes_cli.plugins._get_disabled_plugins",
            return_value={"platforms/discord"},
        ):
            assert is_platform_plugin_disabled("") is False
            assert is_platform_plugin_disabled(None) is False  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────────────
# Gateway runtime: skip disabled platform plugins cleanly
# ────────────────────────────────────────────────────────────────────────────


class _ShouldNotBeCreatedAdapter(BasePlatformAdapter):
    """Sentinel adapter — its instantiation signals a regression.

    If the gateway invokes the platform-creation path for a platform whose
    plugin is disabled, that's the user-visible bug from #30736: connection
    attempts, ERROR-level "No bot token" logs, and reconnect-watcher churn.
    Surfacing the regression as an AssertionError makes the failure mode
    unambiguous in CI.
    """

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token=""), Platform.DISCORD)
        raise AssertionError(
            "Disabled-plugin platform must not reach _create_adapter "
            "— this is the #30736 regression"
        )

    async def connect(self) -> bool:  # pragma: no cover - guarded by __init__
        return False

    async def disconnect(self) -> None:  # pragma: no cover
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        raise NotImplementedError

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return {"id": chat_id}


class _OkAdapter(BasePlatformAdapter):
    """Trivial successful adapter so the gateway has at least one working
    platform alongside the disabled one (keeps ``connected_count`` > 0 so
    the cron-only branch doesn't kick in)."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="ok"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture()
def _isolated_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _make_runner(*platforms: Platform, hermes_home) -> GatewayRunner:
    config = GatewayConfig(
        platforms={p: PlatformConfig(enabled=True, token="x") for p in platforms},
        sessions_dir=hermes_home / "sessions",
    )
    return GatewayRunner(config)


@pytest.mark.asyncio
class TestGatewaySkipsDisabledPlatformPlugin:
    """Gateway startup must not touch an adapter whose plugin is disabled."""

    async def test_disabled_platform_is_skipped_before_create_adapter(
        self, _isolated_hermes_home, caplog
    ):
        """When platforms/discord is disabled, _create_adapter must NOT be
        called for discord, the platform must NOT enter _failed_platforms,
        and a single user-friendly INFO log must explain the skip."""
        runner = _make_runner(
            Platform.DISCORD, Platform.TELEGRAM, hermes_home=_isolated_hermes_home
        )

        # The Telegram side returns a working adapter; the Discord side
        # would explode loudly if it were ever asked to instantiate (see
        # _ShouldNotBeCreatedAdapter).  Combined with the patch on
        # is_platform_plugin_disabled, this asserts that the gateway
        # short-circuits BEFORE reaching _create_adapter for discord.
        def _factory(platform, _platform_config):
            if platform == Platform.TELEGRAM:
                return _OkAdapter()
            if platform == Platform.DISCORD:
                return _ShouldNotBeCreatedAdapter()
            return None

        with patch("gateway.run.GatewayRunner._create_adapter", side_effect=_factory), \
             patch(
                "hermes_cli.plugins.is_platform_plugin_disabled",
                side_effect=lambda name: name == "discord",
             ), \
             caplog.at_level(logging.INFO, logger="gateway.run"):
            ok = await runner.start()

        assert ok is True
        # Disabled platform never gets queued for reconnect — the whole
        # point of the fix is that there's no retry storm.
        assert Platform.DISCORD not in runner._failed_platforms
        # Telegram still connected successfully.
        assert Platform.TELEGRAM in runner.adapters
        # Disabled platform is reflected in the runtime status snapshot.
        state = read_runtime_status()
        assert state["platforms"]["discord"]["state"] == "disabled"

    async def test_skip_emits_actionable_info_log(
        self, _isolated_hermes_home, caplog
    ):
        runner = _make_runner(
            Platform.DISCORD, Platform.TELEGRAM, hermes_home=_isolated_hermes_home
        )

        def _factory(platform, _platform_config):
            if platform == Platform.TELEGRAM:
                return _OkAdapter()
            return _ShouldNotBeCreatedAdapter()

        with patch("gateway.run.GatewayRunner._create_adapter", side_effect=_factory), \
             patch(
                "hermes_cli.plugins.is_platform_plugin_disabled",
                side_effect=lambda name: name == "discord",
             ), \
             caplog.at_level(logging.INFO, logger="gateway.run"):
            await runner.start()

        # Exactly one INFO record explains the skip and names the action
        # that would re-enable the plugin.
        skip_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO
            and "Skipping platform 'discord'" in r.getMessage()
        ]
        assert len(skip_records) == 1, (
            f"Expected exactly one skip INFO log; got: "
            f"{[r.getMessage() for r in skip_records]}"
        )
        msg = skip_records[0].getMessage()
        assert "plugin is disabled" in msg
        assert "hermes plugins enable platforms/discord" in msg

    async def test_no_error_logged_for_disabled_platform(
        self, _isolated_hermes_home, caplog
    ):
        """The original issue's most visible symptom was an ERROR per
        startup ('[Discord] No bot token configured') plus warnings.  The
        skip path must not produce ERROR or WARNING records for the
        disabled platform.
        """
        runner = _make_runner(Platform.DISCORD, hermes_home=_isolated_hermes_home)

        with patch(
            "gateway.run.GatewayRunner._create_adapter",
            side_effect=_ShouldNotBeCreatedAdapter,
        ), patch(
            "hermes_cli.plugins.is_platform_plugin_disabled",
            side_effect=lambda name: name == "discord",
        ), caplog.at_level(logging.DEBUG, logger="gateway.run"):
            await runner.start()

        for rec in caplog.records:
            if rec.levelno < logging.WARNING:
                continue
            msg = rec.getMessage().lower()
            # A warning about discord specifically means we did NOT skip
            # cleanly.  Generic startup warnings unrelated to discord are
            # fine.
            assert "discord" not in msg or "no adapter" not in msg, (
                f"Unexpected warning/error for disabled discord plugin: {rec.getMessage()}"
            )
            assert "discord" not in msg or "no bot token" not in msg, (
                f"Unexpected adapter-level error for disabled discord plugin: {rec.getMessage()}"
            )

    async def test_enabled_platform_still_connects_normally(
        self, _isolated_hermes_home
    ):
        """Sanity check: the skip path must NOT affect enabled platforms."""
        runner = _make_runner(Platform.TELEGRAM, hermes_home=_isolated_hermes_home)

        with patch(
            "gateway.run.GatewayRunner._create_adapter",
            side_effect=lambda *_args, **_kwargs: _OkAdapter(),
        ), patch(
            "hermes_cli.plugins.is_platform_plugin_disabled",
            return_value=False,
        ):
            ok = await runner.start()

        assert ok is True
        assert Platform.TELEGRAM in runner.adapters
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "connected"

    async def test_platform_disabled_via_platform_config_takes_precedence(
        self, _isolated_hermes_home
    ):
        """``platforms.<name>.enabled=False`` already short-circuits the
        loop before the plugin-disabled check runs — the existing 'enabled'
        gate must keep working.  This guards against the new check
        accidentally inverting the order or duplicating behavior.
        """
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=False, token=""),
            },
            sessions_dir=_isolated_hermes_home / "sessions",
        )
        runner = GatewayRunner(config)

        # Both factory and is_platform_plugin_disabled would surface a
        # regression if invoked — but neither should be reached because
        # the platform is disabled at the config level.
        with patch(
            "gateway.run.GatewayRunner._create_adapter",
            side_effect=_ShouldNotBeCreatedAdapter,
        ), patch(
            "hermes_cli.plugins.is_platform_plugin_disabled",
            side_effect=AssertionError(
                "platform-config-disabled platforms must not hit the plugin check"
            ),
        ):
            ok = await runner.start()

        assert ok is True
        assert Platform.DISCORD not in runner.adapters
        assert Platform.DISCORD not in runner._failed_platforms

    async def test_is_platform_plugin_disabled_exception_is_caught(
        self, _isolated_hermes_home, caplog
    ):
        """If the disable lookup itself throws (e.g. corrupt config.yaml),
        the gateway must not crash — it should fall through to the normal
        adapter-creation path and let downstream layers report whatever
        the underlying issue is.
        """
        runner = _make_runner(Platform.TELEGRAM, hermes_home=_isolated_hermes_home)

        with patch(
            "gateway.run.GatewayRunner._create_adapter",
            side_effect=lambda *_a, **_k: _OkAdapter(),
        ), patch(
            "hermes_cli.plugins.is_platform_plugin_disabled",
            side_effect=RuntimeError("boom"),
        ):
            ok = await runner.start()

        assert ok is True
        # Telegram still connected: the exception was swallowed and the
        # adapter path proceeded normally.
        assert Platform.TELEGRAM in runner.adapters
