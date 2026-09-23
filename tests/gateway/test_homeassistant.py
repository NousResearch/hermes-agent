"""Tests for the Home Assistant gateway adapter.

Tests real logic: state change formatting, event filtering pipeline,
cooldown behavior, config integration, and adapter initialization.
"""

import errno
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from plugins.platforms.homeassistant.adapter import (
    HomeAssistantAdapter,
    check_ha_requirements,
    validate_ha_config,
)
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# check_ha_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:


    @patch("plugins.platforms.homeassistant.adapter.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        assert check_ha_requirements() is False

    def test_validate_config_accepts_platform_token(self, monkeypatch):
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        config = PlatformConfig(enabled=True, token="config-token")
        assert validate_ha_config(config) is True


class TestValidateConfig:
    def test_returns_false_without_token_in_config_or_env(self, monkeypatch):
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        assert validate_ha_config(PlatformConfig(enabled=True)) is False


# ---------------------------------------------------------------------------
# _format_state_change - pure function, all domain branches
# ---------------------------------------------------------------------------


class TestFormatStateChange:
    @staticmethod
    def fmt(entity_id, old_state, new_state):
        return HomeAssistantAdapter._format_state_change(entity_id, old_state, new_state)

    def test_climate_includes_temperatures(self):
        msg = self.fmt(
            "climate.thermostat",
            {"state": "off"},
            {"state": "heat", "attributes": {
                "friendly_name": "Main Thermostat",
                "current_temperature": 21.5,
                "temperature": 23,
            }},
        )
        assert "Main Thermostat" in msg
        assert "'off'" in msg and "'heat'" in msg
        assert "21.5" in msg and "23" in msg

    def test_sensor_includes_unit(self):
        msg = self.fmt(
            "sensor.temperature",
            {"state": "22.5"},
            {"state": "25.1", "attributes": {
                "friendly_name": "Living Room Temp",
                "unit_of_measurement": "C",
            }},
        )
        assert "22.5C" in msg and "25.1C" in msg
        assert "Living Room Temp" in msg


    def test_binary_sensor_on(self):
        msg = self.fmt(
            "binary_sensor.motion",
            {"state": "off"},
            {"state": "on", "attributes": {"friendly_name": "Hallway Motion"}},
        )
        assert "triggered" in msg
        assert "Hallway Motion" in msg


    def test_light_turned_on(self):
        msg = self.fmt(
            "light.bedroom",
            {"state": "off"},
            {"state": "on", "attributes": {"friendly_name": "Bedroom Light"}},
        )
        assert "turned on" in msg

    def test_switch_turned_off(self):
        msg = self.fmt(
            "switch.heater",
            {"state": "on"},
            {"state": "off", "attributes": {"friendly_name": "Heater"}},
        )
        assert "turned off" in msg


# ---------------------------------------------------------------------------
# Adapter initialization from config
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_url_and_token_from_config_extra(self, monkeypatch):
        monkeypatch.delenv("HASS_URL", raising=False)
        monkeypatch.delenv("HASS_TOKEN", raising=False)

        config = PlatformConfig(
            enabled=True,
            token="config-token",
            extra={"url": "http://192.168.1.50:8123"},
        )
        adapter = HomeAssistantAdapter(config)
        assert adapter._hass_token == "config-token"
        assert adapter._hass_url == "http://192.168.1.50:8123"


    def test_watch_filters_parsed(self):
        config = PlatformConfig(
            enabled=True, token="***",
            extra={
                "watch_domains": ["climate", "binary_sensor"],
                "watch_entities": ["sensor.special"],
                "ignore_entities": ["sensor.uptime", "sensor.cpu"],
                "cooldown_seconds": 120,
            },
        )
        adapter = HomeAssistantAdapter(config)
        assert adapter._watch_domains == {"climate", "binary_sensor"}
        assert adapter._watch_entities == {"sensor.special"}
        assert adapter._ignore_entities == {"sensor.uptime", "sensor.cpu"}
        assert adapter._watch_all is False
        assert adapter._cooldown_seconds == 120


# ---------------------------------------------------------------------------
# Event filtering pipeline (_handle_ha_event)
#
# We mock handle_message (not our code, it's the base class pipeline) to
# capture the MessageEvent that _handle_ha_event produces.
# ---------------------------------------------------------------------------


def _make_adapter(**extra) -> HomeAssistantAdapter:
    config = PlatformConfig(enabled=True, token="tok", extra=extra)
    adapter = HomeAssistantAdapter(config)
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(entity_id, old_state, new_state, old_attrs=None, new_attrs=None):
    return {
        "data": {
            "entity_id": entity_id,
            "old_state": {"state": old_state, "attributes": old_attrs or {}},
            "new_state": {"state": new_state, "attributes": new_attrs or {"friendly_name": entity_id}},
        }
    }


class TestEventFilteringPipeline:
    @pytest.mark.asyncio
    async def test_ignored_entity_not_forwarded(self):
        adapter = _make_adapter(watch_all=True, ignore_entities=["sensor.uptime"])
        await adapter._handle_ha_event(_make_event("sensor.uptime", "100", "101"))
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unwatched_domain_not_forwarded(self):
        adapter = _make_adapter(watch_domains=["climate"])
        await adapter._handle_ha_event(_make_event("light.bedroom", "off", "on"))
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_watched_domain_forwarded(self):
        adapter = _make_adapter(watch_domains=["climate"], cooldown_seconds=0)
        await adapter._handle_ha_event(
            _make_event("climate.thermostat", "off", "heat",
                        new_attrs={"friendly_name": "Thermostat", "current_temperature": 20, "temperature": 22})
        )
        adapter.handle_message.assert_called_once()

        # Verify the actual MessageEvent text content
        msg_event = adapter.handle_message.call_args[0][0]
        assert "Thermostat" in msg_event.text
        assert "heat" in msg_event.text
        assert msg_event.source.platform == Platform.HOMEASSISTANT
        assert msg_event.source.chat_id == "ha_events"


# ---------------------------------------------------------------------------
# Cooldown behavior
# ---------------------------------------------------------------------------


class TestCooldown:

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        adapter = _make_adapter(watch_all=True, cooldown_seconds=1)

        event = _make_event("sensor.temp", "20", "21",
                            new_attrs={"friendly_name": "Temp"})
        await adapter._handle_ha_event(event)
        assert adapter.handle_message.call_count == 1

        # Simulate time passing beyond cooldown
        adapter._last_event_time["sensor.temp"] = time.time() - 2

        event2 = _make_event("sensor.temp", "21", "22",
                             new_attrs={"friendly_name": "Temp"})
        await adapter._handle_ha_event(event2)
        assert adapter.handle_message.call_count == 2


# ---------------------------------------------------------------------------
# Config integration (env overrides, round-trip)
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_env_override_creates_ha_platform(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "env-token")
        monkeypatch.setenv("HASS_URL", "http://10.0.0.5:8123")
        # Clear other platform tokens
        for v in ["TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN"]:
            monkeypatch.delenv(v, raising=False)

        from gateway.config import load_gateway_config
        config = load_gateway_config()

        assert Platform.HOMEASSISTANT in config.platforms
        ha = config.platforms[Platform.HOMEASSISTANT]
        assert ha.enabled is True
        assert ha.token == "env-token"
        assert ha.extra["url"] == "http://10.0.0.5:8123"


# ---------------------------------------------------------------------------
# send() via REST API
# ---------------------------------------------------------------------------


class TestSendViaRestApi:
    """send() uses REST API (not WebSocket) to avoid race conditions."""

    @staticmethod
    def _mock_aiohttp_session(response_status=200, response_text="OK"):
        """Build a mock aiohttp session + response for async-with patterns.

        aiohttp.ClientSession() is a sync constructor whose return value
        is used as ``async with session:``.  ``session.post(...)`` returns a
        context-manager (not a coroutine), so both layers use MagicMock for
        the call and AsyncMock only for ``__aenter__`` / ``__aexit__``.
        """
        mock_response = MagicMock()
        mock_response.status = response_status
        mock_response.text = AsyncMock(return_value=response_text)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        return mock_session

    @pytest.mark.asyncio
    async def test_send_success(self):
        adapter = _make_adapter()
        mock_session = self._mock_aiohttp_session(200)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events", "Test notification")

        assert result.success is True
        # Verify the REST API was called with correct payload
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]
        assert call_args[1]["json"]["title"] == "Hermes Agent"
        assert call_args[1]["json"]["message"] == "Test notification"
        assert "Bearer tok" in call_args[1]["headers"]["Authorization"]


# ---------------------------------------------------------------------------
# Toolset integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WebSocket URL construction
# ---------------------------------------------------------------------------


class TestWsUrlConstruction:
    def test_http_to_ws(self):
        config = PlatformConfig(enabled=True, token="t", extra={"url": "http://ha:8123"})
        adapter = HomeAssistantAdapter(config)
        ws_url = adapter._hass_url.replace("http://", "ws://").replace("https://", "wss://")
        assert ws_url == "ws://ha:8123"


class TestLocalNetworkConnectHint:
    def test_ehostunreach_outside_launchd_is_a_plain_unreachable_host(self, monkeypatch):
        """The same errno from a Terminal-run gateway (or another OS) must not be blamed on macOS."""
        from plugins.platforms.homeassistant.adapter import _connect_error_detail

        monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
        err = OSError(errno.EHOSTUNREACH, "No route to host")
        assert _connect_error_detail(err) == str(err)
        assert _connect_error_detail(RuntimeError("auth failed")) == "auth failed"

    @pytest.mark.macos_only
    def test_ehostunreach_under_launchd_names_the_remedy(self, monkeypatch):
        """Only the launchd-supervised gateway can be denied by Local Network Privacy (#71206)."""
        from plugins.platforms.homeassistant.adapter import _connect_error_detail

        monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
        err = OSError(errno.EHOSTUNREACH, "No route to host")
        detail = _connect_error_detail(err)
        assert detail.startswith(str(err))
        assert "Local Network" in detail
        assert "hermes gateway install" in detail
        assert "71206" in detail


# ---------------------------------------------------------------------------
# Deliver target config parsing (issue #35060)
# ---------------------------------------------------------------------------


def _make_deliver_adapter(**extra) -> HomeAssistantAdapter:
    """Helper: instantiate adapter with no network, for deliver tests."""
    config = PlatformConfig(enabled=True, token="tok", extra=extra)
    return HomeAssistantAdapter(config)


def _fake_home_channel(chat_id: str):
    """Minimal stand-in for a GatewayConfig home-channel entry."""
    home = MagicMock()
    home.chat_id = chat_id
    return home


class TestDeliverTargetParsing:
    """Configurable deliver-target parsing for HA watch config.

    watch_entities and watch_domains entries accept either a plain string
    (unchanged behavior) or a single-entry dict {entity_or_domain: {"deliver": "<platform>"}}.
    Also reads optional top-level "deliver" / "default_deliver" as the default.
    Precedence: per-entry deliver > top-level deliver > "homeassistant".
    """

    # -- plain-string entries -------------------------------------------------

    def test_plain_string_entities_default_to_homeassistant(self):
        """Plain-string watch_entities entries resolve to 'homeassistant'."""
        adapter = _make_deliver_adapter(
            watch_entities=["sensor.temp", "light.bedroom"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "homeassistant"
        assert adapter.resolve_deliver_target("light.bedroom") == "homeassistant"

    def test_plain_string_domains_default_to_homeassistant(self):
        """Plain-string watch_domains entries resolve to 'homeassistant'."""
        adapter = _make_deliver_adapter(
            watch_domains=["climate", "binary_sensor"],
        )
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"
        assert adapter.resolve_deliver_target("binary_sensor.motion") == "homeassistant"

    # -- dict-form entries ----------------------------------------------------

    def test_dict_form_entity_resolves_per_entity(self):
        """Dict-form watch_entities entry sets deliver target per entity."""
        adapter = _make_deliver_adapter(
            watch_entities=[{"sensor.temp": {"deliver": "whatsapp"}}],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "whatsapp"
        # Other entities still default to "homeassistant"
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"

    def test_dict_form_domain_resolves_by_domain(self):
        """Dict-form watch_domains entry sets deliver target for any entity in that domain."""
        adapter = _make_deliver_adapter(
            watch_domains=[{"climate": {"deliver": "telegram"}}],
        )
        assert adapter.resolve_deliver_target("climate.thermostat") == "telegram"
        assert adapter.resolve_deliver_target("climate.ac") == "telegram"
        # Unknown domain defaults
        assert adapter.resolve_deliver_target("light.bedroom") == "homeassistant"

    # -- top-level deliver / default_deliver ----------------------------------

    def test_top_level_deliver_sets_default(self):
        """Top-level 'deliver' key sets the default deliver target for all watch items."""
        adapter = _make_deliver_adapter(
            deliver="telegram",
            watch_entities=["sensor.temp", "light.bedroom"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "telegram"
        assert adapter.resolve_deliver_target("light.bedroom") == "telegram"

    def test_default_deliver_alias_sets_default(self):
        """Top-level 'default_deliver' alias sets the default deliver target."""
        adapter = _make_deliver_adapter(
            default_deliver="slack",
            watch_entities=["sensor.temp"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "slack"

    # -- precedence -----------------------------------------------------------

    def test_per_entry_overrides_top_level_deliver(self):
        """Per-entry deliver target overrides the top-level default."""
        adapter = _make_deliver_adapter(
            deliver="slack",
            watch_entities=[{"sensor.temp": {"deliver": "whatsapp"}}],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "whatsapp"
        assert adapter.resolve_deliver_target("climate.thermostat") == "slack"

    def test_top_level_deliver_overrides_implicit_default(self):
        """Top-level deliver overrides the implicit 'homeassistant' default."""
        adapter = _make_deliver_adapter(
            deliver="discord",
            watch_entities=["sensor.temp"],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "discord"
        assert adapter.resolve_deliver_target("climate.thermostat") == "discord"

    # -- malformed entries ----------------------------------------------------

    def test_malformed_entry_dict_with_multiple_keys_skipped(self, caplog):
        """Dict with >1 key is skipped with a warning."""
        adapter = _make_deliver_adapter(
            watch_entities=[{"sensor.temp": {"deliver": "whatsapp"}, "extra": "bad"}],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "homeassistant"
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"
        assert "Malformed" in caplog.text or "skipping" in caplog.text

    def test_malformed_entry_deliver_not_str_skipped(self, caplog):
        """Dict-form entry with non-string deliver value is skipped with a warning."""
        adapter = _make_deliver_adapter(
            watch_entities=[{"sensor.temp": {"deliver": 123}}],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.temp") == "homeassistant"
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"
        assert "Malformed" in caplog.text or "skipping" in caplog.text

    def test_malformed_entry_non_str_non_dict_skipped(self, caplog):
        """Entry that is neither str nor dict is skipped with a warning."""
        adapter = _make_deliver_adapter(
            watch_entities=[42, "sensor.valid"],
            watch_domains=["climate"],
        )
        assert adapter.resolve_deliver_target("sensor.valid") == "homeassistant"
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"
        assert "Malformed" in caplog.text or "skipping" in caplog.text

    @pytest.mark.parametrize("bad_entry", [
        {"climate": {"deliver": "whatsapp"}, "extra": "bad"},  # dict with >1 key
        {"climate": {"deliver": 123}},                          # non-string deliver
        42,                                                     # neither str nor dict
    ])
    def test_malformed_domain_entries_skipped(self, caplog, bad_entry):
        """watch_domains malformed entries are skipped with a warning, same as watch_entities."""
        adapter = _make_deliver_adapter(watch_domains=[bad_entry, "cover"])
        assert adapter.resolve_deliver_target("climate.thermostat") == "homeassistant"
        assert adapter.resolve_deliver_target("cover.garage") == "homeassistant"
        assert "Malformed" in caplog.text or "skipping" in caplog.text

    def test_malformed_entries_dont_break_startup(self):
        """Multiple malformed entries don't raise at startup."""
        adapter = _make_deliver_adapter(
            watch_entities=[
                {"sensor.a": {"deliver": "whatsapp", "extra": "bad"}},
                {"sensor.b": {"deliver": 123}},
                42,
                "sensor.valid",
            ],
            watch_domains=[{"climate.ac": {"deliver": "telegram"}}],
        )
        # The adapter should be constructable and queryable
        assert adapter.resolve_deliver_target("sensor.valid") == "homeassistant"

    # -- resolve_deliver_target interface -------------------------------------

    def test_resolve_deliver_target_unknown_entity_uses_default(self):
        """Entity not in any watch list still resolves to the default deliver target."""
        adapter = _make_deliver_adapter(
            deliver="telegram",
            watch_entities=["sensor.temp"],
            watch_domains=["climate"],
        )
        # Unknown entity - not watched, but resolve_deliver_target still returns default
        assert adapter.resolve_deliver_target("light.unknown") == "telegram"

    # -- M2: null-safe parsing and dotless id --------------------------------

    def test_watch_config_none_does_not_crash(self):
        """Extra with watch_entities=None and watch_domains=None does not crash at construction."""
        adapter = _make_deliver_adapter(watch_entities=None, watch_domains=None)
        assert adapter._watch_entities == set()
        assert adapter._watch_domains == set()

    def test_resolve_deliver_target_dotless_id_falls_through(self):
        """resolve_deliver_target with a dotless id falls through to domain lookup / default."""
        adapter = _make_deliver_adapter()
        # "climate" has no dot, no overrides exist — falls through to default
        assert adapter.resolve_deliver_target("climate") == "homeassistant"

    def test_resolve_deliver_target_entity_overrides_domain(self):
        """When both an entity key and its domain key have overrides, the entity (most specific) wins."""
        adapter = _make_deliver_adapter(
            watch_entities=[{"sensor.camera": {"deliver": "signal"}}],
            watch_domains=[{"sensor": {"deliver": "whatsapp"}}],
        )
        # Exact entity match beats the domain match
        assert adapter.resolve_deliver_target("sensor.camera") == "signal"
        # Sibling entities on the watched domain still use the domain override
        assert adapter.resolve_deliver_target("sensor.motion") == "whatsapp"


# ---------------------------------------------------------------------------
# Cross-platform delivery routing in send() (issue #35060)
# ---------------------------------------------------------------------------


class TestDeliverRouting:
    """send() routes cross-platform when chat_id uses the ha_events: prefix."""

    @staticmethod
    def _stub_adapter(send_result=None):
        """Build a minimal stub adapter with async send()."""
        stub = MagicMock()
        if send_result is not None:
            stub.send = AsyncMock(return_value=send_result)
        else:
            stub.send = AsyncMock(return_value=SendResult(success=True))
        return stub

    @staticmethod
    def _make_ha_adapter(**extra) -> HomeAssistantAdapter:
        config = PlatformConfig(enabled=True, token="tok", extra=extra)
        adapter = HomeAssistantAdapter(config)
        return adapter

    def _stub_runner(self, target_platform, target_adapter=None, home_chat_id=None):
        """Build a stub gateway runner that returns a target adapter."""
        runner = MagicMock()
        runner.adapters = {}
        if target_adapter is not None:
            runner.adapters[target_platform] = target_adapter
        runner.config = MagicMock()

        if home_chat_id is not None:

            class _FakeHomeChannel:
                chat_id = home_chat_id

            runner.config.get_home_channel = MagicMock(return_value=_FakeHomeChannel())
        else:
            runner.config.get_home_channel = MagicMock(return_value=None)
        runner._profile_adapters = {}

        def _authorization_adapter(platform, profile=None):
            """Mirror GatewayAuthorizationMixin: own profile's map only, fail closed."""
            if profile and profile != "default":
                return (runner._profile_adapters or {}).get(profile, {}).get(platform)
            return runner.adapters.get(platform)

        runner._authorization_adapter = MagicMock(side_effect=_authorization_adapter)
        return runner

    @pytest.mark.asyncio
    async def test_send_routes_to_target_adapter_when_chat_id_has_prefix(self):
        """send('ha_events:telegram', ...) routes to the telegram adapter."""
        adapter = self._make_ha_adapter()
        target_adapter = self._stub_adapter()
        runner = self._stub_runner(
            Platform.TELEGRAM, target_adapter=target_adapter, home_chat_id="chat_42"
        )
        adapter.gateway_runner = runner

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock()
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events:telegram", "hello from HA")

        assert result.success is True
        # Target adapter should have been called with the home channel's chat_id
        target_adapter.send.assert_called_once_with("chat_42", "hello from HA", metadata=None)

    @pytest.mark.asyncio
    async def test_send_ha_events_no_prefix_stays_local(self):
        """send('ha_events', ...) without colon suffix stays in HA notification path."""
        adapter = self._make_ha_adapter()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events", "direct notification")

        assert result.success is True
        # Verify HA REST API was called
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_ha_when_no_gateway_runner(self):
        """send('ha_events:telegram', ...) falls back to HA notification when gateway_runner is None."""
        adapter = self._make_ha_adapter()
        # gateway_runner is None by default
        assert adapter.gateway_runner is None

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events:telegram", "fallback content")

        assert result.success is True
        # HA notification path should have been used
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_ha_when_target_adapter_missing(self):
        """send('ha_events:telegram', ...) falls back when target adapter is not connected."""
        adapter = self._make_ha_adapter()
        runner = self._stub_runner(Platform.TELEGRAM, target_adapter=None)
        adapter.gateway_runner = runner

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events:telegram", "fallback content")

        assert result.success is True
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_ha_when_no_home_channel(self):
        """send('ha_events:telegram', ...) falls back when home channel is missing."""
        adapter = self._make_ha_adapter()
        target_adapter = self._stub_adapter()
        runner = self._stub_runner(
            Platform.TELEGRAM, target_adapter=target_adapter, home_chat_id=None
        )
        adapter.gateway_runner = runner

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events:telegram", "fallback content")

        assert result.success is True
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_ha_when_unknown_platform(self):
        """send('ha_events:unknown_platform', ...) falls back to HA notification."""
        adapter = self._make_ha_adapter()
        runner = self._stub_runner(None)  # No target registered at all
        adapter.gateway_runner = runner

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
            mock_aiohttp.ClientTimeout = lambda total: total

            result = await adapter.send("ha_events:unknown_platform", "fallback content")

        assert result.success is True
        call_args = mock_session.post.call_args
        assert "/api/services/persistent_notification/create" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_ha_event_sets_tagged_chat_id_for_cross_platform(self):
        """_handle_ha_event sets chat_id to ha_events:telegram when deliver target is telegram."""
        adapter = self._make_ha_adapter(
            watch_entities=[{"sensor.temp": {"deliver": "telegram"}}],
            cooldown_seconds=0,
        )
        adapter.handle_message = AsyncMock()
        await adapter._handle_ha_event(
            _make_event("sensor.temp", "22", "25",
                        new_attrs={"friendly_name": "Temp Sensor", "unit_of_measurement": "C"})
        )
        adapter.handle_message.assert_called_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.source.chat_id == "ha_events:telegram"

    @pytest.mark.asyncio
    async def test_handle_ha_event_leaves_default_chat_id_for_ha(self):
        """_handle_ha_event keeps default chat_id when deliver target is homeassistant."""
        adapter = self._make_ha_adapter(
            watch_entities=["sensor.temp"],
            cooldown_seconds=0,
        )
        adapter.handle_message = AsyncMock()
        await adapter._handle_ha_event(
            _make_event("sensor.temp", "22", "25",
                        new_attrs={"friendly_name": "Temp Sensor", "unit_of_measurement": "C"})
        )
        adapter.handle_message.assert_called_once()
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.source.chat_id == "ha_events"

    @pytest.mark.asyncio
    async def test_send_accepts_capitalized_platform_name(self):
        """send('ha_events:Telegram', ...) normalizes case and routes correctly."""
        adapter = self._make_ha_adapter()
        target_adapter = self._stub_adapter()
        runner = self._stub_runner(
            Platform.TELEGRAM, target_adapter=target_adapter, home_chat_id="chat_42"
        )
        adapter.gateway_runner = runner

        result = await adapter.send("ha_events:Telegram", "capitalized alert")

        assert result.success is True
        target_adapter.send.assert_called_once_with("chat_42", "capitalized alert", metadata=None)

    @pytest.mark.asyncio
    async def test_send_falls_back_to_ha_when_target_adapter_raises(self):
        """A raise from the target adapter falls back to HA notification."""
        adapter = self._make_ha_adapter()
        target_adapter = self._stub_adapter()
        target_adapter.send = AsyncMock(side_effect=RuntimeError("network exploded"))
        runner = self._stub_runner(
            Platform.TELEGRAM, target_adapter=target_adapter, home_chat_id="chat_42"
        )
        adapter.gateway_runner = runner

        with patch(
            "plugins.platforms.homeassistant.adapter.HomeAssistantAdapter._send_ha_notification",
            new_callable=AsyncMock,
            return_value=SendResult(success=True),
        ) as mock_ha_fallback:
            result = await adapter.send("ha_events:telegram", "fallback alert")

        assert result.success is True
        mock_ha_fallback.assert_awaited_once_with("fallback alert")

    @pytest.mark.asyncio
    async def test_send_uses_own_profile_adapter_and_home_channel(self):
        """A secondary profile delivers through ITS OWN adapter and ITS OWN home channel."""
        adapter = self._make_ha_adapter()
        adapter._owner_profile = "profile_1"
        target_adapter = self._stub_adapter()
        runner = self._stub_runner(Platform.TELEGRAM, target_adapter=None)
        runner._profile_adapters = {"profile_1": {Platform.TELEGRAM: target_adapter}}
        adapter.gateway_runner = runner
        adapter._target_home_channel = MagicMock(return_value=_fake_home_channel("chat_42"))

        result = await adapter.send("ha_events:telegram", "profile alert")

        assert result.success is True
        target_adapter.send.assert_called_once_with("chat_42", "profile alert", metadata=None)
        runner._authorization_adapter.assert_called_once_with(Platform.TELEGRAM, "profile_1")
        adapter._target_home_channel.assert_called_once_with(Platform.TELEGRAM, "profile_1")

    @pytest.mark.asyncio
    async def test_send_does_not_borrow_another_profiles_adapter(self):
        """Regression (#65939): a routed alert never egresses through another profile's bot.

        The target platform connected only on a DIFFERENT profile: this adapter's own
        profile has no adapter for it, so the alert degrades to the HA notification
        instead of being posted as the wrong profile's identity.
        """
        adapter = self._make_ha_adapter()
        adapter._owner_profile = "profile_1"
        foreign_adapter = self._stub_adapter()
        runner = self._stub_runner(Platform.TELEGRAM, target_adapter=None, home_chat_id="chat_42")
        runner._profile_adapters = {"profile_2": {Platform.TELEGRAM: foreign_adapter}}
        adapter.gateway_runner = runner
        # A home channel exists for this profile, so borrowing profile_2's adapter WOULD
        # deliver: only the fail-closed resolution can keep the alert off the wrong bot.
        adapter._target_home_channel = MagicMock(return_value=_fake_home_channel("chat_42"))

        with patch(
            "plugins.platforms.homeassistant.adapter.HomeAssistantAdapter._send_ha_notification",
            new_callable=AsyncMock,
            return_value=SendResult(success=True),
        ) as mock_ha_fallback:
            result = await adapter.send("ha_events:telegram", "cross-profile alert")

        assert result.success is True
        foreign_adapter.send.assert_not_called()
        adapter._target_home_channel.assert_not_called()
        mock_ha_fallback.assert_awaited_once_with("cross-profile alert")

    def test_target_home_channel_reads_own_profile_config(self):
        """A secondary profile's home channel comes from ITS config.yaml, not the default's."""
        adapter = self._make_ha_adapter()
        runner = self._stub_runner(Platform.TELEGRAM, home_chat_id="default_chat")
        adapter.gateway_runner = runner
        profile_cfg = MagicMock()
        profile_cfg.get_home_channel = MagicMock(
            return_value=_fake_home_channel("profile_chat"))
        with patch("gateway.config.load_gateway_config", return_value=profile_cfg), patch(
            "gateway.run._profile_runtime_scope"
        ) as mock_scope, patch(
            "hermes_cli.profiles.get_profile_dir", return_value="/tmp/profile_1"
        ):
            home = adapter._target_home_channel(Platform.TELEGRAM, "profile_1")

        assert home.chat_id == "profile_chat"
        profile_cfg.get_home_channel.assert_called_once_with(Platform.TELEGRAM)
        runner.config.get_home_channel.assert_not_called()
        mock_scope.assert_called_once_with("/tmp/profile_1")

    def test_target_home_channel_uses_runner_config_without_profile(self):
        """Primary/default: the runner's own config is the source, no scope switch."""
        adapter = self._make_ha_adapter()
        runner = self._stub_runner(Platform.TELEGRAM, home_chat_id="default_chat")
        adapter.gateway_runner = runner

        home = adapter._target_home_channel(Platform.TELEGRAM, None)

        assert home.chat_id == "default_chat"
        runner.config.get_home_channel.assert_called_once_with(Platform.TELEGRAM)

