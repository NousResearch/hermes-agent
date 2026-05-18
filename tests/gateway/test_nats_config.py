"""Phase 2 (T2.2): config parsing for the NATS gateway adapter.

Covers :class:`NatsAdapterSettings.from_extra` happy/bad paths, the
adapter's fatal-error behaviour on bad config, and the env-variable →
``config.extra`` round-trip in :func:`_apply_env_overrides`. The
``_ensure_synadia_agents_mock`` autouse in ``conftest.py`` installs a mock
``synadia_ai.agents`` module so these tests run without the real SDK.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)
from gateway.platforms.nats import (
    DEFAULT_ACK_KEEPALIVE_INTERVAL_S,
    DEFAULT_AGENT,
    DEFAULT_ATTACHMENTS_OK,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    MAX_ACK_KEEPALIVE_INTERVAL_S,
    NatsAdapter,
    NatsAdapterSettings,
    NatsConfigError,
    check_nats_requirements,
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSettingsFromExtraHappy:
    def test_minimal_servers_config_applies_defaults(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://127.0.0.1:4222"],
                "owner": "rene",
                "session_name": "default",
            }
        )

        assert settings.servers == ["nats://127.0.0.1:4222"]
        assert settings.context is None
        assert settings.agent == DEFAULT_AGENT
        assert settings.owner == "rene"
        assert settings.session_name == "default"
        assert settings.heartbeat_interval_s == DEFAULT_HEARTBEAT_INTERVAL_S
        # max_payload is intentionally None when not set — _on_connect
        # derives it from the broker's negotiated INFO at connect time
        # (PR #41). Hardcoding "1MB" here would have capped a 64MB
        # broker.
        assert settings.max_payload is None
        assert settings.attachments_ok is DEFAULT_ATTACHMENTS_OK
        assert settings.ack_keepalive_interval_s == DEFAULT_ACK_KEEPALIVE_INTERVAL_S

    def test_minimal_context_config_works(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "context": "local-nats",
                "owner": "rene",
                "session_name": "default",
            }
        )

        assert settings.context == "local-nats"
        assert settings.servers is None

    def test_full_config_preserves_all_overrides(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://a:4222", "nats://b:4222"],
                "agent": "hermes",
                "owner": "acme_corp",
                "session_name": "prod-1",
                "heartbeat_interval_s": 15,
                "max_payload": "2MB",
                "attachments_ok": False,
                "ack_keepalive_interval_s": 30,
            }
        )

        assert settings.servers == ["nats://a:4222", "nats://b:4222"]
        assert settings.agent == "hermes"
        assert settings.owner == "acme_corp"
        assert settings.session_name == "prod-1"
        assert settings.heartbeat_interval_s == 15
        assert settings.max_payload == "2MB"
        assert settings.attachments_ok is False
        assert settings.ack_keepalive_interval_s == 30

    def test_servers_string_is_coerced_to_single_url_list(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": "nats://127.0.0.1:4222",
                "owner": "rene",
                "session_name": "default",
            }
        )
        assert settings.servers == ["nats://127.0.0.1:4222"]

    def test_servers_whitespace_is_stripped_and_empty_entries_dropped(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["  nats://a:4222  ", "", "nats://b:4222"],
                "owner": "rene",
                "session_name": "default",
            }
        )
        assert settings.servers == ["nats://a:4222", "nats://b:4222"]

    def test_identity_property_formats_agent_owner_session_name(self):
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://x:4222"],
                "agent": "hermes",
                "owner": "rene",
                "session_name": "default",
            }
        )
        assert settings.identity == "hermes:rene:default"

    def test_case_insensitive_max_payload_accepted(self):
        # The §2.1 size grammar is case-insensitive; callers shouldn't have
        # to match an exact spelling for such a common misconfig.
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://x:4222"],
                "owner": "rene",
                "session_name": "default",
                "max_payload": "4gb",
            }
        )
        assert settings.max_payload == "4gb"

    def test_unset_max_payload_stays_none_for_broker_derivation(self):
        # PR #41: when the user omits max_payload, leave the field unset
        # so _on_connect can derive it from nc.max_payload at connect
        # time. Hardcoding "1MB" here was the bug — it capped 64MB
        # brokers regardless of negotiated capacity.
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://x:4222"],
                "owner": "rene",
                "session_name": "default",
            }
        )
        assert settings.max_payload is None

    def test_blank_max_payload_string_treated_as_unset(self):
        # Empty/whitespace-only is the YAML "I didn't set this" shape.
        settings = NatsAdapterSettings.from_extra(
            {
                "servers": ["nats://x:4222"],
                "owner": "rene",
                "session_name": "default",
                "max_payload": "   ",
            }
        )
        assert settings.max_payload is None


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


class TestSettingsFromExtraBad:
    def test_missing_transport_raises(self):
        with pytest.raises(NatsConfigError, match="exactly one of 'servers'"):
            NatsAdapterSettings.from_extra(
                {"owner": "rene", "session_name": "default"}
            )

    def test_both_servers_and_context_raises(self):
        with pytest.raises(NatsConfigError, match="not both"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "context": "local-nats",
                    "owner": "rene",
                    "session_name": "default",
                }
            )

    def test_empty_servers_list_raises(self):
        with pytest.raises(NatsConfigError, match="at least one non-empty URL"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["   ", ""],
                    "owner": "rene",
                    "session_name": "default",
                }
            )

    def test_servers_wrong_type_raises(self):
        with pytest.raises(NatsConfigError, match="'servers' must be a string or list"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": 42,
                    "owner": "rene",
                    "session_name": "default",
                }
            )

    def test_context_wrong_type_raises(self):
        with pytest.raises(NatsConfigError, match="'context' must be a string"):
            NatsAdapterSettings.from_extra(
                {
                    "context": 123,
                    "owner": "rene",
                    "session_name": "default",
                }
            )

    def test_missing_owner_raises(self):
        with pytest.raises(NatsConfigError, match="'owner' is required"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "session_name": "default",
                }
            )

    def test_missing_session_name_raises(self):
        with pytest.raises(NatsConfigError, match="'session_name' is required"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                }
            )

    def test_owner_non_string_raises(self):
        with pytest.raises(NatsConfigError, match="'owner' must be a string"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": 9000,
                    "session_name": "default",
                }
            )

    def test_invalid_agent_token_raises(self):
        # §2.2 restricts agent to lowercase alphanumeric + hyphens; catch
        # mixed-case or underscores before the SDK's AgentSubject.new().
        with pytest.raises(NatsConfigError, match="'agent'"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "agent": "Hermes_Bot",
                    "owner": "rene",
                    "session_name": "default",
                }
            )

    def test_invalid_max_payload_raises(self):
        with pytest.raises(NatsConfigError, match="'max_payload'"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "max_payload": "one megabyte",
                }
            )

    def test_non_bool_attachments_ok_raises(self):
        with pytest.raises(NatsConfigError, match="'attachments_ok' must be a boolean"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "attachments_ok": "yes",
                }
            )

    def test_non_positive_heartbeat_raises(self):
        with pytest.raises(NatsConfigError, match="'heartbeat_interval_s' must be positive"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "heartbeat_interval_s": 0,
                }
            )

    def test_bool_heartbeat_is_rejected(self):
        # Without an explicit bool guard, Python's int(True) == 1 would
        # silently pass validation.
        with pytest.raises(NatsConfigError, match="'heartbeat_interval_s' must be an integer"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "heartbeat_interval_s": True,
                }
            )

    def test_non_integer_heartbeat_raises(self):
        with pytest.raises(NatsConfigError, match="'heartbeat_interval_s' must be an integer"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "heartbeat_interval_s": "often",
                }
            )

    def test_ack_keepalive_at_protocol_limit_raises(self):
        with pytest.raises(NatsConfigError, match="ack_keepalive_interval_s"):
            NatsAdapterSettings.from_extra(
                {
                    "servers": ["nats://x:4222"],
                    "owner": "rene",
                    "session_name": "default",
                    "ack_keepalive_interval_s": MAX_ACK_KEEPALIVE_INTERVAL_S,
                }
            )


# ---------------------------------------------------------------------------
# Adapter instantiation
# ---------------------------------------------------------------------------


class TestNatsAdapterInit:
    def _pconfig(self, **extra) -> PlatformConfig:
        return PlatformConfig(enabled=True, extra=extra)

    def test_valid_config_no_fatal_error(self):
        adapter = NatsAdapter(
            self._pconfig(
                servers=["nats://127.0.0.1:4222"],
                owner="rene",
                session_name="default",
            )
        )
        assert adapter.has_fatal_error is False
        assert adapter._settings is not None
        assert adapter._settings.owner == "rene"

    def test_invalid_config_sets_fatal_error_nonretryable(self):
        adapter = NatsAdapter(self._pconfig(owner="rene"))
        # No servers / context / session_name — must fail fast.
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "nats_config_error"
        assert adapter.fatal_error_retryable is False
        assert adapter._settings is None

    def test_active_streams_and_service_fields_are_initialised(self):
        # Phases (3, 4) assume these attributes exist regardless of
        # whether init succeeded — guard against regressions.
        adapter = NatsAdapter(
            self._pconfig(
                servers=["nats://x:4222"],
                owner="rene",
                session_name="default",
            )
        )
        assert adapter._active_streams == {}
        assert adapter._nc is None
        assert adapter._service is None

    @pytest.mark.asyncio
    async def test_get_chat_info_shape(self):
        adapter = NatsAdapter(
            self._pconfig(
                servers=["nats://x:4222"],
                owner="rene",
                session_name="default",
            )
        )
        info = await adapter.get_chat_info("any-session-id")
        assert info == {"name": "any-session-id", "type": "dm"}

    @pytest.mark.asyncio
    async def test_disconnect_before_connect_is_idempotent_no_op(self):
        # ``gateway._safe_adapter_disconnect`` calls ``disconnect()``
        # defensively after failed ``connect()`` attempts and during
        # shutdown — must tolerate being called on a never-connected
        # adapter without raising.
        adapter = NatsAdapter(
            self._pconfig(
                servers=["nats://x:4222"],
                owner="rene",
                session_name="default",
            )
        )
        await adapter.disconnect()
        await adapter.disconnect()  # idempotent


# ---------------------------------------------------------------------------
# Env-variable → config.extra round-trip (complements Phase 1 T1.2)
# ---------------------------------------------------------------------------


class TestNatsEnvOverrides:
    def _nats_env(self, **overrides: str) -> dict[str, str]:
        """Build a clean env dict with only NATS_* / HERMES_NATS_* variables."""
        base = {
            "NATS_URL": "",
            "NATS_CONTEXT": "",
            "HERMES_NATS_AGENT": "",
            "HERMES_NATS_OWNER": "",
            "HERMES_NATS_SESSION_NAME": "",
        }
        base.update(overrides)
        return {k: v for k, v in base.items() if v}

    def test_nats_url_enables_and_populates_servers(self):
        config = GatewayConfig()
        with patch.dict(os.environ, self._nats_env(NATS_URL="nats://127.0.0.1:4222"), clear=True):
            _apply_env_overrides(config)

        assert Platform.NATS in config.platforms
        platform_cfg = config.platforms[Platform.NATS]
        assert platform_cfg.enabled is True
        assert platform_cfg.extra["servers"] == ["nats://127.0.0.1:4222"]

    def test_nats_context_enables_and_populates_context(self):
        config = GatewayConfig()
        with patch.dict(os.environ, self._nats_env(NATS_CONTEXT="local-nats"), clear=True):
            _apply_env_overrides(config)

        platform_cfg = config.platforms[Platform.NATS]
        assert platform_cfg.enabled is True
        assert platform_cfg.extra["context"] == "local-nats"
        assert "servers" not in platform_cfg.extra

    def test_identity_env_vars_populate_extra(self):
        config = GatewayConfig()
        env = self._nats_env(
            NATS_URL="nats://127.0.0.1:4222",
            HERMES_NATS_AGENT="hermes",
            HERMES_NATS_OWNER="rene",
            HERMES_NATS_SESSION_NAME="default",
        )
        with patch.dict(os.environ, env, clear=True):
            _apply_env_overrides(config)

        extra = config.platforms[Platform.NATS].extra
        assert extra["agent"] == "hermes"
        assert extra["owner"] == "rene"
        assert extra["session_name"] == "default"

    def test_identity_only_enables_but_stays_disconnected(self):
        # Decision log 2026-04-21: HERMES_NATS_OWNER alone marks the
        # platform enabled but lacks transport, so get_connected_platforms
        # must still filter it out.
        config = GatewayConfig()
        with patch.dict(os.environ, self._nats_env(HERMES_NATS_OWNER="rene"), clear=True):
            _apply_env_overrides(config)

        assert config.platforms[Platform.NATS].enabled is True
        assert Platform.NATS not in config.get_connected_platforms()

    def test_no_env_vars_leaves_platform_absent(self):
        config = GatewayConfig()
        with patch.dict(os.environ, {}, clear=True):
            _apply_env_overrides(config)
        assert Platform.NATS not in config.platforms


# ---------------------------------------------------------------------------
# get_connected_platforms()
# ---------------------------------------------------------------------------


class TestNatsConnectedGate:
    def test_enabled_with_servers_is_connected(self):
        config = GatewayConfig(
            platforms={
                Platform.NATS: PlatformConfig(
                    enabled=True,
                    extra={"servers": ["nats://x:4222"], "owner": "rene", "session_name": "default"},
                )
            }
        )
        assert Platform.NATS in config.get_connected_platforms()

    def test_enabled_with_context_is_connected(self):
        config = GatewayConfig(
            platforms={
                Platform.NATS: PlatformConfig(
                    enabled=True,
                    extra={"context": "local-nats", "owner": "rene", "session_name": "default"},
                )
            }
        )
        assert Platform.NATS in config.get_connected_platforms()

    def test_enabled_without_transport_is_not_connected(self):
        config = GatewayConfig(
            platforms={
                Platform.NATS: PlatformConfig(
                    enabled=True,
                    extra={"owner": "rene", "session_name": "default"},
                )
            }
        )
        assert Platform.NATS not in config.get_connected_platforms()

    def test_disabled_with_servers_is_not_connected(self):
        config = GatewayConfig(
            platforms={
                Platform.NATS: PlatformConfig(
                    enabled=False,
                    extra={"servers": ["nats://x:4222"], "owner": "rene", "session_name": "default"},
                )
            }
        )
        assert Platform.NATS not in config.get_connected_platforms()


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def test_check_nats_requirements_reports_sdk_availability():
    # conftest installs the synadia_ai.agents mock module before any test
    # runs, so the requirements check must report True both in CI (no real
    # SDK) and locally (real SDK installed).
    assert check_nats_requirements() is True
