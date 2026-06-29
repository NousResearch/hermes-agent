"""Tests for GatewayRunner._resolve_gateway_model_tier."""
import pytest
from unittest.mock import MagicMock, patch


def _make_runner(service_tier=None, session_overrides=None):
    """Create a minimal GatewayRunner-like object without full __init__."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._service_tier = service_tier
    runner._session_model_overrides = session_overrides or {}
    # Minimal stubs so _session_key_for_source doesn't blow up
    runner.session_store = None
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    return runner


class TestResolveGatewayModelTierGlobalConfig:
    """Global config (no session override) resolution."""

    def test_returns_none_when_no_tier_configured(self):
        runner = _make_runner(service_tier=None)
        assert runner._resolve_gateway_model_tier() is None

    def test_returns_priority_when_global_tier_is_priority(self):
        runner = _make_runner(service_tier="priority")
        assert runner._resolve_gateway_model_tier() == "priority"

    def test_returns_none_when_global_tier_is_none_string(self):
        # _load_service_tier normalises "normal"/"off" to None before storing
        runner = _make_runner(service_tier=None)
        assert runner._resolve_gateway_model_tier() is None

    def test_session_key_none_falls_back_to_global(self):
        runner = _make_runner(service_tier="priority")
        result = runner._resolve_gateway_model_tier(session_key=None)
        assert result == "priority"

    def test_unknown_session_key_falls_back_to_global(self):
        runner = _make_runner(service_tier="priority")
        result = runner._resolve_gateway_model_tier(session_key="agent:main:telegram:dm:99999")
        assert result == "priority"


class TestResolveGatewayModelTierSessionOverride:
    """Session-scoped override takes precedence over global config."""

    def test_session_priority_overrides_global_none(self):
        runner = _make_runner(
            service_tier=None,
            session_overrides={"sk1": {"service_tier": "priority"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_session_priority_overrides_global_priority(self):
        # Both are priority - result is still priority
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": "priority"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_session_normal_overrides_global_priority(self):
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": "normal"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") is None

    def test_session_off_overrides_global_priority(self):
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": "off"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") is None

    def test_session_fast_alias_returns_priority(self):
        runner = _make_runner(
            service_tier=None,
            session_overrides={"sk1": {"service_tier": "fast"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_session_on_alias_returns_priority(self):
        runner = _make_runner(
            service_tier=None,
            session_overrides={"sk1": {"service_tier": "on"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_session_none_value_returns_none(self):
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": None}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") is None

    def test_session_empty_string_returns_none(self):
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": ""}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") is None

    def test_session_override_without_service_tier_key_falls_back_to_global(self):
        # Override exists but has no service_tier key - use global
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"model": "gpt-4o", "provider": "openai"}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_different_session_keys_are_independent(self):
        runner = _make_runner(
            service_tier=None,
            session_overrides={
                "sk_priority": {"service_tier": "priority"},
                "sk_normal": {"service_tier": "normal"},
            },
        )
        assert runner._resolve_gateway_model_tier(session_key="sk_priority") == "priority"
        assert runner._resolve_gateway_model_tier(session_key="sk_normal") is None

    def test_unknown_session_tier_value_falls_back_to_global(self):
        runner = _make_runner(
            service_tier="priority",
            session_overrides={"sk1": {"service_tier": "turbo_ultra"}},
        )
        # Unknown value → fall through to global "priority"
        assert runner._resolve_gateway_model_tier(session_key="sk1") == "priority"

    def test_unknown_session_tier_value_falls_back_to_global_none(self):
        runner = _make_runner(
            service_tier=None,
            session_overrides={"sk1": {"service_tier": "turbo_ultra"}},
        )
        # Unknown value → fall through to global None
        assert runner._resolve_gateway_model_tier(session_key="sk1") is None


class TestResolveGatewayModelTierWithSource:
    """Resolution via SessionSource (source= kwarg path)."""

    def _make_source(self, platform_value="telegram", chat_id="123", user_id="u1"):
        from gateway.session import SessionSource
        from gateway.config import Platform

        return SessionSource(
            platform=Platform(platform_value),
            chat_id=chat_id,
            user_id=user_id,
            chat_type="dm",
        )

    def test_source_resolves_session_key_and_finds_override(self):
        from gateway.session import build_session_key

        source = self._make_source()
        sk = build_session_key(source, group_sessions_per_user=True, thread_sessions_per_user=False)

        runner = _make_runner(
            service_tier=None,
            session_overrides={sk: {"service_tier": "priority"}},
        )
        # Patch _session_key_for_source to return the known key
        runner._session_key_for_source = lambda s: sk

        result = runner._resolve_gateway_model_tier(source=source)
        assert result == "priority"

    def test_source_with_no_override_falls_back_to_global(self):
        source = self._make_source()
        runner = _make_runner(service_tier="priority")
        runner._session_key_for_source = lambda s: "agent:main:telegram:dm:123"

        result = runner._resolve_gateway_model_tier(source=source)
        assert result == "priority"

    def test_source_key_resolution_exception_falls_back_to_global(self):
        source = self._make_source()
        runner = _make_runner(service_tier="priority")

        def _raise(_):
            raise RuntimeError("session store unavailable")

        runner._session_key_for_source = _raise

        # Should not raise; falls back to global
        result = runner._resolve_gateway_model_tier(source=source)
        assert result == "priority"

    def test_session_key_kwarg_takes_precedence_over_source(self):
        """Explicit session_key kwarg wins over source-derived key."""
        source = self._make_source(chat_id="111")
        runner = _make_runner(
            service_tier=None,
            session_overrides={
                "explicit_key": {"service_tier": "priority"},
                "source_key": {"service_tier": "normal"},
            },
        )
        runner._session_key_for_source = lambda s: "source_key"

        result = runner._resolve_gateway_model_tier(
            source=source, session_key="explicit_key"
        )
        assert result == "priority"


class TestResolveGatewayModelTierReturnTypes:
    """Return value contract: only None or "priority"."""

    @pytest.mark.parametrize(
        "service_tier,expected",
        [
            (None, None),
            ("priority", "priority"),
        ],
    )
    def test_global_tier_return_values(self, service_tier, expected):
        runner = _make_runner(service_tier=service_tier)
        assert runner._resolve_gateway_model_tier() == expected

    @pytest.mark.parametrize(
        "session_tier,expected",
        [
            ("priority", "priority"),
            ("fast", "priority"),
            ("on", "priority"),
            ("normal", None),
            ("off", None),
            ("default", None),
            ("standard", None),
            ("none", None),
            (None, None),
            ("", None),
        ],
    )
    def test_session_tier_normalisation(self, session_tier, expected):
        runner = _make_runner(
            service_tier=None,
            session_overrides={"sk": {"service_tier": session_tier}},
        )
        assert runner._resolve_gateway_model_tier(session_key="sk") == expected
