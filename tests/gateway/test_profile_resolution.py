"""Tests for GatewayRunner._resolve_profile_home_for_source — profile resolution logic."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner
from gateway.profile_routing import ProfileRoute
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter


@pytest.fixture
def mock_runner():
    """Create a minimal mock GatewayRunner with the methods we need."""
    runner = MagicMock(spec=GatewayRunner)
    runner.config = MagicMock(profile_routes=[])
    # Bind the actual methods to the mock
    runner._profile_name_for_source = GatewayRunner._profile_name_for_source.__get__(runner)
    runner._resolve_profile_home_for_source = GatewayRunner._resolve_profile_home_for_source.__get__(runner)
    return runner


@pytest.fixture
def discord_source():
    """Create a basic Discord SessionSource for testing."""
    return SessionSource(
        platform=MagicMock(value="discord"),
        chat_id="123456",
        guild_id="789",
        thread_id=None,
        parent_chat_id=None,
    )


@pytest.fixture
def telegram_source():
    """Create a basic Telegram SessionSource for testing.

    Telegram (like Slack/Feishu/etc.) has no ``guild_id`` — only ``chat_id``.
    Used to prove profile routing is platform-generic, not Discord-only.
    """
    return SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="-1001234567890",
        guild_id=None,
        thread_id=None,
        parent_chat_id=None,
    )


class TestResolutionOrder:
    """Tests that profile resolution follows the correct priority order."""
    
    def test_source_profile_wins_over_routing(self, mock_runner, discord_source):
        """source.profile should be used even if routing would match."""
        discord_source.profile = "from-source"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    mock_get_dir.return_value = Path("/hermes/profiles/from-source")
                    result = mock_runner._resolve_profile_home_for_source(discord_source)
                    
                    assert result == Path("/hermes/profiles/from-source")
                    mock_get_dir.assert_called_once_with("from-source")
    
    def test_routing_wins_over_active_profile(self, mock_runner, discord_source):
        """When source.profile is empty, routing should win over active profile."""
        discord_source.profile = None
        
        # Mock routing to return a profile
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    mock_get_dir.return_value = Path("/hermes/profiles/routed")
                    
                    # Manually set routing to return a profile
                    mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                    
                    result = mock_runner._resolve_profile_home_for_source(discord_source)
                    
                    assert result == Path("/hermes/profiles/routed")
                    mock_get_dir.assert_called_once_with("routed")
    
    def test_active_profile_fallback(self, mock_runner, discord_source):
        """When source.profile and routing both return None, active profile is used."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/active")
                
                # No routing match
                mock_runner._profile_name_for_source = MagicMock(return_value=None)
                
                result = mock_runner._resolve_profile_home_for_source(discord_source)
                
                assert result == Path("/hermes/profiles/active")
                mock_get_dir.assert_called_once_with("active")
    
    def test_default_fallback_when_no_active(self, mock_runner, discord_source):
        """When even active profile is None, 'default' is used."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value=None):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes")
                
                mock_runner._profile_name_for_source = MagicMock(return_value=None)
                
                result = mock_runner._resolve_profile_home_for_source(discord_source)
                
                assert result == Path("/hermes")
                mock_get_dir.assert_called_once_with("default")


class TestMissingProfileWarning:
    """Tests for warning when a profile doesn't exist on disk."""
    
    def test_nonexistent_profile_warning(self, mock_runner, discord_source, caplog):
        """When source.profile points to a nonexistent profile, log a WARNING."""
        discord_source.profile = "nonexistent"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/nonexistent")
                with patch("hermes_cli.profiles.profile_exists", return_value=False):
                    with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                        with caplog.at_level(logging.WARNING):
                            result = mock_runner._resolve_profile_home_for_source(discord_source)
                            
                            # Should fall back to global HERMES_HOME
                            assert result == Path("/hermes")
                            
                            # Should have logged a warning
                            assert len(caplog.records) == 1
                            assert caplog.records[0].levelname == "WARNING"
                            assert "nonexistent" in caplog.records[0].message
                            assert "does not exist" in caplog.records[0].message
                            assert "discord" in caplog.records[0].message
                            assert "123456" in caplog.records[0].message
    
    def test_nonexistent_routing_profile_warning(self, mock_runner, discord_source, caplog):
        """When routing returns a nonexistent profile, log a WARNING."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/routed")
                with patch("hermes_cli.profiles.profile_exists", return_value=False):
                    with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                        # Routing returns a profile that doesn't exist
                        mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                        
                        with caplog.at_level(logging.WARNING):
                            result = mock_runner._resolve_profile_home_for_source(discord_source)
                            
                            # Should fall back to global HERMES_HOME
                            assert result == Path("/hermes")
                            
                            # Should have logged a warning
                            assert len(caplog.records) == 1
                            assert "routed" in caplog.records[0].message
    
    def test_empty_source_profile_no_warning(self, mock_runner, discord_source, caplog):
        """When source.profile is empty, silent fallback to active profile (no warning)."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/active")
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    with caplog.at_level(logging.WARNING):
                        mock_runner._profile_name_for_source = MagicMock(return_value=None)
                        
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        # Should use active profile
                        assert result == Path("/hermes/profiles/active")
                        
                        # No warnings (active profile exists)
                        assert not any(r.levelname == "WARNING" for r in caplog.records)
    
    def test_existing_profile_no_warning(self, mock_runner, discord_source, caplog):
        """When the profile exists, no warning should be logged."""
        discord_source.profile = "existing"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/existing")
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        assert result == Path("/hermes/profiles/existing")
                        
                        # No warnings
                        assert not any(r.levelname == "WARNING" for r in caplog.records)


class TestExceptionHandling:
    """Tests for exception handling in profile resolution."""
    
    def test_get_profile_dir_exception_logs_warning(self, mock_runner, discord_source, caplog):
        """When get_profile_dir raises an exception, log a WARNING with context."""
        discord_source.profile = "bad-profile"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir", side_effect=ValueError("Invalid profile name")):
                with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        # Should fall back to global HERMES_HOME
                        assert result == Path("/hermes")
                        
                        # Should have logged a warning with exception info
                        assert len(caplog.records) == 1
                        assert caplog.records[0].levelname == "WARNING"
                        assert "bad-profile" in caplog.records[0].message
                        assert "Failed to resolve profile directory" in caplog.records[0].message
    
    def test_exception_with_no_profile_name(self, mock_runner, discord_source, caplog):
        """Exception when no profile was set should still log a warning."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value=None):
            with patch("hermes_cli.profiles.get_profile_dir", side_effect=RuntimeError("Filesystem error")):
                with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                    mock_runner._profile_name_for_source = MagicMock(return_value=None)
                    
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        assert result == Path("/hermes")
                        
                        # Warning should mention "(no profile)"
                        assert "(no profile)" in caplog.records[0].message


class TestRoutingConsultation:
    """Tests that _profile_name_for_source is consulted when source.profile is empty."""
    
    def test_routing_consulted_when_source_profile_empty(self, mock_runner, discord_source):
        """_profile_name_for_source should be called when source.profile is empty."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/routed")
                
                mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                
                mock_runner._resolve_profile_home_for_source(discord_source)
                
                # Should have called routing
                mock_runner._profile_name_for_source.assert_called_once_with(discord_source)
    
    def test_routing_not_consulted_when_source_profile_set(self, mock_runner, discord_source):
        """_profile_name_for_source should NOT be called when source.profile is set."""
        discord_source.profile = "from-source"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/from-source")
                
                mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                
                mock_runner._resolve_profile_home_for_source(discord_source)
                
                # Should NOT have called routing
                mock_runner._profile_name_for_source.assert_not_called()


class TestNonDiscordProfileRouting:
    """Profile routing must be platform-generic, not Discord-only.

    Regression coverage for the ``gateway_runner`` injection gap: previously
    only Discord's adapter pre-declared ``gateway_runner``, so only Discord
    ever had ``build_source`` call ``_profile_name_for_source``. Telegram /
    Feishu / Slack / etc. silently fell through to the default profile. These
    tests pin the resolution half for a non-Discord platform (Telegram).
    """

    def test_telegram_route_resolves(self, mock_runner, telegram_source):
        """A configured Telegram route resolves to its profile via the real
        ``_profile_name_for_source`` (bound onto the mock runner)."""
        mock_runner.config.profile_routes = [
            ProfileRoute(name="tg", platform="telegram", profile="tg-profile",
                         chat_id="-1001234567890"),
        ]
        telegram_source.profile = None

        assert mock_runner._profile_name_for_source(telegram_source) == "tg-profile"

    def test_telegram_no_route_returns_none(self, mock_runner, telegram_source):
        """With no matching Telegram route, resolution returns None (caller
        falls back to the default/active profile)."""
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="dc-profile",
                         chat_id="123456"),
        ]
        telegram_source.profile = None

        assert mock_runner._profile_name_for_source(telegram_source) is None


class TestGatewayRunnerInjection:
    """``BasePlatformAdapter`` declares ``gateway_runner`` so the gateway's
    unconditional injection reaches every platform adapter — the foundation
    that makes the routing in TestNonDiscordProfileRouting reachable at runtime.
    """

    def test_base_adapter_declares_gateway_runner(self):
        from gateway.platforms.base import BasePlatformAdapter

        # Class-level attribute exists and defaults to None.
        assert hasattr(BasePlatformAdapter, "gateway_runner")
        assert BasePlatformAdapter.gateway_runner is None

    def test_subclass_inherits_gateway_runner(self):
        from gateway.platforms.base import BasePlatformAdapter

        class _ToyAdapter(BasePlatformAdapter):
            pass

        # No manual declaration — yet the attribute is inherited from the base,
        # so the gateway's ``adapter.gateway_runner = self`` injection reaches
        # every adapter, not just the ones that pre-declared it (Discord).
        assert hasattr(_ToyAdapter, "gateway_runner")
        assert _ToyAdapter.gateway_runner is None


# A concrete adapter we can instantiate without the full platform stack.
# ``build_source`` only reads ``self.platform`` and ``self.gateway_runner``, so a
# bare instance with those two attrs exercises the real BasePlatformAdapter
# method end-to-end. Clearing ``__abstractmethods__`` lets ``__new__`` bypass
# the ABC instantiation guard without stubbing connect/send/get_chat_info/…
class _StubAdapter(BasePlatformAdapter):
    pass


_StubAdapter.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]


def _stub_adapter(platform: Platform, runner) -> "_StubAdapter":
    a = _StubAdapter.__new__(_StubAdapter)
    a.platform = platform
    a.gateway_runner = runner
    return a


class TestAdapterToSessionKeyIntegration:
    """Adapter -> ``source.profile`` -> session-key integration coverage.

    The review asked for integration coverage for Discord AND a non-Discord
    platform. These drive a concrete adapter's real ``build_source``
    (BasePlatformAdapter) with an injected ``gateway_runner``, assert the
    matched route's profile is stamped on the source, and that the resulting
    session key is profile-scoped (``agent:<profile>:...`` rather than the
    shared ``agent:main:...``). The Telegram case is the bug-#2 regression:
    pre-fix it never received ``gateway_runner`` and fell through to default.
    """

    @staticmethod
    def _routes():
        return [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="111", chat_id="222"),
            ProfileRoute(name="tg", platform="telegram", profile="ops",
                         chat_id="-1001234567890"),
        ]

    def test_discord_adapter_stamps_profile_and_scopes_key(self, mock_runner):
        mock_runner.config.profile_routes = self._routes()
        adapter = _stub_adapter(Platform.DISCORD, mock_runner)

        source = adapter.build_source(
            chat_id="222", chat_type="group", guild_id="111", user_id="u1",
        )
        assert source.profile == "coder"

        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:coder:"), key
        # A default-profile key would land in agent:main — must differ.
        assert key != build_session_key(source, profile=None)

    def test_telegram_adapter_stamps_profile_and_scopes_key(self, mock_runner):
        """Non-Discord platform (bug #2). The adapter now receives
        ``gateway_runner``, so ``build_source`` stamps the profile and the
        session key is isolated under ``agent:ops:`` instead of ``agent:main:``."""
        mock_runner.config.profile_routes = self._routes()
        adapter = _stub_adapter(Platform.TELEGRAM, mock_runner)

        source = adapter.build_source(
            chat_id="-1001234567890", chat_type="group", user_id="u1",
        )
        assert source.profile == "ops"
        assert source._transport_adapter_ref() is adapter

        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:ops:"), key
        assert key != build_session_key(source, profile=None)

    @pytest.mark.asyncio
    async def test_chat_route_keeps_shared_adapter_for_delivery(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            profile_routes=self._routes(),
        )
        runner._profile_adapters = {"ops": {}}
        adapter = _stub_adapter(Platform.TELEGRAM, runner)
        adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}

        source = adapter.build_source(
            chat_id="-1001234567890", chat_type="group", user_id="u1",
        )

        assert source.profile == "ops"
        assert runner._adapter_for_source(source) is adapter
        await runner._deliver_platform_notice(source, "routed reply")
        adapter.send.assert_awaited_once_with(
            "-1001234567890",
            "routed reply",
            metadata=None,
        )

    def test_adapter_without_runner_falls_back_to_default_namespace(self, mock_runner):
        """Regression anchor: with no ``gateway_runner`` injected (the pre-fix
        state for non-Discord adapters), ``build_source`` leaves ``profile=None``
        and the session key is the shared ``agent:main:`` namespace — no
        per-profile isolation. This is the silent fallback the fix removes for
        non-Discord platforms."""
        adapter = _stub_adapter(Platform.TELEGRAM, runner=None)

        source = adapter.build_source(
            chat_id="-1001234567890", chat_type="group", user_id="u1",
        )
        assert source.profile is None
        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:main:"), key


class TestMultiplexGate:
    """``profile_routes`` only activates under ``gateway.multiplex_profiles``.

    Routing stamps ``source.profile``, which namespaces session/batch keys —
    but the profile-scoped agent run (``_profile_runtime_scope``) only engages
    when multiplexing is on. Without the gate, a configured route with
    multiplexing off would split batch/session keys into ``agent:<profile>``
    while the agent still served the turn from ``agent:main``'s home.
    """

    def test_routes_ignored_when_multiplex_off(self, mock_runner, discord_source):
        mock_runner.config.multiplex_profiles = False
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="789", chat_id="123456"),
        ]
        discord_source.profile = None

        assert mock_runner._profile_name_for_source(discord_source) is None

    def test_routes_active_when_multiplex_on(self, mock_runner, discord_source):
        mock_runner.config.multiplex_profiles = True
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="789", chat_id="123456"),
        ]
        discord_source.profile = None

        assert mock_runner._profile_name_for_source(discord_source) == "coder"

    def test_build_source_leaves_profile_none_when_multiplex_off(self, mock_runner):
        """End-to-end through the real adapter ``build_source``: with routes
        configured but multiplexing off, no profile is stamped and the session
        key stays in the legacy ``agent:main`` namespace — byte-identical to a
        gateway with no routes at all."""
        mock_runner.config.multiplex_profiles = False
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="111", chat_id="222"),
        ]
        adapter = _stub_adapter(Platform.DISCORD, mock_runner)

        source = adapter.build_source(
            chat_id="222", chat_type="group", guild_id="111", user_id="u1",
        )
        assert source.profile is None
        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:main:"), key
