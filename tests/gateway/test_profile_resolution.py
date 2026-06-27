"""Tests for GatewayRunner._resolve_profile_home_for_source — profile resolution logic."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.session import SessionSource
from gateway.run import GatewayRunner


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
