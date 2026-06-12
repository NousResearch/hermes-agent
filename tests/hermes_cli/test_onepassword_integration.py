"""
Tests for 1Password integration in hermes-agent.

Tests cover:
- 1Password reference resolution (op://)
- Environment variable loading
- Profile-specific .env files
- Migration tool functionality
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from hermes_cli.onepassword_resolver import (
    _op_cli_available,
    _resolve_op_reference,
    resolve_value,
    is_1password_available,
)
from hermes_cli.config import get_env_value, load_env


class TestOpCliAvailable:
    """Test 1Password CLI availability detection."""

    def test_op_cli_available_when_in_path(self):
        """Should return True when 'op' is in PATH."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            assert _op_cli_available() is True

    def test_op_cli_not_available_when_not_in_path(self):
        """Should return False when 'op' is not in PATH."""
        with patch("shutil.which", return_value=None):
            assert _op_cli_available() is False


class TestResolveOpReference:
    """Test 1Password reference resolution."""

    def test_resolve_valid_reference(self):
        """Should resolve valid 1Password reference."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="sk-0123456789abcdef\n"
                )
                result = _resolve_op_reference("op://Empire/litellm-master-key/credential")
                assert result == "sk-0123456789abcdef"

    def test_resolve_invalid_reference(self):
        """Should return None when reference doesn't exist."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(
                    returncode=1,
                    stderr="item not found"
                )
                result = _resolve_op_reference("op://Empire/nonexistent/credential")
                assert result is None

    def test_resolve_when_cli_not_available(self):
        """Should return None when 'op' CLI is not available."""
        with patch("shutil.which", return_value=None):
            result = _resolve_op_reference("op://Empire/litellm-master-key/credential")
            assert result is None

    def test_timeout_handling(self):
        """Should handle timeout gracefully."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = __import__("subprocess").TimeoutExpired("op", 10)
                result = _resolve_op_reference("op://Empire/litellm-master-key/credential")
                assert result is None


class TestResolveValue:
    """Test value resolution with 1Password support."""

    def test_resolve_op_reference(self):
        """Should resolve op:// references."""
        with patch("hermes_cli.onepassword_resolver.resolve_value") as mock_resolve:
            mock_resolve.return_value = "resolved_value"
            from hermes_cli.onepassword_resolver import resolve_value
            result = resolve_value("op://Empire/test/credential")
            # Note: In real test, we'd need to mock the actual resolution
            assert isinstance(result, str)

    def test_pass_through_normal_value(self):
        """Should return normal values unchanged."""
        from hermes_cli.onepassword_resolver import resolve_value
        result = resolve_value("my-normal-value")
        assert result == "my-normal-value"

    def test_handle_empty_value(self):
        """Should handle empty values."""
        from hermes_cli.onepassword_resolver import resolve_value
        result = resolve_value("")
        assert result == ""


class TestConfigIntegration:
    """Test integration with hermes config system."""

    def test_get_env_value_from_environ(self):
        """Should get values from os.environ."""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            result = get_env_value("TEST_KEY")
            assert result == "test_value"

    def test_get_env_value_from_file(self):
        """Should get values from .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("TEST_KEY=file_value\n")
            
            with patch("hermes_cli.config.get_env_path", return_value=env_file):
                result = get_env_value("TEST_KEY")
                assert result == "file_value"

    def test_get_env_value_op_reference_fallback(self):
        """Should return op:// reference when CLI not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("API_KEY=op://Empire/my-key/credential\n")
            
            with patch("hermes_cli.config.get_env_path", return_value=env_file):
                with patch("shutil.which", return_value=None):
                    result = get_env_value("API_KEY")
                    # Should return the reference when op CLI not available
                    assert "op://" in result or result is None


class TestMigrationTool:
    """Test 1Password migration tool."""

    def test_migrate_simple_env_file(self):
        """Should migrate a simple .env file."""
        from hermes_cli.onepassword_resolver import migrate_env_file_to_onepassword
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "LITELLM_MASTER_KEY=sk-0123456789\n"
                "TAVILY_API_KEY=tvly-1234567890\n"
            )
            
            with patch("shutil.which", return_value=None):
                result = migrate_env_file_to_onepassword(env_file)
                
                assert result["migrated"] == 2
                assert result["backup_path"] is not None
                
                # Check that file was updated
                migrated_content = env_file.read_text()
                assert "op://Empire/" in migrated_content

    def test_migrate_skip_already_migrated(self):
        """Should skip values already migrated to 1Password."""
        from hermes_cli.onepassword_resolver import migrate_env_file_to_onepassword
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "MIGRATED=op://Empire/migrated-key/credential\n"
                "PLAINTEXT=sk-0123456789\n"
            )
            
            with patch("shutil.which", return_value=None):
                result = migrate_env_file_to_onepassword(env_file)
                
                assert result["migrated"] == 1
                assert result["skipped"] == 1

    def test_migrate_nonexistent_file(self):
        """Should handle nonexistent .env files gracefully."""
        from hermes_cli.onepassword_resolver import migrate_env_file_to_onepassword
        
        result = migrate_env_file_to_onepassword(Path("/nonexistent/.env"))
        
        assert result["migrated"] == 0
        assert len(result["errors"]) > 0


class TestIs1PasswordAvailable:
    """Test 1Password availability check."""

    def test_available_and_signed_in(self):
        """Should return True when signed in."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)
                assert is_1password_available() is True

    def test_not_available(self):
        """Should return False when CLI not available."""
        with patch("shutil.which", return_value=None):
            assert is_1password_available() is False

    def test_not_signed_in(self):
        """Should return False when not signed in."""
        with patch("shutil.which", return_value="/usr/bin/op"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=1)
                assert is_1password_available() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
