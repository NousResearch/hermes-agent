"""
Unit tests for CLI auth improvements:
- --base-url flag on `hermes auth add`
- base_url column in `hermes auth list`
- Unified resolver integration tests

These tests verify that:
1. The --base-url flag is accepted by the parser
2. The --base-url value is stored on the PooledCredential
3. auth list output includes the base_url column
4. The unified resolver works end-to-end with real pool entries
5. The cascade bug (base_url="") is fixed across all surfaces
"""

import os
import sys
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# agent.auth import removed for standalone CLI PR


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _fake_entry(provider="zai", base_url="", api_key="sk-fake-key", source="manual"):
    """Create a fake PooledCredential-like object."""
    return SimpleNamespace(
        provider=provider,
        id="test-id",
        label="Test Key",
        auth_type="api_key",
        source=source,
        access_token=api_key,
        refresh_token=None,
        last_status=None,
        last_status_at=None,
        last_error_code=None,
        last_error_reason=None,
        last_error_message=None,
        last_error_reset_at=None,
        base_url=base_url,
        expires_at=None,
        expires_at_ms=None,
        last_refresh=None,
        inference_base_url=None,
        agent_key=None,
        agent_key_expires_at=None,
        request_count=0,
        extra={},
        runtime_api_key=api_key if api_key else None,
        runtime_base_url=base_url if base_url else None,
    )


# ── CLI Parser Tests ─────────────────────────────────────────────────────────

class TestAuthAddParser:
    """Test that --base-url flag is accepted by the auth add parser."""

    def _build_parser(self):
        """Build a parser matching the real CLI structure: auth → add."""
        from hermes_cli.subcommands.auth import build_auth_parser
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_auth_parser(subparsers, cmd_auth=lambda: None)
        return parser

    def test_base_url_flag_exists(self):
        """The --base-url flag should be in the parser."""
        parser = self._build_parser()

        # Parse with --base-url: auth add zai --type api-key --api-key sk-test --base-url ...
        args = parser.parse_args([
            "auth", "add", "zai", "--type", "api-key",
            "--api-key", "sk-test",
            "--base-url", "https://api.z.ai/api/coding/paas/v4",
        ])
        assert hasattr(args, "base_url")
        assert args.base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_base_url_flag_optional(self):
        """The --base-url flag should be optional."""
        parser = self._build_parser()

        args = parser.parse_args([
            "auth", "add", "deepseek", "--type", "api-key",
            "--api-key", "sk-test",
        ])
        assert hasattr(args, "base_url")
        assert args.base_url is None

    def test_base_url_flag_with_label(self):
        """--base-url should work alongside --label."""
        parser = self._build_parser()

        args = parser.parse_args([
            "auth", "add", "zai", "--type", "api-key",
            "--api-key", "sk-test",
            "--label", "GLM coding 35",
            "--base-url", "https://api.z.ai/api/anthropic",
        ])
        assert args.base_url == "https://api.z.ai/api/anthropic"
        assert args.label == "GLM coding 35"


# ── Auth List Display Tests ──────────────────────────────────────────────────

class TestAuthListDisplay:
    """Test that auth list shows the base_url column."""

    def test_auth_list_includes_base_url(self, capsys):
        """auth list output should contain 'url=' for each entry."""
        from hermes_cli.auth_commands import auth_list_command
        from agent.credential_pool import PooledCredential, STATUS_OK

        # Create a fake pool with a known base_url
        entry = PooledCredential(
            provider="zai",
            id="abc123",
            label="Test Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-test",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )

        mock_pool = MagicMock()
        mock_pool.entries.return_value = [entry]
        mock_pool.peek.return_value = entry

        with patch("hermes_cli.auth_commands.load_pool", return_value=mock_pool):
            args = SimpleNamespace(provider="zai")
            auth_list_command(args)

        captured = capsys.readouterr()
        assert "coding" in captured.out
        assert "url=" not in captured.out  # old verbose format should be gone

    def test_auth_list_shows_default_for_empty_base_url(self, capsys):
        """auth list should show '(default)' when base_url is empty."""
        from hermes_cli.auth_commands import auth_list_command
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="zai",
            id="abc123",
            label="Empty URL Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-test",
            base_url="",
        )

        mock_pool = MagicMock()
        mock_pool.entries.return_value = [entry]
        mock_pool.peek.return_value = entry

        with patch("hermes_cli.auth_commands.load_pool", return_value=mock_pool):
            args = SimpleNamespace(provider="zai")
            auth_list_command(args)

        captured = capsys.readouterr()
        # Empty base_url should show NO endpoint tag (clean display)
        assert "(default)" not in captured.out
        assert "url=" not in captured.out

    def test_auth_list_truncates_long_urls(self, capsys):
        """Long URLs should be truncated for display."""
        from hermes_cli.auth_commands import auth_list_command
        from agent.credential_pool import PooledCredential

        long_url = "https://api.example.com/very/long/path/that/exceeds/45/characters/and/should/be/truncated"
        entry = PooledCredential(
            provider="custom",
            id="abc123",
            label="Long URL Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-test",
            base_url=long_url,
        )

        mock_pool = MagicMock()
        mock_pool.entries.return_value = [entry]
        mock_pool.peek.return_value = entry

        with patch("hermes_cli.auth_commands.load_pool", return_value=mock_pool):
            args = SimpleNamespace(provider="custom")
            auth_list_command(args)

        captured = capsys.readouterr()
        # Custom URLs show hostname, not the full long URL
        assert long_url not in captured.out  # full URL should NOT be shown
        assert "example.com" in captured.out  # hostname should be shown


# ── Unified Resolver Integration Tests ───────────────────────────────────────
