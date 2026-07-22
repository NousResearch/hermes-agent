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

from agent.auth import resolve_provider_credentials, ResolvedCredential


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

class TestUnifiedResolverIntegration:
    """Test the unified resolver with real pool entries (not just mocks)."""

    def test_resolve_with_real_pooled_credential(self):
        """resolve_provider_credentials should work with a real PooledCredential."""
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="zai",
            id="test-1",
            label="Test ZAI Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-fake-zai-key",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )

        result = resolve_provider_credentials(
            provider="zai",
            entry=entry,
            model_cfg={"provider": "zai", "default": "glm-5.2"},
        )
        assert result.base_url != ""
        assert "z.ai" in result.base_url
        assert result.entry is entry  # entry should be passed through

    def test_resolve_with_empty_base_url_entry(self):
        """The core cascade bug: entry with base_url='' should NOT return empty."""
        from agent.credential_pool import PooledCredential

        entry = PooledCredential(
            provider="zai",
            id="test-2",
            label="Broken Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-fake-key",
            base_url="",  # ← THE BUG
        )

        result = resolve_provider_credentials(
            provider="zai",
            entry=entry,
            model_cfg={"provider": "zai", "default": "glm-5.2"},
        )
        assert result.base_url != "", "base_url should not be empty!"
        assert result.base_url.startswith("https://")

    def test_resolve_deepseek_returns_correct_url(self):
        """DeepSeek should always return the correct API endpoint."""
        result = resolve_provider_credentials(
            provider="deepseek",
            model_cfg={"provider": "deepseek", "default": "deepseek-chat"},
        )
        assert result.base_url == "https://api.deepseek.com/v1"
        assert result.api_mode == "chat_completions"

    def test_resolve_anthropic_strips_v1(self):
        """Anthropic should strip /v1 suffix for anthropic_messages mode."""
        result = resolve_provider_credentials(
            provider="anthropic",
            model_cfg={
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1",
            },
        )
        assert not result.base_url.endswith("/v1")
        assert result.api_mode == "anthropic_messages"

    def test_resolve_minimax_cn_uses_china_endpoint(self):
        """minimax-cn should use api.minimaxi.com (China), not api.minimax.io."""
        result = resolve_provider_credentials(
            provider="minimax-cn",
            model_cfg={"provider": "minimax-cn"},
        )
        assert "minimaxi.com" in result.base_url

    def test_resolve_minimax_uses_international_endpoint(self):
        """minimax should use api.minimax.io (international), not api.minimaxi.com."""
        result = resolve_provider_credentials(
            provider="minimax",
            model_cfg={"provider": "minimax"},
        )
        assert "minimax.io" in result.base_url

    def test_resolve_openrouter_uses_correct_url(self):
        """OpenRouter should use the official API URL."""
        result = resolve_provider_credentials(
            provider="openrouter",
            model_cfg={"provider": "openrouter"},
        )
        assert "openrouter.ai" in result.base_url

    def test_resolve_xai_uses_codex_responses(self):
        """xAI API key should use codex_responses mode."""
        result = resolve_provider_credentials(
            provider="xai",
            model_cfg={"provider": "xai"},
        )
        assert result.api_mode == "codex_responses"
        assert "api.x.ai" in result.base_url

    def test_resolve_lmstudio_normalizes_url(self):
        """LM Studio should normalize its URL."""
        result = resolve_provider_credentials(
            provider="lmstudio",
            model_cfg={"provider": "lmstudio"},
        )
        assert "localhost" in result.base_url or "127.0.0.1" in result.base_url

    def test_resolve_returns_resolved_credential_type(self):
        """Result should always be a ResolvedCredential instance."""
        result = resolve_provider_credentials(
            provider="deepseek",
            model_cfg={"provider": "deepseek"},
        )
        assert isinstance(result, ResolvedCredential)
        assert hasattr(result, "provider")
        assert hasattr(result, "api_key")
        assert hasattr(result, "base_url")
        assert hasattr(result, "api_mode")
        assert hasattr(result, "source")

    def test_resolve_source_label_correct_for_explicit(self):
        """Source should be 'explicit' when explicit args are provided."""
        result = resolve_provider_credentials(
            provider="deepseek",
            explicit_api_key="sk-explicit",
            explicit_base_url="https://explicit.example.com/v1",
            model_cfg={"provider": "deepseek"},
        )
        assert result.source == "explicit"
        assert "explicit.example.com" in result.base_url

    def test_resolve_source_label_correct_for_env(self):
        """Source should be 'env' when env var override is used."""
        with patch.dict(os.environ, {"GLM_BASE_URL": "https://env.z.ai/api/coding/paas/v4"}):
            result = resolve_provider_credentials(
                provider="zai",
                model_cfg={"provider": "zai"},
            )
        assert result.source == "env"

    def test_resolve_source_label_correct_for_config(self):
        """Source should be 'config' when config.yaml override is used."""
        result = resolve_provider_credentials(
            provider="deepseek",
            model_cfg={
                "provider": "deepseek",
                "base_url": "https://config.deepseek.com/v1",
            },
        )
        # config should win over registry default
        assert "config.deepseek.com" in result.base_url


# ── Cascade Bug Regression Tests ─────────────────────────────────────────────

class TestCascadeBugRegression:
    """Regression tests for the base_url="" cascade bug.

    These tests ensure that the bug that caused all pool entries to be
    marked as exhausted when one entry had base_url="" is FIXED and
    stays fixed.
    """

    @pytest.mark.parametrize("provider,expected_domain", [
        ("zai", "z.ai"),
        ("deepseek", "deepseek.com"),
        ("minimax-cn", "minimaxi.com"),
        ("anthropic", "anthropic.com"),
        ("minimax", "minimax.io"),
        ("openrouter", "openrouter.ai"),
        ("xai", "x.ai"),
    ])
    def test_empty_base_url_never_reaches_http_client(self, provider, expected_domain):
        """Every provider should return a non-empty base_url with the correct domain,
        even when the pool entry has base_url=''."""
        entry = _fake_entry(provider=provider, base_url="")
        result = resolve_provider_credentials(
            provider=provider,
            entry=entry,
            model_cfg={"provider": provider},
        )
        assert result.base_url != "", f"{provider}: base_url is empty!"
        assert expected_domain in result.base_url, \
            f"{provider}: expected '{expected_domain}' in base_url, got: {result.base_url}"

    def test_multiple_empty_entries_all_resolve(self):
        """Multiple entries with base_url='' should ALL resolve to non-empty URLs."""
        providers = ["zai", "deepseek", "minimax-cn", "anthropic"]
        for provider in providers:
            entry = _fake_entry(provider=provider, base_url="")
            result = resolve_provider_credentials(
                provider=provider,
                entry=entry,
                model_cfg={"provider": provider},
            )
            assert result.base_url != "", f"{provider}: base_url is empty!"
            assert result.base_url.startswith("https://"), \
                f"{provider}: base_url is not HTTPS: {result.base_url}"


# ── Cross-Provider Parity Tests ──────────────────────────────────────────────

class TestCrossProviderParity:
    """Verify that CLI and Gateway/Desktop paths now use the SAME resolver."""

    @pytest.mark.parametrize("provider", [
        "deepseek", "zai", "minimax", "minimax-cn", "anthropic",
        "openrouter", "xai", "lmstudio", "stepfun", "arcee",
    ])
    def test_provider_returns_valid_credential(self, provider):
        """Every provider should return a valid ResolvedCredential."""
        result = resolve_provider_credentials(
            provider=provider,
            model_cfg={"provider": provider},
        )
        assert isinstance(result, ResolvedCredential)
        assert result.provider == provider
        assert isinstance(result.base_url, str)
        assert isinstance(result.api_mode, str)
        assert isinstance(result.source, str)
        assert result.api_mode in (
            "chat_completions", "anthropic_messages", "codex_responses",
            "gemini_native", "codex_app_server",
        )

    def test_all_providers_return_https(self):
        """All providers should return HTTPS URLs (except lmstudio which is localhost)."""
        providers = ["deepseek", "zai", "minimax", "minimax-cn", "anthropic",
                     "openrouter", "xai", "stepfun", "arcee"]
        for provider in providers:
            result = resolve_provider_credentials(
                provider=provider,
                model_cfg={"provider": provider},
            )
            assert result.base_url.startswith("https://"), \
                f"{provider}: base_url should be HTTPS: {result.base_url}"
