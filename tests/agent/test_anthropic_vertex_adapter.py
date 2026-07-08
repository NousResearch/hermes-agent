"""Tests for agent/anthropic_vertex_adapter.py — Anthropic on Vertex AI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# Anthropic-vertex reuses the vertex adapter's credential resolution helpers.
# Tests here mock at the seam between our adapter and the google-auth /
# anthropic SDKs, so they don't hit either dependency at runtime.


def _reset_anthropic_sdk_cache():
    """Clear the cached ``_anthropic_sdk`` sentinel between tests.

    The adapter caches the imported SDK module (or ``None`` when the import
    fails) after the first access. Tests that patch the SDK need to reset
    the sentinel so each test resolves it independently.
    """
    from agent import anthropic_adapter

    anthropic_adapter._anthropic_sdk = ...  # sentinel


# ---------------------------------------------------------------------------
# Base URL builder
# ---------------------------------------------------------------------------


class TestBuildAnthropicVertexBaseUrl:
    def test_global_uses_bare_host(self):
        from agent.anthropic_vertex_adapter import build_anthropic_vertex_base_url

        url = build_anthropic_vertex_base_url("my-proj", "global")
        assert url == (
            "https://aiplatform.googleapis.com/v1/projects/my-proj"
            "/locations/global/publishers/anthropic"
        )

    def test_regional_uses_prefixed_host(self):
        from agent.anthropic_vertex_adapter import build_anthropic_vertex_base_url

        url = build_anthropic_vertex_base_url("my-proj", "us-east5")
        assert url == (
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-proj"
            "/locations/us-east5/publishers/anthropic"
        )


# ---------------------------------------------------------------------------
# Credentials resolution
# ---------------------------------------------------------------------------


class TestResolveGoogleCredentials:
    def test_missing_google_auth_returns_none(self):
        """When google-auth is not installed, resolver returns (None, None)."""
        with patch("agent.anthropic_vertex_adapter.google", None):
            from agent.anthropic_vertex_adapter import _resolve_google_credentials

            creds, project_id = _resolve_google_credentials()
            assert creds is None
            assert project_id is None

    def test_adc_returns_credentials_and_project(self):
        """With ADC available, resolver returns (creds, project_id)."""
        mock_creds = MagicMock()
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_credentials_path",
                return_value=None,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_project_override",
                return_value=None,
            ),
            patch("agent.anthropic_vertex_adapter.google") as mock_google,
        ):
            mock_google.auth.default.return_value = (mock_creds, "test-project-42")

            from agent.anthropic_vertex_adapter import _resolve_google_credentials

            creds, project_id = _resolve_google_credentials()
            assert creds is mock_creds
            assert project_id == "test-project-42"

    def test_explicit_project_override_wins(self):
        """VERTEX_PROJECT_ID env / config.yaml override wins over embedded project."""
        mock_creds = MagicMock()
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_credentials_path",
                return_value=None,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_project_override",
                return_value="override-project",
            ),
            patch("agent.anthropic_vertex_adapter.google") as mock_google,
        ):
            mock_google.auth.default.return_value = (mock_creds, "embedded-project")

            from agent.anthropic_vertex_adapter import _resolve_google_credentials

            _creds, project_id = _resolve_google_credentials()
            assert project_id == "override-project"


class TestGetAnthropicVertexConfig:
    def test_returns_project_and_default_region(self):
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(MagicMock(), "test-proj"),
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_region",
                return_value="global",
            ),
        ):
            from agent.anthropic_vertex_adapter import get_anthropic_vertex_config

            project_id, region = get_anthropic_vertex_config()
            assert project_id == "test-proj"
            assert region == "global"

    def test_no_project_returns_none_none(self):
        """When credentials resolve but embedded project is empty, return (None, None)."""
        with patch(
            "agent.anthropic_vertex_adapter._resolve_google_credentials",
            return_value=(MagicMock(), None),
        ):
            from agent.anthropic_vertex_adapter import get_anthropic_vertex_config

            project_id, region = get_anthropic_vertex_config()
            assert project_id is None
            assert region is None

    def test_explicit_region_argument_wins(self):
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(MagicMock(), "test-proj"),
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_region",
                side_effect=lambda explicit=None: explicit or "global",
            ),
        ):
            from agent.anthropic_vertex_adapter import get_anthropic_vertex_config

            _p, region = get_anthropic_vertex_config(region="us-east5")
            assert region == "us-east5"


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestBuildAnthropicVertexClient:
    def setup_method(self):
        _reset_anthropic_sdk_cache()

    def teardown_method(self):
        _reset_anthropic_sdk_cache()

    def test_missing_sdk_raises(self):
        with patch("agent.anthropic_adapter._anthropic_sdk", None):
            from agent.anthropic_vertex_adapter import build_anthropic_vertex_client

            with pytest.raises(ImportError, match="anthropic"):
                build_anthropic_vertex_client("proj", "global")

    def test_sdk_without_anthropic_vertex_raises(self):
        """Older SDK versions without AnthropicVertex class fail clearly."""
        mock_sdk = MagicMock()
        del mock_sdk.AnthropicVertex  # attribute absent
        with (
            patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(MagicMock(), "proj"),
            ),
        ):
            from agent.anthropic_vertex_adapter import build_anthropic_vertex_client

            with pytest.raises(ImportError, match="AnthropicVertex not available"):
                build_anthropic_vertex_client("proj", "global")

    def test_missing_credentials_raises(self):
        mock_sdk = MagicMock()
        mock_sdk.AnthropicVertex = MagicMock()
        with (
            patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(None, None),
            ),
        ):
            from agent.anthropic_vertex_adapter import build_anthropic_vertex_client

            with pytest.raises(RuntimeError, match="credentials could not be resolved"):
                build_anthropic_vertex_client("proj", "global")

    def test_client_constructed_with_expected_kwargs(self):
        mock_sdk = MagicMock()
        mock_sdk.AnthropicVertex = MagicMock()
        mock_creds = MagicMock()
        with (
            patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(mock_creds, "creds-proj"),
            ),
        ):
            from agent.anthropic_vertex_adapter import build_anthropic_vertex_client

            build_anthropic_vertex_client("explicit-proj", "us-east5", timeout=120.0)

        kwargs = mock_sdk.AnthropicVertex.call_args[1]
        # Explicit project_id wins over credentials' embedded project.
        assert kwargs["project_id"] == "explicit-proj"
        assert kwargs["region"] == "us-east5"
        assert kwargs["credentials"] is mock_creds
        # Hermes disables SDK-level retries so its own outer loop can honor
        # Retry-After. Same contract as bedrock.
        assert kwargs["max_retries"] == 0
        # Common Anthropic beta headers are attached; context-1m is NOT
        # (subscriptions without the long-context beta reject it).
        betas = kwargs["default_headers"]["anthropic-beta"]
        assert "interleaved-thinking-2025-05-14" in betas
        assert "fine-grained-tool-streaming-2025-05-14" in betas
        assert "context-1m-2025-08-07" not in betas

    def test_credentials_project_used_when_explicit_project_falsy(self):
        mock_sdk = MagicMock()
        mock_sdk.AnthropicVertex = MagicMock()
        with (
            patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(MagicMock(), "creds-proj"),
            ),
        ):
            from agent.anthropic_vertex_adapter import build_anthropic_vertex_client

            build_anthropic_vertex_client("", "global")

        kwargs = mock_sdk.AnthropicVertex.call_args[1]
        assert kwargs["project_id"] == "creds-proj"


# ---------------------------------------------------------------------------
# Fast credential-present check
# ---------------------------------------------------------------------------


class TestHasAnthropicVertexCredentials:
    def test_service_account_path_returns_true(self):
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_credentials_path",
                return_value="/tmp/sa.json",
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_project_override",
                return_value=None,
            ),
        ):
            from agent.anthropic_vertex_adapter import has_anthropic_vertex_credentials

            assert has_anthropic_vertex_credentials() is True

    def test_project_override_returns_true(self):
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_credentials_path",
                return_value=None,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_project_override",
                return_value="my-proj",
            ),
        ):
            from agent.anthropic_vertex_adapter import has_anthropic_vertex_credentials

            assert has_anthropic_vertex_credentials() is True

    def test_no_config_returns_false(self):
        with (
            patch(
                "agent.anthropic_vertex_adapter._resolve_credentials_path",
                return_value=None,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_project_override",
                return_value=None,
            ),
        ):
            from agent.anthropic_vertex_adapter import has_anthropic_vertex_credentials

            assert has_anthropic_vertex_credentials() is False


# ---------------------------------------------------------------------------
# Model classifier — dispatches ``vertex`` provider onto anthropic_messages
# ---------------------------------------------------------------------------


class TestIsAnthropicVertexModel:
    """``is_anthropic_vertex_model`` is the runtime dispatch classifier.

    Called by ``resolve_runtime_provider`` when the requested provider
    is ``vertex`` — a True return routes through the ``AnthropicVertex``
    SDK (anthropic_messages wire), a False return routes through Vertex's
    OpenAI-compat aggregator (chat_completions wire, same code path as
    Gemini-on-Vertex).

    Behavior contract (STRICT — no legacy shortcuts):

    * ``anthropic/<anything>`` → True. The vendor prefix is REQUIRED.
    * Anything else — including bare ``claude-*`` — → False.

    The strict form is deliberate. Vertex Model Garden is multi-vendor
    from day one, so a bare model name has no unambiguous meaning:
    ``claude-opus-4-8`` under ``provider=vertex`` could plausibly be
    misrouting a Gemini setup that accidentally shipped a Claude ID,
    and the right behavior is to surface that as a Vertex 404 pointing
    at the misconfiguration rather than silently guessing at the
    intended wire protocol. Contrast Bedrock, which accepts bare
    ``claude-*`` as a legacy shortcut from the era when Bedrock was
    Anthropic-only.
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic/claude-opus-4-8",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-fable-5",
            # Version-suffixed IDs — the ``@YYYYMMDD`` form Vertex also accepts.
            "anthropic/claude-opus-4-5@20250929",
            # Case-insensitive.
            "ANTHROPIC/claude-opus-4-8",
            "Anthropic/Claude-Opus-4-8",
            # Whitespace-tolerant (defensive against config-file whitespace).
            "  anthropic/claude-opus-4-8  ",
        ],
    )
    def test_vendor_prefixed_anthropic_matches(self, model_id):
        from agent.anthropic_vertex_adapter import is_anthropic_vertex_model

        assert is_anthropic_vertex_model(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            # Bare Claude names must NOT match — vendor prefix is required
            # (see class docstring). These will fall through to the
            # OpenAI-compat aggregator and 404 with an actionable error.
            "claude-opus-4-8",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-fable-5",
            "CLAUDE-OPUS-4-8",
        ],
    )
    def test_bare_claude_rejected_without_vendor_prefix(self, model_id):
        """Regression guard: strict vendor-prefix requirement.

        Historically an earlier draft of the classifier accepted bare
        ``claude-*`` as a convenience. That was dropped: Vertex is a
        multi-vendor surface, so a bare name has no unambiguous meaning.
        """
        from agent.anthropic_vertex_adapter import is_anthropic_vertex_model

        assert is_anthropic_vertex_model(model_id) is False

    @pytest.mark.parametrize(
        "model_id",
        [
            # Gemini on Vertex — must take the OpenAI-compat path.
            "google/gemini-3.1-pro-preview",
            "google/gemini-3-pro-preview",
            "google/gemma-3-27b-it",
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            # OpenRouter-style Anthropic slug (different provider entirely
            # — should NOT reach the vertex dispatch, but the classifier
            # correctly returns False for the negative case too).
            "openrouter/anthropic/claude-opus-4-8",
            # Non-Claude Anthropic-ish strings we shouldn't accidentally match.
            "claudius-something",
        ],
    )
    def test_non_anthropic_models_reject(self, model_id):
        from agent.anthropic_vertex_adapter import is_anthropic_vertex_model

        assert is_anthropic_vertex_model(model_id) is False

    @pytest.mark.parametrize("value", ["", "   ", None, 42, 0, [], {}])
    def test_empty_or_non_string_reject(self, value):
        """Defensive: unexpected inputs must return False, not raise."""
        from agent.anthropic_vertex_adapter import is_anthropic_vertex_model

        assert is_anthropic_vertex_model(value) is False
