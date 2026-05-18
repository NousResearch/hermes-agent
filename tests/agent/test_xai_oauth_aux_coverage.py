"""Regression coverage for xai-oauth auxiliary client resolution.

Usage:
    uv run pytest tests/agent/test_xai_oauth_aux_coverage.py -v
"""

from unittest.mock import patch, MagicMock

XAI_OAUTH_MODEL = "grok-4-1-fast-non-reasoning"

# ── Mock helpers ───────────────────────────────────────────────────────────


class _FakeCredential:
    """Simulates a PooledCredential with a valid xai-oauth token."""

    def __init__(self):
        self.access_token = "xai-oauth-fake-access-token-abc123"
        self.refresh_token = "xai-oauth-fake-refresh-token-xyz789"
        self.runtime_api_key = "xai-oauth-fake-access-token-abc123"
        self.runtime_base_url = None
        self.base_url = "https://api.x.ai/v1"


class _FakePool:
    """Simulates CredentialPool returning a single valid xai-oauth entry."""

    def __init__(self):
        self._entry = _FakeCredential()

    def has_credentials(self):
        return True

    def has_available(self):
        return True

    def select(self):
        return self._entry

    def entries(self):
        return [self._entry]


# ── Test runner ────────────────────────────────────────────────────────────


def _make_mock_openai():
    """Return a MagicMock that looks like an OpenAI client instance."""
    mock_client = MagicMock()
    mock_client.api_key = "xai-oauth-fake-access-token-abc123"
    mock_client.base_url = "https://api.x.ai/v1"
    return mock_client


def test_resolve_provider_client_uses_xai_oauth_builder():
    """resolve_provider_client returns a concrete xai-oauth client/model tuple."""
    fake_pool = _FakePool()
    mock_client = _make_mock_openai()

    with patch("agent.auxiliary_client.load_pool", return_value=fake_pool), \
         patch("agent.auxiliary_client.OpenAI", return_value=mock_client):
        from agent.auxiliary_client import resolve_provider_client

        result = resolve_provider_client("xai-oauth", XAI_OAUTH_MODEL)

    assert result is not None
    client, model = result
    assert client is mock_client
    assert model == XAI_OAUTH_MODEL


def test_xai_oauth_returns_client_and_model():
    """Verify that resolve_provider_client("xai-oauth", "grok-4-1-fast-non-reasoning")
    returns the expected (client, model) tuple through _build_xai_oauth_aux_client."""
    fake_pool = _FakePool()
    mock_client = _make_mock_openai()

    with patch("agent.auxiliary_client.load_pool", return_value=fake_pool):
        with patch("agent.auxiliary_client.OpenAI", return_value=mock_client):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("xai-oauth", XAI_OAUTH_MODEL)

            assert client is not None, "Expected a valid client, got None"
            assert model == XAI_OAUTH_MODEL, (
                f"Expected model {XAI_OAUTH_MODEL!r}, got {model!r}"
            )
            # xAI uses Chat Completions, not the Responses API.
            assert hasattr(client, "chat"), "Client missing .chat attribute"
            assert hasattr(client.chat, "completions"), (
                "Client missing .chat.completions attribute"
            )
            assert hasattr(client, "api_key"), "Client missing .api_key attribute"
            assert hasattr(client, "base_url"), "Client missing .base_url attribute"



def test_vision_lane_resolves_xai_oauth():
    """Verify that the vision lane can resolve xai-oauth credentials
    through the resolve_vision_provider_client path."""
    fake_pool = _FakePool()
    mock_client = _make_mock_openai()

    with patch("agent.auxiliary_client.load_pool", return_value=fake_pool):
        with patch("agent.auxiliary_client.OpenAI", return_value=mock_client):
            from agent.auxiliary_client import resolve_vision_provider_client

            # Test resolve_vision_provider_client with explicit xai-oauth provider
            requested, client, model = resolve_vision_provider_client(
                "xai-oauth", XAI_OAUTH_MODEL
            )

            assert client is not None, (
                "Vision: resolve_vision_provider_client('xai-oauth', model) "
                "returned None for client"
            )
            assert model == XAI_OAUTH_MODEL, (
                f"Vision: expected model {XAI_OAUTH_MODEL!r}, got {model!r}"
            )
            assert requested == "xai-oauth", (
                f"Vision: expected provider 'xai-oauth', got {requested!r}"
            )



def test_strict_vision_backend_does_not_support_xai_oauth():
    """Verify _resolve_strict_vision_backend('xai-oauth', model) returns
    (None, None) — this backend dispatches to specific providers only."""
    from agent.auxiliary_client import _resolve_strict_vision_backend

    client, model = _resolve_strict_vision_backend("xai-oauth", XAI_OAUTH_MODEL)

    assert client is None, (
        "_resolve_strict_vision_backend should return None for xai-oauth "
        "(no dedicated dispatch arm)"
    )
    assert model is None, (
        "_resolve_strict_vision_backend should return None model for xai-oauth"
    )



def test_empty_xai_oauth_runtime_uses_config_gated_api_key_fallback(monkeypatch):
    """Empty xai-oauth runtime creds should use XAI_API_KEY only when opted in."""
    from agent.auxiliary_client import _resolve_xai_oauth_for_aux

    monkeypatch.setenv("XAI_API_KEY", "xai-api-key-fallback")
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    with patch("agent.auxiliary_client.load_pool", return_value=None), \
         patch("hermes_cli.auth.resolve_xai_oauth_runtime_credentials", return_value={"api_key": "", "base_url": ""}), \
         patch("hermes_cli.config.load_config", return_value={"auxiliary": {"xai_fallback_to_api_key": True}}):
        assert _resolve_xai_oauth_for_aux() == (
            "xai-api-key-fallback",
            "https://api.x.ai/v1",
        )


def test_nous_runtime_never_uses_xai_api_key_fallback(monkeypatch):
    """Nous auxiliary auth must not return xAI credentials on empty Nous creds."""
    from agent.auxiliary_client import _resolve_nous_runtime_api

    monkeypatch.setenv("XAI_API_KEY", "xai-api-key-fallback")

    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", return_value={"api_key": "", "base_url": ""}), \
         patch("hermes_cli.config.load_config", return_value={"auxiliary": {"xai_fallback_to_api_key": True}}):
        assert _resolve_nous_runtime_api() is None
