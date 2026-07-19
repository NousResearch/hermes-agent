"""Bearer-token vs SigV4 auth regression tests for the AnthropicBedrock path.

Hermes routes Claude-on-Bedrock through ``build_anthropic_bedrock_client``
(``agent/anthropic_adapter.py``), which constructs ``AnthropicBedrock`` with
no explicit ``api_key`` and relies on the SDK's credential chain. Bearer-token
auth (``AWS_BEARER_TOKEN_BEDROCK``, short-term ``ABSK…`` keys) is only read by
the SDK starting with ``anthropic`` 0.88.0 — on 0.87.0 the client is
SigV4-only and bearer-token users fail at request time with
``RuntimeError: could not resolve credentials from session``.

These tests exercise the REAL installed SDK (no mocks) so a future pin
downgrade below 0.88.0 fails loudly here instead of at runtime.
"""

import pytest

anthropic = pytest.importorskip("anthropic")

_SDK_HAS_BEARER_SUPPORT = tuple(
    int(part) for part in anthropic.__version__.split(".")[:2]
) >= (0, 88)


class TestBedrockBearerTokenAuth:
    """``AWS_BEARER_TOKEN_BEDROCK`` must reach the AnthropicBedrock client."""

    def test_installed_sdk_supports_bearer_tokens(self):
        """The pinned SDK must be >= 0.88.0, the first release whose
        AnthropicBedrock reads AWS_BEARER_TOKEN_BEDROCK. Guards against the
        pin (pyproject / tools/lazy_deps.py / uv.lock) drifting back below."""
        assert _SDK_HAS_BEARER_SUPPORT, (
            f"anthropic=={anthropic.__version__} predates Bedrock bearer-token "
            "support (needs >= 0.88.0); bearer-key users regress to "
            "'could not resolve credentials from session'"
        )

    def test_bedrock_client_picks_up_bearer_token_env(self, monkeypatch):
        """With the env var set, the client authenticates via bearer token."""
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer-token")

        from agent.anthropic_adapter import build_anthropic_bedrock_client

        client = build_anthropic_bedrock_client(region="us-east-1")
        assert client.api_key == "test-bearer-token", (
            "AnthropicBedrock did not read AWS_BEARER_TOKEN_BEDROCK — "
            "bearer-token Bedrock auth is broken (SDK pin below 0.88.0?)"
        )

    def test_bedrock_client_defaults_to_sigv4_without_bearer_token(self, monkeypatch):
        """Without the env var, the client falls back to the SigV4/boto3
        credential chain (api_key unset) — bearer support must not break
        existing IAM/SSO users."""
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

        from agent.anthropic_adapter import build_anthropic_bedrock_client

        client = build_anthropic_bedrock_client(region="us-east-1")
        assert client.api_key is None, (
            "api_key should be unset without AWS_BEARER_TOKEN_BEDROCK so the "
            "SDK uses the SigV4/boto3 credential chain"
        )
