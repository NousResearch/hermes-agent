"""Regression guard: Codex Cloudflare 403 mitigation headers.

The ``chatgpt.com/backend-api/codex`` endpoint sits behind a Cloudflare layer
that whitelists a small set of first-party originators (``codex_cli_rs``,
``codex_vscode``, ``codex_sdk_ts``, ``Codex*``). Requests from non-residential
IPs (VPS, always-on servers, some corporate egress) that don't advertise an
allowed originator are served 403 with ``cf-mitigated: challenge`` regardless
of auth correctness.

``_codex_cloudflare_headers`` in ``agent.auxiliary_client`` centralizes the
header set so the primary chat client (``run_agent.AIAgent.__init__`` +
``_apply_client_headers_for_base_url``) and the auxiliary client paths
(``_try_codex`` and the ``raw_codex`` branch of ``resolve_provider_client``)
all emit the same headers.

These tests pin:
- the originator value (must be ``codex_cli_rs`` — the whitelisted one)
- the User-Agent shape (codex_cli_rs-prefixed)
- ``ChatGPT-Account-ID`` extraction from the OAuth JWT (canonical casing,
  from codex-rs ``auth.rs``)
- graceful handling of malformed tokens (drop the account-ID header, don't
  raise)
- primary-client wiring at both entry points in ``run_agent.py``
- aux-client wiring at both entry points in ``agent/auxiliary_client.py``
"""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_codex_jwt(account_id: str = "acct-test-123") -> str:
    """Build a syntactically valid Codex-style JWT with the account_id claim."""
    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
    header = b64url(b'{"alg":"RS256","typ":"JWT"}')
    claims = {
        "sub": "user-xyz",
        "exp": 9999999999,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": "plus",
        },
    }
    payload = b64url(json.dumps(claims).encode())
    sig = b64url(b"fake-sig")
    return f"{header}.{payload}.{sig}"


# ---------------------------------------------------------------------------
# _codex_cloudflare_headers — the shared helper
# ---------------------------------------------------------------------------

class TestCodexCloudflareHeaders:
    def test_originator_is_codex_cli_rs(self):
        """Cloudflare whitelists codex_cli_rs — any other value is 403'd."""
        from agent.auxiliary_client import _codex_cloudflare_headers
        headers = _codex_cloudflare_headers(_make_codex_jwt())
        assert headers["originator"] == "codex_cli_rs"

    def test_user_agent_advertises_codex_cli_rs(self):
        from agent.auxiliary_client import _codex_cloudflare_headers
        headers = _codex_cloudflare_headers(_make_codex_jwt())
        assert headers["User-Agent"].startswith("codex_cli_rs/")

    def test_account_id_extracted_from_jwt(self):
        from agent.auxiliary_client import _codex_cloudflare_headers
        headers = _codex_cloudflare_headers(_make_codex_jwt("acct-abc-999"))
        # Canonical casing — matches codex-rs auth.rs
        assert headers["ChatGPT-Account-ID"] == "acct-abc-999"

    def test_canonical_header_casing(self):
        """Upstream codex-rs uses PascalCase with trailing -ID. Match exactly."""
        from agent.auxiliary_client import _codex_cloudflare_headers
        headers = _codex_cloudflare_headers(_make_codex_jwt())
        assert "ChatGPT-Account-ID" in headers
        # The lowercase/titlecase variants MUST NOT be used — pin to be explicit
        assert "chatgpt-account-id" not in headers
        assert "ChatGPT-Account-Id" not in headers

    def test_malformed_token_drops_account_id_without_raising(self):
        from agent.auxiliary_client import _codex_cloudflare_headers
        for bad in ["not-a-jwt", "", "only.one", "  ", "...."]:
            headers = _codex_cloudflare_headers(bad)
            # Still returns base headers — never raises
            assert headers["originator"] == "codex_cli_rs"
            assert "ChatGPT-Account-ID" not in headers

    def test_non_string_token_handled(self):
        from agent.auxiliary_client import _codex_cloudflare_headers
        headers = _codex_cloudflare_headers(None)  # type: ignore[arg-type]
        assert headers["originator"] == "codex_cli_rs"
        assert "ChatGPT-Account-ID" not in headers

    def test_jwt_without_chatgpt_account_id_claim(self):
        """A valid JWT that lacks the account_id claim should still return headers."""
        from agent.auxiliary_client import _codex_cloudflare_headers
        import base64 as _b64, json as _json

        def b64url(data: bytes) -> str:
            return _b64.urlsafe_b64encode(data).rstrip(b"=").decode()
        payload = b64url(_json.dumps({"sub": "user-xyz", "exp": 9999999999}).encode())
        token = f"{b64url(b'{}')}.{payload}.{b64url(b'sig')}"
        headers = _codex_cloudflare_headers(token)
        assert headers["originator"] == "codex_cli_rs"
        assert "ChatGPT-Account-ID" not in headers


# ---------------------------------------------------------------------------
# Primary chat client wiring (run_agent.AIAgent)
# ---------------------------------------------------------------------------

class TestPrimaryClientWiring:
    def test_init_wires_codex_headers_for_chatgpt_base_url(self):
        from run_agent import AIAgent
        token = _make_codex_jwt("acct-primary-init")
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            AIAgent(
                api_key=token,
                base_url="https://chatgpt.com/backend-api/codex",
                provider="openai-codex",
                model="gpt-5.4",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            headers = mock_openai.call_args.kwargs.get("default_headers") or {}
            assert headers.get("originator") == "codex_cli_rs"
            assert headers.get("ChatGPT-Account-ID") == "acct-primary-init"
            assert headers.get("User-Agent", "").startswith("codex_cli_rs/")

    def test_apply_client_headers_on_base_url_change(self):
        """Credential-rotation / base-url change path must also emit codex headers."""
        from run_agent import AIAgent
        token = _make_codex_jwt("acct-rotation")
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            agent = AIAgent(
                api_key="placeholder-openrouter-key",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            # Simulate rotation into a Codex credential
            agent._client_kwargs["api_key"] = token
            agent._apply_client_headers_for_base_url(
                "https://chatgpt.com/backend-api/codex"
            )
            headers = agent._client_kwargs.get("default_headers") or {}
            assert headers.get("originator") == "codex_cli_rs"
            assert headers.get("ChatGPT-Account-ID") == "acct-rotation"
            assert headers.get("User-Agent", "").startswith("codex_cli_rs/")

    def test_apply_client_headers_clears_codex_headers_off_chatgpt(self):
        """Switching AWAY from chatgpt.com must drop the codex headers."""
        from run_agent import AIAgent
        token = _make_codex_jwt()
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            agent = AIAgent(
                api_key=token,
                base_url="https://chatgpt.com/backend-api/codex",
                provider="openai-codex",
                model="gpt-5.4",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            # Sanity: headers are set initially
            assert "originator" in (agent._client_kwargs.get("default_headers") or {})
            agent._apply_client_headers_for_base_url(
                "https://api.anthropic.com"
            )
            # default_headers should be popped for anthropic base
            assert "default_headers" not in agent._client_kwargs

    def test_openrouter_base_url_does_not_get_codex_headers(self):
        from run_agent import AIAgent
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            AIAgent(
                api_key="sk-or-test",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            headers = mock_openai.call_args.kwargs.get("default_headers") or {}
            assert headers.get("originator") != "codex_cli_rs"


# ---------------------------------------------------------------------------
# Auxiliary client wiring (agent.auxiliary_client)
# ---------------------------------------------------------------------------

class TestAuxiliaryClientWiring:
    def test_try_codex_passes_codex_headers(self, monkeypatch):
        """_try_codex builds the OpenAI client used for compression / vision /
        title generation when routed through Codex. Must emit codex headers."""
        from agent import auxiliary_client
        token = _make_codex_jwt("acct-aux-try-codex")

        # Force _select_pool_entry to return "no pool" so we fall through to
        # _read_codex_access_token.
        monkeypatch.setattr(
            auxiliary_client, "_select_pool_entry",
            lambda provider: (False, None),
        )
        monkeypatch.setattr(
            auxiliary_client, "_read_codex_access_token",
            lambda: token,
        )
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = auxiliary_client._try_codex()
            assert client is not None
            headers = mock_openai.call_args.kwargs.get("default_headers") or {}
            assert headers.get("originator") == "codex_cli_rs"
            assert headers.get("ChatGPT-Account-ID") == "acct-aux-try-codex"
            assert headers.get("User-Agent", "").startswith("codex_cli_rs/")

    def test_resolve_provider_client_raw_codex_passes_codex_headers(self, monkeypatch):
        """The ``raw_codex=True`` branch (used by the main agent loop for direct
        responses.stream() access) must also emit codex headers."""
        from agent import auxiliary_client
        token = _make_codex_jwt("acct-aux-raw-codex")
        monkeypatch.setattr(
            auxiliary_client, "_read_codex_access_token",
            lambda: token,
        )
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = auxiliary_client.resolve_provider_client(
                "openai-codex", raw_codex=True,
            )
            assert client is not None
            headers = mock_openai.call_args.kwargs.get("default_headers") or {}
            assert headers.get("originator") == "codex_cli_rs"
            assert headers.get("ChatGPT-Account-ID") == "acct-aux-raw-codex"
            assert headers.get("User-Agent", "").startswith("codex_cli_rs/")
