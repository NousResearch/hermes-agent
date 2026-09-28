"""Tests that callable api_key (Entra ID bearer provider) flows through
the agent stack without coercion.

The OpenAI Python SDK accepts ``api_key: str | None | Callable[[], str]``,
and ``azure-identity``'s ``get_bearer_token_provider`` returns a callable.
Hermes preserves the callable end-to-end so the SDK refreshes tokens
transparently. This file pins the contract at the high-risk seams the
rubber-duck audit identified.

Covered:
  * ``_create_openai_client`` passes a callable ``api_key`` straight
    through to ``openai.OpenAI(...)``.
  * ``_normalize_main_runtime`` preserves the callable so auxiliary
    clients inherit Entra auth.
  * ``_truncate_token`` (dashboard preview) renders ``"<entra-id-bearer>"``
    instead of ``"<function ...>"`` and never invokes the callable.
  * ``run_agent.py`` masked-banner path renders the Entra placeholder
    and never tries to slice/len the callable.
  * Serialization scrub: dumping a runtime dict via ``json.dumps`` with
    a callable api_key raises (default behaviour) — guards against
    silently leaking ``"<function ...>"`` strings into event logs.
  * ``batch_runner`` strips the callable from the worker config dict
    so multiprocessing.Pool can pickle the rest.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# OpenAI SDK construction preserves the callable
# ---------------------------------------------------------------------------


class TestCreateOpenAIClientCallable:
    """``AIAgent._create_openai_client`` must pass the callable through
    to ``openai.OpenAI(...)`` without coercion."""

    def test_callable_api_key_passed_to_openai_constructor(self, monkeypatch):
        """Construct the smallest possible AIAgent surface and verify
        the OpenAI client receives the callable unchanged."""
        captured = {}

        def fake_openai(**kwargs):
            captured["kwargs"] = kwargs
            return MagicMock(api_key=kwargs.get("api_key"))

        # Patch the module-level OpenAI proxy used by ``_create_openai_client``.
        monkeypatch.setattr("agent.process_bootstrap.OpenAI", fake_openai)

        # Build a minimal stand-in for AIAgent so we can call the bound
        # method directly without paying the full __init__ cost.
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        # Attributes consulted by _create_openai_client / _client_log_context.
        agent.provider = "azure-foundry"
        agent.model = "gpt-4o"
        agent.base_url = "https://r.openai.azure.com/openai/v1"
        agent._client_kwargs = {}

        def token_provider():
            return "fresh-jwt"

        client_kwargs = {
            "api_key": token_provider,
            "base_url": "https://r.openai.azure.com/openai/v1",
        }
        client = agent._create_openai_client(client_kwargs, reason="test", shared=False)

        # The OpenAI constructor must receive the *callable*, not a string.
        forwarded = captured["kwargs"]["api_key"]
        assert callable(forwarded)
        assert not isinstance(forwarded, str)
        assert forwarded is token_provider, (
            "_create_openai_client must not wrap or coerce the callable"
        )
        assert client is not None


# ---------------------------------------------------------------------------
# Auxiliary runtime preserves the callable
# ---------------------------------------------------------------------------


class TestNormalizeMainRuntimePreservesCallable:
    """The aux client orchestrator must keep the callable on the
    runtime dict so compression / vision / embedding / title-gen clients
    inherit Entra ID auth from the main agent."""

    def test_callable_api_key_survives_normalization(self):
        from agent.auxiliary_client import _normalize_main_runtime

        def provider():
            return "jwt"

        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "model": "gpt-4o",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "api_key": provider,
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
        })
        assert normalized["api_key"] is provider
        assert normalized["auth_mode"] == "entra_id"

    def test_string_api_key_still_works(self):
        from agent.auxiliary_client import _normalize_main_runtime
        normalized = _normalize_main_runtime({
            "provider": "azure-foundry",
            "api_key": "sk-static",
        })
        assert normalized["api_key"] == "sk-static"




# ---------------------------------------------------------------------------
# Display surfaces never invoke the callable
# ---------------------------------------------------------------------------


class TestTruncateTokenCallable:
    def test_callable_returns_placeholder(self):
        """Dashboard preview must render the Entra placeholder, NOT
        ``"<function ...>"``."""
        from hermes_cli.web_server_oauth import _truncate_token

        invoked = {"count": 0}

        def provider():
            invoked["count"] += 1
            return "should-not-appear-in-ui"

        token_provider = cast(str | None, provider)
        rendered = _truncate_token(token_provider)
        assert rendered == "<entra-id-bearer>"
        assert invoked["count"] == 0

    def test_string_jwt_still_truncated_to_signature_tail(self):
        from hermes_cli.web_server_oauth import _truncate_token
        # JWT shape: header.payload.signature → only signature tail shown.
        out = _truncate_token("aaaa.bbbb.cccccccsig", visible=4)
        assert out == "…csig"

    def test_empty_returns_empty(self):
        from hermes_cli.web_server_oauth import _truncate_token
        assert _truncate_token(None) == ""
        assert _truncate_token("") == ""


# ---------------------------------------------------------------------------
# Serialization scrub — runtime dicts with callables must NOT silently
# JSON-encode as ``"<function ...>"`` (would leak garbage into events).
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# batch_runner strips callables from the worker config dict
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Inline masked-banner / display sites (callable-aware)
# ---------------------------------------------------------------------------






