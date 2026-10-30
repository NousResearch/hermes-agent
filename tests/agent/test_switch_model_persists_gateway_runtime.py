"""Regression test: an in-place model switch (auto-raise, rate-limit/entitlement fallback)
must persist the same model_config route the explicit ``/model`` commit path writes — top-level
keys AND nested ``gateway_runtime`` — not just the billing_* columns.

Before this fix, ``_persist_switch_billing_route()`` only called
``update_session_billing_route()``. ``session_gateway_runtime()`` reads nested
``gateway_runtime`` with top priority over everything else (by design, #125942/#125956), so a
cold-started agent after an in-place switch recombined the CURRENT model with the PREVIOUS
provider's stale ``gateway_runtime`` entry and sent an invalid pairing (e.g. a non-OpenAI model
id to the Codex endpoint), 400/404ing before the request reached the right host.

Reported: a webui session alternated between openai-codex/gpt-6.1-sol (correct) and
anthropic/gpt-6-1-sol (a stale leftover, not a real model id) every time an automatic switch —
Codex context-size auto-raise, or ollama-cloud rate-limit fallback — fired.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(session_db):
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openai-codex"
    agent.model = "gpt-6-astra-900k"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "codex-key"
    agent.api_mode = "codex_responses"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "codex-key", "base_url": "https://chatgpt.com/backend-api/codex"}
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._session_db = session_db
    agent.session_id = "local-test-session"

    return agent


def test_inplace_switch_persists_gateway_runtime_route():
    """A rate-limit fallback switch (openai-codex -> ollama-cloud) must write BOTH the billing
    route and the model_config route (gateway_runtime), mirroring the explicit /model path."""
    session_db = MagicMock()
    agent = _make_agent(session_db)

    with (
        patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="kimi-k3",
            new_provider="ollama-cloud",
            api_key="",
            base_url="https://ollama.com/v1",
            api_mode="chat_completions",
        )

    session_db.update_session_billing_route.assert_called_once_with(
        "local-test-session", provider="ollama-cloud", base_url="https://ollama.com/v1",
        billing_mode="chat_completions",
    )
    # The fix: the same switch must ALSO refresh model_config (top-level + gateway_runtime),
    # or the next cold-started agent pairs the new model with the OLD provider.
    session_db.update_session_model.assert_called_once_with(
        "local-test-session", "kimi-k3", provider="ollama-cloud",
        base_url="https://ollama.com/v1", api_mode="chat_completions",
    )


def test_inplace_switch_billing_failure_does_not_block_route_persist():
    """A billing-route persist failure must not prevent the gateway_runtime persist — the two
    are independent durability concerns and either one failing silently must not sink the other."""
    session_db = MagicMock()
    session_db.update_session_billing_route.side_effect = RuntimeError("db locked")
    agent = _make_agent(session_db)

    with (
        patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="kimi-k3",
            new_provider="ollama-cloud",
            api_key="",
            base_url="https://ollama.com/v1",
            api_mode="chat_completions",
        )

    session_db.update_session_model.assert_called_once_with(
        "local-test-session", "kimi-k3", provider="ollama-cloud",
        base_url="https://ollama.com/v1", api_mode="chat_completions",
    )
