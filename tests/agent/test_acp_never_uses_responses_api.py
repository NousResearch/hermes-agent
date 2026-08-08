"""ACP providers must never be routed to the Responses API.

``CopilotACPClient`` speaks a chat-shaped JSON-RPC protocol over a spawned
subprocess and exposes only a ``.chat`` surface — it has no ``.responses``
attribute. A gpt-5.x model reached over ACP (``copilot-acp`` /
``acp://copilot``) matches the "GPT-5 → Responses API" rule, so upgrading it
makes the loop dispatch ``client.responses.create`` and raise
``AttributeError: 'CopilotACPClient' object has no attribute 'responses'``
instead of completing the turn.

``agent_init.py`` guards the primary path. The fallback-activation path in
``chat_completion_helpers`` recomputes the api_mode from scratch, so the rule
has to hold in the shared helper too — otherwise a Copilot ACP entry
configured purely as a *fallback* still crashes.
"""

from run_agent import AIAgent


requires_responses = AIAgent._provider_model_requires_responses_api


class TestAcpNeverUsesResponsesApi:
    def test_copilot_acp_gpt5_model_stays_on_chat(self):
        """The exact crashing combination: gpt-5.x over copilot-acp."""
        assert requires_responses("gpt-5.6-sol", provider="copilot-acp") is False

    def test_copilot_acp_rejects_every_gpt5_variant(self):
        for model in ("gpt-5", "gpt-5-mini", "gpt-5.6-sol", "gpt-6-future"):
            assert requires_responses(model, provider="copilot-acp") is False, model

    def test_provider_match_is_case_insensitive(self):
        assert requires_responses("gpt-5.6-sol", provider="COPILOT-ACP") is False
        assert requires_responses("gpt-5.6-sol", provider="  Copilot-ACP  ") is False

    def test_non_acp_copilot_is_not_affected(self):
        """The regular HTTP copilot provider keeps its own routing logic."""
        assert requires_responses("gpt-5.6-sol", provider="copilot") is True
