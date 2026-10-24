"""The agent-level fallback must not switch to a model that cannot hold the session.

Incident: t_e94832a1 (2026-10-23). MiniMax hit its plan usage cap, the ``local-llama``
fallback took the load, and the gateway switched long sessions onto a 32K-context local
model. The server rejected every one of them::

    E srv send_error: task id = 864, error: request (44622 tokens) exceeds the available
    context size (32768 tokens)

The agent-level fallback chain screened entitlement, credential-pool exhaustion, local
usability and backend identity — but never the candidate's CONTEXT CAPACITY against the
session it is about to replay. These tests pin that screen.
"""

from unittest.mock import MagicMock, patch

import pytest


# The real llama.cpp window on ironman (APP_CTX_SIZE=32768) and the real conversation size
# from the incident log (session 20261007_031610_9596ddb5: 244 messages, ~182,472 tokens).
LOCAL_CTX = 32768
INCIDENT_TOKENS = 182_472
LOCAL_MODEL = "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf"


def _agent(prompt_tokens=INCIDENT_TOKENS):
    """A minimal agent stand-in carrying just what the screen reads."""
    agent = MagicMock(name="agent")
    agent.messages = [{"role": "user", "content": "x" * 4000}]
    agent._usage_anchor = None
    agent.context_compressor = MagicMock(name="compressor")
    agent.context_compressor.last_real_prompt_tokens = prompt_tokens
    agent.context_compressor.last_prompt_tokens = prompt_tokens
    agent.provider = "minimax"
    agent.model = "MiniMax-M3"
    agent.base_url = "https://api.minimax.io/v1"
    return agent


# ── 1. The session size is read from the provider's own usage ──────────────────────────

class TestSessionContextTokens:
    """The screen needs the size of the prompt it is about to replay."""

    def test_prefers_the_anchored_provider_usage(self):
        from agent.chat_completion_helpers import _session_context_tokens

        agent = _agent(prompt_tokens=1000)
        with patch("agent.usage_anchor.anchored_context_tokens", return_value=182_472):
            assert _session_context_tokens(agent) == 182_472

    def test_falls_back_to_the_compressor_count(self):
        from agent.chat_completion_helpers import _session_context_tokens

        agent = _agent(prompt_tokens=55_000)
        with patch("agent.usage_anchor.anchored_context_tokens", return_value=None):
            assert _session_context_tokens(agent) == 55_000

    def test_unknown_when_nothing_is_known(self):
        from agent.chat_completion_helpers import _session_context_tokens

        agent = _agent(prompt_tokens=0)
        with patch("agent.usage_anchor.anchored_context_tokens", return_value=None):
            assert _session_context_tokens(agent) is None


# ── 2. The candidate is screened against the session size ─────────────────────────────

class TestFallbackCandidateContextScreen:
    """A candidate whose window cannot hold the session must be skipped."""

    def _entry(self, base_url=""):
        return {"provider": "local-llama", "model": LOCAL_MODEL, "base_url": base_url}

    def test_skips_a_candidate_too_small_for_the_session(self):
        from agent.chat_completion_helpers import _fallback_candidate_context_too_small

        agent = _agent(prompt_tokens=INCIDENT_TOKENS)
        with patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length", return_value=LOCAL_CTX):
            reason = _fallback_candidate_context_too_small(
                agent, self._entry(), "local-llama", LOCAL_MODEL)

        assert reason is not None, (
            "a 32K local model was accepted for a 182K-token session — the request "
            "would be forwarded and rejected server-side"
        )
        assert str(LOCAL_CTX) in reason
        assert str(INCIDENT_TOKENS) in reason

    def test_allows_a_candidate_that_fits(self):
        from agent.chat_completion_helpers import _fallback_candidate_context_too_small

        agent = _agent(prompt_tokens=20_000)
        with patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length", return_value=1_000_000):
            reason = _fallback_candidate_context_too_small(
                agent, self._entry(), "local-llama", LOCAL_MODEL)

        assert reason is None

    def test_unknown_session_size_never_blocks(self):
        """We cannot prove a candidate is too small — do not gut the fallback surface."""
        from agent.chat_completion_helpers import _fallback_candidate_context_too_small

        agent = _agent(prompt_tokens=0)
        with patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length", return_value=LOCAL_CTX):
            reason = _fallback_candidate_context_too_small(
                agent, self._entry(), "local-llama", LOCAL_MODEL)

        assert reason is None

    def test_unknown_candidate_window_never_blocks(self):
        """An unrecognised model's window is unknown — do not refuse it."""
        from agent.chat_completion_helpers import _fallback_candidate_context_too_small

        agent = _agent(prompt_tokens=INCIDENT_TOKENS)
        with patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length", return_value=None):
            reason = _fallback_candidate_context_too_small(
                agent, self._entry(), "local-llama", LOCAL_MODEL)

        assert reason is None

    def test_uses_the_candidates_configured_endpoint(self):
        """The window must come from the REAL endpoint, not the catalog's name guess.

        ``/app/models/qwen2.5-7b-instruct-q4_k_m.gguf`` matches the hardcoded ``qwen``
        catalog entry (131072) even though the llama-server runs at 32768. Without the
        endpoint the screen believes the local model has 128K and waves a 182K request
        through.
        """
        from agent.chat_completion_helpers import _fallback_candidate_context_too_small

        agent = _agent(prompt_tokens=INCIDENT_TOKENS)
        with patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length",
                   return_value=LOCAL_CTX) as ctx:
            _fallback_candidate_context_too_small(agent, self._entry(), "local-llama", LOCAL_MODEL)

        assert ctx.call_args.kwargs.get("base_url") == "https://llama.loc.wallacearizona.us/v1"


# ── 3. The screen is wired into the candidate-skip gate ────────────────────────────────

class TestSkipGateWiring:
    """``_should_skip_fallback_candidate`` is the gate every chain entry passes through."""

    def _skip(self, agent, fb, ctx):
        from agent.chat_completion_helpers import _should_skip_fallback_candidate

        with patch("agent.fallback_cooldown._is_entitlement_rejected", return_value=False), \
             patch("agent.chat_completion_helpers._candidate_pool_exhausted", return_value=False), \
             patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                   return_value=None), \
             patch("agent.auxiliary_health._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.model_metadata.get_model_context_length", return_value=ctx):
            return _should_skip_fallback_candidate(
                agent, fb, ("local-llama", LOCAL_MODEL, ""), "local-llama", LOCAL_MODEL, set())

    def test_the_gate_skips_a_too_small_candidate(self):
        agent = _agent(prompt_tokens=INCIDENT_TOKENS)
        fb = {"provider": "local-llama", "model": LOCAL_MODEL, "base_url": ""}
        assert self._skip(agent, fb, LOCAL_CTX) is True

    def test_the_gate_admits_a_candidate_that_fits(self):
        agent = _agent(prompt_tokens=20_000)
        fb = {"provider": "local-llama", "model": LOCAL_MODEL, "base_url": ""}
        assert self._skip(agent, fb, 1_000_000) is False

    def test_the_skip_is_logged_with_the_reason(self, caplog):
        agent = _agent(prompt_tokens=INCIDENT_TOKENS)
        fb = {"provider": "local-llama", "model": LOCAL_MODEL, "base_url": ""}
        with caplog.at_level("WARNING", logger="agent.chat_completion_helpers"):
            self._skip(agent, fb, LOCAL_CTX)

        assert "Fallback skip" in caplog.text
        assert str(LOCAL_CTX) in caplog.text, (
            f"the log does not name the candidate's window; log text was: {caplog.text!r}"
        )
