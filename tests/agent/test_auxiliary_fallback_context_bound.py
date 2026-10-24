"""Gateway must not forward an oversized prompt to a local provider that cannot hold it.

Incident: t_e94832a1 (2026-10-23). MiniMax hit its plan usage cap, the ``local-llama``
fallback took the load, and the auxiliary/compression path — whose prompt IS the whole
conversation — replayed 182K-token sessions onto a 32K-context local model. The server
rejected every one of them::

    E srv send_error: task id = 864, error: request (44622 tokens) exceeds the available
    context size (32768 tokens)

The leak was the ``_try_main_agent_model_fallback`` rung: an auxiliary task with an EXPLICIT
provider (so ``is_auto`` is False) whose provider ran out of credit skipped both configured
chains and landed on the main agent's model — with NO context screen at all. The main agent
had itself fallen back to the 32K local model, so compression forwarded the whole session to
it.

These tests pin the two properties that close it:

  1. every runtime fallback rung screens the candidate by its OWN context window, and
  2. the screen compares that window against the ACTUAL request size, not just a fixed floor.
"""

from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────────────

def _entry(provider="local-llama", model="/app/models/qwen2.5-7b-instruct-q4_k_m.gguf",
           base_url="https://llama.loc.wallacearizona.us/v1", api_key="k"):
    return {"provider": provider, "model": model, "base_url": base_url, "api_key": api_key}


# The real llama.cpp window on ironman (APP_CTX_SIZE=32768), and the real conversation size
# from the incident log (session 20261007_031610_9596ddb5: 244 messages, ~182,472 tokens).
LOCAL_CTX = 32768
INCIDENT_TOKENS = 182_472


# ── 1. The request-size screen exists and measures the outgoing prompt ────────────────

class TestRequestTokenRequirement:
    """``_request_token_requirement`` sizes the outgoing request so a candidate can be
    screened against it. A candidate that cannot hold the request must never be sent."""

    def test_measures_the_outgoing_prompt(self):
        from agent.auxiliary_client import _request_token_requirement

        kwargs = {"messages": [{"role": "user", "content": "x" * 40_000}]}
        tokens = _request_token_requirement(kwargs)
        assert tokens is not None
        assert tokens >= 10_000, f"expected ~10K tokens for 40K chars, got {tokens}"

    def test_returns_none_for_empty_or_malformed_kwargs(self):
        from agent.auxiliary_client import _request_token_requirement

        assert _request_token_requirement(None) is None
        assert _request_token_requirement({}) is None
        assert _request_token_requirement({"messages": []}) is None
        assert _request_token_requirement({"messages": "not-a-list"}) is None

    def test_never_raises_on_a_broken_message_list(self):
        """A sizing failure must degrade to "unknown" (pass-through), never crash the call."""
        from agent.auxiliary_client import _request_token_requirement

        with patch("agent.model_metadata.estimate_messages_tokens_rough",
                   side_effect=RuntimeError("boom")):
            assert _request_token_requirement({"messages": [{"role": "user", "content": "x"}]}) is None


# ── 2. The main-agent fallback rung is screened (THE LEAK) ─────────────────────────────

class TestMainAgentFallbackContextBound:
    """``_try_main_agent_model_fallback`` is the rung the incident actually fell through.
    It must refuse a main model whose window cannot hold the request."""

    def _patched(self, ctx, required_tokens, *, main_provider="local-llama"):
        main_model = "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf"
        client = MagicMock(name="main_client")
        return (
            patch("agent.auxiliary_client._read_main_provider", return_value=main_provider),
            patch("agent.auxiliary_client._read_main_model", return_value=main_model),
            patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
            patch("agent.auxiliary_client._custom_health_base_url",
                  return_value="https://llama.loc.wallacearizona.us/v1"),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(client, main_model)),
            patch("agent.auxiliary_client.get_model_context_length", return_value=ctx),
        )

    def test_refuses_a_main_model_too_small_for_the_request(self):
        """RED→GREEN: the 32K local model must NOT be handed a 182K-token compression prompt."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        patches = self._patched(LOCAL_CTX, INCIDENT_TOKENS)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            client, model, label = _try_main_agent_model_fallback(
                "minimax", task="compression", reason="payment error",
                required_tokens=INCIDENT_TOKENS,
            )

        assert client is None, (
            "main-agent fallback returned a client whose 32K window cannot hold a 182K-token "
            "request — the request would be forwarded and rejected server-side"
        )
        assert model is None
        assert label == ""

    def test_refuses_a_main_model_below_the_compression_floor(self):
        """The pre-existing floor still applies: a 32K model is not a sane compressor."""
        from agent.auxiliary_client import _try_main_agent_model_fallback
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

        patches = self._patched(LOCAL_CTX, None)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            client, _model, _label = _try_main_agent_model_fallback(
                "minimax", task="compression", reason="payment error",
            )

        assert client is None, (
            f"a {LOCAL_CTX}-token window is below the compression floor "
            f"({MINIMUM_CONTEXT_LENGTH}) and must not be selected"
        )

    def test_allows_a_main_model_that_fits(self):
        """A big enough main model is still selected — the screen must not disable the rung."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        patches = self._patched(1_000_000, INCIDENT_TOKENS)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            client, model, label = _try_main_agent_model_fallback(
                "minimax", task="compression", reason="payment error",
                required_tokens=INCIDENT_TOKENS,
            )

        assert client is not None
        assert label == "main-agent(local-llama)"

    def test_unknown_window_passes_through(self):
        """We cannot prove an unrecognised model is too small — do not gut the fallback surface."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        patches = self._patched(None, INCIDENT_TOKENS)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            client, _model, _label = _try_main_agent_model_fallback(
                "minimax", task="compression", reason="payment error",
                required_tokens=INCIDENT_TOKENS,
            )

        assert client is not None

    def test_non_compression_task_is_unaffected_by_the_floor(self):
        """A task with no floor still gets the request-size screen, but no floor of its own."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        # 50K window: below MINIMUM_CONTEXT_LENGTH, but a 2K request fits fine.
        patches = self._patched(50_000, 2_000)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            client, _model, _label = _try_main_agent_model_fallback(
                "minimax", task="title_generation", reason="payment error",
                required_tokens=2_000,
            )

        assert client is not None


# ── 3. The configured chains screen against the request size too ───────────────────────

class TestConfiguredChainRequestBound:
    """The two configured chains already had a floor screen; it now also compares the
    candidate's window against the ACTUAL request size."""

    def test_configured_chain_skips_a_candidate_too_small_for_the_request(self):
        """A 32K candidate must be skipped when the request needs 182K, and the next
        (large enough) candidate wins."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        small_client = MagicMock(name="small_client")
        large_client = MagicMock(name="large_client")
        entries = [_entry(provider="local-llama"), _entry(provider="big", model="huge-1m")]

        def fake_resolve(entry):
            if entry is entries[0]:
                return small_client, "small"
            return large_client, "huge-1m"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"small": LOCAL_CTX, "huge-1m": 1_048_576}.get(model, 256_000)

        with patch("agent.auxiliary_client._get_auxiliary_task_config",
                   return_value={"fallback_chain": entries}), \
             patch("agent.auxiliary_client._resolve_fallback_entry", side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length", side_effect=fake_ctx):
            client, model, _label = _try_configured_fallback_chain(
                task="compression", failed_provider="minimax", required_tokens=INCIDENT_TOKENS)

        assert client is large_client, (
            "the chain returned a 32K candidate for a 182K-token request"
        )
        assert model == "huge-1m"

    def test_main_chain_skips_a_candidate_too_small_for_the_request(self):
        """Same screen on the main (fallback_providers) chain."""
        from agent.auxiliary_client import _try_main_fallback_chain

        small_client = MagicMock(name="small_client")
        entries = [_entry(provider="local-llama")]

        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}), \
             patch("hermes_cli.fallback_config.get_fallback_chain", return_value=entries), \
             patch("hermes_cli.config.load_config_readonly", return_value={}), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client._resolve_fallback_entry",
                   return_value=(small_client, "small")), \
             patch("agent.auxiliary_client.get_model_context_length", return_value=LOCAL_CTX):
            client, model, label = _try_main_fallback_chain(
                task="compression", failed_provider="minimax", required_tokens=INCIDENT_TOKENS)

        assert client is None, (
            "the main chain returned a 32K candidate for a 182K-token request"
        )
        assert model is None
        assert label == ""


# ── 4. The screen must resolve the candidate's REAL endpoint ───────────────────────────

class TestScreenResolvesTheRealEndpoint:
    """A named custom provider carries its endpoint in the CONFIG, not in the fallback entry.

    Without resolving it, the window lookup falls through to the hardcoded catalog, which
    guesses from the model NAME: ``/app/models/qwen2.5-7b-instruct-q4_k_m.gguf`` matches the
    ``qwen`` catalog entry (131072) even though the llama-server runs at 32768
    (APP_CTX_SIZE). The screen then believes the local model has 128K and waves a
    182K-token request through to a server that rejects it — which is exactly why
    "context too small" never appeared in the 2026-10-23 logs.
    """

    def test_candidate_window_uses_the_providers_configured_endpoint(self):
        from agent.auxiliary_client import _candidate_context_window

        with patch("agent.auxiliary_client._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.auxiliary_client.get_model_context_length",
                   return_value=LOCAL_CTX) as ctx:
            window = _candidate_context_window(
                "local-llama", "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf",
                base_url="", api_key="")

        assert window == LOCAL_CTX
        # The lookup MUST have been given the real endpoint, not an empty one.
        assert ctx.call_args.kwargs.get("base_url") == "https://llama.loc.wallacearizona.us/v1"

    def test_explicit_entry_base_url_still_wins(self):
        """An entry that names its own endpoint is not overridden by the provider's config."""
        from agent.auxiliary_client import _candidate_context_window

        with patch("agent.auxiliary_client._custom_health_base_url",
                   return_value="https://config.example.com/v1"), \
             patch("agent.auxiliary_client.get_model_context_length",
                   return_value=LOCAL_CTX) as ctx:
            _candidate_context_window(
                "local-llama", "some-model", base_url="https://entry.example.com/v1", api_key="")

        assert ctx.call_args.kwargs.get("base_url") == "https://entry.example.com/v1"

    def test_screen_rejects_the_local_model_on_its_real_window(self):
        """End-to-end shape of the incident: a 32K local model must be rejected for a
        182K-token compression request, even though the catalog says 131072."""
        from agent.auxiliary_client import _context_too_small

        with patch("agent.auxiliary_client._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.auxiliary_client.get_model_context_length", return_value=LOCAL_CTX):
            note = _context_too_small(
                {"provider": "local-llama"}, "local-llama",
                "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf", 64000,
                task="compression", label="fallback_providers[0](local-llama)",
                required_tokens=INCIDENT_TOKENS,
            )

        assert note is not None, "the 32K local model was accepted for a 182K-token request"
        assert "context too small" in note


# ── 5. The screen reports the reason in the log ────────────────────────────────────────

class TestScreenLogging:
    """An operator reading the log must be able to see WHY a candidate was skipped."""

    def test_skip_is_logged_with_the_request_size(self, caplog):
        from agent.auxiliary_client import _try_main_agent_model_fallback

        main_model = "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf"
        with patch("agent.auxiliary_client._read_main_provider", return_value="local-llama"), \
             patch("agent.auxiliary_client._read_main_model", return_value=main_model), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client._custom_health_base_url",
                   return_value="https://llama.loc.wallacearizona.us/v1"), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(MagicMock(), main_model)), \
             patch("agent.auxiliary_client.get_model_context_length", return_value=LOCAL_CTX), \
             caplog.at_level("INFO", logger="agent.auxiliary_client"):
            _try_main_agent_model_fallback(
                "minimax", task="compression", reason="payment error",
                required_tokens=INCIDENT_TOKENS)

        text = caplog.text
        assert "context too small" in text or "cannot hold this request" in text, (
            f"the skip was not logged; log text was: {text!r}"
        )
        assert str(INCIDENT_TOKENS) in text, (
            f"the log does not name the request size; log text was: {text!r}"
        )
