"""Finding D (#4): Fallback/restore-primary post-transition failure leaves host+engine coherent.

When ``restore_primary_runtime`` runs, the sequence is:

  1. Host state → primary (model/provider/client/api_key/...)
  2. ``transition_model_context`` → engine → primary (COMMITTED)
  3. Credential pool rebind
  4. ``_swap_credential(entry)`` → can RAISE
  5. reasoning_config, fallback reset, prompt identity rewrite

If step 4 raises, the ``except`` block restores the **host** to fallback
(snapshot) but the **engine** (committed in step 2) is left on primary →
the agent is left with a fallback host + primary engine — incoherent.

The fix: ``_swap_credential`` failure is NON-FATAL (the documented
"keep the snapshot key" behavior).  The exception is caught locally,
logged, and the restore completes with the host on primary (coherent with
the engine).  The pool entry swap is simply skipped.

RED   — before the fix, ``_swap_credential`` raising propagates to the
         outer ``except``, which restores the host to fallback. The engine
         stays on primary → assertions FAIL (host≠engine).
GREEN — after the fix, ``_swap_credential`` failure is caught locally;
         the restore completes with host on primary, coherent with the
         engine → assertions PASS.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_agent_d() -> "MagicMock":
    """Agent on fallback with a primary snapshot to restore."""
    agent = MagicMock()

    # CURRENT state: on FALLBACK (the state we're restoring FROM)
    agent.model = "fallback-model"
    agent.provider = "custom"
    agent.base_url = "http://fallback:8080/v1"
    agent.api_key = "key-fallback"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="FallbackClient")
    agent._client_kwargs = {"api_key": "key-fallback"}
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._pre_cap_context_length = 65536
    agent._reasoning_echo_flag = False
    agent._transport_cache = {}

    # Primary snapshot (what we're restoring TO)
    primary = {
        "model": "primary-model",
        "provider": "custom",
        "base_url": "http://primary:8080/v1",
        "api_key": "key-primary",
        "api_mode": "chat_completions",
        "client_kwargs": {"api_key": "key-primary"},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "anthropic_api_key": "",
        "anthropic_base_url": None,
        "is_anthropic_oauth": False,
        "reasoning_echo_flag": False,
        "compressor_context_length": 131072,
        "compressor_model": "primary-model",
    }
    agent._primary_runtime = primary

    # Fallback state
    agent._fallback_activated = True
    agent._fallback_index = 0
    agent._fallback_chain = [{"model": "fallback-model", "provider": "custom"}]
    agent._rate_limited_until = 0
    agent._restore_wait_logged = False
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._cache_disabled = False
    agent._rate_limit_backoff_count = 0

    # Engine: on FALLBACK initially
    engine = MagicMock(name="ContextEngine")
    engine.model = "fallback-model"
    engine.provider = "custom"
    engine.base_url = "http://fallback:8080/v1"
    engine.api_key = "key-fallback"
    engine.api_mode = "chat_completions"
    engine.context_length = 65536
    engine.pre_cap_context_length = 65536
    engine._handles_context_ceiling = True
    engine.update_model = MagicMock()
    agent.context_compressor = engine

    # _create_openai_client returns a new client
    new_client = MagicMock(name="PrimaryClient")
    agent._create_openai_client = MagicMock(return_value=new_client)

    return agent


class TestRestorePrimaryPostTransitionFailure:
    """Finding #4: _swap_credential raise after engine transition must not leave host+engine incoherent."""

    def test_swap_credential_failure_keeps_host_and_engine_coherent(self):
        """When _swap_credential raises (after transition_model_context succeeds),
        the agent must be left in a coherent state — NOT host=fallback +
        engine=primary."""
        agent = _make_agent_d()

        # A credential pool entry that will be selected
        pool = MagicMock()
        pool.has_available = MagicMock(return_value=True)
        entry = MagicMock()
        entry.provider = "custom"
        entry.runtime_api_key = "key-from-pool"
        entry.access_token = "key-from-pool"
        pool.select = MagicMock(return_value=entry)
        agent._credential_pool = pool

        # _swap_credential RAISES (the post-transition failure)
        def swap_boom(_entry):
            raise RuntimeError("simulated credential swap failure")
        agent._swap_credential = swap_boom

        # transition_model_context succeeds (engine transitions to primary)
        with patch("agent.agent_runtime_helpers.transition_model_context") as mock_tm:
            mock_tm.return_value = 131072
            with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
                with patch("agent.chat_completion_helpers._reset_stale_streak", return_value=None):
                    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity", return_value=None):
                        from agent.agent_runtime_helpers import restore_primary_runtime
                        result = restore_primary_runtime(agent)

        # The engine transition WAS committed (transition_model_context was
        # called for the primary) — this is the "engine on primary" side of
        # the coherence invariant, evidenced by the actual call.
        assert mock_tm.called, "transition_model_context must run (engine → primary)"
        _tm_kwargs = mock_tm.call_args
        _tm_model = _tm_kwargs.kwargs.get("model") or (_tm_kwargs.args[1] if len(_tm_kwargs.args) > 1 else None)
        assert _tm_model == "primary-model", (
            f"engine must transition to primary-model, got {_tm_model!r}"
        )

        # The restore must have COMPLETED (result=True) because
        # _swap_credential failure is NON-FATAL (keep snapshot key).
        assert result is True, (
            f"restore_primary_runtime should return True (swap failure is non-fatal), "
            f"got {result!r} — the outer except restored the host to fallback, "
            f"leaving host=fallback + engine=primary (INCOHERENT)"
        )

        # HOST must be on PRIMARY (coherent with the engine that was committed)
        assert agent.model == "primary-model", (
            f"host model must be primary (coherent with engine), got {agent.model!r}"
        )
        assert agent.base_url == "http://primary:8080/v1", (
            f"host base_url must be primary, got {agent.base_url!r}"
        )
        assert agent.api_key == "key-primary", (
            f"host api_key must be primary, got {agent.api_key!r}"
        )

        # The key invariant: host stays on primary, matching the engine's
        # committed primary-model transition (no rollback to fallback).
        assert agent.model == "primary-model" and _tm_model == "primary-model", (
            f"host ({agent.model!r}) and engine ({_tm_model!r}) must agree — "
            f"incoherent state (host=fallback, engine=primary) detected"
        )
