"""Regression tests that ``AIAgent.switch_model`` swaps ``_credential_pool``.

The /model pipeline now resolves a target-provider credential pool and
surfaces it on ``ModelSwitchResult.credential_pool``.  Both the gateway's
``_on_model_selected`` and the CLI's ``_apply_model_switch_result`` /
``_handle_model_switch`` pass that pool through to ``agent.switch_model()``.

Without these tests, a regression that drops the new ``credential_pool``
kwarg from ``AIAgent.switch_model`` would silently strand the new
provider's pool and break 429/402 rotation after a /model switch — the
exact bug fixed in #16678.  Pin the contract.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent_minimal(initial_pool=None) -> AIAgent:
    """Build a minimal AIAgent that can run ``switch_model`` end-to-end.

    Skips ``__init__`` (which builds clients, fallback chain, prompt
    cache, compressor, etc.) and seeds only the fields ``switch_model``
    reads or writes.  Mirrors the pattern in ``test_switch_model_context.py``.
    """
    agent = AIAgent.__new__(AIAgent)

    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    agent._credential_pool = initial_pool
    agent._config_context_length = None
    agent._primary_runtime = {}
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._cached_system_prompt = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._client_kwargs = {}
    # No context_compressor attribute — switch_model guards on hasattr.

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
def test_switch_model_swaps_credential_pool(mock_ctx_len):
    """Passing ``credential_pool`` updates ``self._credential_pool``.

    Reproduces the in-place swap that the /model command performs after
    resolving a new provider's pool — the CLI and gateway both call
    ``agent.switch_model(..., credential_pool=result.credential_pool)``
    and rely on the agent picking up the new pool for next-turn rotation.
    """
    old_pool = MagicMock(name="openrouter_pool")
    new_pool = MagicMock(name="codex_pool")
    agent = _make_agent_minimal(initial_pool=old_pool)

    # Mock client builder so we don't hit the network on the real
    # _create_openai_client path.
    with patch.object(agent, "_create_openai_client", return_value=MagicMock()):
        agent.switch_model(
            "gpt-5.4",
            "openai-codex",
            api_key="sk-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="chat_completions",
            credential_pool=new_pool,
        )

    # The new pool must replace the old one — without this, 429s on the
    # new provider rotate against the wrong pool (or skip rotation).
    assert agent._credential_pool is new_pool


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
def test_switch_model_without_credential_pool_preserves_existing(mock_ctx_len):
    """Omitting ``credential_pool`` leaves the existing one in place.

    Many switch paths legitimately have no pool (single-key custom
    endpoints, providers without rotation).  The agent must NOT clobber
    its existing pool to ``None`` in that case — only replace when a new
    pool is explicitly supplied.  This matches the ``if credential_pool
    is not None`` guard in switch_model and prevents stale callers (or
    older code paths that don't yet pass the kwarg) from silently
    disabling rotation.
    """
    existing_pool = MagicMock(name="existing_pool")
    agent = _make_agent_minimal(initial_pool=existing_pool)

    with patch.object(agent, "_create_openai_client", return_value=MagicMock()):
        agent.switch_model(
            "gpt-5.4",
            "openai-codex",
            api_key="sk-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="chat_completions",
        )

    assert agent._credential_pool is existing_pool


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
def test_switch_model_explicit_none_preserves_existing(mock_ctx_len):
    """``credential_pool=None`` is treated as "no change", not "clear".

    The CLI passes ``result.credential_pool`` directly; that field is
    ``None`` for providers without a saved pool.  We deliberately do NOT
    clear the agent's existing pool in that case — clearing on a switch
    to a single-key provider would leave the agent rotation-less for the
    rest of the session even if the user later switches back to a pooled
    provider whose pool is then re-set on the next switch.  Matches the
    ``if credential_pool is not None`` guard.
    """
    existing_pool = MagicMock(name="existing_pool")
    agent = _make_agent_minimal(initial_pool=existing_pool)

    with patch.object(agent, "_create_openai_client", return_value=MagicMock()):
        agent.switch_model(
            "gpt-5.4",
            "openai-codex",
            api_key="sk-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="chat_completions",
            credential_pool=None,
        )

    assert agent._credential_pool is existing_pool
