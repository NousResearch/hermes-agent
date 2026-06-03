"""Regression tests for the Azure Foundry transient-401 retry path.

Some Azure-API-Management-fronted Foundry endpoints intermittently reject a
fully valid static API key with a spurious ``401 Invalid token`` (empirically
~20% of identical requests on some resources). Before the fix, a single
transient 401 aborted the whole turn AND marked the sole credential
``exhausted`` in the credential pool.

The fix (``agent/conversation_loop.py``) retries the same key in-place a
bounded number of times BEFORE the credential pool can mutate state, but only
for static-string-key chat-completions on ``azure-foundry``. Entra ID auth (a
callable bearer-token provider) is excluded because there a 401 is a real
RBAC/token failure that retrying cannot fix.

These tests lock in:
  * a transient 401 followed by success recovers WITHOUT touching the pool
  * Entra ID (callable key) does NOT use the transient retry path
  * a persistent 401 still gives up after the bounded retries
  * non-azure providers are unaffected by the new block
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.conversation_loop import AZURE_FOUNDRY_TRANSIENT_401_MAX_RETRIES


def _make_agent(provider, api_mode="chat_completions", api_key="valid-azure-key-123"):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key=api_key,
            base_url="https://claude-44.services.ai.azure.com/openai/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a.provider = provider
    a.api_mode = api_mode
    a.api_key = api_key
    a._credential_pool = None
    a._persist_session = lambda *args, **kwargs: None
    a._save_trajectory = lambda *args, **kwargs: None
    a.suppress_status_output = True
    return a


class _Auth401Error(Exception):
    status_code = 401

    def __str__(self):
        return "Error code: 401 - {'error': {'code': 'unauthorized', 'message': 'Invalid token'}}"


def _mock_response(content="Recovered"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        role="assistant",
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="Kimi-K2.6-1", usage=None)


def _sequenced_api_call(responses):
    def _call(api_kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return _call


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    # jittered_backoff -> 0 makes the interrupt-aware sleep loop exit
    # immediately; patch time.sleep too as belt-and-suspenders.
    with (
        patch("agent.conversation_loop.jittered_backoff", return_value=0.0),
        patch("agent.conversation_loop.time.sleep", return_value=None),
    ):
        yield


class TestAzureFoundryTransient401:
    def test_transient_401_then_success_recovers_without_pool(self):
        """A single transient 401 retries in-place and recovers, never
        invoking credential-pool recovery (so the key isn't marked exhausted)."""
        agent = _make_agent("azure-foundry")
        agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
        agent._interruptible_api_call = _sequenced_api_call(
            [_Auth401Error(), _mock_response("Recovered")]
        )

        result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        # The transient retry pre-empts pool recovery entirely.
        assert agent._recover_with_credential_pool.call_count == 0

    def test_multiple_transient_401s_within_budget_recover(self):
        """Several consecutive transient 401s (within the retry budget) still
        recover without touching the pool."""
        agent = _make_agent("azure-foundry")
        agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
        errs = [_Auth401Error() for _ in range(AZURE_FOUNDRY_TRANSIENT_401_MAX_RETRIES)]
        agent._interruptible_api_call = _sequenced_api_call(
            errs + [_mock_response("Recovered")]
        )

        result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        assert agent._recover_with_credential_pool.call_count == 0

    def test_persistent_401_gives_up_after_max_retries(self):
        """A genuinely-rejected key still aborts after the bounded retries,
        falling through to the normal pool-recovery / abort path."""
        agent = _make_agent("azure-foundry")
        agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
        # One more 401 than the retry budget guarantees the fall-through.
        errs = [_Auth401Error() for _ in range(AZURE_FOUNDRY_TRANSIENT_401_MAX_RETRIES + 2)]
        agent._interruptible_api_call = _sequenced_api_call(errs + [_mock_response("late")])

        result = agent.run_conversation("hello")

        # Falls through to pool recovery (which can't help a single bad key).
        assert agent._recover_with_credential_pool.called
        assert result.get("completed") is not True

    def test_entra_callable_key_skips_transient_retry(self):
        """Entra ID auth (callable token provider) must NOT use the transient
        retry path — a 401 there is a real auth failure."""
        token_provider = lambda: "minted-bearer-jwt"  # noqa: E731
        agent = _make_agent("azure-foundry", api_key=token_provider)
        agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
        agent._interruptible_api_call = _sequenced_api_call(
            [_Auth401Error(), _mock_response("should-not-reach")]
        )

        result = agent.run_conversation("hello")

        # No transient retry: the first 401 goes straight to pool recovery.
        assert agent._recover_with_credential_pool.called
        assert result.get("completed") is not True

    def test_non_azure_provider_unaffected(self):
        """A non-azure provider 401 must not enter the azure transient path —
        it goes straight to pool recovery as before."""
        agent = _make_agent("openrouter")
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
        agent._interruptible_api_call = _sequenced_api_call(
            [_Auth401Error(), _mock_response("nope")]
        )

        result = agent.run_conversation("hello")

        assert agent._recover_with_credential_pool.called
        assert result.get("completed") is not True
