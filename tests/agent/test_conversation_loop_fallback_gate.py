"""Loop-level regression: a ``should_fallback=False`` classification must not
activate the provider fallback chain.

Context (PR #62765 review by teknium1/hermes-sweeper): the classifier now marks
LM Studio "model unloaded" errors as ``model_not_found, retryable=False,
should_fallback=False`` (see ``TestLMStudioModelUnloaded`` in
``test_error_classifier.py``). But the recovery path in
``agent/conversation_loop.py`` (the ``if is_client_error:`` block, ~L3722-3742)
calls ``agent._try_activate_fallback()`` **without** consulting
``classified.should_fallback`` — so today the fallback chain is walked anyway.

The existing tests only assert classifier *output*. This test asserts *loop
behaviour*: with a no-fallback classification, ``_try_activate_fallback`` is
never entered.

The guard that makes this true lives in Omar's PR (NousResearch/hermes-agent
#34076: ``if classified.should_fallback and agent._try_activate_fallback():`` in
the same region), which is approved but not yet on ``main``. So this test is
marked ``xfail(strict=True)``:

  * today (no gate) → ``_try_activate_fallback`` IS called → ``assert_not_called``
    raises → xfail catches it (reported ``xfailed``).
  * once #34076 lands → the gate short-circuits → the call is suppressed → the
    assertion passes → ``XPASS`` → ``strict=True`` turns that into a visible CI
    failure. That is the intended "living merge detector".

FLIP PLAN — as soon as #34076 is merged: remove the ``xfail`` marker, keep the
hard assertion, and rebase this branch onto the gated ``main``.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.conversation_loop import run_conversation


class _LMStudioAPIError(Exception):
    """Minimal stand-in for an OpenAI SDK APIStatusError, shaped exactly like
    the ``MockAPIError`` used in ``test_error_classifier.py`` so the *real*
    ``classify_api_error`` maps it to ``model_not_found`` with
    ``should_fallback=False`` — we deliberately do NOT patch the classifier, to
    close the loop back to the PR's actual behaviour.
    """

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _make_agent(fallback_model=None):
    """Minimal AIAgent with an optional fallback chain (mirrors the helper in
    tests/run_agent/test_provider_fallback.py)."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://lmstudio.local/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


@pytest.mark.xfail(
    strict=True,
    reason=(
        "conversation_loop.py ~L3736 calls _try_activate_fallback() without a "
        "classified.should_fallback guard; the guard lands in "
        "NousResearch/hermes-agent#34076. Until it does, a no-fallback "
        "classification still walks the fallback chain. When #34076 merges this "
        "XPASSes (strict) -> remove xfail, keep the assert, rebase."
    ),
)
def test_no_fallback_activation_for_model_unloaded():
    # A fallback chain must exist, otherwise there is nothing to (wrongly)
    # activate and the assertion would be vacuous.
    agent = _make_agent(
        fallback_model=[{"provider": "openai", "model": "gpt-4o"}]
    )
    assert agent._has_pending_fallback() is True

    # The real LM Studio "no models loaded" failure: a 404 whose body message
    # the real classifier routes to model_not_found / retryable=False /
    # should_fallback=False.
    err = _LMStudioAPIError("No models loaded", status_code=404)

    # Stub only the leaf API call (both streaming and non-streaming paths) so
    # the error flows through the real classifier into the recovery branch.
    # Force _try_activate_fallback to a no-op returning False so, if the loop
    # does call it, it aborts cleanly instead of resolving a real provider.
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=err),
        patch.object(agent, "_interruptible_streaming_api_call", side_effect=err),
        patch.object(
            agent, "_try_activate_fallback", return_value=False
        ) as spy_fallback,
    ):
        run_conversation(
            agent,
            "trigger a model-unloaded error",
            conversation_history=[],
            task_id="t",
        )

    # The heart of the regression: a should_fallback=False classification must
    # not reach fallback activation at all.
    spy_fallback.assert_not_called()
