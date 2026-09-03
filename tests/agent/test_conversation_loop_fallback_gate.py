"""Loop-level regression: a ``should_fallback=False`` classification must not
activate the provider fallback chain — for BOTH the model-unloaded case (#62765)
and the 409 model-pool-lock case (NousResearch/hermes-agent #34076).

Context (PR #62765 review by teknium1/hermes-sweeper): the recovery path in
``agent/conversation_loop.py`` (the ``if is_client_error:`` block, ~L3722-3736)
calls ``agent._try_activate_fallback()`` **without** consulting
``classified.should_fallback`` — so today the chain is walked even for a
no-fallback classification. The guard that fixes this
(``if classified.should_fallback and agent._try_activate_fallback():``) lands in
Omar's #34076, approved but not yet on ``main``. The guard is error-type-agnostic.

This test asserts *loop behaviour* (not classifier output): a
``should_fallback=False`` classification must not reach ``_try_activate_fallback``
AND must leave the fallback chain index unchanged.

Two parametrizations exercise the SAME guard:

  * ``model_unloaded`` — a real ``404 "No models loaded"`` through the **real**
    classifier (#62765 maps it to ``model_not_found, should_fallback=False``). No
    classifier patch — closes the loop back to the PR's actual behaviour.
  * ``conflict_409`` — the 409 model-pool-lock case. The 409→``should_fallback=
    False`` mapping lives in #34076, NOT on this branch, so here we **mock**
    ``classify_api_error`` to return that classification. Transparent mock
    boundary: once #34076 lands, lift this to a real 409 trigger (drop the patch).

Both are ``xfail(strict=True)`` — the shared merge-detector for #34076: today the
ungated loop calls ``_try_activate_fallback`` → ``assert_not_called`` raises →
``xfailed``; once #34076 gates it → call suppressed → asserts pass → ``XPASS`` →
strict turns that into a visible CI failure.

FLIP PLAN when #34076 merges: remove ``xfail``, keep the asserts, lift the 409
branch to a real trigger, rebase onto the gated ``main``.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.conversation_loop import run_conversation
from agent.error_classifier import ClassifiedError, FailoverReason


class _LMStudioAPIError(Exception):
    """Minimal stand-in for an OpenAI SDK APIStatusError, shaped like the
    ``MockAPIError`` in ``test_error_classifier.py`` so the *real*
    ``classify_api_error`` maps the model-unloaded case to
    ``model_not_found`` / ``should_fallback=False``."""

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


# The 409 model-pool-lock classification #34076 will produce. Mocked here because
# the 409 -> should_fallback=False mapping is in #34076, not on this branch. reason
# is a stand-in (model_not_found: same non-retryable/no-fallback form, and it takes
# the identical loop path as the model-unloaded case); the loop gate keys on
# should_fallback, not on the reason.
_CONFLICT_409_NO_FALLBACK = ClassifiedError(
    reason=FailoverReason.model_not_found,
    status_code=409,
    message="model pool locked",
    retryable=False,
    should_compress=False,
    should_fallback=False,
)


@pytest.mark.parametrize(
    "err, mocked_classification",
    [
        pytest.param(
            _LMStudioAPIError("No models loaded", status_code=404),
            None,  # real classifier (#62765)
            id="model_unloaded_404_real_classifier",
        ),
        pytest.param(
            _LMStudioAPIError("model pool locked", status_code=409),
            _CONFLICT_409_NO_FALLBACK,  # mocked, pending #34076
            id="conflict_409_mocked_pending_34076",
        ),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "conversation_loop.py ~L3736 calls _try_activate_fallback() without a "
        "classified.should_fallback guard; the guard lands in "
        "NousResearch/hermes-agent#34076. Until it does, a no-fallback "
        "classification still walks the fallback chain. When #34076 merges both "
        "params XPASS (strict) -> remove xfail, keep asserts, lift 409 to a real "
        "trigger, rebase."
    ),
)
def test_no_fallback_activation_when_should_fallback_false(err, mocked_classification):
    # A fallback chain must exist, otherwise there is nothing to (wrongly)
    # activate and the assertion would be vacuous.
    agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
    assert agent._has_pending_fallback() is True
    index_before = agent._fallback_index

    # Stub only the leaf API call (streaming + non-streaming) so the error flows
    # through the recovery branch. Force _try_activate_fallback to a no-op
    # returning False so, if the loop reaches it, it aborts cleanly instead of
    # resolving a real provider.
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=err))
        stack.enter_context(patch.object(agent, "_interruptible_streaming_api_call", side_effect=err))
        spy_fallback = stack.enter_context(
            patch.object(agent, "_try_activate_fallback", return_value=False)
        )
        if mocked_classification is not None:
            # 409 branch: inject the #34076 classification (real classifier on this
            # branch does not yet map 409 -> should_fallback=False).
            stack.enter_context(
                patch("agent.conversation_loop.classify_api_error",
                      return_value=mocked_classification)
            )
        run_conversation(
            agent, "trigger a no-fallback error",
            conversation_history=[], task_id="t",
        )

    # The heart of the regression: a should_fallback=False classification must not
    # reach fallback activation ...
    spy_fallback.assert_not_called()
    # ... and the fallback chain must be untouched (Sweeper's explicit wording).
    # _fallback_index only advances inside _try_activate_fallback, so not-calling
    # it and an unchanged index are two faces of the same guarantee.
    assert agent._fallback_index == index_before
    assert agent._has_pending_fallback() is True
