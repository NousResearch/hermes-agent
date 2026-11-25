"""A delisted-model 404 must walk the configured task fallback chain.

Incident: OpenRouter removed ``nex-agi/nex-n2.5-mini:free`` from its catalogue;
every auxiliary call pinned to it failed with HTTP 404 ``No endpoints found for
<model>.`` The recovery ladder must read that bare form as model_not_found so
``fallback_label`` routes explicit requests through their configured task chain.
"""

import httpx
import openai

from agent import auxiliary_error_policy as policy
from agent.error_classifier import FailoverReason, classify_api_error

PRIMARY_MODEL = "nex-agi/nex-n2.5-mini:free"
_REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _delisted_error():
    body = {"message": f"No endpoints found for {PRIMARY_MODEL}.", "code": 404}
    return openai.NotFoundError(
        f"Error code: 404 - {{'error': {body!r}}}",
        response=httpx.Response(404, request=_REQ),
        body=body,
    )


def test_delisted_model_404_is_model_not_found():
    """The bare OpenRouter delisted-model 404 belongs to the not-found bucket."""
    verdict = classify_api_error(_delisted_error(), provider="openrouter")
    assert verdict.reason is FailoverReason.model_not_found
    assert verdict.should_fallback is True


def test_delisted_404_reaches_configured_task_fallback():
    """The policy layer names the failure so the ladder consults the task chain."""
    verdict = classify_api_error(_delisted_error(), provider="openrouter")
    failure = policy.Failure(_delisted_error(), verdict)
    assert policy.fallback_label(failure) == policy.MODEL_NOT_FOUND
    assert policy.ladder_accepts(failure) is True
