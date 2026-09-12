"""Safety wording is presentation, not permission to change recovery policy."""
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.run_agent.test_primary_runtime_restore import _make_agent
from tests.run_agent.test_provider_fallback import _make_agent as _fallback_agent, _mock_client


SAFETY_MESSAGE = (
    "This request was blocked by our safety systems. "
    "Reason: Potentially unintended activity."
)


def safety_error(status):
    error = openai.APIError(
        SAFETY_MESSAGE, request=httpx.Request("POST", "https://example.test/v1/responses"), body=None,
    )
    if status is not None:
        error.status_code = status
    return error


@pytest.mark.parametrize("status", [None, 400, 403, 429, 500])
def test_display_reason_does_not_change_recovery(status):
    error = safety_error(status)
    classified = classify_api_error(error)
    before = asdict(classified)
    assert classified.display_reason == FailoverReason.content_policy_blocked
    assert asdict(classified) == before
    control = openai.APIError("ordinary provider failure", request=error.request, body=None)
    if status is not None:
        control.status_code = status
    baseline = classify_api_error(control)
    for field in ("reason", "retryable", "should_compress", "should_rotate_credential", "should_fallback"):
        assert getattr(classified, field) == getattr(baseline, field)


@pytest.mark.parametrize("status", [None, 400, 403])
def test_safety_api_error_does_not_enter_transport_recovery(status):
    agent = _make_agent(provider="custom")
    agent._vprint = MagicMock()
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()), patch("time.sleep") as sleep:
        assert not agent._try_recover_primary_transport(safety_error(status), retry_count=3, max_retries=3)
    sleep.assert_not_called()
    agent._vprint.assert_not_called()


@pytest.mark.parametrize("status", [None, 400, 403, 429, 500])
def test_fallback_display_keeps_routing_reason_and_cooldown(status):
    classified = classify_api_error(safety_error(status))
    agent = _fallback_agent(fallback_model={"provider": "zai", "model": "glm-5.2"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "glm-5.2")),
        patch("agent.fallback_cooldown._arm_rate_limit_cooldown", return_value=None) as cooldown,
    ):
        assert agent._try_activate_fallback(reason=classified.reason, display_reason=classified.display_reason)
    cooldown.assert_called_once_with(agent, classified.reason)
    assert "content policy blocked" in agent._pending_fallback_notice[-1]
    assert agent.model == "glm-5.2"


def test_ordinary_error_display_unchanged():
    error = openai.APIError("ordinary provider failure", request=httpx.Request("POST", "https://example.test"), body=None)
    classified = classify_api_error(error)
    assert classified.display_reason == classified.reason
