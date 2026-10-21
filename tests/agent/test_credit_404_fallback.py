"""Regression for #115702: a paid Nous model behind an empty credit balance answers HTTP 404
``insufficient_credits_for_paid_model``. The code must classify as billing (fallback chain armed,
no retry burn) and the resulting switch must be a WARNING naming the failing profile and remedy.
"""

import logging

from agent.chat_completion_helpers import _log_fallback_activated
from agent.error_classifier import FailoverReason, classify_api_error
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _StatusError(Exception):
    def __init__(self, message, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_404_insufficient_credits_code_is_billing_with_fallback():
    # Structured code only, message carries no billing wording (message-pattern rules cannot save it).
    err = _StatusError(
        "Not Found", 404, {"error": {"code": "insufficient_credits_for_paid_model", "message": "Not Found"}},
    )
    verdict = classify_api_error(err, provider="nous", model="z-ai/glm-5.2")
    assert verdict.reason == FailoverReason.billing
    assert verdict.should_fallback and not verdict.retryable
    # Control: an unrelated 404 body keeps its generic verdict — nothing to fall back for.
    other = classify_api_error(
        _StatusError("Not Found", 404, {"error": {"code": "route_not_found", "message": "Not Found"}}),
        provider="nous", model="z-ai/glm-5.2",
    )
    assert other.reason == FailoverReason.unknown


def test_billing_fallback_warning_names_failing_profile_and_remedy(tmp_path, caplog):
    """Under multiplex the log runs in the failing profile's home scope: A then B name themselves,
    never the launch profile; a non-billing switch stays INFO."""
    seen = {}
    for name in ("alpha", "beta"):
        token = set_hermes_home_override(tmp_path / ".hermes" / "profiles" / name)
        try:
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="agent.chat_completion_helpers"):
                _log_fallback_activated(None, FailoverReason.billing, "z-ai/glm-5.2", "nous", "free/model", "nous")
        finally:
            reset_hermes_home_override(token)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        seen[name] = warnings[0].getMessage()
    assert "Profile alpha:" in seen["alpha"] and "hermes -p alpha model" in seen["alpha"]
    assert "Profile beta:" in seen["beta"] and "alpha" not in seen["beta"]
    for text in seen.values():
        assert "z-ai/glm-5.2" in text and "free/model" in text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.chat_completion_helpers"):
        _log_fallback_activated(None, FailoverReason.server_error, "a", "p", "b", "q")
    assert [r.levelno for r in caplog.records] == [logging.INFO]
