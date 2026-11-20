"""Provider safety-filter refusals classify as content_policy_blocked, never as a retryable outage."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _err(message, **fields):
    """An OpenAI-style ``{"error": {...}}`` body carrying *message*."""
    return {"error": {"message": message, **fields}}


_ALIBABA_OUT = "Upstream error from Alibaba: Output data may contain inappropriate content."
_CYBER = "This content was flagged for possible cybersecurity risk. If this seems wrong, try rephrasing your request."
_TOU = "Request rejected in accordance with our terms of use and policies."
_SAFETY = ("Your request was rejected by the safety system. If you believe this is an error, contact us at "
           "help.openai.com and include the request ID req_123.")
_FLAGGED = "Your input was flagged"
_USAGE = ("Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again "
          "with a different prompt.")
_ZAI = "System detected potentially unsafe or sensitive content in input or generation."
_GOOGLE = "Request blocked due to safety violations (harmful content). Please modify your input and retry."


@pytest.mark.parametrize("status, message, body", [
    # Output guardrail relayed by OpenRouter / the Nous Portal as an in-band 502 marked provider_unavailable.
    (502, _ALIBABA_OUT, None),
    (None, _ALIBABA_OUT, None),
    # Input guardrail: DashScope's documented 400 (code data_inspection_failed).
    (400, "Error code: 400 - {'error': {'message': 'Input data may contain inappropriate content.', "
          "'type': 'data_inspection_failed', 'code': 'data_inspection_failed'}}", None),
    # z-ai content filter 403 (2026-09-12: 61 of these were routed as auth).
    (403, "Error code: 403 - Request blocked by content filter: [SECRET:google-oauth-client-secret]", None),
    # opencode's provider-error corpus (anomalyco/opencode#64844, #64910 and the cases beside them).
    (400, _USAGE, _err(_USAGE, type="invalid_request_error", code="invalid_prompt")),
    (400, "Output blocked by content filtering policy",
     _err("Output blocked by content filtering policy", type="invalid_request_error")),
    (403, _FLAGGED, _err(_FLAGGED, code=403, metadata={"error_type": "content_policy_violation"})),
    (None, _FLAGGED, _err(_FLAGGED, code="image_content_policy_violation")),
    (None, "Claude refused to respond",
     _err("Claude refused to respond", type="invalid_request_error", error_type="refusal")),
    (400, _CYBER, _err(_CYBER, type="invalid_request_error", code="cyber_policy")),
    (None, "Request blocked.", _err("Request blocked.", code="cyber_policy")),  # stream frame, code only
    (400, "Green net check failed for text (input).", {"code": "DataInspectionFailed", "message": "Green net check"}),
    (400, _SAFETY, _err(_SAFETY, type="image_generation_user_error", code="moderation_blocked")),
    (400, _SAFETY, _err(_SAFETY, type="AI_APICallError")),  # Vercel AI Gateway drops the code
    (403, _TOU, _err(_TOU, code=403, status="PERMISSION_DENIED")),  # Gemini: was auth
    (403, f"Upstream request failed: [403] {_TOU}", _err(f"Upstream request failed: [403] {_TOU}", type="server_error")),
    (400, _ZAI, _err(_ZAI, code="1301")),
    (400, _GOOGLE, _err(_GOOGLE, code="safety")),
])
def test_provider_refusal_is_a_content_policy_block_not_an_outage(status, message, body):
    """A safety refusal repeats for the unchanged request, so it must not ride the retry ladder,
    rotate credentials, or read as a bad key (403) / malformed request (400)."""
    result = classify_api_error(_APIError(message, status, body), provider="openrouter", model="m")
    assert result.reason == FailoverReason.content_policy_blocked
    assert result.retryable is False
    assert result.should_rotate_credential is False


@pytest.mark.parametrize("status, message, body, expected", [
    (401, "Incorrect API key provided", _err("Incorrect API key provided", code="invalid_api_key"), FailoverReason.auth),
    # Echoed config naming the filter (space variant) is not a block.
    (403, "Error code: 403 - config uses content filter mode", None, FailoverReason.auth),
    (403, "Error code: 403 - Key limit exceeded", None, FailoverReason.billing),
    (400, "Invalid 'messages[1].content': string too long",
     _err("Invalid 'messages[1].content': string too long", code="invalid_request_error"), FailoverReason.format_error),
])
def test_non_refusals_keep_their_class(status, message, body, expected):
    assert classify_api_error(_APIError(message, status, body), provider="openrouter", model="m").reason == expected
