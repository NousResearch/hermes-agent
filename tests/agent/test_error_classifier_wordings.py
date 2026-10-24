"""Provider wordings that name a cause without the generic phrases the classifier keys on:
invalid keys sent as HTTP 400, and context overflows phrased as input-length validation errors."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_error_classifier import MockAPIError


class TestInvalidKeyAs400:
    """xAI and Google report a bad API key as HTTP 400, not 401; the credential is still the fault
    (port of anomalyco/opencode#52112, #51950)."""

    @pytest.mark.parametrize("message, body, provider", [
        ('Error code: 400 - {"code":"invalid-argument","error":"Incorrect API key provided. You can obtain '
         'an API key from https://console.x.ai."}',
         {"code": "invalid-argument", "error": "Incorrect API key provided."}, "xai"),
        ('Error code: 400 - {"error":{"code":400,"message":"API key not valid. Please pass a valid API key.",'
         '"status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}',
         {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}, "gemini"),
    ])
    def test_400_invalid_key_rotates_like_401(self, message, body, provider):
        result = classify_api_error(MockAPIError(message, status_code=400, body=body), provider=provider)
        assert result.reason == FailoverReason.auth
        assert result.should_rotate_credential is True
        assert result.should_compress is False

    def test_unrelated_400_invalid_argument_stays_format_error(self):
        e = MockAPIError('Error code: 400 - {"code":"invalid-argument","error":"Unknown parameter: foo"}',
                         status_code=400, body={"code": "invalid-argument", "error": "Unknown parameter: foo"})
        assert classify_api_error(e, provider="xai").reason == FailoverReason.format_error


class TestInputLengthOverflowWordings:
    """DeepInfra and HF TGI word a context overflow without any 'input token(s)' / 'context length'
    phrase (port of anomalyco/opencode#52132, #52133)."""

    @pytest.mark.parametrize("status,message", [
        (400, 'Error code: 400 - {"detail":{"error":"requested input length 135000 exceeds maximum input length 131072"}}'),
        (400, 'Error code: 400 - {"error":{"message":"Input validation error: `inputs` tokens + `max_new_tokens` must '
              'be <= 131073. Given: 140000 `inputs` tokens and 4096 `max_new_tokens`","type":"invalid_request_error"}}'),
        # TGI/TEI answer validation errors with 422 (openclaw#158161).
        (422, 'Input validation error: `inputs` must have less than 4096 tokens. Given: 5000'),
        (422, 'Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16000. Given: 220616 `inputs` tokens'),
        # DashScope / Model Studio (openclaw#171712).
        (400, '<400> InternalError.Algo.InvalidParameter: Range of input length should be [1, 129024]'),
    ])
    def test_input_length_rejection_compresses(self, status, message):
        result = classify_api_error(MockAPIError(message, status_code=status), provider="custom")
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    @pytest.mark.parametrize("status,message", [
        (422, 'Input validation error: `temperature` must be strictly positive'),
        (400, '<400> InvalidParameter: Range of top_p should be (0.0, 1.0]'),
    ])
    def test_sibling_validation_rejections_stay_format_errors(self, status, message):
        result = classify_api_error(MockAPIError(message, status_code=status), provider="custom")
        assert result.reason == FailoverReason.format_error
