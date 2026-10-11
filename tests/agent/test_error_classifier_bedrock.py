"""Bedrock (botocore ``ClientError``) status reaches the shared classifier (#121294).

botocore keeps the HTTP status in ``response["ResponseMetadata"]["HTTPStatusCode"]``, not an
attribute, so a deterministic 400 used to classify as ``unknown`` and be retried. The SDK-free
stand-in mirrors botocore's shape (boto3 is not in the unit-test environment); the real-SDK case
is ``tests/e2e/core/providers/test_native_bedrock_converse_faults.py``.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_failure_copy import nonretryable_copy


class ClientError(Exception):
    def __init__(self, code: str, status: int):
        self.response = {
            "Error": {"Code": code, "Message": "Bedrock rejected the request"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }
        super().__init__(f"An error occurred ({code}) when calling the ConverseStream operation")


@pytest.mark.parametrize(
    ("code", "status", "reason"),
    [("ValidationException", 400, FailoverReason.format_error), ("AccessDeniedException", 403, FailoverReason.auth)],
)
def test_botocore_client_error_is_terminal_and_not_reported_as_unavailable(code, status, reason):
    result = classify_api_error(ClientError(code, status), provider="bedrock", model="test-model")

    assert (result.status_code, result.reason, result.retryable) == (status, reason, False)
    copy = nonretryable_copy(result, provider="bedrock", model="test-model", summary=result.message)
    assert "temporarily unavailable" not in copy.lower()
