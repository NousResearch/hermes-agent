"""Native botocore responses must obey the same recovery contract as HTTP SDKs."""
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize("code,status,reason,retryable", [
    ("ValidationException", 400, FailoverReason.format_error, False),
    ("AccessDeniedException", 403, FailoverReason.auth, False),
    ("ResourceNotFoundException", 404, FailoverReason.model_not_found, False),
    ("ThrottlingException", 429, FailoverReason.rate_limit, True),
    ("ServiceUnavailableException", 503, FailoverReason.overloaded, True),
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_response_classification(code, status, reason, retryable, wrapped):
    ClientError = pytest.importorskip("botocore.exceptions").ClientError
    error = ClientError({"Error": {"Code": code, "Message": "Bedrock request rejected MARK-294"},
                         "ResponseMetadata": {"HTTPStatusCode": status}}, "ConverseStream")
    if wrapped:
        outer = RuntimeError("provider call failed")
        outer.__cause__ = error
        error = outer
    result = classify_api_error(error, provider="bedrock", model="deepseek.v3-v1:0")
    assert result.status_code == status
    assert result.reason == reason
    assert result.retryable is retryable
    assert "MARK-294" in result.message
    assert result.should_fallback is (not retryable or status == 429)


def test_descriptive_validation_is_not_large_context_overflow():
    ClientError = pytest.importorskip("botocore.exceptions").ClientError
    error = ClientError({"Error": {"Code": "ValidationException", "Message": "The model returned malformed input request: MARK-294"},
                         "ResponseMetadata": {"HTTPStatusCode": 400}}, "ConverseStream")
    result = classify_api_error(error, provider="bedrock", approx_tokens=150000, num_messages=200)
    assert result.reason == FailoverReason.format_error
    assert not result.should_compress


@pytest.mark.parametrize("response", [None, {}, {"ResponseMetadata": None},
    {"ResponseMetadata": {"HTTPStatusCode": "400"}}, {"Error": []}])
def test_malformed_or_empty_response_preserves_unknown(response):
    error = RuntimeError("unclassified")
    error.response = response
    result = classify_api_error(error, provider="bedrock")
    assert result.reason == FailoverReason.unknown
    assert result.status_code is None
    assert result.retryable


def test_explicit_http_sdk_fields_keep_precedence():
    error = RuntimeError("rejected")
    error.status_code = 429
    error.body = {"message": "Too many requests"}
    error.response = {"ResponseMetadata": {"HTTPStatusCode": 400},
                      "Error": {"Code": "ValidationException", "Message": "invalid"}}
    result = classify_api_error(error, provider="bedrock")
    assert result.reason == FailoverReason.rate_limit
    assert result.status_code == 429
    assert result.message == "Too many requests"


@pytest.mark.parametrize("code,status,reason", [
    ("ValidationException", 400, FailoverReason.format_error),
    ("AccessDeniedException", 403, FailoverReason.auth),
    ("ResourceNotFoundException", 404, FailoverReason.model_not_found),
])
def test_real_boto3_wire_error_reaches_classifier(code, status, reason):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from tests.fakes.providers.bedrock_converse import (
        ACCESS_KEY, REGION, SECRET_KEY, FakeBedrock, HttpError, seq,
    )

    with FakeBedrock(seq(HttpError(code, "Malformed or inaccessible request MARK-294"))) as fake:
        client = boto3.client("bedrock-runtime", endpoint_url=fake.endpoint, region_name=REGION,
                              aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY,
                              config=Config(retries={"total_max_attempts": 1}, proxies={}))
        try:
            with pytest.raises(ClientError) as caught:
                client.converse_stream(modelId="deepseek.v3-v1:0", messages=[
                    {"role": "user", "content": [{"text": "hello"}]}])
            result = classify_api_error(caught.value, provider="bedrock")
            assert result.status_code == status
            assert result.reason == reason
            assert not result.retryable
            assert "MARK-294" in result.message
            assert len(fake.snapshot()) == 1
            assert not fake.snapshot()[0].get("rejected")
        finally:
            client.close()
