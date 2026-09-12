"""Tests for the AWS/botocore credential-chain family in agent.error_classifier.

On a Bedrock install whose AWS session is refreshable but has lapsed (SSO,
``credential_process``, an assumed role), botocore fails while *signing* the
request, before any HTTP status exists. Every member of that family — the
credential-chain exceptions, the Bedrock-side STS/token ``ClientError``
codes, and the OpenAI-SDK-wrapped (Mantle) variant — must classify as
``auth`` with ``should_rotate_credential=True`` / ``should_fallback=True``,
the same verdict this file already gives a status-less "token expired"
message and a 401.

The carve-outs are a family exception whose text names a *refreshable*
metadata provider (``iam-role`` / ``container-role``, botocore's "Credential
refresh failed, response did not contain" shape) or embeds a network failure
from a ``credential_process`` helper ("timed out", "connection refused"), and
the two "no credentials resolvable" ``RuntimeError`` texts, which are raised
by code that re-resolves the credential chain on every request: nothing there
proves the session is dead, so those keep the retryable verdict they already
get on main.

Exceptions are built by class name via ``type(name, (Exception,), {})``,
exactly like ``tests/agent/test_error_classifier.py::TestRateLimitErrorWithoutStatusCode``
does, so botocore is not a hard import dependency here;
``test_real_botocore_exceptions_are_auth`` exercises the real classes when the
``bedrock`` extra is installed.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (mirrors test_error_classifier.py)."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


def _mk(class_name: str, message: str) -> Exception:
    """An exception instance whose *type name* is ``class_name``, without importing botocore."""
    return type(class_name, (Exception,), {})(message)


def _wrapped(outer_class_name: str, outer_message: str, cause: BaseException) -> Exception:
    """An exception with the given type name and ``__cause__``, mirroring an SDK's
    ``raise Outer(...) from err`` wrapping."""
    exc = type(outer_class_name, (Exception,), {})(outer_message)
    exc.__cause__ = cause
    return exc


def _assert_is_auth_rotate(result) -> None:
    assert result.reason == FailoverReason.auth
    assert result.retryable is False
    assert result.should_rotate_credential is True
    assert result.should_fallback is True
    assert result.should_compress is False


# ── botocore credential-chain exception types ───────────────────────────

class TestCredentialChainErrorsAreAuth:
    def test_credential_retrieval_error_is_auth(self):
        """The operator's actual shape: a credential_process helper's session expired."""
        e = _mk(
            "CredentialRetrievalError",
            "Error when retrieving credentials from custom-process: "
            "profile [my-sso-profile] expired 41s ago. Refresh it on the HOST.",
        )
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)

    def test_no_credentials_error_is_auth(self):
        e = _mk("NoCredentialsError", "Unable to locate credentials")
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)

    def test_partial_credentials_error_is_auth(self):
        e = _mk(
            "PartialCredentialsError",
            "Partial credentials found in env, missing: AWS_SECRET_ACCESS_KEY",
        )
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)

    @pytest.mark.parametrize(
        "class_name,message",
        [
            (
                "TokenRetrievalError",
                "Error when retrieving token from sso: Token has expired and refresh failed",
            ),
            (
                "SSOTokenLoadError",
                "The SSO session associated with this profile has expired or is otherwise invalid",
            ),
            (
                "UnauthorizedSSOTokenError",
                "The SSO session associated with this profile has expired or is "
                "otherwise invalid. Please re-authenticate.",
            ),
            (
                "RefreshWithMFAUnsupportedError",
                "Refreshing temporary credentials with MFA is not supported.",
            ),
        ],
    )
    def test_sso_token_errors_are_auth(self, class_name, message):
        e = _mk(class_name, message)
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)


# ── Bedrock-side ClientError STS/token codes ────────────────────────────

class TestBedrockClientErrorCodesAreAuth:
    @pytest.mark.parametrize(
        "code,operation",
        [
            (
                "ExpiredTokenException",
                "An error occurred (ExpiredTokenException) when calling the ConverseStream "
                "operation: The security token included in the request is expired",
            ),
            (
                "UnrecognizedClientException",
                "An error occurred (UnrecognizedClientException) when calling the ConverseStream "
                "operation: The security token included in the request is invalid.",
            ),
            (
                "InvalidSignatureException",
                "An error occurred (InvalidSignatureException) when calling the ConverseStream "
                "operation: The request signature we calculated does not match the signature "
                "you provided.",
            ),
            (
                "InvalidClientTokenId",
                "An error occurred (InvalidClientTokenId) when calling the GetCallerIdentity "
                "operation: The security token included in the request is invalid.",
            ),
        ],
    )
    def test_bedrock_client_error_codes_are_auth(self, code, operation):
        """Native Converse path: botocore's ``.response`` is a dict, so this is
        status-less too — ``_status_of``/``_body_of`` see neither status nor body."""
        e = _mk("ClientError", operation)
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)


# ── Bedrock Mantle (OpenAI SDK) wrapper ─────────────────────────────────

class TestOpenAISDKWrappedCredentialError:
    def test_openai_sdk_wrapped_credential_error_is_auth(self):
        """On the Mantle (OpenAI SDK) path botocore raises inside httpx's auth
        flow; the SDK wraps it as ``APIConnectionError("Connection error.") from
        err``. ``APIConnectionError`` is in ``_TRANSPORT_ERROR_TYPES``, so without
        the new stage running first this would misclassify as ``timeout``."""
        cause = _mk(
            "CredentialRetrievalError",
            "Error when retrieving credentials from custom-process: profile expired",
        )
        e = _wrapped("APIConnectionError", "Connection error.", cause)
        result = classify_api_error(e, provider="bedrock")
        _assert_is_auth_rotate(result)


# ── Refreshable providers and helper network failures stay retryable ────

_IMDS_REFRESH_FAILED = (
    "Error when retrieving credentials from iam-role: Credential refresh failed, "
    "response did not contain: access_key, secret_key, token, expiry_time"
)


class TestTransientCredentialFailuresStayRetryable:
    @pytest.mark.parametrize(
        "message",
        [
            _IMDS_REFRESH_FAILED,
            "Error when retrieving credentials from container-role: Connection refused",
            # A Vault/1Password/aws-vault helper that could not reach its backend.
            "Error when retrieving credentials from custom-process: Read timed out",
            "Error when retrieving credentials from custom-process: connection refused",
        ],
    )
    def test_refreshable_provider_or_helper_network_failure_is_not_terminal(self, message):
        """An instance/task role whose IMDS refresh blipped, or a credential_process
        helper whose stderr reports a network failure, is not evidence of an expired
        session: the next attempt a few seconds later normally succeeds. These must
        keep a retryable verdict (never ``auth``) so the existing backoff applies."""
        e = _mk("CredentialRetrievalError", message)
        result = classify_api_error(e, provider="bedrock")
        assert result.reason != FailoverReason.auth
        assert result.retryable is True

    def test_wrapped_refreshable_provider_failure_stays_timeout(self):
        """The Mantle wrapper around an IMDS refresh blip must keep the transport
        verdict ``_TRANSPORT_ERROR_TYPES`` already gives ``APIConnectionError``."""
        cause = _mk("CredentialRetrievalError", _IMDS_REFRESH_FAILED)
        e = _wrapped("APIConnectionError", "Connection error.", cause)
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True


# ── "No credentials resolvable" RuntimeErrors stay retryable ────────────

_NO_CREDENTIALS_RUNTIME_ERRORS = [
    # anthropic/lib/bedrock/_auth.py: session.get_credentials() on every sign.
    "could not resolve credentials from session",
    # agent/bedrock_adapter.py BedrockOpenAISigV4Auth.auth_flow: a fresh
    # botocore Session per request.
    "No AWS credentials available for Bedrock OpenAI Responses. Configure "
    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, SSO, or an instance/task role.",
]


class TestNoCredentialsRuntimeErrorsStayRetryable:
    @pytest.mark.parametrize("message", _NO_CREDENTIALS_RUNTIME_ERRORS)
    def test_no_credentials_runtime_errors_are_not_terminal(self, message):
        """Both texts are raised by code that re-runs the credential chain on
        EVERY request (a fresh ``botocore.session.get_session()`` in
        ``auth_flow``; ``session.get_credentials()`` per sign in the Bedrock
        SDK, and botocore's ``Session.get_credentials`` only caches a non-None
        result). A first-request IMDS miss therefore clears on the next attempt,
        so these must keep the retryable ``unknown`` verdict main gives them —
        unlike ``NoCredentialsError`` from the cached boto3 Converse client,
        whose signer holds ``credentials=None`` for the client's lifetime."""
        e = RuntimeError(message)
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True
        assert result.should_rotate_credential is False

    def test_wrapped_no_credentials_runtime_error_stays_timeout(self):
        """The Mantle path raises the RuntimeError inside httpx's auth flow and
        the OpenAI SDK wraps it as ``APIConnectionError``; that keeps the
        transport verdict main already gives it."""
        e = _wrapped("APIConnectionError", "Connection error.", RuntimeError(_NO_CREDENTIALS_RUNTIME_ERRORS[1]))
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True


# ── Real botocore classes, when the extra is installed ──────────────────

class TestRealBotocoreExceptions:
    def test_real_botocore_exceptions_are_auth(self):
        botocore_exceptions = pytest.importorskip("botocore.exceptions")
        credential_error = botocore_exceptions.CredentialRetrievalError(
            provider="custom-process", error_msg="profile expired 41s ago"
        )
        result = classify_api_error(credential_error, provider="bedrock")
        _assert_is_auth_rotate(result)

        client_error = botocore_exceptions.ClientError(
            error_response={
                "Error": {
                    "Code": "ExpiredTokenException",
                    "Message": "The security token included in the request is expired",
                }
            },
            operation_name="ConverseStream",
        )
        result = classify_api_error(client_error, provider="bedrock")
        _assert_is_auth_rotate(result)


# ── Guards: behaviour this PR must NOT change ────────────────────────────

class TestGuardsUnchangedBehaviour:
    def test_wrapped_transport_failure_stays_timeout(self):
        """A non-credential cause inside the same wrapper shape must still hit
        ``_TRANSPORT_ERROR_TYPES`` and classify as ``timeout``."""
        cause = _mk("EndpointConnectionError", "Could not connect to the endpoint URL")
        e = _wrapped("APIConnectionError", "Connection error.", cause)
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.timeout

    def test_bedrock_throttling_client_error_stays_rate_limit(self):
        """Control: a sibling ``ClientError`` already lands on ``rate_limit``
        through the status-less message path — proving the gap is specific to
        credentials, not to status-less Bedrock ``ClientError``s in general."""
        e = _mk(
            "ClientError",
            "An error occurred (ThrottlingException) when calling the "
            "ConverseStream operation: Too many requests, please wait",
        )
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    def test_http_403_expired_token_keeps_status_verdict(self):
        """When an HTTP 403 status IS present, ``_status_403`` must keep
        deciding the verdict (auth, but WITHOUT credential rotation) — the new
        stage sits after ``_by_status`` and must not shadow it."""
        e = MockAPIError(
            "The security token included in the request is expired",
            status_code=403,
        )
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.auth
        assert result.should_rotate_credential is False

    @pytest.mark.parametrize(
        "message",
        [
            "something exploded",
            "credentials saved to profile",
        ],
    )
    def test_unrelated_error_stays_unknown(self, message):
        """Prose that merely mentions "credentials" without any family type
        name or STS/token code must not be swept into this stage."""
        e = RuntimeError(message)
        result = classify_api_error(e, provider="bedrock")
        assert result.reason == FailoverReason.unknown
