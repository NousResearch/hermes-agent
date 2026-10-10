"""Cause-specific acquisition failures for programmatic web consumers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from typing import Annotated, Literal

import httpx
import requests
from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict


WebFailureKind = Literal[
    "authentication",
    "rate-limited",
    "quota-exhausted",
    "request-rejected",
    "unavailable",
    "timeout",
    "connection",
    "invalid-response",
    "dependency-missing",
    "credentials-missing",
    "configuration",
    "unsupported-capability",
    "cancelled",
    "secret-url",
    "private-address",
    "website-policy",
    "page-not-found",
    "result-missing",
    "attribution-invalid",
    "content-coverage",
    "unclassified",
    "payment-required",
    "website-unsupported",
    "managed-gateway-unavailable",
    "page-fetch",
]
WebFailureScope = Literal["provider", "page", "input", "policy"]
WebRetry = Literal["never", "transient", "unknown"]
ContentCoverage = Literal["full", "partial", "unknown"]


class WebFailureData(TypedDict):
    kind: WebFailureKind
    retry: WebRetry
    scope: WebFailureScope
    status: NotRequired[Annotated[int, Field(ge=400, le=599)]]
    retry_after_ms: NotRequired[Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    timeout_seconds: NotRequired[Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    dependency: NotRequired[str]
    credential: NotRequired[str]
    capability: NotRequired[str]
    coverage: NotRequired[ContentCoverage]


class WebAcquisitionError(Exception, ABC):
    """A failure whose safe facts can cross the acquisition boundary."""

    retry: WebRetry = "never"
    scope: WebFailureScope = "provider"

    @property
    @abstractmethod
    def kind(self) -> WebFailureKind:
        """The acquisition cause, independent of diagnostic wording."""

    def __init__(self, *, cause: BaseException | None = None):
        super().__init__(self.diagnostic)
        self.__cause__ = cause

    @property
    def diagnostic(self) -> str:
        return f"Web acquisition failed: {self.kind}"

    def to_failure(self) -> WebFailureData:
        """Return safe facts without the upstream exception or its diagnostic."""
        return {"kind": self.kind, "retry": self.retry, "scope": self.scope}


class _WebHttpError(WebAcquisitionError, ABC):
    def __init__(
        self,
        status: int | None = None,
        retry_after_ms: float | None = None,
        *,
        cause: BaseException | None = None,
    ):
        self.status = status
        self.retry_after_ms = retry_after_ms
        super().__init__(cause=cause)

    @property
    def diagnostic(self) -> str:
        detail = f" (HTTP {self.status})" if self.status is not None else ""
        return f"Web acquisition failed: {self.kind}{detail}"

    def to_failure(self) -> WebFailureData:
        failure = super().to_failure()
        if self.status is not None:
            failure["status"] = self.status
        if self.retry_after_ms is not None:
            failure["retry_after_ms"] = self.retry_after_ms
        return failure


class WebAuthenticationError(_WebHttpError):
    kind = "authentication"


class WebRateLimitedError(_WebHttpError):
    kind = "rate-limited"
    retry = "transient"


class WebQuotaExhaustedError(_WebHttpError):
    kind = "quota-exhausted"


class WebPaymentRequiredError(_WebHttpError):
    kind = "payment-required"


class WebWebsiteUnsupportedError(_WebHttpError):
    kind = "website-unsupported"
    scope = "page"


class WebRequestRejectedError(_WebHttpError):
    kind = "request-rejected"


class WebServiceUnavailableError(_WebHttpError):
    kind = "unavailable"
    retry = "transient"


class WebPageNotFoundError(_WebHttpError):
    kind = "page-not-found"
    scope = "page"


class WebPageFetchError(_WebHttpError):
    kind = "page-fetch"
    scope = "page"

    def __init__(
        self, status: int | None = None, *, cause: BaseException | None = None
    ):
        self.retry = "unknown"
        if status is not None:
            self.retry = (
                "transient" if status in {408, 429} or status >= 500 else "never"
            )
        super().__init__(status, cause=cause)


class WebTimeoutError(WebAcquisitionError):
    kind = "timeout"
    retry = "transient"

    def __init__(
        self,
        timeout_seconds: float | None = None,
        *,
        cause: BaseException | None = None,
    ):
        super().__init__(cause=cause)
        self.timeout_seconds = timeout_seconds

    def to_failure(self) -> WebFailureData:
        failure = super().to_failure()
        if self.timeout_seconds is not None:
            failure["timeout_seconds"] = self.timeout_seconds
        return failure


class WebConnectionError(WebAcquisitionError):
    kind = "connection"
    retry = "transient"


class WebInvalidResponseError(WebAcquisitionError):
    kind = "invalid-response"


class WebDependencyMissingError(WebAcquisitionError):
    kind = "dependency-missing"

    def __init__(
        self, dependency: str | None = None, *, cause: BaseException | None = None
    ):
        super().__init__(cause=cause)
        self.dependency = dependency

    def to_failure(self) -> WebFailureData:
        failure = super().to_failure()
        if self.dependency is not None:
            failure["dependency"] = self.dependency
        return failure


class WebCredentialsMissingError(WebAcquisitionError):
    kind = "credentials-missing"

    def __init__(self, credential: str, *, cause: BaseException | None = None):
        self.credential = credential
        super().__init__(cause=cause)

    @property
    def diagnostic(self) -> str:
        return f"{self.credential} is not set"

    def to_failure(self) -> WebFailureData:
        return {**super().to_failure(), "credential": self.credential}


class WebConfigurationError(WebAcquisitionError):
    kind = "configuration"


class WebManagedGatewayUnavailableError(WebConfigurationError):
    kind = "managed-gateway-unavailable"

    @property
    def diagnostic(self) -> str:
        return "Nous Tool Gateway is unavailable. Run `hermes tools` to configure web access."


class WebCapabilityUnsupportedError(WebAcquisitionError):
    kind = "unsupported-capability"

    def __init__(self, capability: str, *, cause: BaseException | None = None):
        super().__init__(cause=cause)
        self.capability = capability

    def to_failure(self) -> WebFailureData:
        return {**super().to_failure(), "capability": self.capability}


class WebCancelledError(WebAcquisitionError):
    kind = "cancelled"


class WebSecretUrlDeniedError(WebAcquisitionError):
    kind = "secret-url"
    scope = "input"


class WebPrivateAddressDeniedError(WebAcquisitionError):
    kind = "private-address"
    scope = "policy"


class WebWebsitePolicyDeniedError(WebAcquisitionError):
    kind = "website-policy"
    scope = "policy"


class WebResultMissingError(WebAcquisitionError):
    kind = "result-missing"
    scope = "page"


class WebAttributionInvalidError(WebAcquisitionError):
    kind = "attribution-invalid"
    scope = "page"


class WebContentCoverageError(WebAcquisitionError):
    kind = "content-coverage"
    scope = "page"

    def __init__(self, coverage: ContentCoverage):
        super().__init__()
        self.coverage = coverage

    def to_failure(self) -> WebFailureData:
        return {**super().to_failure(), "coverage": self.coverage}


class WebUnclassifiedFailure(WebAcquisitionError):
    kind = "unclassified"
    retry = "unknown"


_FAILURE_ADAPTER = TypeAdapter(WebFailureData)
_HTTP_FAILURES: dict[WebFailureKind, type[_WebHttpError]] = {
    "authentication": WebAuthenticationError,
    "rate-limited": WebRateLimitedError,
    "quota-exhausted": WebQuotaExhaustedError,
    "request-rejected": WebRequestRejectedError,
    "unavailable": WebServiceUnavailableError,
    "page-not-found": WebPageNotFoundError,
    "payment-required": WebPaymentRequiredError,
    "website-unsupported": WebWebsiteUnsupportedError,
}
_SIMPLE_FAILURES: dict[WebFailureKind, type[WebAcquisitionError]] = {
    "connection": WebConnectionError,
    "invalid-response": WebInvalidResponseError,
    "configuration": WebConfigurationError,
    "managed-gateway-unavailable": WebManagedGatewayUnavailableError,
    "cancelled": WebCancelledError,
    "secret-url": WebSecretUrlDeniedError,
    "private-address": WebPrivateAddressDeniedError,
    "website-policy": WebWebsitePolicyDeniedError,
    "result-missing": WebResultMissingError,
    "attribution-invalid": WebAttributionInvalidError,
    "unclassified": WebUnclassifiedFailure,
}


def failure_from_data(value: object) -> WebAcquisitionError:
    """Validate safe wire facts and reconstruct the corresponding cause-specific error."""
    try:
        failure = _FAILURE_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc

    kind = failure["kind"]
    http_class = _HTTP_FAILURES.get(kind)
    simple_class = _SIMPLE_FAILURES.get(kind)
    error: WebAcquisitionError
    if http_class is not None:
        error = http_class(failure.get("status"), failure.get("retry_after_ms"))
    elif simple_class is not None:
        error = simple_class()
    elif kind == "timeout":
        error = WebTimeoutError(failure.get("timeout_seconds"))
    elif kind == "dependency-missing":
        error = WebDependencyMissingError(failure.get("dependency"))
    elif kind == "credentials-missing" and "credential" in failure:
        error = WebCredentialsMissingError(failure["credential"])
    elif kind == "unsupported-capability" and "capability" in failure:
        error = WebCapabilityUnsupportedError(failure["capability"])
    elif kind == "content-coverage" and "coverage" in failure:
        error = WebContentCoverageError(failure["coverage"])
    elif kind == "page-fetch":
        error = WebPageFetchError(failure.get("status"))
    else:
        raise WebInvalidResponseError()

    if error.to_failure() != value:
        raise WebInvalidResponseError()
    return error


def _retry_after_ms(headers: object) -> float | None:
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("retry-after", headers.get("Retry-After"))
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    delay = max(0, seconds) * 1000
    return delay if math.isfinite(seconds) and math.isfinite(delay) else None


def http_failure(
    status: int, headers: object = None, *, cause: BaseException | None = None
) -> WebAcquisitionError:
    """Classify a provider endpoint status; this does not imply a target-page status."""
    delay = _retry_after_ms(headers)
    if status in {401, 403}:
        return WebAuthenticationError(status, cause=cause)
    if status == 429:
        return WebRateLimitedError(status, delay, cause=cause)
    if status == 408 or status >= 500:
        return WebServiceUnavailableError(status, delay, cause=cause)
    return WebRequestRejectedError(status, cause=cause)


FailureClassifier = Callable[[BaseException], WebAcquisitionError | None]


def acquisition_error(
    error: BaseException, *, classify: FailureClassifier | None = None
) -> WebAcquisitionError:
    """Preserve available exception facts without interpreting diagnostic text."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, WebAcquisitionError):
            return current
        if classify is not None:
            classified = classify(current)
            if classified is not None:
                classified.__cause__ = error
                return classified
        if isinstance(
            current, (TimeoutError, httpx.TimeoutException, requests.Timeout)
        ):
            return WebTimeoutError(cause=error)
        if isinstance(
            current, (ConnectionError, httpx.RequestError, requests.ConnectionError)
        ):
            return WebConnectionError(cause=error)
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", getattr(current, "status_code", None))
        if type(status) is int and 400 <= status <= 599:
            return http_failure(status, getattr(response, "headers", None), cause=error)
        if isinstance(current, (json.JSONDecodeError, UnicodeError)):
            return WebInvalidResponseError(cause=error)
        if isinstance(current, ImportError):
            return WebDependencyMissingError(cause=error)
        current = current.__cause__
    return WebUnclassifiedFailure(cause=error)
