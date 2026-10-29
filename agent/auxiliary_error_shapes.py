"""Shape readers for auxiliary-call errors: where an SDK error carries its HTTP status, and the
status-less failures (HTTP-200 malformed answers, SSE ``error`` events) that still end a route.

botocore's ``ClientError`` keeps its status in ``response["ResponseMetadata"]`` rather than on an
attribute; the Bedrock client makes one attempt per call, so the auxiliary recovery ladder has to
read it there to retry a 503 or treat a ``ThrottlingException`` as a rate limit.
"""

from __future__ import annotations

from typing import Any


def _botocore_error(exc: Exception) -> tuple[Any, Any]:
    """``(HTTP status, error code)`` of a botocore ``ClientError``, whose ``response`` is the parsed
    dict; ``(None, None)`` for any other exception. The Bedrock client makes one attempt per call,
    so these classifiers own its retries the way they own the OpenAI SDK's (``max_retries=0``)."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None, None
    return response.get("ResponseMetadata", {}).get("HTTPStatusCode"), response.get("Error", {}).get("Code")


def _exc_http_status(exc: Exception) -> Any:
    """HTTP status on the exception itself, its ``response``, or a botocore error's response metadata."""
    return (getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            or _botocore_error(exc)[0])


def _is_invalid_aux_response_error(exc: Exception) -> bool:
    """HTTP-200 empty/malformed ChatCompletions — a capability failure routed like model incompatibility."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "auxiliary " in msg and "llm returned invalid response" in msg and "choices[0].message" in msg


def _is_statusless_structured_provider_error(exc: Exception) -> bool:
    """Detect a structured provider failure that has no HTTP status.

    OpenAI-compatible relays may commit SSE with status 200, then send an
    OpenAI-style ``error`` event. The SDK raises a status-less ``APIError`` with
    ``body=data["error"]`` — the INNER error object or a bare string (an
    ``{"error": ...}`` wrapper is accepted too). Any non-empty structured error in
    that status-less shape is a route failure; ordinary HTTP errors keep their
    existing status-based classifiers, and message text alone is insufficient.
    """
    if _exc_http_status(exc) is not None:
        return False
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) and "error" in body else body
    if isinstance(err, str):
        return bool(err.strip())
    return isinstance(err, dict) and any(err.get(k) for k in ("type", "code", "message"))
