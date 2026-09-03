"""Bounded, transport-agnostic review checkpoint runner.

This module intentionally does not resolve credentials or call a provider on
its own.  A trusted backend supplies both operations.  The runner's job is to
enforce the review contract around one exact, no-tools completion call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any


_MAX_PACKET_BYTES = 64 * 1024
_VALID_PHASES = frozenset({"plan", "recovery", "final"})
_VALID_VERDICTS = frozenset({"PASS", "REVISE", "ASK_USER", "BLOCK"})
_REDACTED = "[REDACTED]"
_SECRET_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_KEY_RE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")


@dataclass(frozen=True)
class ReviewRequest:
    """A bounded checkpoint packet plus an exact requested review route."""

    checkpoint_id: str
    session_id: str
    phase: str
    objective: str
    constraints: tuple[str, ...]
    candidate: Mapping[str, Any]
    provider: str
    model: str
    main_provider: str
    main_model: str
    attempt: int = 0
    credential_policy: str = "subscription_oauth_only"
    fallback_policy: str = "none"
    require_distinct_from_main: bool = True
    timeout_seconds: float = 45.0


@dataclass(frozen=True)
class ResolvedReviewRoute:
    """Sanitized route provenance plus a backend-owned opaque credential."""

    profile: str
    provider: str
    model: str
    credential_kind: str
    credential_handle: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ReviewResult:
    """Outcome safe to return to the caller or display in the shell."""

    checkpoint_id: str
    status: str
    verdict: str | None = None
    summary: str = ""
    feedback: tuple[Any, ...] = ()
    actual_route: Mapping[str, str] | None = None
    usage: Mapping[str, Any] | None = None
    unavailable_reason: str | None = None


def _actual_route(route: ResolvedReviewRoute) -> dict[str, str]:
    return {
        "profile": route.profile,
        "provider": route.provider,
        "model": route.model,
        "credential_kind": route.credential_kind,
    }


def _unavailable(
    request: ReviewRequest,
    reason: str,
    *,
    summary: str = "",
    status: str = "unavailable",
    actual_route: Mapping[str, str] | None = None,
) -> ReviewResult:
    return ReviewResult(
        checkpoint_id=request.checkpoint_id,
        status=status,
        summary=summary,
        actual_route=actual_route,
        unavailable_reason=reason,
    )


def _is_secret_key(key: object) -> bool:
    normalized = str(key).casefold().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_secret_key(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_error(error: Exception) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ")[:300]
    message = _BEARER_RE.sub("Bearer [REDACTED]", message)
    return _KEY_RE.sub(_REDACTED, message)


def _validate_request(request: ReviewRequest) -> str | None:
    if request.credential_policy != "subscription_oauth_only":
        return "unsupported_policy"
    if request.fallback_policy != "none":
        return "unsupported_policy"
    if (
        not request.checkpoint_id.strip()
        or not request.session_id.strip()
        or request.phase not in _VALID_PHASES
        or not request.objective.strip()
        or not request.provider.strip()
        or not request.model.strip()
        or request.attempt < 0
        or request.timeout_seconds <= 0
    ):
        return "invalid_request"
    return None


def _build_messages(request: ReviewRequest) -> tuple[list[dict[str, str]], int]:
    packet = _redact({
        "checkpoint_id": request.checkpoint_id,
        "session_id": request.session_id,
        "phase": request.phase,
        "attempt": request.attempt,
        "objective": request.objective,
        "constraints": request.constraints,
        "candidate": request.candidate,
    })
    serialized = json.dumps(
        packet,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Review the bounded checkpoint packet. Do not execute actions "
                "or request tools. Return only one JSON object, with no markdown "
                "or surrounding text, using this exact shape: "
                '{"verdict":"PASS|REVISE|ASK_USER|BLOCK",'
                '"summary":"short explanation","feedback":[]}. '
                "The verdict must be exactly PASS, REVISE, ASK_USER, or BLOCK."
            ),
        },
        {"role": "user", "content": serialized},
    ]
    return messages, len(serialized.encode("utf-8"))


def run_review(
    request: ReviewRequest,
    *,
    resolve_route: Callable[..., ResolvedReviewRoute],
    complete: Callable[..., Mapping[str, Any]],
) -> ReviewResult:
    """Run one strict review completion, or return a truthful unavailable state."""

    invalid_reason = _validate_request(request)
    if invalid_reason is not None:
        return _unavailable(request, invalid_reason)

    messages, packet_bytes = _build_messages(request)
    if packet_bytes > _MAX_PACKET_BYTES:
        return _unavailable(request, "packet_too_large")

    try:
        route = resolve_route(
            provider=request.provider,
            model=request.model,
            credential_policy=request.credential_policy,
            fallback_policy=request.fallback_policy,
        )
    except Exception as error:
        return _unavailable(
            request,
            "route_unavailable",
            summary=f"Review route unavailable: {_safe_error(error)}",
        )

    attestation = _actual_route(route)
    if route.credential_kind != "subscription_oauth":
        return _unavailable(
            request,
            "credential_policy_mismatch",
            actual_route=attestation,
        )
    if route.provider != request.provider or route.model != request.model:
        return _unavailable(
            request,
            "route_mismatch",
            actual_route=attestation,
        )
    if request.require_distinct_from_main and (
        route.provider == request.main_provider and route.model == request.main_model
    ):
        return _unavailable(
            request,
            "review_route_matches_main",
            actual_route=attestation,
        )

    try:
        response = complete(
            credential_handle=route.credential_handle,
            provider=route.provider,
            model=route.model,
            messages=messages,
            tools=[],
            tool_choice="none",
            timeout=request.timeout_seconds,
            idempotency_key=request.checkpoint_id,
        )
    except TimeoutError as error:
        return _unavailable(
            request,
            "timeout",
            status="timed_out",
            summary=_safe_error(error),
            actual_route=attestation,
        )
    except Exception as error:
        return _unavailable(
            request,
            "completion_failed",
            summary=_safe_error(error),
            actual_route=attestation,
        )

    if not isinstance(response, Mapping) or response.get("verdict") not in _VALID_VERDICTS:
        return _unavailable(
            request,
            "invalid_response",
            actual_route=attestation,
        )

    feedback = response.get("feedback", ())
    if not isinstance(feedback, Sequence) or isinstance(feedback, (str, bytes)):
        feedback = ()
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        usage = None

    return ReviewResult(
        checkpoint_id=request.checkpoint_id,
        status="completed",
        verdict=str(response["verdict"]),
        summary=str(response.get("summary", "")),
        feedback=tuple(feedback),
        actual_route=attestation,
        usage=dict(usage) if usage is not None else None,
    )
