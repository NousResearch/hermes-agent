"""Shared recovery-policy primitives for agent/runtime failures.

This module is the first foundation block for the broader
"autonomous operator" work:
- normalize common failure shapes across API and tool paths
- map them to explicit next actions
- keep the policy data-first and easily testable before wiring it into
  higher-level loops

The existing ``agent.error_classifier`` already does deep provider/API
classification.  This module sits one layer above it and answers the
practical question: *what should Hermes try next?*
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from agent.error_classifier import ClassifiedError, FailoverReason


class FailureClass(enum.Enum):
    """Cross-runtime failure buckets suitable for recovery policy."""

    auth = "auth"
    billing = "billing"
    rate_limit = "rate_limit"
    overloaded = "overloaded"
    timeout = "timeout"
    server_error = "server_error"
    context_overflow = "context_overflow"
    payload_too_large = "payload_too_large"
    content_policy = "content_policy"
    tool_unavailable = "tool_unavailable"
    tool_failed = "tool_failed"
    empty_result = "empty_result"
    parse_error = "parse_error"
    unsupported = "unsupported"
    unknown = "unknown"


class RecoveryAction(enum.Enum):
    """Preferred next move after a classified failure."""

    retry_same = "retry_same"
    retry_smaller_scope = "retry_smaller_scope"
    compress_context = "compress_context"
    rotate_credential = "rotate_credential"
    fallback_provider = "fallback_provider"
    fallback_tool = "fallback_tool"
    degrade_mode = "degrade_mode"
    ask_user = "ask_user"
    abort = "abort"


@dataclass(frozen=True)
class RecoveryDecision:
    """A small, explicit recovery-plan object."""

    failure_class: FailureClass
    primary_action: RecoveryAction
    reason: str
    retryable: bool = False
    secondary_actions: tuple[RecoveryAction, ...] = field(default_factory=tuple)


_API_REASON_TO_FAILURE_CLASS: dict[FailoverReason, FailureClass] = {
    FailoverReason.auth: FailureClass.auth,
    FailoverReason.auth_permanent: FailureClass.auth,
    FailoverReason.billing: FailureClass.billing,
    FailoverReason.rate_limit: FailureClass.rate_limit,
    FailoverReason.overloaded: FailureClass.overloaded,
    FailoverReason.server_error: FailureClass.server_error,
    FailoverReason.timeout: FailureClass.timeout,
    FailoverReason.context_overflow: FailureClass.context_overflow,
    FailoverReason.payload_too_large: FailureClass.payload_too_large,
    FailoverReason.image_too_large: FailureClass.payload_too_large,
    FailoverReason.provider_policy_blocked: FailureClass.unsupported,
    FailoverReason.content_policy_blocked: FailureClass.content_policy,
    FailoverReason.model_not_found: FailureClass.unsupported,
    FailoverReason.format_error: FailureClass.parse_error,
    FailoverReason.invalid_encrypted_content: FailureClass.parse_error,
    FailoverReason.multimodal_tool_content_unsupported: FailureClass.unsupported,
    FailoverReason.thinking_signature: FailureClass.parse_error,
    FailoverReason.long_context_tier: FailureClass.unsupported,
    FailoverReason.oauth_long_context_beta_forbidden: FailureClass.unsupported,
    FailoverReason.llama_cpp_grammar_pattern: FailureClass.unsupported,
    FailoverReason.unknown: FailureClass.unknown,
}


def decide_api_recovery(error: ClassifiedError) -> RecoveryDecision:
    """Map a classified provider/API error to a concrete recovery decision."""

    failure_class = _API_REASON_TO_FAILURE_CLASS.get(error.reason, FailureClass.unknown)

    if error.reason == FailoverReason.context_overflow:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.compress_context,
            secondary_actions=(RecoveryAction.retry_smaller_scope,),
            retryable=True,
            reason="Prompt/context exceeded the provider limit.",
        )

    if error.reason in {FailoverReason.payload_too_large, FailoverReason.image_too_large}:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.retry_smaller_scope,
            secondary_actions=(RecoveryAction.compress_context,),
            retryable=True,
            reason="Request payload is too large; shrink the payload before retrying.",
        )

    if error.reason in {FailoverReason.auth, FailoverReason.billing}:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.rotate_credential,
            secondary_actions=(RecoveryAction.fallback_provider,),
            retryable=error.should_fallback,
            reason="Credential/account path failed; rotate first, then fall back to another provider if needed.",
        )

    if error.reason == FailoverReason.rate_limit:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.fallback_provider,
            secondary_actions=(RecoveryAction.retry_same,),
            retryable=True,
            reason="Provider is temporarily rate-limited; prefer another provider before retrying the same one.",
        )

    if error.reason in {FailoverReason.overloaded, FailoverReason.server_error, FailoverReason.timeout}:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.retry_same,
            secondary_actions=(RecoveryAction.fallback_provider,),
            retryable=True,
            reason="Transient provider failure; retry once, then fall back if the path keeps failing.",
        )

    if error.reason in {
        FailoverReason.provider_policy_blocked,
        FailoverReason.model_not_found,
        FailoverReason.multimodal_tool_content_unsupported,
        FailoverReason.long_context_tier,
        FailoverReason.oauth_long_context_beta_forbidden,
        FailoverReason.llama_cpp_grammar_pattern,
    }:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.degrade_mode,
            secondary_actions=(RecoveryAction.fallback_provider,),
            retryable=True,
            reason="Current provider/path is unsupported for this request shape; degrade or switch paths.",
        )

    if error.reason in {
        FailoverReason.format_error,
        FailoverReason.invalid_encrypted_content,
        FailoverReason.thinking_signature,
    }:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.degrade_mode,
            secondary_actions=(RecoveryAction.retry_same,),
            retryable=True,
            reason="Request/response formatting can often be repaired in-place before retrying.",
        )

    if error.reason == FailoverReason.content_policy_blocked:
        return RecoveryDecision(
            failure_class=failure_class,
            primary_action=RecoveryAction.ask_user,
            retryable=False,
            reason="The provider blocked this request on policy grounds; do not retry unchanged.",
        )

    return RecoveryDecision(
        failure_class=failure_class,
        primary_action=RecoveryAction.retry_same if error.retryable else RecoveryAction.ask_user,
        secondary_actions=(RecoveryAction.fallback_provider,) if error.should_fallback else tuple(),
        retryable=error.retryable,
        reason="Unknown provider failure; start with the least-destructive retry path.",
    )


_TOOL_UNAVAILABLE_PATTERNS = (
    "not available",
    "not enabled",
    "unknown tool",
    "missing required",
    "requires",
)

_TOOL_PARSE_PATTERNS = (
    "json",
    "schema",
    "parse",
    "invalid arguments",
)


def decide_tool_recovery(tool_name: str, result: Optional[str]) -> RecoveryDecision:
    """Classify a Hermes tool result into a conservative recovery decision.

    The tool ecosystem is intentionally heterogeneous, so this function only
    treats obviously structured failures as signals.  Ambiguous plain text is
    left in the ``unknown`` bucket rather than over-classifying.
    """

    data = _parse_json_object(result)
    if data is not None:
        if data.get("success") is False:
            message = _tool_message(data)
            return _decision_from_tool_message(tool_name, message)
        if _has_nonzero_exit_code(data):
            return RecoveryDecision(
                failure_class=FailureClass.tool_failed,
                primary_action=RecoveryAction.degrade_mode,
                secondary_actions=(RecoveryAction.fallback_tool,),
                retryable=True,
                reason=f"{tool_name} exited non-zero; prefer a safer variant or fallback tool.",
            )
        if _looks_empty(data):
            return RecoveryDecision(
                failure_class=FailureClass.empty_result,
                primary_action=RecoveryAction.retry_smaller_scope,
                secondary_actions=(RecoveryAction.fallback_tool,),
                retryable=True,
                reason=f"{tool_name} returned no useful payload; narrow the query or switch tools.",
            )
        return RecoveryDecision(
            failure_class=FailureClass.unknown,
            primary_action=RecoveryAction.retry_same,
            retryable=True,
            reason=f"{tool_name} produced structured output with no obvious failure marker.",
        )

    text = (result or "").strip()
    if not text:
        return RecoveryDecision(
            failure_class=FailureClass.empty_result,
            primary_action=RecoveryAction.retry_smaller_scope,
            secondary_actions=(RecoveryAction.fallback_tool,),
            retryable=True,
            reason=f"{tool_name} returned an empty result.",
        )

    lowered = text.lower()
    if any(p in lowered for p in _TOOL_UNAVAILABLE_PATTERNS):
        return RecoveryDecision(
            failure_class=FailureClass.tool_unavailable,
            primary_action=RecoveryAction.fallback_tool,
            secondary_actions=(RecoveryAction.ask_user,),
            retryable=False,
            reason=f"{tool_name} appears unavailable in this environment.",
        )
    if any(p in lowered for p in _TOOL_PARSE_PATTERNS):
        return RecoveryDecision(
            failure_class=FailureClass.parse_error,
            primary_action=RecoveryAction.degrade_mode,
            secondary_actions=(RecoveryAction.retry_same,),
            retryable=True,
            reason=f"{tool_name} failed because of argument/format issues that may be fixable.",
        )

    return RecoveryDecision(
        failure_class=FailureClass.unknown,
        primary_action=RecoveryAction.retry_same,
        secondary_actions=(RecoveryAction.fallback_tool,),
        retryable=True,
        reason=f"{tool_name} returned unstructured output; retry once, then use a different tool path.",
    )


def _parse_json_object(result: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not isinstance(result, str) or not result.strip():
        return None
    try:
        parsed = json.loads(result)
    except Exception:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _has_nonzero_exit_code(data: Mapping[str, Any]) -> bool:
    code = data.get("exit_code")
    return isinstance(code, int) and code != 0


def _looks_empty(data: Mapping[str, Any]) -> bool:
    if isinstance(data.get("results"), list) and len(data["results"]) == 0:
        return True
    if isinstance(data.get("matches"), list) and len(data["matches"]) == 0:
        return True
    if isinstance(data.get("files"), list) and len(data["files"]) == 0:
        return True
    content = data.get("content")
    if isinstance(content, str) and not content.strip():
        return True
    output = data.get("output")
    if isinstance(output, str) and not output.strip() and data.get("exit_code") == 0:
        return True
    return False


def _tool_message(data: Mapping[str, Any]) -> str:
    for key in ("error", "message", "detail", "output"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _decision_from_tool_message(tool_name: str, message: str) -> RecoveryDecision:
    lowered = (message or "").lower()
    if any(p in lowered for p in _TOOL_UNAVAILABLE_PATTERNS):
        return RecoveryDecision(
            failure_class=FailureClass.tool_unavailable,
            primary_action=RecoveryAction.fallback_tool,
            secondary_actions=(RecoveryAction.ask_user,),
            retryable=False,
            reason=f"{tool_name} is unavailable or misconfigured.",
        )
    if any(p in lowered for p in _TOOL_PARSE_PATTERNS):
        return RecoveryDecision(
            failure_class=FailureClass.parse_error,
            primary_action=RecoveryAction.degrade_mode,
            secondary_actions=(RecoveryAction.retry_same,),
            retryable=True,
            reason=f"{tool_name} rejected the request shape; adjust arguments and retry.",
        )
    return RecoveryDecision(
        failure_class=FailureClass.tool_failed,
        primary_action=RecoveryAction.degrade_mode,
        secondary_actions=(RecoveryAction.fallback_tool,),
        retryable=True,
        reason=f"{tool_name} reported a structured failure.",
    )
