"""Strict wire helpers for certified Context Governor subprocess calls."""

from __future__ import annotations

import json
import re
from typing import Any

FAILURE_SCHEMA = "ContextGovernorFailureV1"
FAILURE_FLAG = "--failure-envelope-v1"
FAILURE_STREAM = "stderr"
_MAX_FAILURE_BYTES = 16 * 1024
_MAX_DETAILS_BYTES = 8 * 1024
_CODE_RE = re.compile(r"^[a-z0-9_]{1,128}$")


class ContextGovernorProtocolError(RuntimeError):
    """The governor binary violated the negotiated machine contract."""


class ContextGovernorCancellationIndeterminate(ContextGovernorProtocolError):
    """A timed-out subprocess could not be proven quiescent."""


class ContextGovernorCommandError(RuntimeError):
    """A certified governor command returned a typed, machine-readable failure."""

    def __init__(self, operation: str, code: str, details: dict[str, Any]) -> None:
        self.operation = operation
        self.code = code
        self.details = details
        super().__init__(f"context-governor {operation} failed [{code}]")


def require_failure_capability(capabilities: dict[str, Any]) -> None:
    expected = {
        "schema": FAILURE_SCHEMA,
        "flag": FAILURE_FLAG,
        "stream": FAILURE_STREAM,
    }
    if capabilities.get("failure_envelope") != expected:
        raise ContextGovernorProtocolError(
            "governor does not advertise the required typed failure contract"
        )


def parse_failure_envelope(stderr: str, expected_operation: str) -> ContextGovernorCommandError:
    encoded = stderr.encode("utf-8", errors="replace")
    if not stderr or len(encoded) > _MAX_FAILURE_BYTES:
        raise ContextGovernorProtocolError("invalid Context Governor failure envelope size")
    try:
        value = json.loads(stderr)
    except json.JSONDecodeError as exc:
        raise ContextGovernorProtocolError(
            "Context Governor returned non-JSON certified failure output"
        ) from exc
    if not isinstance(value, dict) or set(value) - {"schema", "operation", "code", "details"}:
        raise ContextGovernorProtocolError("invalid Context Governor failure envelope shape")
    if value.get("schema") != FAILURE_SCHEMA or value.get("operation") != expected_operation:
        raise ContextGovernorProtocolError("Context Governor failure envelope identity mismatch")
    code = value.get("code")
    details = value.get("details", {})
    if not isinstance(code, str) or _CODE_RE.fullmatch(code) is None:
        raise ContextGovernorProtocolError("invalid Context Governor failure code")
    if not isinstance(details, dict) or len(details) > 32:
        raise ContextGovernorProtocolError("invalid Context Governor failure details")
    try:
        details_size = len(
            json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    except (TypeError, ValueError) as exc:
        raise ContextGovernorProtocolError("unserializable Context Governor failure details") from exc
    if details_size > _MAX_DETAILS_BYTES:
        raise ContextGovernorProtocolError("Context Governor failure details exceed bound")
    return ContextGovernorCommandError(expected_operation, code, details)


def validated_finalized_tokens(receipt: Any, target_tokens: int) -> int:
    if not isinstance(receipt, dict):
        raise ContextGovernorProtocolError("finalize-v2 returned no receipt object")
    value = receipt.get("compacted_approx_tokens")
    if type(value) is not int or value < 0:
        raise ContextGovernorProtocolError(
            "finalize-v2 returned an invalid compacted_approx_tokens field"
        )
    if value > target_tokens:
        raise ContextGovernorProtocolError(
            "finalize-v2 emitted a transcript above the admitted target"
        )
    return value
