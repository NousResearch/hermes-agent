"""Standard error taxonomy for Hermes OS runtime delegation."""

from dataclasses import dataclass


RUNTIME_UNAVAILABLE = "runtime_unavailable"
RUNTIME_TIMEOUT = "runtime_timeout"
VALIDATION_ERROR = "validation_error"
PERMISSION_DENIED = "permission_denied"
TOOL_FAILURE = "tool_failure"
PROCESS_ERROR = "process_error"
STATE_CONFLICT = "state_conflict"

RETRYABLE_CODES = {
    RUNTIME_UNAVAILABLE,
    RUNTIME_TIMEOUT,
    PROCESS_ERROR,
}


@dataclass(frozen=True)
class AdapterError:
    code: str
    message: str
    retryable: bool = False


def adapter_error(code, message):
    return AdapterError(
        code=code,
        message=message,
        retryable=code in RETRYABLE_CODES,
    )
