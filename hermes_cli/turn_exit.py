"""Turn-result exit policy shared by CLI views and authority-owned workers.

Worker identity is supplied by the caller, not inferred from a daemon's environment.
"""
from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes of a finite run (``hermes -z`` / ``hermes chat -q``); documented in
    website/docs/reference/cli-commands.md#exit-codes-for-one-shot-runs."""
    OK = 0
    FAILED = 1
    USAGE_OR_PARTIAL = 2
    INPUT_REQUIRED = 3
    APPROVAL_PENDING = 5
    KANBAN_RETRY = 75
    KANBAN_TERMINAL = 78
    TIMEOUT = 124
    INTERRUPTED = 130


# What each code means, per entry point. `chat -q` and `-z` keep their historical, different 1/2
# meanings; 5 and 124 come only from the opt-in --on-approval fail / --timeout flags.
EXIT_CODES = {
    ExitCode.OK: "the turn completed",
    ExitCode.FAILED: "chat -q: the turn failed, stopped partway or never ran; -z: it completed with no reply text",
    ExitCode.USAGE_OR_PARTIAL: "usage error before the run started; -z: the turn failed or stopped partway",
    ExitCode.INPUT_REQUIRED: "the session needs a person (a pending prompt or a lost turn); detached without cancelling",
    ExitCode.APPROVAL_PENDING: "--on-approval fail: a flagged action needed approval; it was denied and the run stopped",
    ExitCode.KANBAN_RETRY: "Kanban worker only: a retryable provider wall (rate limit, overload, 5xx, billing)",
    ExitCode.KANBAN_TERMINAL: "Kanban worker only: a provider failure that needs an operator (auth, missing model)",
    ExitCode.TIMEOUT: "--timeout: the run did not finish in time and was interrupted",
    ExitCode.INTERRUPTED: "the turn was interrupted (Ctrl-C, /stop, gateway restart)",
}

# Provider walls do not spend a Kanban task's retry budget.
TRANSIENT_PROVIDER_REASONS = frozenset({
    "rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout",
})
# These failures need an operator change; the dispatcher parks the task immediately.
TERMINAL_PROVIDER_REASONS = frozenset({
    "auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked",
})


def turn_exit_code(
    result, *, kanban_worker: bool, credentials_rate_limited: bool = False,
    credentials_terminal: bool = False,
    transient_reasons=TRANSIENT_PROVIDER_REASONS, terminal_reasons=TERMINAL_PROVIDER_REASONS,
) -> int:
    """Preserve completion, interruption and retryable/terminal provider outcomes."""
    if not isinstance(result, dict):
        if credentials_rate_limited and kanban_worker:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
            return KANBAN_RATE_LIMIT_EXIT_CODE
        if credentials_terminal and kanban_worker:
            from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
            return KANBAN_TERMINAL_PROVIDER_EXIT_CODE
        return 1
    if result.get("interrupted"):
        return 130
    if not (result.get("failed") or result.get("partial") or result.get("completed") is False):
        return 0
    if kanban_worker:
        reason = result.get("failure_reason")
        if reason in transient_reasons:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
            return KANBAN_RATE_LIMIT_EXIT_CODE
        if reason in terminal_reasons:
            from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
            return KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    return 1


def credential_failure_flags(exc) -> dict:
    """``turn_exit_code`` credential kwargs for an exception raised before any turn result exists.

    Same predicates as the one-shot CLI's ``_ensure_runtime_credentials``: a quota/rate-limit
    ``AuthError`` is transient, only an explicit re-authentication requirement is terminal. The
    cause chain is walked because agent construction wraps the resolver's error."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, AuthError):
            return {"credentials_rate_limited": is_rate_limited_auth_error(exc),
                    "credentials_terminal": bool(exc.relogin_required)}
        exc = exc.__cause__ or exc.__context__
    return {}
