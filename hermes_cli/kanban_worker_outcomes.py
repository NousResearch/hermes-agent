"""Exit protocol for retryable and terminal Kanban provider failures."""

# Worker exit "provider rate-limited": released WITHOUT counting a failure (the
# breaker must never trip on a throttle). 75 == BSD EX_TEMPFAIL.
KANBAN_RATE_LIMIT_EXIT_CODE = 75
KANBAN_PROVIDER_UNAVAILABLE_EXIT_CODE = 76  # transient transport/server failure, not quota

# Worker exit "provider rejected the configuration": credential revoked (401/403), model gone
# (404), TLS chain broken — a retry cannot fix it, so the dispatcher parks the card blocked on
# the FIRST occurrence instead of spending ``failure_limit`` identical spawns. 78 == BSD EX_CONFIG.
KANBAN_TERMINAL_PROVIDER_EXIT_CODE = 78


def exit_code_kind(code: int) -> tuple[str, int]:
    """``(kind, code)`` for a worker's exit code, however it was observed."""
    if code == 0:
        return "clean_exit", code
    if code == KANBAN_RATE_LIMIT_EXIT_CODE:
        return "rate_limited", code
    if code == KANBAN_PROVIDER_UNAVAILABLE_EXIT_CODE:
        return "provider_unavailable", code
    if code == KANBAN_TERMINAL_PROVIDER_EXIT_CODE:
        return "terminal_provider", code
    return "nonzero_exit", code
