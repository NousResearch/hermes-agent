"""Closed approval-gate vocabulary shared by producers and result consumers."""
APPROVAL_OUTCOME_NOTICES = {
    "denied": "The user explicitly denied an approval request. Do not retry or bypass it; ask for a new authorization only if the user requests the action again.",
    "cancelled": "The approval request was withdrawn before the user answered. It is no longer live; obtain fresh authorization before retrying.",
    "timeout": "The approval request expired without a user response. It is no longer live; obtain fresh authorization before retrying.",
    "notify_failed": "The approval request could not be delivered to the user. Do not retry or bypass it; obtain fresh authorization before retrying.",
    "blocked": "The action was blocked because approval could not be obtained in this context. Do not retry or bypass it; obtain fresh authorization before retrying.",
}
APPROVAL_OUTCOMES = frozenset(APPROVAL_OUTCOME_NOTICES)
