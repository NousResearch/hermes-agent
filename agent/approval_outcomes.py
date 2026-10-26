"""Closed approval-gate vocabulary shared by producers and result consumers."""
APPROVAL_OUTCOME_NOTICES = {
    "denied": "The user explicitly denied an approval request. Do not retry or bypass it; ask for a new authorization only if the user requests the action again.",
    "cancelled": "The approval request was withdrawn before the user answered. It is no longer live; obtain fresh authorization before retrying.",
    "timeout": "The approval request expired without a user response. It is no longer live; obtain fresh authorization before retrying.",
    "notify_failed": "The approval request could not be delivered to the user. Do not retry or bypass it; obtain fresh authorization before retrying.",
    "blocked": "The action was blocked because approval could not be obtained in this context. Do not retry or bypass it; obtain fresh authorization before retrying.",
}
APPROVAL_OUTCOMES = frozenset(APPROVAL_OUTCOME_NOTICES)


# This module is dependency-free: both CLI resolution and agent dispatch import it.
# These types are internal provenance, not a plugin protocol or a Python sandbox.
class _NativeApprovalBlock(str):
    """String-compatible rejection minted only by the native pre-tool gate resolver."""

    approval_outcome: str

    def __new__(cls, message, outcome):
        if type(outcome) is not str or outcome not in APPROVAL_OUTCOMES:
            raise ValueError("Invalid native approval outcome")
        obj = super().__new__(cls, message)
        obj.approval_outcome = outcome
        return obj

    def __reduce__(self):
        return type(self), (str(self), self.approval_outcome)


class _NativeApprovalResult(str):
    """Serialized tool error retaining native provenance until the live row is built."""

    approval_outcome: str

    def __new__(cls, content, outcome):
        obj = super().__new__(cls, content)
        obj.approval_outcome = outcome
        return obj

    def __reduce__(self):
        return type(self), (str(self), self.approval_outcome)


def native_approval_outcome(value):
    """Never promote JSON, duck-typed attributes, or subclasses into trusted status."""
    if type(value) not in (_NativeApprovalBlock, _NativeApprovalResult):
        return None
    outcome = value.approval_outcome
    return outcome if type(outcome) is str and outcome in APPROVAL_OUTCOMES else None


def pre_tool_block_result(message, *, error_builder=None):
    """Serialize a block, preserving the caller's error builder and native provenance."""
    import json

    extra = {}
    outcome = native_approval_outcome(message) if type(message) is _NativeApprovalBlock else None
    if outcome is not None:
        extra["approval_outcome"] = outcome
    result = (error_builder(message, **extra) if error_builder is not None else
              json.dumps({"error": str(message), **extra}, ensure_ascii=False))
    return _NativeApprovalResult(result, outcome) if outcome is not None else result
