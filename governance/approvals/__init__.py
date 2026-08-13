"""Approval state machine for KENSEI triage.

Public API:
    ApprovalLedger       — state machine for PENDING → APPROVED/REJECTED/AMENDING
    ApprovalWorkflowManager — orchestrates approval actions + Kanban task creation
    generate_fingerprint — SHA-256 fingerprint for dedup detection
"""

from governance.approvals.ledger import ApprovalLedger, generate_fingerprint

# ApprovalWorkflowManager requires _board_compat and kanban_db at import time.
# Import lazily to avoid ModuleNotFoundError in test environments where these
# dependencies are not available. Use:
#   from governance.approvals.manager import ApprovalWorkflowManager
# explicitly when needed.
def __getattr__(name):
    if name == "ApprovalWorkflowManager":
        from governance.approvals.manager import ApprovalWorkflowManager
        return ApprovalWorkflowManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")