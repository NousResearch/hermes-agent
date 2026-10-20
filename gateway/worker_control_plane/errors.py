"""Domain errors exposed through a single safe envelope."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class WorkerControlPlaneError(Exception):
    code: str
    status: int = 400
    retryable: bool = False
    message: str = "Request rejected"

ERRORS = {
 "invalid_credential": (401, "Authentication failed"), "worker_revoked": (403, "Worker is revoked"),
 "worker_not_authorized": (403, "Worker is not authorized"), "task_not_found": (404, "Task not found"),
 "duplicate_active_instance": (409, "An active instance already exists"), "state_conflict": (409, "State conflict"),
 "idempotency_conflict": (409, "Idempotency key conflicts with the original request"),
 "stale_delivery": (409, "Stale delivery"), "registration_expired": (410, "Registration expired"),
 "lease_expired": (410, "Lease expired"), "payload_too_large": (413, "Payload too large"),
 "unsupported_protocol": (422, "Unsupported protocol"), "unsupported_capability": (422, "Unsupported capability"),
 "invalid_task_payload": (422, "Invalid task payload"), "invalid_result": (422, "Invalid result"),
}
def error(code: str, *, status: int | None = None, message: str | None = None) -> WorkerControlPlaneError:
    default_status, default_message = ERRORS.get(code, (400, "Malformed request"))
    return WorkerControlPlaneError(code, status or default_status, False, message or default_message)
