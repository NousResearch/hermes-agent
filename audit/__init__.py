"""Audit logging package following K8s audit.k8s.io/v1 patterns.

Provides K8s-style audit logging for hermes-agent with:
- Four audit levels: None, Metadata, Request, RequestResponse
- First-match-wins policy rules
- Stage-based request lifecycle (RequestReceived, ResponseComplete)
- Async batched JSONL file backend with rotation
- SHA256 hash chain for tamper protection

Usage:
    from audit import setup_audit_logging, get_audit_engine

    # Initialize from config
    engine = setup_audit_logging(audit_config)

    # Or use directly
    from audit import AuditEngine, AuditPolicy, OperationContext
    from audit.events import Stage
    from audit.levels import AuditLevel

    context = OperationContext(
        user="wecom:zhangsan",
        verb="execute",
        resource="docker",
        operation_type="Mutate",
        platform="wecom",
        channel="group:ops",
        session_id="abc123",
    )

    engine = get_audit_engine()
    engine.emit_request_received(context, request_body={"command": "docker ps"})
    # ... execute operation ...
    engine.emit_response_complete(context, response_body={"result": "ok"}, response_code=0)
"""

from audit.backends.base import AuditBackend
from audit.chain import AuditHashChain
from audit.engine import (
    AuditEngine,
    get_audit_engine,
    setup_audit_logging,
)
from audit.events import AuditEvent, OperationContext, Stage
from audit.levels import AuditLevel
from audit.policy import AuditPolicy, GroupVersionResource, PolicyRule

__all__ = [
    # Core
    "AuditEngine",
    "get_audit_engine",
    "setup_audit_logging",
    # Events
    "AuditEvent",
    "OperationContext",
    "Stage",
    # Levels
    "AuditLevel",
    # Policy
    "AuditPolicy",
    "PolicyRule",
    "GroupVersionResource",
    # Backends
    "AuditBackend",
    "AuditHashChain",
]