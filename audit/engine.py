"""Audit engine - policy evaluation and event dispatch."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from audit.backends.base import AuditBackend
from audit.backends.log_backend import LogBackend
from audit.chain import AuditHashChain
from audit.events import AuditEvent, OperationContext, Stage
from audit.levels import AuditLevel
from audit.policy import AuditPolicy

logger = logging.getLogger(__name__)


class AuditEngine:
    """
    Central audit engine that:
    1. Evaluates audit policy to determine level for an operation
    2. Creates audit events at each stage
    3. Dispatches to configured backends
    4. Maintains hash chain for tamper protection
    """

    _instance: Optional["AuditEngine"] = None
    _lock = logging.getLogger(__name__ + "._lock")

    def __init__(
        self,
        policy: AuditPolicy,
        backends: List[AuditBackend],
        chain: Optional[AuditHashChain] = None,
    ):
        """
        Initialize audit engine.

        Args:
            policy: Audit policy with rules
            backends: List of backends to emit events to
            chain: Optional hash chain for tamper protection
        """
        self._policy = policy
        self._backends = backends
        self._chain = chain

    @classmethod
    def get_instance(cls) -> Optional["AuditEngine"]:
        """Get the singleton instance if initialized."""
        return cls._instance

    @classmethod
    def initialize(
        cls,
        policy: AuditPolicy,
        backends: List[AuditBackend],
        chain: Optional[AuditHashChain] = None,
    ) -> "AuditEngine":
        """
        Initialize the singleton audit engine.

        Args:
            policy: Audit policy with rules
            backends: List of backends to emit events to
            chain: Optional hash chain for tamper protection

        Returns:
            The initialized AuditEngine instance
        """
        cls._instance = cls(policy, backends, chain)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance:
            for backend in cls._instance._backends:
                backend.close()
        cls._instance = None

    @property
    def policy(self) -> AuditPolicy:
        return self._policy

    def evaluate_level(
        self,
        verb: str,
        resource: str,
        user: str = "",
        user_groups: Optional[List[str]] = None,
        namespace: str = "",
        op_type: str = "",
        channel: str = "",
    ) -> AuditLevel:
        """
        Evaluate the audit policy to determine the level for an operation.

        Args:
            verb: HTTP verb or operation type (get, list, create, etc.)
            resource: Resource or tool name
            user: Username
            user_groups: List of user groups
            namespace: Namespace (usually channel for hermes)
            op_type: Operation type (Mutate, Read)
            channel: Channel/chat identifier

        Returns:
            The audit level to use for this operation
        """
        return self._policy.evaluate_level(
            verb, resource, user, user_groups, namespace, op_type, channel
        )

    def emit(
        self,
        stage: Stage,
        context: OperationContext,
        level: AuditLevel,
        request_body: Any = None,
        response_body: Any = None,
        response_code: int = 0,
    ) -> None:
        """
        Emit an audit event at a specific stage.

        Args:
            stage: The stage in the request lifecycle
            context: Operation context
            level: Audit level (None means don't emit)
            request_body: Request payload (for Request/ResponseResponse levels)
            response_body: Response payload (for RequestResponse level)
            response_code: HTTP status code or RPC code
        """
        if level == AuditLevel.NONE:
            return

        # Build event
        event = self._build_event(
            stage, context, level, request_body, response_body, response_code
        )

        # Apply hash chain if enabled
        if self._chain:
            event = self._chain.write_entry(event)
        else:
            # Just add hash fields as empty if no chain
            event["prev_hash"] = ""
            event["hash"] = ""

        # Dispatch to all backends
        for backend in self._backends:
            try:
                backend.emit(event)
            except Exception as e:
                logger.warning("Audit backend %s failed: %s", backend, e)

    def _build_event(
        self,
        stage: Stage,
        context: OperationContext,
        level: AuditLevel,
        request_body: Any,
        response_body: Any,
        response_code: int,
    ) -> Dict[str, Any]:
        """Build an audit event dict from parameters."""
        import json

        event = AuditEvent(
            ts=AuditEvent.now_iso(),
            stage=stage.value,
            stage_ts=AuditEvent.now_iso(),
            user=context.user,
            user_groups=context.user_groups,
            platform=context.platform,
            channel=context.channel,
            session_id=context.session_id,
            verb=context.verb,
            resource=context.resource,
            op_type=context.op_type,
            source=context.source,
            event_id=AuditEvent.new_uuid(),
            response_code=response_code,
        )

        # Include request body only for appropriate levels
        if level.includes_request_body() and request_body is not None:
            if isinstance(request_body, str):
                event.request_body = request_body
            else:
                event.request_body = json.dumps(request_body, ensure_ascii=False)

        # Include response body only for RequestResponse level
        if level.includes_response_body() and response_body is not None:
            if isinstance(response_body, str):
                event.response_body = response_body
            else:
                event.response_body = json.dumps(response_body, ensure_ascii=False)

        return event.to_dict()

    def emit_request_received(
        self,
        context: OperationContext,
        request_body: Any = None,
    ) -> None:
        """Convenience method to emit a RequestReceived event."""
        level = self.evaluate_level(
            context.verb,
            context.resource,
            context.user,
            context.user_groups,
            "",
            context.op_type,
            context.channel,
        )
        self.emit(
            Stage.REQUEST_RECEIVED,
            context,
            level,
            request_body=request_body,
        )

    def emit_response_complete(
        self,
        context: OperationContext,
        response_body: Any = None,
        response_code: int = 0,
    ) -> None:
        """Convenience method to emit a ResponseComplete event."""
        level = self.evaluate_level(
            context.verb,
            context.resource,
            context.user,
            context.user_groups,
            "",
            context.op_type,
            context.channel,
        )
        self.emit(
            Stage.RESPONSE_COMPLETE,
            context,
            level,
            response_body=response_body,
            response_code=response_code,
        )


def get_audit_engine() -> Optional[AuditEngine]:
    """Get the current audit engine instance."""
    return AuditEngine.get_instance()


def setup_audit_logging(
    audit_config: Dict[str, Any],
    log_dir: Optional[Path] = None,
) -> Optional[AuditEngine]:
    """
    Set up audit logging from configuration.

    Args:
        audit_config: Audit configuration dict from YAML
        log_dir: Optional override for log directory

    Returns:
        Initialized AuditEngine or None if audit is disabled
    """
    import os
    from pathlib import Path

    # Check if audit is enabled
    enabled = audit_config.get("enabled", False)
    if not enabled:
        return None

    # Load policy
    policy_data = audit_config.get("policy", {})
    policy = AuditPolicy.from_dict({"policy": policy_data, "log": audit_config.get("log", {}), "tamper_protection": audit_config.get("tamper_protection", {})})

    # Determine log path
    log_path = audit_config.get("log", {}).get(
        "path", "~/.hermes/logs/audit/"
    )
    if log_dir:
        log_path = str(log_dir / "audit")

    # Create backend
    log_config = audit_config.get("log", {})
    rotation_config = log_config.get("rotation", {})
    backend = LogBackend(
        log_path=log_path,
        max_size_mb=rotation_config.get("max_size_mb", 100),
        max_age_hours=rotation_config.get("max_age_hours", 24),
        max_backups=rotation_config.get("max_backups", 720),
        compress=rotation_config.get("compress", True),
        timezone_name=rotation_config.get("timezone_name", "Asia/Shanghai"),
    )

    # Create hash chain if enabled
    chain = None
    tamper_config = audit_config.get("tamper_protection", {})
    if tamper_config.get("enabled", True):
        chain = AuditHashChain(log_path=log_path)

    # Initialize engine
    return AuditEngine.initialize(
        policy=policy,
        backends=[backend],
        chain=chain,
    )