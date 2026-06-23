"""Generic runtime guard data types.

These primitives intentionally do not encode any deployment-specific policy.
They give embedders a stable shape for deciding whether a runtime action may
continue, while the default runtime guard remains disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GuardContext:
    """Context passed to a runtime guard decision point."""

    guard_name: str
    action: str = ""
    session_id: str = ""
    task_id: str = ""
    tool_call_id: str = ""
    turn_id: str = ""
    api_request_id: str = ""
    platform: str = ""
    tool_name: str = ""
    tool_args: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return non-sensitive context metadata for blocked tool results."""
        data: dict[str, Any] = {
            "guard_name": self.guard_name,
            "action": self.action,
        }
        for key in (
            "session_id",
            "task_id",
            "tool_call_id",
            "turn_id",
            "api_request_id",
            "platform",
            "tool_name",
        ):
            value = getattr(self, key)
            if value:
                data[key] = value
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class GuardDecision:
    """Decision returned by a runtime guard."""

    allowed: bool
    reason: str = ""
    message: str = ""
    code: str = ""
    replacement_text: str | None = None
    context: GuardContext | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, reason: str = "allowed", *, context: GuardContext | None = None) -> "GuardDecision":
        return cls(True, reason=reason, code="allowed", context=context)

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        message: str | None = None,
        code: str = "runtime_guard_block",
        replacement_text: str | None = None,
        context: GuardContext | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GuardDecision":
        return cls(
            False,
            reason=reason,
            message=message or reason,
            code=code,
            replacement_text=replacement_text,
            context=context,
            metadata=dict(metadata or {}),
        )

    def with_context(self, context: GuardContext) -> "GuardDecision":
        if self.context is context:
            return self
        return GuardDecision(
            allowed=self.allowed,
            reason=self.reason,
            message=self.message,
            code=self.code,
            replacement_text=self.replacement_text,
            context=context,
            metadata=self.metadata,
        )

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "allowed": self.allowed,
            "reason": self.reason,
            "message": self.message or self.reason,
            "code": self.code or ("allowed" if self.allowed else "runtime_guard_block"),
        }
        if self.context is not None:
            data["context"] = self.context.to_metadata()
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data
