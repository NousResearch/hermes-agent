"""Generic pre-delivery decision gate for completed assistant responses.

This module deliberately knows nothing about the evaluator repository. A policy
callback supplied by the integration layer decides whether a completed response
may be delivered. In strict mode, a malformed or failed policy result is a safe
stop; it never falls back to unvalidated delivery.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol

Mode = Literal["legacy", "shadow", "strict"]
Status = Literal["passed", "blocked", "inconclusive"]


class Policy(Protocol):
    def __call__(self, *, final_text: str, metadata: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class PreDeliveryDecision:
    allowed: bool
    final_text: str | None
    status: Status
    evidence_ref: str | None = None
    reason: str | None = None


class PreDeliveryGate:
    """Evaluate a completed response before the platform send boundary."""

    def __init__(self, *, mode: Mode = "legacy", policy: Policy | None = None) -> None:
        if mode not in {"legacy", "shadow", "strict"}:
            raise ValueError("mode must be legacy, shadow, or strict")
        if mode != "legacy" and policy is None:
            raise ValueError("shadow and strict modes require a policy")
        self.mode = mode
        self.policy = policy

    async def evaluate(self, *, final_text: str, metadata: dict[str, Any] | None = None) -> PreDeliveryDecision:
        if not isinstance(final_text, str):
            return PreDeliveryDecision(False, None, "inconclusive", reason="final_text must be a string")
        if self.mode == "legacy":
            return PreDeliveryDecision(True, final_text, "passed", reason="legacy mode")
        assert self.policy is not None
        try:
            result = self.policy(final_text=final_text, metadata=dict(metadata or {}))
            if inspect.isawaitable(result):
                result = await result
            decision = self._coerce(result)
        except Exception as exc:
            decision = PreDeliveryDecision(False, None, "inconclusive", reason=f"policy failed: {exc}")
        if self.mode == "shadow":
            # Shadow observes but never changes legacy visibility.
            return PreDeliveryDecision(True, final_text, decision.status, decision.evidence_ref, decision.reason)
        if not decision.allowed or decision.status != "passed" or not isinstance(decision.final_text, str):
            return PreDeliveryDecision(False, None, decision.status, decision.evidence_ref, decision.reason or "policy did not approve delivery")
        return decision

    def evaluate_sync(self, *, final_text: str, metadata: dict[str, Any] | None = None) -> PreDeliveryDecision:
        """Evaluate from the agent worker thread at the synchronous finish boundary."""
        result = self.evaluate(final_text=final_text, metadata=metadata)
        if inspect.isawaitable(result):
            import asyncio
            return asyncio.run(result)
        return result

    @staticmethod
    def _coerce(value: Any) -> PreDeliveryDecision:
        if not isinstance(value, dict):
            raise ValueError("policy result must be an object")
        allowed = value.get("allowed")
        status = value.get("status")
        final_text = value.get("final_text")
        if not isinstance(allowed, bool) or status not in {"passed", "blocked", "inconclusive"}:
            raise ValueError("policy result has invalid allowed/status")
        if final_text is not None and not isinstance(final_text, str):
            raise ValueError("policy final_text must be a string or null")
        evidence_ref = value.get("evidence_ref")
        reason = value.get("reason")
        if evidence_ref is not None and not isinstance(evidence_ref, str):
            raise ValueError("policy evidence_ref must be a string or null")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("policy reason must be a string or null")
        return PreDeliveryDecision(allowed, final_text, status, evidence_ref, reason)
