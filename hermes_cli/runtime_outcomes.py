"""Typed runtime outcomes shared by provider, launcher, worker, and goal paths."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeOutcome:
    """Bounded execution result used for retry and budget accounting."""

    kind: str
    retryable: bool
    counts_against_failure_budget: bool
    reason: str = ""
    source: str = ""

    @property
    def is_transient(self) -> bool:
        return self.retryable and not self.counts_against_failure_budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "retryable": self.retryable,
            "counts_against_failure_budget": self.counts_against_failure_budget,
            "reason": self.reason,
            "source": self.source,
        }

    @classmethod
    def provider_overload(cls, reason: str = "provider overloaded") -> "RuntimeOutcome":
        return cls("provider_overload", True, False, reason, "provider")

    @classmethod
    def launcher_transport_failure(
        cls, reason: str = "launcher transport failure"
    ) -> "RuntimeOutcome":
        return cls("launcher_transport_failure", True, False, reason, "launcher")

    @classmethod
    def ex_tempfail(cls, reason: str = "worker exited EX_TEMPFAIL") -> "RuntimeOutcome":
        return cls("ex_tempfail", True, False, reason, "worker")

    @classmethod
    def code_failure(cls, reason: str = "worker code failure") -> "RuntimeOutcome":
        return cls("code_failure", False, True, reason, "worker")

    @classmethod
    def from_value(cls, value: Any) -> "RuntimeOutcome":
        """Normalize legacy outcome/reason strings, failing closed."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            value = value.get("kind") or value.get("failure_reason")
        raw = getattr(value, "value", value)
        raw = str(raw or "").strip().lower()
        if raw in {"overloaded", "upstream_rate_limit", "provider_overload"}:
            return cls.provider_overload(reason=raw)
        if raw in {"rate_limit", "billing"}:
            return cls(raw, True, False, raw, "provider")
        if raw in {"transport", "launcher_transport_failure", "judge_transport_failure"}:
            return cls(raw, True, False, raw, "launcher" if raw != "judge_transport_failure" else "goal_judge")
        if raw in {"rate_limited", "tempfail", "ex_tempfail"}:
            return cls.ex_tempfail(reason=raw)
        if raw in {"crashed", "code_failure", "failure", "spawn_failed", "timed_out"}:
            return cls.code_failure(reason=raw)
        if raw in {"completed", "success"}:
            return cls("completed", False, False, raw, "worker")
        return cls.code_failure(reason=raw or "unknown runtime failure")


def outcome_for_provider_reason(reason: Any) -> RuntimeOutcome:
    """Map the provider classifier's reason to runtime accounting."""
    raw = getattr(reason, "value", reason)
    raw = str(raw or "").strip().lower()
    if raw in {"overloaded", "upstream_rate_limit"}:
        return RuntimeOutcome.provider_overload(reason=raw)
    if raw in {"rate_limit", "billing"}:
        return RuntimeOutcome(raw, True, False, raw, "provider")
    return RuntimeOutcome.code_failure(reason=raw or "unknown provider failure")


def outcome_for_launcher_exception(exc: BaseException) -> RuntimeOutcome:
    """Classify launcher transport failures without hiding code failures."""
    if isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError, subprocess.TimeoutExpired)):
        return RuntimeOutcome.launcher_transport_failure(
            reason=str(exc) or exc.__class__.__name__
        )
    return RuntimeOutcome("spawn_failure", False, True, str(exc) or exc.__class__.__name__, "launcher")


def outcome_for_worker_exit(kind: str, code: int | None) -> RuntimeOutcome:
    """Map the existing worker-exit classifier to a typed outcome."""
    if kind == "clean_exit":
        return RuntimeOutcome("completed", False, False, "worker exited successfully", "worker")
    if kind == "rate_limited" or code == 75:
        return RuntimeOutcome.ex_tempfail(reason="worker exited EX_TEMPFAIL (75)")
    return RuntimeOutcome.code_failure(reason=f"worker exit kind={kind}, code={code}")


__all__ = [
    "RuntimeOutcome",
    "outcome_for_launcher_exception",
    "outcome_for_provider_reason",
    "outcome_for_worker_exit",
]
