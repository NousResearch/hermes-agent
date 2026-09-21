"""Typed research failure and checkpoint primitives shared by agent/runtime paths.

The dispatcher owns lifecycle transitions, while the agent owns the evidence it has
collected. This module only carries the small, JSON-safe envelope between them:
classification, evidence presence, and the synthesis-only recovery mode.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS = "timeout_before_synthesis"
FAILURE_CLASS_RESEARCH_BUDGET_EXHAUSTED = "research_budget_exhausted"
FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION = "terminal_protocol_violation"
FINALIZATION_PROTOCOL_FAILURE = "FINALIZATION_PROTOCOL_FAILURE"

RECOVERY_FAILURE_CLASSES = frozenset({
    FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS,
    FAILURE_CLASS_RESEARCH_BUDGET_EXHAUSTED,
    FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION,
})

SYNTHESIS_ONLY_MODE = "synthesis_only"
RESEARCH_MODE_ENV = "HERMES_KANBAN_RESEARCH_MODE"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ResearchFailure:
    """Durable classification attached to a terminal worker run."""

    failure_class: str
    failure_code: str
    evidence_present: bool
    checkpoint_path: str | None = None
    recovery_mode: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "failure_class": self.failure_class,
            "failure_code": self.failure_code,
            "evidence_present": self.evidence_present,
            "research_recovery": True,
        }
        if self.checkpoint_path:
            data["checkpoint_path"] = self.checkpoint_path
        if self.recovery_mode:
            data["recovery_mode"] = self.recovery_mode
        return data


def failure_code(failure_class: str) -> str:
    """Return the stable machine-facing code for a failure class."""
    if failure_class == FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION:
        return FINALIZATION_PROTOCOL_FAILURE
    return failure_class.upper()


def classify_failure(
    *,
    outcome: str | None = None,
    research_budget: Mapping[str, Any] | None = None,
    protocol_violation: bool = False,
    timed_out: bool = False,
) -> str | None:
    """Classify from authoritative runtime evidence, never task prose.

    Protocol evidence wins over a budget marker. A configured research turn that
    timed out before the guardrail transitioned is distinct from an explicit
    collection-budget exhaustion; ordinary tasks return ``None`` and retain the
    native dispatcher behavior.
    """
    if protocol_violation:
        return FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION
    if isinstance(research_budget, Mapping):
        if bool(research_budget.get("exhausted")):
            return FAILURE_CLASS_RESEARCH_BUDGET_EXHAUSTED
        if timed_out and bool(research_budget.get("enabled")):
            return FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS
    return None


def research_policy_enabled(policy: Mapping[str, Any] | None) -> bool:
    """Whether a persisted task/profile policy actually enables a bounded turn."""
    if not isinstance(policy, Mapping):
        return False
    return any(
        policy.get(name) is not None
        for name in (
            "web_search_max",
            "browser_extract_max",
            "collection_deadline_seconds",
            "synthesis_reserve_seconds",
        )
    )


def checkpoint_path(task_id: str, *, board: str | None = None) -> Path:
    """Resolve the same per-task checkpoint path injected into worker env."""
    from hermes_cli import kanban_db

    return Path(kanban_db.workspaces_root(board=board)).parent / "checkpoints" / task_id / "latest.yaml"


def _read_checkpoint_path(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def read_checkpoint(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Read a JSON-compatible YAML checkpoint, failing closed on bad input."""
    if not path:
        return {}
    return _read_checkpoint_path(Path(path).expanduser())


def checkpoint_evidence(path: str | os.PathLike[str] | None) -> bool:
    """Return true only for a valid checkpoint that records preserved evidence."""
    data = read_checkpoint(path)
    if not data:
        return False
    if bool(data.get("evidence_present")):
        return True
    try:
        return int(data.get("evidence_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def evidence_present(
    research_budget: Mapping[str, Any] | None = None,
    *,
    checkpoint: Mapping[str, Any] | None = None,
) -> bool:
    """Derive evidence presence from counters/checkpoint, without storing content."""
    if isinstance(research_budget, Mapping):
        if bool(research_budget.get("evidence_present")):
            return True
        try:
            if int(research_budget.get("evidence_count") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
    if isinstance(checkpoint, Mapping):
        if bool(checkpoint.get("evidence_present")):
            return True
        try:
            return int(checkpoint.get("evidence_count") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def checkpoint_for_task(
    task_id: str,
    *,
    board: str | None = None,
    explicit_path: str | os.PathLike[str] | None = None,
) -> Path:
    return Path(explicit_path).expanduser() if explicit_path else checkpoint_path(task_id, board=board)


def write_research_checkpoint(
    *,
    task_id: str | None = None,
    research_budget: Mapping[str, Any] | None = None,
    failure_class: str | None = None,
    evidence: bool | None = None,
    path: str | os.PathLike[str] | None = None,
) -> str | None:
    """Atomically persist bounded research progress for a future recovery run.

    Only counters and state are written; raw search/browser output stays in the
    normal session transcript. Existing evidence is never replaced by a later
    empty checkpoint, so a finalization failure cannot erase the recovery proof.
    """
    resolved = checkpoint_for_task(
        task_id or os.environ.get("HERMES_KANBAN_TASK", ""),
        explicit_path=path or os.environ.get("HERMES_KANBAN_CHECKPOINT"),
    ) if (path or task_id or os.environ.get("HERMES_KANBAN_TASK")) else None
    if resolved is None:
        return None
    previous = _read_checkpoint_path(resolved) if resolved.exists() else {}
    budget = dict(research_budget) if isinstance(research_budget, Mapping) else None
    current_evidence = evidence_present(budget, checkpoint=previous)
    if evidence is not None:
        current_evidence = bool(evidence) or current_evidence
    payload: dict[str, Any] = dict(previous)
    payload.update({
        "version": CHECKPOINT_VERSION,
        "task_id": task_id or os.environ.get("HERMES_KANBAN_TASK") or None,
        "run_id": os.environ.get("HERMES_KANBAN_RUN_ID") or None,
        "updated_at": int(time.time()),
        "evidence_present": current_evidence,
        "evidence_count": _evidence_count(budget, previous),
    })
    if failure_class in RECOVERY_FAILURE_CLASSES:
        payload["research_recovery"] = True
    if failure_class:
        payload["failure_class"] = failure_class
        payload["failure_code"] = failure_code(failure_class)
    if budget is not None:
        payload["research_budget"] = budget
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, resolved)
    except OSError:
        return None
    return str(resolved)


def _evidence_count(
    budget: Mapping[str, Any] | None,
    previous: Mapping[str, Any],
) -> int:
    counts: list[int] = []
    for source in (budget, previous):
        if not isinstance(source, Mapping):
            continue
        try:
            if source.get("evidence_present"):
                counts.append(1)
            counts.append(int(source.get("evidence_count") or 0))
        except (TypeError, ValueError):
            continue
    return max(counts, default=0)


def recovery_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    checkpoint_path_value: str | None = None,
) -> ResearchFailure | None:
    """Return synthesis-only/blocked recovery facts for a named prior failure."""
    if not isinstance(metadata, Mapping):
        metadata = {}
    if not bool(metadata.get("research_recovery")) and not (
        isinstance(checkpoint, Mapping) and bool(checkpoint.get("research_recovery"))
    ):
        return None
    failure_class_value = metadata.get("failure_class")
    if failure_class_value not in RECOVERY_FAILURE_CLASSES and isinstance(checkpoint, Mapping):
        failure_class_value = checkpoint.get("failure_class")
    if failure_class_value not in RECOVERY_FAILURE_CLASSES:
        return None
    has_evidence = evidence_present(metadata, checkpoint=checkpoint)
    failure_class_text = str(failure_class_value)
    return ResearchFailure(
        failure_class=failure_class_text,
        failure_code=str(metadata.get("failure_code") or failure_code(failure_class_text)),
        evidence_present=has_evidence,
        checkpoint_path=checkpoint_path_value,
        recovery_mode=SYNTHESIS_ONLY_MODE if has_evidence else None,
    )
