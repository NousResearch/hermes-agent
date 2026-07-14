"""Trajectory Collector — captures agent execution traces as structured YAML for analysis."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Max steps before we truncate (prevent runaway traces)
MAX_TRACE_STEPS = 500
# Max summary chars per step
MAX_SUMMARY_CHARS = 500


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """A single step in an agent trajectory."""
    step: int
    type: str  # "model_call", "tool_execution", "thinking"
    duration_ms: int = 0
    status: str = "success"  # "success", "error", "timeout"
    summary: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "type": self.type,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "summary": self.summary[:MAX_SUMMARY_CHARS],
        }
        d.update(self.extra)
        return d


@dataclass
class EvalCheck:
    """Result of a single evaluation check."""
    type: str
    passed: bool
    detail: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "passed": self.passed, "detail": self.detail, "score": self.score}


@dataclass
class EvalResult:
    """Aggregate evaluation result for a trajectory."""
    passed: bool = False
    score: float = 0.0
    checks: List[EvalCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class Trajectory:
    """A complete agent trajectory for a task attempt."""
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    task_name: str = ""
    run_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    status: str = "pending"  # "success", "failed", "timeout", "error"
    total_turns: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    steps: List[TraceStep] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    eval_result: Optional[EvalResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "task_name": self.task_name,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "steps": [s.to_dict() for s in self.steps],
            "errors": self.errors,
        }
        if self.eval_result:
            d["eval_result"] = self.eval_result.to_dict()
        return d

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class TrajectoryCollector:
    """Captures agent execution traces during a task attempt.

    Thread-safe. One collector per task attempt. Hooks are called from
    the agent's turn loop via the EvolutionManager.
    """

    def __init__(
        self,
        task_name: str = "",
        run_id: str = "",
        max_steps: int = MAX_TRACE_STEPS,
    ):
        self._trajectory = Trajectory(task_name=task_name, run_id=run_id)
        self._max_steps = max_steps
        self._step_counter = 0
        self._turn_counter = 0
        self._tool_call_counter = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._active = False

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Begin collecting trajectory data."""
        with self._lock:
            self._active = True
            self._start_time = time.monotonic()
            self._trajectory.started_at = datetime.now(timezone.utc).isoformat()
            self._trajectory.status = "attempting"

    def stop(self, status: str = "completed") -> Trajectory:
        """Stop collecting and return the completed trajectory."""
        with self._lock:
            self._active = False
            self._trajectory.completed_at = datetime.now(timezone.utc).isoformat()
            self._trajectory.status = status
            self._trajectory.total_turns = self._turn_counter
            self._trajectory.total_tool_calls = self._tool_call_counter
            return self._trajectory

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def is_active(self) -> bool:
        return self._active

    # -- Event hooks (called from agent loop) ----------------------------------

    def record_model_call(
        self,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        tool_calls: Optional[List[str]] = None,
        summary: str = "",
        thinking_summary: str = "",
    ) -> None:
        """Record a model inference call."""
        if not self._active or self._step_counter >= self._max_steps:
            return
        with self._lock:
            self._step_counter += 1
            self._turn_counter += 1
            self._trajectory.total_tokens += input_tokens + output_tokens
            step = TraceStep(
                step=self._step_counter,
                type="model_call",
                duration_ms=duration_ms,
                status="success",
                summary=summary[:MAX_SUMMARY_CHARS],
                extra={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tool_calls": tool_calls or [],
                    "thinking_summary": thinking_summary[:MAX_SUMMARY_CHARS] if thinking_summary else "",
                },
            )
            self._trajectory.steps.append(step)

    def record_tool_call(
        self,
        tool_name: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
        status: str = "success",
        error_message: str = "",
        result_summary: str = "",
    ) -> None:
        """Record a tool execution."""
        if not self._active or self._step_counter >= self._max_steps:
            return
        with self._lock:
            self._step_counter += 1
            self._tool_call_counter += 1
            step = TraceStep(
                step=self._step_counter,
                type="tool_execution",
                duration_ms=duration_ms,
                status=status,
                summary=result_summary[:MAX_SUMMARY_CHARS],
                extra={
                    "tool": tool_name,
                    "tool_args_summary": _summarize_args(tool_args),
                },
            )
            self._trajectory.steps.append(step)
            if status != "success" and error_message:
                self._trajectory.errors.append({
                    "step": self._step_counter,
                    "tool": tool_name,
                    "message": error_message[:MAX_SUMMARY_CHARS],
                })

    def record_thinking(
        self,
        duration_ms: int = 0,
        summary: str = "",
    ) -> None:
        """Record a reasoning/thinking phase."""
        if not self._active or self._step_counter >= self._max_steps:
            return
        with self._lock:
            self._step_counter += 1
            step = TraceStep(
                step=self._step_counter,
                type="thinking",
                duration_ms=duration_ms,
                status="success",
                summary=summary[:MAX_SUMMARY_CHARS],
            )
            self._trajectory.steps.append(step)

    def record_error(self, error_message: str, step: Optional[int] = None) -> None:
        """Record an error that occurred during execution."""
        with self._lock:
            self._trajectory.errors.append({
                "step": step or self._step_counter,
                "message": error_message[:MAX_SUMMARY_CHARS],
            })

    def set_eval_result(self, result: EvalResult) -> None:
        """Attach evaluation results to the trajectory."""
        with self._lock:
            self._trajectory.eval_result = result
            self._trajectory.status = "success" if result.passed else "failed"

    # -- Persistence -----------------------------------------------------------

    def save(self, base_dir: Optional[Path] = None) -> Path:
        """Save the trajectory to disk. Returns the file path."""
        if base_dir is None:
            base_dir = get_hermes_home() / "evolution" / "traces"
        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trace_dir = base_dir / date_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{self._trajectory.task_name}_{self._trajectory.trace_id}.trace.yaml"
        self._trajectory.to_yaml(path)
        logger.debug("Saved trajectory to %s", path)
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_args(args: Optional[Dict[str, Any]]) -> str:
    """Create a short summary of tool arguments for trace storage."""
    if not args:
        return "{}"
    # Keep it brief — full args are in the conversation history
    keys = list(args.keys())
    if len(keys) <= 3:
        return json.dumps({k: _truncate_arg(args[k]) for k in keys}, default=str)
    return json.dumps({k: _truncate_arg(args[k]) for k in keys[:3]}, default=str) + "..."


def _truncate_arg(value: Any, max_len: int = 80) -> Any:
    """Truncate a tool arg value for summary display."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    if isinstance(value, dict):
        return {k: _truncate_arg(v, max_len // 2) for k, v in list(value.items())[:3]}
    if isinstance(value, list):
        return [_truncate_arg(v, max_len // 2) for v in value[:3]]
    return value


# ---------------------------------------------------------------------------
# Trace file utilities
# ---------------------------------------------------------------------------


def get_trace_dir() -> Path:
    """Return the trace storage directory."""
    return get_hermes_home() / "evolution" / "traces"


def list_traces(task_name: Optional[str] = None, days: int = 30) -> List[Path]:
    """List recent trace files, optionally filtered by task name."""
    trace_dir = get_trace_dir()
    if not trace_dir.is_dir():
        return []
    cutoff = time.time() - (days * 86400)
    traces = []
    for date_dir in sorted(trace_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for trace_file in date_dir.glob("*.trace.yaml"):
            try:
                if trace_file.stat().st_mtime < cutoff:
                    continue
                if task_name and task_name not in trace_file.stem:
                    continue
                traces.append(trace_file)
            except OSError:
                continue
    return traces


def load_trace(path: Path) -> Optional[Trajectory]:
    """Load a trajectory from a YAML file."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load trace %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    return _trajectory_from_dict(data)


def _trajectory_from_dict(d: Dict[str, Any]) -> Trajectory:
    """Reconstruct a Trajectory from a dict."""
    steps = []
    for s in d.get("steps", []):
        extra = {k: v for k, v in s.items() if k not in ("step", "type", "duration_ms", "status", "summary")}
        steps.append(TraceStep(
            step=s.get("step", 0),
            type=s.get("type", ""),
            duration_ms=s.get("duration_ms", 0),
            status=s.get("status", "success"),
            summary=s.get("summary", ""),
            extra=extra,
        ))
    eval_result = None
    if "eval_result" in d:
        er = d["eval_result"]
        checks = [EvalCheck(**c) for c in er.get("checks", [])]
        eval_result = EvalResult(passed=er.get("passed", False), score=er.get("score", 0.0), checks=checks)
    return Trajectory(
        trace_id=d.get("trace_id", ""),
        task_name=d.get("task_name", ""),
        run_id=d.get("run_id", ""),
        started_at=d.get("started_at", ""),
        completed_at=d.get("completed_at", ""),
        status=d.get("status", "pending"),
        total_turns=d.get("total_turns", 0),
        total_tool_calls=d.get("total_tool_calls", 0),
        total_tokens=d.get("total_tokens", 0),
        steps=steps,
        errors=d.get("errors", []),
        eval_result=eval_result,
    )
