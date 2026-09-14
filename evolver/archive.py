"""Versioned pathology records and adapters for existing Hermes fix-run JSON."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA_VERSION = 1
_HEX_RE = re.compile(r"^[0-9a-f]+$")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_otel_id(value: str, size: int, name: str) -> str:
    value = _require_text(value, name).lower()
    if len(value) != size or not _HEX_RE.fullmatch(value) or set(value) == {"0"}:
        raise ValueError(f"{name} must be a non-zero {size}-character hexadecimal ID")
    return value


def _stable_hex(payload: object, size: int) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:size]


@dataclass(frozen=True)
class Task:
    """The replayable work item associated with a failure trace."""

    task_id: str
    harness_version: str
    input: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task.task_id")
        _require_text(self.harness_version, "task.harness_version")
        _require_text(self.input, "task.input")


@dataclass(frozen=True)
class Span:
    """An OpenTelemetry-shaped span without requiring the OTEL SDK."""

    trace_id: str
    span_id: str
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    parent_span_id: str | None = None
    status_code: str = "UNSET"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_otel_id(self.trace_id, 32, "span.trace_id")
        _require_otel_id(self.span_id, 16, "span.span_id")
        if self.parent_span_id is not None:
            _require_otel_id(self.parent_span_id, 16, "span.parent_span_id")
        _require_text(self.name, "span.name")
        if self.start_time_unix_nano < 0 or self.end_time_unix_nano < self.start_time_unix_nano:
            raise ValueError("span timestamps must be non-negative and ordered")
        if self.status_code not in {"UNSET", "OK", "ERROR"}:
            raise ValueError("span.status_code must be UNSET, OK, or ERROR")


@dataclass(frozen=True)
class Trace:
    trace_id: str
    spans: tuple[Span, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = _require_otel_id(self.trace_id, 32, "trace.trace_id")
        if not self.spans:
            raise ValueError("trace.spans must contain at least one span")
        if any(span.trace_id != trace_id for span in self.spans):
            raise ValueError("every span must belong to trace.trace_id")
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span IDs must be unique within a trace")


@dataclass(frozen=True)
class PathologyRecord:
    task: Task
    trace: Trace
    failure_class: str
    observed_at: str
    source: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported pathology schema version: {self.schema_version}")
        _require_text(self.failure_class, "failure_class")
        try:
            parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("observed_at must include a timezone")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PathologyRecord":
        task_raw = raw["task"]
        trace_raw = raw["trace"]
        spans = tuple(Span(**span) for span in trace_raw["spans"])
        return cls(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            task=Task(**task_raw),
            trace=Trace(trace_id=trace_raw["trace_id"], spans=spans, attributes=trace_raw.get("attributes", {})),
            failure_class=raw["failure_class"],
            observed_at=raw["observed_at"],
            source=raw.get("source"),
        )


def _trajectory_from_result(result: Mapping[str, Any], failure_class: str) -> list[Mapping[str, Any]]:
    trajectory = result.get("conversations") or result.get("trajectory")
    if failure_class == "runner_error" and not trajectory:
        error = _require_text(result.get("error"), "runner error")
        return [{"role": "runner", "content": error}]
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("fix result must contain a non-empty conversations or trajectory list")
    if not all(isinstance(turn, Mapping) for turn in trajectory):
        raise ValueError("trajectory turns must be JSON objects")
    return trajectory


def _canonical_observed_at(value: object) -> str:
    """Normalize producer timestamps while keeping canonical records timezone-aware.

    Legacy mini/batch producers emit naive ISO timestamps. They do not retain the
    host offset, so the adapter deterministically treats those legacy values as UTC.
    """

    observed = _require_text(value, "observed_at")
    try:
        parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _failure_class(result: Mapping[str, Any], explicit: str | None) -> str | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    declared = explicit or result.get("failure_class") or metadata.get("failure_class")
    if declared:
        return _require_text(declared, "failure_class")
    if result.get("error"):
        return "runner_error"
    if result.get("partial"):
        return "invalid_tool_call"
    error_counts = result.get("tool_error_counts")
    if isinstance(error_counts, Mapping) and any(isinstance(count, int) and count > 0 for count in error_counts.values()):
        return "tool_error"
    if result.get("completed") is False:
        return "incomplete"
    return None


def record_from_fix_result(
    result: Mapping[str, Any],
    *,
    task_id: str | None = None,
    harness_version: str | None = None,
    failure_class: str | None = None,
    source: str | None = None,
) -> PathologyRecord | None:
    """Convert a mini_swe_runner/batch_runner record; successful runs are ignored."""

    failure = _failure_class(result, failure_class)
    if failure is None:
        return None
    trajectory = _trajectory_from_result(result, failure)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    prompt = result.get("prompt") or result.get("task")
    if not prompt:
        human = next((turn for turn in trajectory if turn.get("from") in {"human", "user"}), None)
        prompt = human.get("value") or human.get("content") if human else None
    if not prompt and failure == "runner_error":
        prompt = "[task unavailable: runner failed before trajectory capture]"
    prompt = _require_text(prompt, "task input")
    resolved_harness = str(harness_version or metadata.get("harness_version") or metadata.get("model") or "unknown")
    fallback_task_id = f"task-{_stable_hex({'harness_version': resolved_harness, 'input': prompt}, 12)}"
    resolved_task_id = str(task_id or result.get("task_id") or fallback_task_id)
    trace_id = _stable_hex({"task_id": resolved_task_id, "trajectory": trajectory}, 32)
    spans: list[Span] = []
    parent: str | None = None
    for index, turn in enumerate(trajectory):
        role = str(turn.get("from") or turn.get("role") or "unknown")
        value = turn.get("value", turn.get("content", ""))
        span_id = _stable_hex({"trace_id": trace_id, "index": index}, 16)
        spans.append(
            Span(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent,
                name=f"hermes.trajectory.{role}",
                start_time_unix_nano=index,
                end_time_unix_nano=index + 1,
                status_code="ERROR" if index == len(trajectory) - 1 else "UNSET",
                attributes={"hermes.turn_index": index, "hermes.role": role, "hermes.content": value},
            )
        )
        parent = span_id
    observed = _canonical_observed_at(
        metadata.get("timestamp") or result.get("timestamp") or datetime.now(timezone.utc).isoformat()
    )
    return PathologyRecord(
        task=Task(resolved_task_id, resolved_harness, prompt, dict(metadata)),
        trace=Trace(trace_id, tuple(spans), {"hermes.source_format": "sharegpt"}),
        failure_class=failure,
        observed_at=observed,
        source=source,
    )


class PathologyArchive:
    """Append-only JSONL archive with adapters for factory fix-run directories."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: PathologyRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __iter__(self) -> Iterator[PathologyRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield PathologyRecord.from_dict(json.loads(line))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(f"invalid pathology record at {self.path}:{line_number}: {exc}") from exc

    def ingest_fix_results(self, root: str | Path, *, harness_version: str | None = None) -> int:
        """Read existing per-run ``.json``/``.jsonl`` files without modifying the producer."""

        root = Path(root)
        paths: Iterable[Path] = [root] if root.is_file() else sorted((*root.rglob("*.json"), *root.rglob("*.jsonl")))
        written = 0
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()] if path.suffix == ".jsonl" else [json.load(handle)]
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError(f"fix result must be a JSON object: {path}")
                record = record_from_fix_result(row, harness_version=harness_version, source=str(path))
                if record is not None:
                    self.append(record)
                    written += 1
        return written
