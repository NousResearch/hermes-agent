"""Pathology-keyed failure archive: the evolver's memory.

A failure record is ``(task, trace, failure_class)`` plus provenance. The
trace uses OpenTelemetry-flavored spans (name, timestamps, status, flat
attributes) so the same schema works across harnesses — it is the
cross-harness lingua franca the pathology archive depends on.

Storage is append-only JSONL, one record per line, so readers never need to
load the whole archive and writers never corrupt it with a partial rewrite.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from . import SCHEMA_VERSION

# Controlled vocabulary for failure_class. Keep it small on purpose: the
# archive is keyed by pathology, and a thousand bespoke classes is the same
# as no classes. Extend only when a new class changes what patch to propose.
FAILURE_CLASSES = frozenset({
    "test_failure",   # a regression test fails (red on base)
    "exception",      # unhandled exception / crash
    "timeout",        # exceeded a time budget
    "wrong_output",   # completed but produced the wrong result
    "regression",     # previously passing behavior broke
    "infra_flake",    # failure attributable to the environment, not the code
    "unknown",
})

TASK_KINDS = frozenset({"bugfix", "feature", "eval_task", "trap", "manual"})

OUTCOME_STATUSES = frozenset({"failed", "passed", "flaky", "unknown"})

_REQUIRED_TOP = ("schema_version", "record_id", "created_at", "task",
                 "trace", "failure_class", "source")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_span(name: str, status: str = "ok",
              duration_s: float | None = None, **attributes) -> dict:
    """Build one OpenTelemetry-flavored span dict.

    ``duration_s`` records how long the span took; it is what makes the
    "timeout" failure class measurable rather than a label.
    """
    started = datetime.now(timezone.utc)
    ended = (started + timedelta(seconds=duration_s)
             if duration_s is not None else datetime.now(timezone.utc))
    span = {
        "name": name,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "status": {"code": status},
        "attributes": {k: str(v) for k, v in attributes.items()},
    }
    if duration_s is not None:
        span["attributes"]["duration_s"] = str(duration_s)
    return span


def make_record(task: dict, trace_spans: list, failure_class: str,
                source: dict, outcome: dict | None = None) -> dict:
    """Build a validated failure record dict (raises ValueError if invalid)."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": uuid.uuid4().hex,
        "created_at": utcnow_iso(),
        "task": task,
        "trace": {"spans": list(trace_spans)},
        "failure_class": failure_class,
        "outcome": outcome or {"status": "unknown"},
        "source": source,
    }
    errors = validate_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_record(record: dict) -> list:
    """Return a list of validation errors (empty = valid). Pure function."""
    errors = []
    if not isinstance(record, dict):
        return ["record must be a dict"]
    for key in _REQUIRED_TOP:
        if key not in record:
            errors.append(f"missing required key: {key}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {record.get('schema_version')!r}")
    task = record.get("task")
    if isinstance(task, dict):
        if not task.get("task_id"):
            errors.append("task.task_id is required")
        if task.get("kind") not in TASK_KINDS:
            errors.append(f"task.kind must be one of {sorted(TASK_KINDS)}")
    trace = record.get("trace")
    if isinstance(trace, dict):
        spans = trace.get("spans")
        if not isinstance(spans, list) or not spans:
            errors.append("trace.spans must be a non-empty list")
        else:
            for i, span in enumerate(spans):
                if not isinstance(span, dict) or not span.get("name"):
                    errors.append(f"trace.spans[{i}] must have a name")
                code = (span.get("status") or {}).get("code")
                if code not in ("ok", "error", "unset"):
                    errors.append(
                        f"trace.spans[{i}].status.code must be ok|error|unset")
    if record.get("failure_class") not in FAILURE_CLASSES:
        errors.append(f"failure_class must be one of {sorted(FAILURE_CLASSES)}")
    outcome = record.get("outcome") or {}
    if outcome.get("status") not in OUTCOME_STATUSES:
        errors.append(f"outcome.status must be one of {sorted(OUTCOME_STATUSES)}")
    source = record.get("source")
    if isinstance(source, dict) and not source.get("type"):
        errors.append("source.type is required")
    return errors


class TraceArchive:
    """Append-only JSONL archive of failure records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: dict) -> str:
        errors = validate_record(record)
        if errors:
            raise ValueError("; ".join(errors))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record["record_id"]

    def iter_records(self, failure_class: str | None = None,
                     task_kind: str | None = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # never let one bad line kill a scan
                if failure_class and record.get("failure_class") != failure_class:
                    continue
                if task_kind and (record.get("task") or {}).get("kind") != task_kind:
                    continue
                yield record

    def count(self) -> int:
        return sum(1 for _ in self.iter_records())
