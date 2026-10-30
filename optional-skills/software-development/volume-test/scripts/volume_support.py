"""Small application-independent helpers; no target discovery or write adapters."""

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def stable_integer(seed, index, field):
    """Content depends on logical identity, never run IDs or batch boundaries."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{index}:{field}".encode()).digest()[:8], "big")


def checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compare_rows(actual, expected, fields):
    """Compare independently derived ordered rows, including multiplicity and content."""
    def key(row):
        return tuple(str(row[field]) for field in fields)

    observed, reference = [key(row) for row in actual], [key(row) for row in expected]
    return {
        "correct": observed == reference,
        "actual_count": len(observed),
        "expected_count": len(reference),
        "missing_count": sum((Counter(reference) - Counter(observed)).values()),
        "unexpected_count": sum((Counter(observed) - Counter(reference)).values()),
        "order_correct": observed == reference if Counter(observed) == Counter(reference) else None,
    }


def summarize(samples):
    """Timings include failed requests; errors and timeouts are not dropped."""
    durations = [s["duration_ms"] for s in samples if s.get("duration_ms") is not None]
    if any(not math.isfinite(d) or d < 0 for d in durations):
        raise ValueError("Durations must be finite nonnegative milliseconds")
    return {
        "samples": len(samples), "timed_samples": len(durations),
        "median_ms": statistics.median(durations) if durations else None,
        "min_ms": min(durations) if durations else None,
        "max_ms": max(durations) if durations else None,
        "errors": sum(bool(s.get("error")) for s in samples),
        "timeouts": sum(bool(s.get("timeout")) for s in samples),
        "incorrect": sum(s.get("correct") is False for s in samples),
    }
