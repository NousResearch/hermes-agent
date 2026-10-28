#!/usr/bin/env python3
"""Summarize a log dump: severity counts, top error signatures, and spike detection.

Stdlib only, no network calls. Reads a log file or stdin and prints a JSON
summary suitable for incident triage:

    {
      "total_lines": int,
      "counts": {"ERROR": int, "WARN": int, ...},
      "top_signatures": [{"signature": str, "count": int, "level": str, "example": str}, ...],
      "spike": {"detected": bool, "recent_rate": float, "baseline_rate": float, "ratio": float|None}
    }

Usage:
    python log_triage.py --file app.log
    cat app.log | python log_triage.py
    python log_triage.py --selftest
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

LEVEL_KEYWORDS = [
    ("CRITICAL", re.compile(r"\b(CRITICAL|FATAL|PANIC(?:KED)?)\b", re.IGNORECASE)),
    ("ERROR", re.compile(r"\b(ERRORS?|ERR|EXCEPTIONS?|TRACEBACKS?|FAILED)\b", re.IGNORECASE)),
    ("WARN", re.compile(r"\bWARN(?:ING)?S?\b", re.IGNORECASE)),
    ("INFO", re.compile(r"\b(INFO|DEBUG)\b", re.IGNORECASE)),
]

# Timestamp patterns tried in order; first match wins.
_TS_ISO = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_TS_SYSLOG = re.compile(r"(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")
TIMESTAMP_PATTERNS = [_TS_ISO, _TS_SYSLOG]

# Tokens normalized away when building a "signature" so similar lines group together.
_NORMALIZERS = [
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<uuid>",
    ),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (
        re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
        "<ts>",
    ),
    (re.compile(r"\b\d+\b"), "<n>"),
]

_CRITICAL_WORDS = ("CRITICAL", "FATAL", "PANIC")


def classify_level(line: str) -> str:
    """Return the highest-severity level keyword found in a log line, or OTHER."""
    for level, pattern in LEVEL_KEYWORDS:
        if pattern.search(line):
            return level
    return "OTHER"


def extract_timestamp(line: str):
    for pattern in TIMESTAMP_PATTERNS:
        m = pattern.search(line)
        if m:
            return m.group("ts")
    return None


def _parse_dt(ts: str, now_year: int):
    """Best-effort parse of a matched timestamp string to a naive UTC datetime, or None."""
    ts_norm = ts.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts_norm, fmt)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            continue
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts_norm, fmt)
        except ValueError:
            continue
    try:
        # syslog timestamps have no year — assume current year.
        return datetime.strptime(f"{now_year} {ts}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None


def fingerprint(line: str) -> str:
    """Normalize volatile tokens (ids, ips, numbers, timestamps) so similar errors group together."""
    sig = line.strip()
    for pattern, placeholder in _NORMALIZERS:
        sig = pattern.sub(placeholder, sig)
    return sig[:200]


def _detect_spike(timestamps, window_minutes: int) -> dict:
    """Compare the error rate in the most recent window against the rest of the span."""
    if len(timestamps) < 2:
        return {"detected": False, "recent_rate": 0.0, "baseline_rate": 0.0, "ratio": 0.0}

    timestamps = sorted(timestamps)
    latest = timestamps[-1]
    window = timedelta(minutes=window_minutes)
    recent = [t for t in timestamps if t > latest - window]
    baseline = [t for t in timestamps if t <= latest - window]

    recent_rate = len(recent) / max(window_minutes, 1)
    total_span_minutes = max((timestamps[-1] - timestamps[0]).total_seconds() / 60.0, 1e-9)
    baseline_minutes = max(total_span_minutes - window_minutes, window_minutes)
    baseline_rate = (len(baseline) / baseline_minutes) if baseline else 0.0

    if baseline_rate > 0:
        ratio = recent_rate / baseline_rate
    else:
        ratio = float("inf") if recent_rate > 0 else 0.0

    detected = bool(baseline) and ratio >= 3.0 and recent_rate > 0
    # No usable baseline (all errors fall in the recent window) — fall back to an
    # absolute-count heuristic so a genuine burst of new errors still flags.
    if not baseline and len(recent) >= 5:
        detected = True

    return {
        "detected": detected,
        "recent_rate": round(recent_rate, 3),
        "baseline_rate": round(baseline_rate, 3),
        "ratio": None if ratio == float("inf") else round(ratio, 3),
    }


def analyze(text: str, window_minutes: int = 5, top: int = 10) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    counts = Counter()
    signatures = Counter()
    examples = {}
    error_timestamps = []
    now_year = datetime.now(timezone.utc).year

    for line in lines:
        level = classify_level(line)
        counts[level] += 1
        if level in ("ERROR", "CRITICAL"):
            sig = fingerprint(line)
            signatures[sig] += 1
            examples.setdefault(sig, line.strip())
            ts_raw = extract_timestamp(line)
            if ts_raw:
                dt = _parse_dt(ts_raw, now_year)
                if dt:
                    error_timestamps.append(dt)

    top_signatures = [
        {
            "signature": sig,
            "count": count,
            "level": "CRITICAL"
            if any(w in examples[sig].upper() for w in _CRITICAL_WORDS)
            else "ERROR",
            "example": examples[sig],
        }
        for sig, count in signatures.most_common(top)
    ]

    return {
        "total_lines": len(lines),
        "counts": dict(counts),
        "top_signatures": top_signatures,
        "spike": _detect_spike(error_timestamps, window_minutes),
    }


_SELFTEST_LOG = """\
2026-07-08T10:00:00Z INFO service started
2026-07-08T10:01:00Z INFO handled request 1 from 10.0.0.5
2026-07-08T10:15:00Z WARNING disk usage at 80% on /data
2026-07-08T10:30:00Z ERROR request 123 failed for user 45 at 10.0.0.5
2026-07-08T10:31:00Z ERROR request 124 failed for user 46 at 10.0.0.6
2026-07-08T10:31:05Z ERROR request 125 failed for user 47 at 10.0.0.7
2026-07-08T10:31:10Z ERROR request 126 failed for user 48 at 10.0.0.8
2026-07-08T10:31:20Z ERROR request 127 failed for user 49 at 10.0.0.9
2026-07-08T10:31:30Z CRITICAL out of memory, restarting worker
"""


def _selftest() -> int:
    result = analyze(_SELFTEST_LOG)
    assert result["total_lines"] == 9
    assert result["counts"].get("ERROR", 0) == 5
    assert result["counts"].get("CRITICAL", 0) == 1
    assert result["counts"].get("WARN", 0) == 1
    assert result["top_signatures"], "expected at least one error signature"
    assert result["top_signatures"][0]["count"] == 5
    assert result["spike"]["detected"] is True
    print(json.dumps({"selftest": "ok"}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Path to a log file. Reads stdin if omitted.")
    parser.add_argument(
        "--window-minutes", type=int, default=5, help="Recent window size for spike detection."
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top error signatures to report."
    )
    parser.add_argument("--selftest", action="store_true", help="Run an internal self-test and exit.")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result = analyze(text, window_minutes=args.window_minutes, top=args.top)
    print(json.dumps(result, indent=2))

    has_errors = result["counts"].get("ERROR", 0) or result["counts"].get("CRITICAL", 0)
    return 1 if (has_errors or result["spike"]["detected"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
