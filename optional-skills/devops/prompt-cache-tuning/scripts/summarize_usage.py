#!/usr/bin/env python3
"""Read usage-only JSONL locally; report token-weighted cache reads, not cost."""

import argparse
import json
import sys
from pathlib import Path


SHAPES = {
    "chat": {
        "input": (("prompt_tokens",), ("input_tokens",)),
        "read": (("prompt_tokens_details", "cached_tokens"), ("cache_read_input_tokens",),
                 ("prompt_cache_hit_tokens",), ("cached_tokens",)),
        "write": (("prompt_tokens_details", "cache_write_tokens"),
                  ("prompt_tokens_details", "cache_creation_input_tokens"),
                  ("cache_creation_input_tokens",), ("cache_write_tokens",)),
    },
    "responses": {
        "input": (("input_tokens",),),
        "read": (("input_tokens_details", "cached_tokens"),),
        "write": (("input_tokens_details", "cache_write_tokens"),
                  ("input_tokens_details", "cache_creation_tokens")),
    },
    "anthropic": {
        "input": (("input_tokens",),),
        "read": (("cache_read_input_tokens",),),
        "write": (("cache_creation_input_tokens",),),
    },
}


def token_count(usage, paths):
    """First present counter wins, including an explicit zero; absent is unknown."""
    for path in paths:
        value = usage
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError("token counters must be non-negative integers")
        return value
    return None


def normalize(record, usage_format):
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    usage = record.get("usage", record)
    if not isinstance(usage, dict):
        raise ValueError("usage must be a JSON object")
    shape = SHAPES[usage_format]
    input_tokens = token_count(usage, shape["input"])
    read = token_count(usage, shape["read"])
    write = token_count(usage, shape["write"])
    total = input_tokens
    if usage_format == "anthropic":
        # Native Anthropic input excludes cache reads and writes. A missing
        # bucket makes the denominator unknown, not an assumed zero.
        total = None if None in (input_tokens, read, write) else input_tokens + read + write
    elif total is not None and (read or 0) + (write or 0) > total:
        raise ValueError("cache buckets exceed the reported prompt total; check --format")
    return {"prompt_tokens": total, "cache_read_tokens": read, "cache_write_tokens": write}


def summarize(lines, usage_format):
    calls = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"line {line_number}: invalid JSON") from None
        try:
            calls.append(normalize(record, usage_format))
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from None
    if not calls:
        raise ValueError("no usage records found")
    measured = [c for c in calls if c["prompt_tokens"] is not None and c["cache_read_tokens"] is not None]
    prompt = sum(c["prompt_tokens"] for c in measured)
    read = sum(c["cache_read_tokens"] for c in measured)
    return {
        "calls": len(calls),
        "measured_calls": len(measured),
        "unmeasured_calls": len(calls) - len(measured),
        "measured_prompt_tokens": prompt,
        "measured_cache_read_tokens": read,
        "cache_read_fraction": read / prompt if prompt else None,
        "calls_with_reported_cache_reads": sum((c["cache_read_tokens"] or 0) > 0 for c in calls),
        "calls_with_missing_cache_reads": sum(c["cache_read_tokens"] is None for c in calls),
        "per_call": calls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="usage-only JSONL file, or - for stdin")
    parser.add_argument("--format", choices=tuple(SHAPES), required=True,
                        help="wire usage shape; use anthropic only for native Messages usage")
    args = parser.parse_args(argv)
    try:
        if args.path == "-":
            report = summarize(sys.stdin, args.format)
        else:
            with Path(args.path).expanduser().open(encoding="utf-8-sig") as stream:
                report = summarize(stream, args.format)
    except (OSError, UnicodeError):
        parser.exit(2, "error: cannot read usage file as UTF-8\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
