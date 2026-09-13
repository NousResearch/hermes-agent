#!/usr/bin/env python3
"""Benchmark the Unicode-Cf guard used by Codex request preflight.

Reports three scopes separately so numbers are not mistaken for one another:
(1) the detector alone, (2) token neutralization, and (3) complete request
preflight across instructions, user/assistant content, tool arguments/results,
and a tool schema. Warm-up runs are excluded from repeated samples.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import timeit
import unicodedata
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.codex_responses_adapter as adapter  # noqa: E402


def _legacy_has_format_control(text: str) -> bool:
    return any(unicodedata.category(char) == "Cf" for char in text)


def _legacy_neutralize_harmony_tokens(text: str) -> str:
    original = adapter._has_format_control
    adapter._has_format_control = _legacy_has_format_control
    try:
        return adapter._neutralize_harmony_tokens(text)
    finally:
        adapter._has_format_control = original


def _request(text: str) -> dict:
    return {
        "model": "gpt-5-codex",
        "instructions": text,
        "input": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": text},
            {
                "type": "function_call",
                "call_id": "call_args",
                "name": "terminal",
                "arguments": text,
            },
            {
                "type": "function_call_output",
                "call_id": "call_result",
                "output": text,
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "inspect_text",
                "description": text,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": text}},
                },
            }
        ],
        "store": False,
    }


def _preflight_with(detector: Callable[[str], bool], request: dict) -> None:
    original = adapter._has_format_control
    adapter._has_format_control = detector
    try:
        adapter._preflight_codex_api_kwargs(request, sanitize_harmony_tokens=True)
    finally:
        adapter._has_format_control = original


def _measure(
    function: Callable[[], object], *, warmup: int, repeats: int, loops: int
) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    samples = timeit.repeat(function, repeat=repeats, number=loops)
    per_call_ms = [sample * 1000 / loops for sample in samples]
    return statistics.median(per_call_ms), statistics.pstdev(per_call_ms)


def _report(
    scope: str,
    old: Callable[[], object],
    new: Callable[[], object],
    args: argparse.Namespace,
) -> None:
    old_ms, old_stddev = _measure(
        old, warmup=args.warmup, repeats=args.repeats, loops=args.loops
    )
    new_ms, new_stddev = _measure(
        new, warmup=args.warmup, repeats=args.repeats, loops=args.loops
    )
    speedup = old_ms / new_ms if new_ms else float("inf")
    print(
        f"scope={scope} old={old_ms:.3f}ms±{old_stddev:.3f} "
        f"new={new_ms:.3f}ms±{new_stddev:.3f} speedup={speedup:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-kib", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loops", type=int, default=10)
    args = parser.parse_args()

    base = "code <tag|value> and non-ASCII Türkçe text " * 2048
    text = (base * ((args.size_kib * 1024 // len(base)) + 1))[: args.size_kib * 1024]
    request = _request(text)
    print(
        f"input={len(text) / 1024:.0f}KiB warmup={args.warmup} "
        f"repeats={args.repeats} loops={args.loops} unicode={unicodedata.unidata_version}"
    )
    _report(
        "guard-only",
        lambda: _legacy_has_format_control(text),
        lambda: adapter._has_format_control(text),
        args,
    )
    _report(
        "neutralizer-only",
        lambda: _legacy_neutralize_harmony_tokens(text),
        lambda: adapter._neutralize_harmony_tokens(text),
        args,
    )
    _report(
        "complete-request-preflight",
        lambda: _preflight_with(_legacy_has_format_control, request),
        lambda: _preflight_with(adapter._has_format_control, request),
        args,
    )


if __name__ == "__main__":
    main()
