#!/usr/bin/env python3
"""Verify upgrade-critical final-response contracts.

This deterministic script complements focused pytest coverage by checking the
source-level wiring that has regressed during upstream rebases: final-only
classic CLI config, the display-policy import/use, and route-bar finalizer
persistence ownership.  It exits non-zero if those seams disappear.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    (
        "hermes_cli/config.py",
        '"assistant_body_streaming": False',
        "DEFAULT_CONFIG must keep classic assistant answer-body streaming final-only by default",
    ),
    (
        "cli.py",
        "decide_final_response_display",
        "classic CLI must route final panel decisions through the durable policy helper",
    ),
    (
        "agent/turn_finalizer.py",
        "apply_route_depth_bar",
        "turn finalizer must apply the runtime-owned route/depth bar",
    ),
    (
        "agent/turn_finalizer.py",
        "_ensure_canonical_final_message",
        "turn finalizer must persist the transformed canonical assistant message",
    ),
    (
        "run_agent.py",
        "_current_visible_streamed_assistant_text",
        "agent must track received stream text separately from visibly rendered stream text",
    ),
]


def main() -> int:
    failures: list[str] = []
    for rel, needle, message in CHECKS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if needle not in text:
            failures.append(f"{rel}: missing {needle!r} — {message}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("OK: final-response display and route-bar persistence contracts are wired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
