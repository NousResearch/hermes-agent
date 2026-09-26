#!/usr/bin/env python3
"""Hermes CI fixture: prints a pass-shaped decision, then crashes.

Regression fixture for the reviewer-found gap in
SubprocessEvaluatorPolicy.__call__: a process that writes a well-formed
"allowed": true JSON payload to stdout and then dies with a nonzero exit
code must NOT be treated as an authoritative passed decision.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.load(sys.stdin)
    request_id = request.get("request_id") if isinstance(request, dict) else None
    decision = {
        "schema_version": 1,
        "request_id": request_id,
        "status": "passed",
        "allowed": True,
        "final_text": "unverified",
        "evidence_ref": "fixture:evaluator-policy-crash",
        "reason": None,
    }
    print(json.dumps(decision, ensure_ascii=False))
    sys.stdout.flush()
    # Simulate a crash AFTER the pass-shaped output was already written.
    return 17


if __name__ == "__main__":
    raise SystemExit(main())
