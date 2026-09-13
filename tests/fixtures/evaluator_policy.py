#!/usr/bin/env python3
"""Hermes CI fixture for the subprocess policy bridge.

This deliberately mirrors the JSON decision contract without importing the
Evaluator repository, so the Hermes test suite remains self-contained.
"""
from __future__ import annotations

import json
import sys



def main() -> int:
    request = json.load(sys.stdin)
    final_text = request.get("final_text") if isinstance(request, dict) else None
    passed = isinstance(final_text, str) and "evidence" in final_text
    decision = {
        "schema_version": 1,
        "request_id": request.get("request_id") if isinstance(request, dict) else None,
        "status": "passed" if passed else "blocked",
        "allowed": passed,
        "final_text": final_text if passed else None,
        "evidence_ref": "fixture:evaluator-policy",
        "reason": None if passed else "required marker is missing",
        "attempt_count": 1,
        "correction_count": 0,
        "issues": [] if passed else [{"message": "required marker is missing"}],
    }
    print(json.dumps(decision, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
