#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from agent.minions_reference import enqueue_job


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"error": "expected JSON envelope on stdin"}))
        return 1
    envelope = json.loads(raw)
    ack = enqueue_job(envelope)
    print(json.dumps(ack, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
