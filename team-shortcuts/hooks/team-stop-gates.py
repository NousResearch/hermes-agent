#!/usr/bin/env python3
"""Run the three team response gates from one Stop hook."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HOOKS_DIR = Path(__file__).resolve().parent
GATES = (
    "validate-thai-language.py",
    "enforce-codex-review.py",
    "enforce-prompt-evidence.py",
)


def main() -> int:
    payload = sys.stdin.read()
    blockers: list[str] = []
    for name in GATES:
        path = HOOKS_DIR / name
        if not path.is_file():
            blockers.append(f"ไม่พบด่านตรวจ {path}")
            continue
        proc = subprocess.run(
            [sys.executable, str(path)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 2:
            blockers.append((proc.stderr or proc.stdout or name).strip())
        elif proc.returncode != 0:
            blockers.append(f"ด่าน {name} ทำงานผิดปกติ exit={proc.returncode}")

    if blockers:
        print("[Hermes Team Stop Gate] ไม่อนุญาตให้ส่งคำตอบรอบนี้", file=sys.stderr)
        for item in blockers:
            print(f"- {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
