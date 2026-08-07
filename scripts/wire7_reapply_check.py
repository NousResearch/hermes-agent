#!/usr/bin/env python3
"""Wire #7 re-apply check (Mothership #47 Ruling C).

DETECTOR, not a patcher. Patch-on-launch is rejected by ruling. This script
confirms the caller-side warn/block severity split is still present in
agent/prompt_builder.py. If an upstream pull stomps the fix, this exits
nonzero and screams so the operator re-applies by hand (until the upstream
PR makes it moot).

Run after any `git pull` of hermes-agent main.
"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "agent" / "prompt_builder.py"

# Two sentinels that together prove the fix is intact.
SENTINELS = (
    "_WARN_CLASS_FINDINGS = frozenset",          # the warn-set definition
    "block_class = [f for f in findings",         # the default-block split logic
)

def main() -> int:
    if not TARGET.exists():
        print(f"WIRE7 RE-APPLY CHECK: FAIL — target missing: {TARGET}", file=sys.stderr)
        return 2

    text = TARGET.read_text()
    missing = [s for s in SENTINELS if s not in text]

    if missing:
        print("=" * 70, file=sys.stderr)
        print("WIRE7 RE-APPLY CHECK: FAIL", file=sys.stderr)
        print("The caller-side warn/block severity split has been STOMPED.", file=sys.stderr)
        print("An upstream pull likely overwrote agent/prompt_builder.py.", file=sys.stderr)
        print("Missing sentinels:", file=sys.stderr)
        for s in missing:
            print(f"    - {s!r}", file=sys.stderr)
        print("", file=sys.stderr)
        print("RE-APPLY the Wire #7 fix before running any agent. Context-scope", file=sys.stderr)
        print("C2 findings will hard-block whole SOUL/AGENTS files until fixed.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 1

    print("WIRE7 RE-APPLY CHECK: PASS — warn/block severity split intact.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
