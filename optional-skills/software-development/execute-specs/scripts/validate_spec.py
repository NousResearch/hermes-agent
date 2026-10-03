#!/usr/bin/env python3
"""
Validation script for execute-specs skill.
Verifies that a SPEC SDD file exists, is in 'Approved' status,
and contains numbered requirements before starting TDD execution.
"""

import sys
import os
import re
from pathlib import Path

def validate_spec(spec_path: str) -> bool:
    path = Path(spec_path)
    if not path.exists():
        print(f"❌ Error: SPEC file not found at '{spec_path}'", file=sys.stderr)
        return False

    content = path.read_text(encoding="utf-8")
    
    # Check status
    status_match = re.search(r'Status\s*\|\s*`?([A-Za-z]+)`?', content, re.IGNORECASE)
    if not status_match:
        status_match = re.search(r'status:\s*["\']?([A-Za-z]+)["\']?', content, re.IGNORECASE)

    status = status_match.group(1).strip() if status_match else "Unknown"
    if status.lower() != "approved":
        print(f"❌ Error: SPEC status is '{status}'. Must be 'Approved' before running execute-specs.", file=sys.stderr)
        return False

    # Check for requirements
    reqs = re.findall(r'RF-\d+', content)
    if not reqs:
        print("⚠️ Warning: No numbered requirements (RF-###) found in SPEC.", file=sys.stderr)
    else:
        print(f"✅ Found {len(set(reqs))} unique requirement tag(s): {', '.join(sorted(set(reqs)))}")

    print(f"✅ SPEC '{path.name}' is Approved and ready for execution.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_spec.py <path-to-spec-file>")
        sys.exit(1)

    success = validate_spec(sys.argv[1])
    sys.exit(0 if success else 1)
