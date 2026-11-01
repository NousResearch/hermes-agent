#!/usr/bin/env python3
"""Verify proxy-tester code integrity — checks that the credential URL bug is fixed.

Ensures build_session() includes the scheme:// prefix when constructing
credentialed proxy URLs. Run this after any skill update or when debugging
proxy connection failures.

Exit codes:
  0 — all checks passed
  1 — bug detected or file unparseable
"""

import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = SKILL_DIR / "scripts" / "proxy_tester.py"

def check_credential_url_fix() -> bool:
    """Verify that the credentialed proxy URL includes the scheme."""
    code = SCRIPT_PATH.read_text()

    # The buggy pattern: f"{username}:{password}@{host}:{port}"
    buggy_pattern = r'f"{proxy_cfg\[["\']username["\']\]:.*?@'
    # The fixed pattern: f"{scheme}://{username}:{password}@{host}:{port}"
    fixed_pattern = r'f"{proxy_cfg\[["\']scheme["\']\]}://'

    has_buggy = re.search(buggy_pattern, code) is not None
    has_fixed = re.search(fixed_pattern, code) is not None

    if has_buggy and not has_fixed:
        print(f"❌ Bug detected in {SCRIPT_PATH.name}: credentialed proxy URL missing scheme://")
        print("   Expected: f\"{scheme}://{username}:{password}@{host}:{port}\"")
        return False
    elif has_fixed:
        print(f"✅ Credential URL fix verified in {SCRIPT_PATH.name}")
        return True
    else:
        print(f"⚠ Could not determine fix status — pattern not found in {SCRIPT_PATH.name}")
        print("   Manual review recommended.")
        return False

if __name__ == "__main__":
    ok = check_credential_url_fix()
    sys.exit(0 if ok else 1)
