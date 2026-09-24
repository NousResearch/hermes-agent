"""Suspicious: standalone script with no imports or package integration.

This file is isolated — it doesn't import anything from the project
and nothing imports it. A UA scan should flag it as an orphan."""

import os
import sys

def suspicious_operation():
    """Does something standalone. No one calls this."""
    base = os.path.expanduser("~")
    return f"Working from {base}"

if __name__ == "__main__":
    print(suspicious_operation())
    sys.exit(0)
