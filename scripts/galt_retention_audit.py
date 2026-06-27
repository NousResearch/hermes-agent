#!/usr/bin/env python3
"""CLI wrapper for cleanup.retention_audit."""
import sys

from cleanup.retention_audit import main

if __name__ == "__main__":
    argv = sys.argv[1:] or ["autonomous-run"]
    raise SystemExit(main(argv))
