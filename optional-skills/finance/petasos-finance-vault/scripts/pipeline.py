#!/usr/bin/env python3
"""Petasos — daily pipeline CLI: fetch mail → parse → classify → append →
verify → balances → state → gates → digest.

Usage: pipeline.py [--dry]   (default live; --dry writes NOTHING, prints the
plan only — INV-1). All configuration is env-driven (PETASOS_*, C7); exit
codes per petasos.pipeline docstring (0 ok/skip · 1 gates/chain · 2 config
· 3 lock).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from petasos.pipeline import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
