#!/usr/bin/env python3
"""
reindex.py — Rebuild Robin's review index from topic files

Usage:
  python3 reindex.py --state-dir /path/to/data/robin

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import reindex_main


def main():
    reindex_main()


if __name__ == "__main__":
    main()
