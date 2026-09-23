#!/usr/bin/env python3
"""
entries.py - Move or delete Robin entries

Usage:
  python3 entries.py --state-dir /path/to/data/robin --delete ENTRY_ID
  python3 entries.py --state-dir /path/to/data/robin --move ENTRY_ID --topic "New Topic"

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import entries_main


def main():
    entries_main()


if __name__ == "__main__":
    main()
