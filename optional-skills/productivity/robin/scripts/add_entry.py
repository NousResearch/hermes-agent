#!/usr/bin/env python3
"""
add_entry.py — Robin add entry script

Usage:
  python3 add_entry.py --state-dir /path/to/data/robin --topic "AI Reasoning" --content "..." --description "..." [--source URL] [--note "..."] [--tags tag1,tag2]

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import add_main


def main():
    add_main()


if __name__ == "__main__":
    main()
