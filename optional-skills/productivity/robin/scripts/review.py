#!/usr/bin/env python3
"""
review.py — Robin review script

Usage:
  python3 review.py --state-dir /path/to/data/robin                    # Scheduled recall; no pending rating
  python3 review.py --state-dir /path/to/data/robin --active-review    # Active review; await a rating
  python3 review.py --state-dir /path/to/data/robin --rate ID 5        # Rate an item (overwrites previous)
  python3 review.py --state-dir /path/to/data/robin --status           # Show review stats without surfacing

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import review_main


def main():
    review_main()


if __name__ == "__main__":
    main()
