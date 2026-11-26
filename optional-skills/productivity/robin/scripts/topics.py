#!/usr/bin/env python3
"""
topics.py — List all Robin topics

Usage:
  python3 topics.py --state-dir /path/to/data/robin         # List all topics with stats
  python3 topics.py --state-dir /path/to/data/robin --json  # Output as JSON

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import topics_main


def main():
    topics_main()


if __name__ == "__main__":
    main()
