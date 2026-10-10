#!/usr/bin/env python3
"""
doctor.py - Check a Robin library for common health issues

Usage:
  python3 doctor.py --state-dir /path/to/data/robin
  python3 doctor.py --state-dir /path/to/data/robin --json

Or set:
  ROBIN_STATE_DIR=/path/to/data/robin
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robin.cli import doctor_main


def main():
    doctor_main()


if __name__ == "__main__":
    main()
