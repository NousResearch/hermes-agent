#!/usr/bin/env python3
"""Source-checkout wrapper for the packaged local execution broker."""

import sys
from pathlib import Path

# Direct script execution puts ``scripts/`` rather than the repository root on
# sys.path. The installed service uses ``python -m tools.local_exec_broker``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.local_exec_broker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
