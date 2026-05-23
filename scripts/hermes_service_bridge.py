#!/usr/bin/env python3
"""Run or validate the jcode -> Hermes service bridge."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.jcode_bridge.hermes_service import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
